#!/usr/bin/env python3
"""Interactive ZED mini capture session for building the reference point-cloud
map used later for localization testing.


Controls (with the live preview window focused):
    s  -- save current frame as a keyframe (image, depth, pose) and fold its
          points into the running fused map
    q  -- quit, write the final fused map + index.json, and exit

Usage:
    python capture_map_zed.py --outdir map_capture --voxel_size 0.005

Output layout:
    outdir/
      intrinsics.json
      index.json                 -- list of saved keyframes in order
      frames/frame_00000.png ...
      depth/frame_00000.npy ...  -- float32 depth map, meters
      poses/frame_00000.json ... -- 4x4 camera-to-world matrix (ZED tracking)
      map.ply                    -- fused, voxel-downsampled point cloud
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

try:
    import pyzed.sl as sl
except ImportError as e:
    raise SystemExit(
        "pyzed not found. Install the ZED SDK and its Python API "
        "(https://www.stereolabs.com/developers) before running this script."
    ) from e

try:
    import open3d as o3d
except ImportError as e:
    raise SystemExit("This script needs open3d for point-cloud fusion/saving: "
                      "pip install open3d") from e


RESOLUTIONS = {
    "HD720": sl.RESOLUTION.HD720,
    "HD1080": sl.RESOLUTION.HD1080,
    "HD2K": sl.RESOLUTION.HD2K,
}
DEPTH_MODES = {
    "NEURAL": sl.DEPTH_MODE.NEURAL,
    "ULTRA": sl.DEPTH_MODE.ULTRA,
    "QUALITY": sl.DEPTH_MODE.QUALITY,
}


def open_zed(resolution, depth_mode):
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = RESOLUTIONS[resolution]
    init.depth_mode = DEPTH_MODES[depth_mode]
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed: {status}")

    tracking_params = sl.PositionalTrackingParameters()
    status = zed.enable_positional_tracking(tracking_params)
    if status != sl.ERROR_CODE.SUCCESS:
        zed.close()
        raise RuntimeError(f"Positional tracking failed to start: {status}")

    return zed


def get_intrinsics(zed):
    calib = zed.get_camera_information().camera_configuration
    left = calib.calibration_parameters.left_cam
    return {
        "fx": left.fx, "fy": left.fy, "cx": left.cx, "cy": left.cy,
        "width": calib.resolution.width, "height": calib.resolution.height,
    }


def get_pose_matrix(zed, pose_holder):
    """Returns (tracking_state_ok: bool, 4x4 camera-to-world np.ndarray)."""
    state = zed.get_position(pose_holder, sl.REFERENCE_FRAME.WORLD)
    m = pose_holder.pose_data(sl.Transform()).m  # 4x4, row-major
    return state == sl.POSITIONAL_TRACKING_STATE.OK, np.array(m, dtype=np.float64)


def transform_points(points_xyz, T_cam_to_world):
    """points_xyz: [N, 3] in camera frame. Returns [N, 3] in world frame."""
    ones = np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)
    homog = np.hstack([points_xyz, ones])            # [N, 4]
    world = (T_cam_to_world @ homog.T).T              # [N, 4]
    return world[:, :3]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--resolution", choices=RESOLUTIONS.keys(), default="HD1080")
    ap.add_argument("--depth_mode", choices=DEPTH_MODES.keys(), default="NEURAL")
    ap.add_argument("--voxel_size", type=float, default=0.005,
                     help="fused-map downsample voxel size in meters (default 5mm)")
    ap.add_argument("--stride_px", type=int, default=2,
                     help="subsample the per-frame point cloud every Nth pixel before "
                          "fusing, to keep memory/time bounded (default: every 2nd pixel)")
    ap.add_argument("--max_depth_m", type=float, default=3.0,
                     help="drop points farther than this from the camera (noisy stereo tail)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "frames").mkdir(parents=True, exist_ok=True)
    (outdir / "depth").mkdir(parents=True, exist_ok=True)
    (outdir / "poses").mkdir(parents=True, exist_ok=True)

    zed = open_zed(args.resolution, args.depth_mode)
    intrinsics = get_intrinsics(zed)
    with open(outdir / "intrinsics.json", "w") as f:
        json.dump(intrinsics, f, indent=2)

    runtime = sl.RuntimeParameters()
    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    cloud_mat = sl.Mat()
    pose_holder = sl.Pose()

    fused_points, fused_colors = [], []
    index = []
    frame_idx = 0

    print("Live preview open. Press 's' to save a keyframe, 'q' to quit.")
    try:
        while True:
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            bgr = cv2.cvtColor(image_mat.get_data(), cv2.COLOR_BGRA2BGR)
            cv2.imshow("ZED left (s=save keyframe, q=quit)", bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key != ord("s"):
                continue

            tracking_ok, T_cam_to_world = get_pose_matrix(zed, pose_holder)
            if not tracking_ok:
                print(f"  [frame {frame_idx:05d}] tracking not OK yet, skipping save")
                continue

            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            zed.retrieve_measure(cloud_mat, sl.MEASURE.XYZRGBA)
            depth_np = depth_mat.get_data().copy()          # float32, meters, camera frame
            cloud_np = cloud_mat.get_data()                 # [H, W, 4] float32: X,Y,Z,rgba(packed)

            name = f"frame_{frame_idx:05d}"
            cv2.imwrite(str(outdir / "frames" / f"{name}.png"), bgr)
            np.save(outdir / "depth" / f"{name}.npy", depth_np)
            with open(outdir / "poses" / f"{name}.json", "w") as f:
                json.dump({"T_camera_to_world": T_cam_to_world.tolist()}, f, indent=2)

            # Fold this frame's points into the fused map.
            xyz = cloud_np[::args.stride_px, ::args.stride_px, :3].reshape(-1, 3)
            valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0) & (xyz[:, 2] < args.max_depth_m)
            xyz_valid = xyz[valid]
            xyz_world = transform_points(xyz_valid, T_cam_to_world)

            rgba_packed = cloud_np[::args.stride_px, ::args.stride_px, 3].reshape(-1)[valid]
            rgba_bytes = rgba_packed.copy().view(np.uint8).reshape(-1, 4)
            rgb01 = rgba_bytes[:, :3].astype(np.float64) / 255.0

            fused_points.append(xyz_world)
            fused_colors.append(rgb01)
            index.append({"frame": name, "n_points_added": int(xyz_world.shape[0])})

            print(f"  saved {name}  (+{xyz_world.shape[0]} points, "
                  f"fused total so far: {sum(p.shape[0] for p in fused_points)})")
            frame_idx += 1
    finally:
        cv2.destroyAllWindows()
        zed.disable_positional_tracking()
        zed.close()

    if fused_points:
        all_xyz = np.concatenate(fused_points, axis=0)
        all_rgb = np.concatenate(fused_colors, axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_xyz)
        pcd.colors = o3d.utility.Vector3dVector(all_rgb)
        if args.voxel_size > 0:
            pcd = pcd.voxel_down_sample(args.voxel_size)
        o3d.io.write_point_cloud(str(outdir / "map.ply"), pcd)
        print(f"\nWrote fused map: {outdir/'map.ply'} ({len(pcd.points)} points after downsampling)")
    else:
        print("\nNo keyframes were saved -- nothing to fuse.")

    with open(outdir / "index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"Done. {frame_idx} keyframe(s) saved under {outdir}/")


if __name__ == "__main__":
    main()