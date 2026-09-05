#!/usr/bin/env python3
"""Interactive ZED mini capture with live YOLO detection overlay, for building
a query-frame test set for the pinhole localization script.

For every detection this also reads the ZED's own stereo depth at the
detection centroid. That's not needed for the pinhole method itself, but
since there's no robot in this test to provide ground truth, the ZED's
stereo depth at the same pixel is a very useful independent reference to
check the monocular pinhole depth estimate against later.

Controls (with the live preview window focused):
    s  -- save current frame (image + detections + pose)
    q  -- quit and write index.json

Usage:
    python capture_query_yolo.py --outdir query_capture --weights fixtures.pt

Output layout:
    outdir/
      intrinsics.json
      index.json
      frames/frame_00000.png ...
      poses/frame_00000.json ...       -- 4x4 camera-to-world (ZED tracking), if available
      detections/frame_00000.json ...  -- list of detections for that frame
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
    from ultralytics import YOLO
except ImportError as e:
    raise SystemExit("This script needs ultralytics: pip install ultralytics") from e


RESOLUTIONS = {
    "HD720": sl.RESOLUTION.HD720,
    "HD1080": sl.RESOLUTION.HD1080,
    "HD2K": sl.RESOLUTION.HD2K,
}


def open_zed(resolution):
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = RESOLUTIONS[resolution]
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed: {status}")

    status = zed.enable_positional_tracking(sl.PositionalTrackingParameters())
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"  warning: positional tracking failed to start ({status}); "
              f"poses will not be saved for this session")
    return zed


def get_intrinsics(zed):
    calib = zed.get_camera_information().camera_configuration
    left = calib.calibration_parameters.left_cam
    return {
        "fx": left.fx, "fy": left.fy, "cx": left.cx, "cy": left.cy,
        "width": calib.resolution.width, "height": calib.resolution.height,
    }


def depth_at_pixel(depth_np, u, v, patch=2):
    """Median depth (meters) in a small patch around (u, v), ignoring invalid values."""
    h, w = depth_np.shape
    u0, u1 = max(0, u - patch), min(w, u + patch + 1)
    v0, v1 = max(0, v - patch), min(h, v + patch + 1)
    patch_vals = depth_np[v0:v1, u0:u1]
    valid = patch_vals[np.isfinite(patch_vals) & (patch_vals > 0)]
    return float(np.median(valid)) if valid.size else None


def draw_overlay(bgr, detections):
    out = bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox_xyxy"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        u, v = d["centroid_px"]
        cv2.circle(out, (u, v), 4, (0, 0, 255), -1)
        label = f"{d['class_name']} {d['confidence']:.2f}"
        if d.get("zed_depth_m") is not None:
            label += f" | zed_z={d['zed_depth_m']:.2f}m"
        cv2.putText(out, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--weights", required=True, help="path to your trained YOLO .pt weights")
    ap.add_argument("--resolution", choices=RESOLUTIONS.keys(), default="HD1080")
    ap.add_argument("--conf_thresh", type=float, default=0.4)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "frames").mkdir(parents=True, exist_ok=True)
    (outdir / "poses").mkdir(parents=True, exist_ok=True)
    (outdir / "detections").mkdir(parents=True, exist_ok=True)

    zed = open_zed(args.resolution)
    intrinsics = get_intrinsics(zed)
    with open(outdir / "intrinsics.json", "w") as f:
        json.dump(intrinsics, f, indent=2)

    model = YOLO(args.weights)

    runtime = sl.RuntimeParameters()
    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    pose_holder = sl.Pose()

    index = []
    frame_idx = 0

    print("Live preview open. Press 's' to save a query frame, 'q' to quit.")
    try:
        while True:
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            bgr = cv2.cvtColor(image_mat.get_data(), cv2.COLOR_BGRA2BGR)
            depth_np = depth_mat.get_data()

            results = model.predict(bgr, conf=args.conf_thresh, verbose=False)[0]
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                u, v = (x1 + x2) // 2, (y1 + y2) // 2
                cls_id = int(box.cls[0])
                detections.append({
                    "class_id": cls_id,
                    "class_name": model.names[cls_id],
                    "confidence": float(box.conf[0]),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "centroid_px": [u, v],
                    "zed_depth_m": depth_at_pixel(depth_np, u, v),
                })

            cv2.imshow("YOLO query capture (s=save, q=quit)", draw_overlay(bgr, detections))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key != ord("s"):
                continue

            name = f"frame_{frame_idx:05d}"
            cv2.imwrite(str(outdir / "frames" / f"{name}.png"), bgr)
            with open(outdir / "detections" / f"{name}.json", "w") as f:
                json.dump(detections, f, indent=2)

            pose_holder_state = zed.get_position(pose_holder, sl.REFERENCE_FRAME.WORLD)
            if pose_holder_state == sl.POSITIONAL_TRACKING_STATE.OK:
                T = np.array(pose_holder.pose_data(sl.Transform()).m, dtype=np.float64)
                with open(outdir / "poses" / f"{name}.json", "w") as f:
                    json.dump({"T_camera_to_world": T.tolist()}, f, indent=2)

            index.append({"frame": name, "n_detections": len(detections)})
            print(f"  saved {name}  ({len(detections)} detection(s))")
            frame_idx += 1
    finally:
        cv2.destroyAllWindows()
        zed.disable_positional_tracking()
        zed.close()

    with open(outdir / "index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"Done. {frame_idx} query frame(s) saved under {outdir}/")


if __name__ == "__main__":
    main()