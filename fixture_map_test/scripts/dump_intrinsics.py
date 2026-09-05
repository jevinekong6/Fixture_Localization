#!/usr/bin/env python3
"""Write the ZED's calibration to intrinsics.json, for check_pose.py.

Run it wherever the camera actually works -- which, without USB passthrough,
means Windows rather than WSL. The JSON is a plain text file; copy it across.

    python3 scripts/dump_intrinsics.py --resolution HD720
    python3 scripts/dump_intrinsics.py --svo bench.svo
    python3 scripts/dump_intrinsics.py --board-half-height 0.24

Output is the same shape capture_map_3D.py and capture_query.py already write
-- {fx, fy, cx, cy, width, height} -- so check_pose.py reads it directly:

    python3 scripts/check_pose.py --intrinsics intrinsics.json --cam-x -0.60

WHY NOT JUST HARDCODE THE NUMBERS
---------------------------------
The ZED calibrates itself at every startup and reports per-resolution values,
so there is no single correct fx to write down. And these numbers are not
decoration: fy sets the vertical field of view, which sets how far back the
tripod has to stand for the board to fit in frame. Guessing fx=700 when the
camera reports 540 moves that answer by more than 10 cm.

THE RESOLUTION MUST MATCH WHAT YOU ACTUALLY RUN
-----------------------------------------------
Intrinsics scale with resolution. Capturing this at HD720 and then running the
ROS wrapper at HD1080 gives you numbers that are wrong by 1.5x in every term.
Whatever you pass here must be the resolution the pipeline runs at.

RECTIFIED, NOT RAW
------------------
This reads calibration_parameters (rectified), not calibration_parameters_raw.
The pipeline consumes image_rect_color and the pinhole model assumes an
undistorted image, so rectified is the correct set -- the same one
capture_map_3D.py reads.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

RESOLUTIONS = ("HD2K", "HD1080", "HD720", "VGA")


def open_camera(svo: str | None, resolution: str):
    """Open a live ZED, or an SVO recording. pyzed is imported lazily so this
    module stays importable (and testable) on a machine without the SDK."""
    try:
        import pyzed.sl as sl
    except ImportError as exc:
        raise SystemExit(
            "pyzed not found. Install the ZED SDK and its Python API "
            "(https://www.stereolabs.com/developers), and run this on the "
            "machine the camera is actually attached to."
        ) from exc

    init = sl.InitParameters()
    init.coordinate_units = sl.UNIT.METER
    # Depth is irrelevant here -- we only want the calibration. Turning it off
    # makes this start in about a second and removes any CUDA requirement.
    init.depth_mode = sl.DEPTH_MODE.NONE

    if svo:
        path = Path(svo)
        if not path.is_file():
            raise SystemExit(f"SVO not found: {path}")
        init.set_from_svo_file(str(path))
        source = f"SVO {path.name}"
    else:
        init.camera_resolution = getattr(sl.RESOLUTION, resolution)
        source = f"live camera @ {resolution}"

    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(
            f"ZED open failed: {status}\n"
            "  - live: is the camera plugged into a USB 3 port, and not held "
            "open by another process?\n"
            "  - under WSL: USB passthrough must be attached first, which "
            "needs a one-time 'usbipd bind' from an admin shell."
        )
    return cam, source


def read_intrinsics(cam, side: str) -> dict:
    """Pull the rectified intrinsics for one eye, plus the image size.

    Handles both SDK layouts: 4.x nests everything under camera_configuration,
    3.x hangs it off camera_information directly.
    """
    info = cam.get_camera_information()
    conf = getattr(info, "camera_configuration", info)      # 4.x, else 3.x
    calib = conf.calibration_parameters                     # RECTIFIED

    eye = calib.left_cam if side == "left" else calib.right_cam

    res = getattr(conf, "resolution", None) or getattr(info, "camera_resolution")
    model = getattr(info, "camera_model", "unknown")

    return {
        "fx": float(eye.fx), "fy": float(eye.fy),
        "cx": float(eye.cx), "cy": float(eye.cy),
        "width": int(res.width), "height": int(res.height),
        # Not read by check_pose.py -- carried so a stale file can be
        # identified later, which matters because intrinsics are per-resolution
        # and per-camera and nothing downstream can tell you they are stale.
        "camera_model": str(model),
        "side": side,
    }


def describe(d: dict, board_half_height: float | None) -> None:
    """Report what the numbers mean for where the tripod has to stand."""
    fx, fy, cx, cy = d["fx"], d["fy"], d["cx"], d["cy"]
    half_h = math.degrees(math.atan(cx / fx))
    half_v = math.degrees(math.atan(cy / fy))

    print(f"  camera      {d.get('camera_model','?')}  ({d['side']} eye, rectified)")
    print(f"  image       {d['width']} x {d['height']}")
    print(f"  intrinsics  fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}")
    print(f"  field of view   horizontal {2*half_h:.1f} deg   vertical {2*half_v:.1f} deg")
    print(f"  half-FOV        h {half_h:.1f} deg   v {half_v:.1f} deg   "
          f"<- vertical is usually the binding one")

    if abs(cx - d["width"] / 2.0) > 0.05 * d["width"] or \
       abs(cy - d["height"] / 2.0) > 0.05 * d["height"]:
        print("  NOTE: the principal point is well off centre. Unusual but not "
              "wrong; just don't substitute width/2 for cx anywhere.")

    if board_half_height:
        need = board_half_height / math.tan(math.radians(half_v))
        print()
        print(f"  For a target half-height of {board_half_height:.2f} m "
              f"(so {2*board_half_height:.2f} m tall):")
        print(f"     minimum standoff for it to FIT in frame:  {need:.2f} m")
        print(f"     with ~20% margin:                         {need*1.2:.2f} m")
        print("     Subtract the fixture plane's x from these to get cam_x.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="intrinsics.json", type=Path,
                    help="where to write the JSON (default intrinsics.json)")
    ap.add_argument("--svo", default=None,
                    help="read from an SVO recording instead of a live camera; "
                         "the recording carries its own resolution, so "
                         "--resolution is ignored")
    ap.add_argument("--resolution", choices=RESOLUTIONS, default="HD720",
                    help="MUST match the resolution the pipeline will run at "
                         "(default HD720)")
    ap.add_argument("--side", choices=("left", "right"), default="left",
                    help="fixture_map_test consumes the LEFT rectified image; "
                         "only change this if you have remapped it")
    ap.add_argument("--board-half-height", type=float, default=0.292,
                    help="half the vertical extent of your fixture board, in "
                         "metres. Prints the minimum tripod standoff.")
    args = ap.parse_args()

    cam, source = open_camera(args.svo, args.resolution)
    try:
        intr = read_intrinsics(cam, args.side)
    finally:
        cam.close()

    # Report BEFORE writing. Opening the camera is the slow, failure-prone
    # part; a bad --out path must not throw away a reading that already
    # succeeded -- the numbers are on screen either way.
    print(f"ZED intrinsics  [{source}]")
    describe(intr, args.board_half_height)
    print()

    out = args.out
    if out.is_dir() or str(out).endswith(("/", "\\")):
        # A directory was given rather than a file. Do the obvious thing
        # instead of raising PermissionError, which is what open() on a
        # directory reports and which reads like a filesystem problem.
        out = out / "intrinsics.json"

    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(intr, f, indent=2)
            f.write("\n")
    except OSError as exc:
        print(f"  COULD NOT WRITE {out}: {exc}")
        print("  The calibration above is still correct -- copy it by hand, or")
        print("  re-run with --out pointing at a writable FILE path.")
        return 1

    print(f"  wrote {out}")
    print()
    print("  Next:")
    print(f"     python3 scripts/check_pose.py --intrinsics {args.out} \\")
    print("         --cam-x <measured> --cam-y <measured> --cam-z <measured>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
