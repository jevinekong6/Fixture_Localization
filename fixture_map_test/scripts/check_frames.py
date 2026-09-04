#!/usr/bin/env python3
"""Standalone frame-convention and range-calibration checker. No ROS needed.

Run it before you trust anything RViz shows you:

    python3 scripts/check_frames.py

and, once you have a tape measure on a real fixture:

    python3 scripts/check_frames.py \\
        --bbox 812 275 96 94 --real-width 0.09 --true-range 1.85 \\
        --fx 700 --cx 640 --cy 360

Exits nonzero if any check fails.

Section 4 is the reason this script exists. When a landmark lands in the wrong
place there are two candidate culprits -- a wrong ``width_m`` and a wrong frame
-- and they are indistinguishable from the marker alone. Section 4 prints the
``width_m`` that WOULD have made the estimate exact. If that number is a
plausible physical size for the thing the detector is boxing, your frames are
fine and your config is wrong. If it is absurd (three times the real fixture,
or negative in spirit), the geometry is not your problem and you should be
looking at transforms.
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np

# The constant body->optical rotation, as a (x, y, z, w) quaternion. This is
# the same quaternion that appears in launch/tripod_test.launch.py. It is a
# fixed property of the ROS convention pair, not a tunable.
Q_BODY_TO_OPTICAL = (-0.5, 0.5, -0.5, 0.5)

TOL = 1e-9
_failures = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global _failures
    if not ok:
        _failures += 1
    status = "PASS" if ok else "FAIL"
    line = f"  [{status}] {label}"
    if detail:
        line += f"\n           {detail}"
    print(line)


def quat_to_matrix(q) -> np.ndarray:
    """(x, y, z, w) -> 3x3 rotation matrix.

    For a TF whose parent is the body frame and child is the optical frame,
    this matrix maps a vector expressed in OPTICAL coordinates into BODY
    coordinates.
    """
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def yaw_matrix(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def backproject(u, v, z, fx, fy, cx, cy) -> np.ndarray:
    """Optical-frame back-projection: +x right, +y down, +z forward."""
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, float(z)])


# --------------------------------------------------------------------------- #
# 1. Optical back-projection
# --------------------------------------------------------------------------- #
def section_optical() -> None:
    print("\n1. Optical back-projection (+x right, +y DOWN, +z FORWARD)")
    fx = fy = 700.0
    cx, cy = 640.0, 360.0
    d = 2.5

    p = backproject(cx, cy, d, fx, fy, cx, cy)
    check("on-axis fixture back-projects to (0, 0, d)",
          abs(p[0]) < TOL and abs(p[1]) < TOL and abs(p[2] - d) < TOL,
          f"got [{p[0]:+.6f} {p[1]:+.6f} {p[2]:+.6f}], want [+0.000000 +0.000000 {d:+.6f}]")

    p = backproject(cx + 100.0, cy, d, fx, fy, cx, cy)
    check("fixture RIGHT of centre gives +x",
          p[0] > 0.0 and abs(p[1]) < TOL,
          f"u = cx + 100 -> x = {p[0]:+.4f} m")

    p = backproject(cx, cy + 100.0, d, fx, fy, cx, cy)
    check("fixture BELOW centre gives +y (y is down)",
          p[1] > 0.0 and abs(p[0]) < TOL,
          f"v = cy + 100 -> y = {p[1]:+.4f} m")


# --------------------------------------------------------------------------- #
# 2. The body->optical quaternion constant
# --------------------------------------------------------------------------- #
def section_quaternion() -> np.ndarray:
    print("\n2. Body->optical rotation, quaternion "
          f"(x, y, z, w) = {Q_BODY_TO_OPTICAL}")
    R = quat_to_matrix(Q_BODY_TO_OPTICAL)

    z_opt = R @ np.array([0.0, 0.0, 1.0])
    check("optical +z maps to body +x (forward)",
          np.allclose(z_opt, [1.0, 0.0, 0.0], atol=1e-9),
          f"[0 0 1]_optical -> [{z_opt[0]:+.3f} {z_opt[1]:+.3f} {z_opt[2]:+.3f}]_body")

    y_opt = R @ np.array([0.0, 1.0, 0.0])
    check("optical +y maps to body -z (down)",
          np.allclose(y_opt, [0.0, 0.0, -1.0], atol=1e-9),
          f"[0 1 0]_optical -> [{y_opt[0]:+.3f} {y_opt[1]:+.3f} {y_opt[2]:+.3f}]_body")

    x_opt = R @ np.array([1.0, 0.0, 0.0])
    check("optical +x maps to body -y (right)",
          np.allclose(x_opt, [0.0, -1.0, 0.0], atol=1e-9),
          f"[1 0 0]_optical -> [{x_opt[0]:+.3f} {x_opt[1]:+.3f} {x_opt[2]:+.3f}]_body")

    check("rotation is orthonormal with det +1",
          np.allclose(R @ R.T, np.eye(3), atol=1e-9)
          and abs(np.linalg.det(R) - 1.0) < 1e-9,
          f"det = {np.linalg.det(R):+.6f}")
    return R


# --------------------------------------------------------------------------- #
# 3. End to end
# --------------------------------------------------------------------------- #
def section_end_to_end(R_body_optical: np.ndarray) -> None:
    print("\n3. End to end: camera at map (0, 0, 1.2), yaw 0, fixture on axis at 2.5 m")
    fx = fy = 700.0
    cx, cy = 640.0, 360.0
    d = 2.5
    t_map_body = np.array([0.0, 0.0, 1.2])
    R_map_body = yaw_matrix(0.0)

    p_opt = backproject(cx, cy, d, fx, fy, cx, cy)
    p_body = R_body_optical @ p_opt
    p_map = R_map_body @ p_body + t_map_body

    print(f"     optical [{p_opt[0]:+.3f} {p_opt[1]:+.3f} {p_opt[2]:+.3f}]"
          f"  ->  body [{p_body[0]:+.3f} {p_body[1]:+.3f} {p_body[2]:+.3f}]"
          f"  ->  map [{p_map[0]:+.3f} {p_map[1]:+.3f} {p_map[2]:+.3f}]")

    check("fixture lands on map +x (straight ahead of the tripod)",
          p_map[0] > 0.0 and abs(p_map[0] - d) < 1e-9,
          f"map x = {p_map[0]:+.4f} m, want {d:+.4f} m")
    check("no lateral offset in map y",
          abs(p_map[1]) < 1e-9, f"map y = {p_map[1]:+.6f} m")
    check("map z unchanged at the camera height",
          abs(p_map[2] - t_map_body[2]) < 1e-9,
          f"map z = {p_map[2]:+.4f} m, want {t_map_body[2]:+.4f} m")

    # A yawed camera is the cheapest test that the rotation composes correctly.
    p_map_90 = yaw_matrix(math.pi / 2.0) @ p_body + t_map_body
    check("yaw +90 deg swings the same fixture onto map +y",
          abs(p_map_90[0]) < 1e-9 and abs(p_map_90[1] - d) < 1e-9,
          f"map [{p_map_90[0]:+.3f} {p_map_90[1]:+.3f} {p_map_90[2]:+.3f}]")


# --------------------------------------------------------------------------- #
# 4. Range calibration against a tape measure
# --------------------------------------------------------------------------- #
def section_range(args) -> None:
    print("\n4. Range check against tape measure")
    u, v, w_px, h_px = args.bbox
    fx = args.fx
    fy = args.fy if args.fy is not None else args.fx
    cx, cy = args.cx, args.cy
    w_real = args.real_width
    truth = args.true_range

    if w_px <= 0.0:
        check("bbox width is positive", False, f"got {w_px}")
        return

    z = fx * w_real / w_px
    sigma = (z * z / (fx * w_real)) * args.bbox_sigma_px
    p_opt = backproject(u, v, z, fx, fy, cx, cy)
    err = z - truth
    pct = 100.0 * err / truth if truth > 0.0 else float("nan")

    # The width_m that would have made the estimate land exactly on the tape.
    w_exact = truth * w_px / fx

    print(f"     bbox        centre ({u:.1f}, {v:.1f}) px, {w_px:.1f} x {h_px:.1f} px")
    print(f"     estimate    {z:.3f} +/- {sigma:.3f} m   (1 sigma, from "
          f"{args.bbox_sigma_px:.1f} px of box noise)")
    print(f"     optical     [{p_opt[0]:+.3f} {p_opt[1]:+.3f} {p_opt[2]:+.3f}] m")
    print(f"     tape        {truth:.3f} m")
    print(f"     error       {err:+.3f} m  ({pct:+.1f}%)")
    print(f"     width_m that would have made this exact:  {w_exact:.4f} m "
          f"(configured: {w_real:.4f} m, ratio {w_exact / w_real:.3f})")
    print("     If that number is a plausible extent for what the detector boxes,")
    print("     the frames are fine and the config is wrong. If it is absurd, stop")
    print("     tuning sizes and go look at the transforms.")

    check(f"estimate is within 1 sigma of the tape measure ({sigma:.3f} m)",
          abs(err) <= sigma,
          f"|{err:+.3f}| vs sigma {sigma:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("CX", "CY", "W", "H"),
                    help="detection box: centre x, centre y, width, height, in pixels")
    ap.add_argument("--real-width", type=float,
                    help="configured width_m for that class, in metres")
    ap.add_argument("--true-range", type=float,
                    help="tape-measured range to the fixture, in metres")
    ap.add_argument("--fx", type=float, help="focal length in pixels")
    ap.add_argument("--fy", type=float, default=None, help="defaults to fx")
    ap.add_argument("--cx", type=float, help="principal point x, pixels")
    ap.add_argument("--cy", type=float, help="principal point y, pixels")
    ap.add_argument("--bbox-sigma-px", type=float, default=3.0,
                    help="assumed 1-sigma bounding-box noise, pixels (default 3)")
    args = ap.parse_args()

    print("fixture_map_test frame checks")
    section_optical()
    R = section_quaternion()
    section_end_to_end(R)

    wants_range = any(v is not None for v in
                      (args.bbox, args.real_width, args.true_range, args.fx, args.cx, args.cy))
    if wants_range:
        missing = [name for name, val in (
            ("--bbox", args.bbox), ("--real-width", args.real_width),
            ("--true-range", args.true_range), ("--fx", args.fx),
            ("--cx", args.cx), ("--cy", args.cy)) if val is None]
        if missing:
            print("\n4. Range check skipped -- missing " + " ".join(missing))
            return 2
        section_range(args)
    else:
        print("\n4. Range check skipped (pass --bbox --real-width --true-range "
              "--fx --cx --cy to run it)")

    print(f"\n{'FAILED' if _failures else 'ALL CHECKS PASSED'}"
          f"{f' -- {_failures} check(s) failed' if _failures else ''}")
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
