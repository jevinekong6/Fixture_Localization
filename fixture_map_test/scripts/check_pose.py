#!/usr/bin/env python3
"""Pre-flight check of YOUR numbers: tripod pose + apriori map + intrinsics.

No ROS needed. Run it after you edit config/apriori_map.yaml or move the
tripod, BEFORE any hardware is powered on.

    python3 scripts/check_pose.py --intrinsics query_capture/intrinsics.json \\
        --cam-z 0.38 --cam-pitch 10 --degrees

WHAT THIS IS FOR, AND HOW IT DIFFERS FROM check_frames.py
---------------------------------------------------------
check_frames.py validates the PACKAGE's conventions. It takes no inputs and
always gives the same answer: is the optical convention right, is the
body->optical quaternion right, does a point compose through to the map frame
correctly. It is a test of the code.

check_pose.py validates YOUR NUMBERS -- the measured tripod pose and the
measured fixture positions, which nothing else in this package checks. It runs
each apriori landmark BACKWARDS through the same transform chain the node uses
forwards, projects it into the image, and asks: given where you say the camera
is and where you say the fixtures are, would the camera actually SEE them, and
could an observation of them ever be CONFIRMED?

Both are cheap. Run both.

THE CHECK THAT MATTERS MOST
---------------------------
sigma_Z > confirm_radius means the landmark can NEVER be confirmed, even by a
perfect detection of a perfectly-placed fixture, because pixel noise alone
exceeds the confirmation gate at that range. Since

    sigma_Z = (Z^2 / (fx * W)) * sigma_px

the confirmable range for a 9 cm fixture at fx=700 with 3 px of box noise is
only about 1.8 m -- while max_range defaults to 6.0 m. Between those two
numbers is a band where observations are accepted, associated, and then
essentially guaranteed to FLAG with everything working correctly. This script
tells you whether your fixtures sit in that band before you spend an afternoon
debugging it.

WHAT THIS CANNOT CHECK
----------------------
Whether width_m is correct (that needs a tape measure -- use
check_frames.py --bbox --true-range), whether the detector will fire at all,
occlusion, lighting, or motion blur. This proves your numbers are
self-consistent and the fixtures are geometrically visible. It does not prove
the vision works.

Exits nonzero if any FAIL check trips.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent          # scripts/ and config/ are siblings, in the
                                  # source tree AND in share/fixture_map_test/

sys.path.insert(0, str(_HERE))
from check_frames import Q_BODY_TO_OPTICAL, quat_to_matrix  # noqa: E402

try:
    from fixture_map_test.fixture_geometry import BBox, Intrinsics, estimate_range
    from fixture_map_test.fixture_registry import Landmark
except ImportError:  # running from the source tree, package not installed
    sys.path.insert(0, str(_PKG_ROOT))
    from fixture_map_test.fixture_geometry import BBox, Intrinsics, estimate_range
    from fixture_map_test.fixture_registry import Landmark


FAIL, WARN, OK = "FAIL", "WARN", "  ok"
_fails = 0
_warns = 0


def note(level: str, msg: str) -> None:
    global _fails, _warns
    if level == FAIL:
        _fails += 1
    elif level == WARN:
        _warns += 1
    print(f"        [{level}] {msg}")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_intrinsics(path: Path):
    """Accept either the intrinsics.json written by the capture scripts
    ({fx, fy, cx, cy, width, height}) or a `ros2 topic echo camera_info --once`
    YAML dump ({k: [...], width, height}).

    The ZED recalibrates every run, so hardcoding these would be a lie that
    drifts. Point this at whatever the camera actually reported.
    """
    text = path.read_text()
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        doc = yaml.safe_load(text)
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: expected a mapping of intrinsics")

    if "k" in doc or "K" in doc:
        intr = Intrinsics.from_k(doc.get("k") or doc.get("K"))
    else:
        missing = [k for k in ("fx", "fy", "cx", "cy") if k not in doc]
        if missing:
            raise ValueError(f"{path}: missing {missing} (and no 'k' matrix)")
        intr = Intrinsics(fx=float(doc["fx"]), fy=float(doc["fy"]),
                          cx=float(doc["cx"]), cy=float(doc["cy"]))
    w = int(doc.get("width") or 0)
    h = int(doc.get("height") or 0)
    if not w or not h:
        # Fall back to twice the principal point, the usual case for a centred
        # sensor. Stated out loud because it is an assumption, not a reading.
        w, h = int(round(2 * intr.cx)), int(round(2 * intr.cy))
        print(f"  note: no width/height in {path.name}; assuming {w}x{h} "
              f"from the principal point")
    return intr, w, h


def load_classes(path: Path) -> dict:
    doc = yaml.safe_load(path.read_text()) or {}
    return {str(k): dict(v) for k, v in (doc.get("classes") or {}).items()}


def load_map(path: Path):
    doc = yaml.safe_load(path.read_text()) or {}
    return str(doc.get("frame_id", "map")), [
        Landmark.from_dict(e) for e in (doc.get("landmarks") or [])]


# --------------------------------------------------------------------------- #
# Transforms -- the SAME chain the node uses, run backwards
# --------------------------------------------------------------------------- #
def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def map_to_optical(p_map, t_map_body, R_map_body, R_body_optical):
    """map point -> optical point. Inverse of what fixture_map_node does.

    static_transform_publisher composes yaw-pitch-roll as intrinsic Z-Y-X, so
    R_map_body = Rz(yaw) @ Ry(pitch) @ Rx(roll). Both rotations here are
    orthonormal, so the inverse is the transpose.
    """
    p_body = R_map_body.T @ (np.asarray(p_map, dtype=float) - t_map_body)
    return R_body_optical.T @ p_body


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--intrinsics", type=Path,
                    help="intrinsics.json from the capture scripts, or a "
                         "camera_info YAML dump. Omit to use --fx/--fy/--cx/--cy.")
    ap.add_argument("--fx", type=float)
    ap.add_argument("--fy", type=float)
    ap.add_argument("--cx", type=float)
    ap.add_argument("--cy", type=float)
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)

    ap.add_argument("--map", type=Path, default=_PKG_ROOT / "config" / "apriori_map.yaml")
    ap.add_argument("--classes", type=Path,
                    default=_PKG_ROOT / "config" / "fixture_classes.yaml")

    ap.add_argument("--cam-x", type=float, default=0.0)
    ap.add_argument("--cam-y", type=float, default=0.0)
    ap.add_argument("--cam-z", type=float, default=0.0)
    ap.add_argument("--cam-yaw", type=float, default=0.0)
    ap.add_argument("--cam-pitch", type=float, default=0.0)
    ap.add_argument("--cam-roll", type=float, default=0.0)
    ap.add_argument("--degrees", action="store_true",
                    help="read the three angles as DEGREES. The launch file "
                         "wants radians; this script prints the conversion.")

    # Gates -- keep these in step with the node's parameters.
    ap.add_argument("--confirm-radius", type=float, default=0.15)
    ap.add_argument("--assoc-radius", type=float, default=0.30)
    ap.add_argument("--max-range", type=float, default=6.0)
    ap.add_argument("--bbox-sigma-px", type=float, default=3.0)
    ap.add_argument("--min-box-px", type=float, default=15.0,
                    help="warn below this predicted box width (default 15)")
    args = ap.parse_args()

    # ---- intrinsics ---------------------------------------------------- #
    if args.intrinsics:
        intr, img_w, img_h = load_intrinsics(args.intrinsics)
        src = str(args.intrinsics)
    else:
        if None in (args.fx, args.cx, args.cy):
            ap.error("give --intrinsics, or at least --fx --cx --cy")
        intr = Intrinsics(fx=args.fx, fy=args.fy or args.fx,
                          cx=args.cx, cy=args.cy)
        img_w = args.width or int(round(2 * intr.cx))
        img_h = args.height or int(round(2 * intr.cy))
        src = "command line"

    classes = load_classes(args.classes)
    frame_id, landmarks = load_map(args.map)

    scale = math.radians(1.0) if args.degrees else 1.0
    yaw, pitch, roll = args.cam_yaw * scale, args.cam_pitch * scale, args.cam_roll * scale

    t_mb = np.array([args.cam_x, args.cam_y, args.cam_z], dtype=float)
    R_mb = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    R_bo = quat_to_matrix(Q_BODY_TO_OPTICAL)

    print("fixture_map_test pose / map pre-flight")
    print(f"  intrinsics   fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.cx:.1f} "
          f"cy={intr.cy:.1f}  image {img_w}x{img_h}   [{src}]")
    print(f"  camera       xyz=({args.cam_x:+.3f}, {args.cam_y:+.3f}, {args.cam_z:+.3f}) m  "
          f"yaw/pitch/roll=({yaw:+.4f}, {pitch:+.4f}, {roll:+.4f}) rad")
    print(f"  map          {args.map}  frame_id={frame_id!r}  "
          f"{len(landmarks)} landmark(s)")
    print(f"  gates        confirm={args.confirm_radius} assoc={args.assoc_radius} "
          f"max_range={args.max_range} bbox_sigma={args.bbox_sigma_px}px")

    if not landmarks:
        print("\n  no landmarks in the map -- nothing to check")
        return 1

    half_fov_h = math.degrees(math.atan(intr.cx / intr.fx))

    print("\nPer landmark")
    print("-" * 78)
    for lm in landmarks:
        p_opt = map_to_optical(lm.position, t_mb, R_mb, R_bo)
        x, y, z = float(p_opt[0]), float(p_opt[1]), float(p_opt[2])
        euclid = float(np.linalg.norm(lm.position - t_mb))

        print(f"  {lm.id}  [{lm.cls}]  map={np.round(lm.position, 3).tolist()}")

        if z <= 0.0:
            note(FAIL, f"BEHIND the camera (optical z={z:+.3f} m). Check the "
                       f"datum, cam_yaw, or the sign of a map coordinate.")
            continue

        u = intr.fx * x / z + intr.cx
        v = intr.fy * y / z + intr.cy
        ang_h = math.degrees(math.atan2(x, z))
        ang_v = math.degrees(math.atan2(y, z))

        print(f"        optical [{x:+.3f} {y:+.3f} {z:+.3f}]  "
              f"depth {z:.3f} m  (straight-line {euclid:.3f} m)")
        print(f"        pixel   u={u:7.1f} v={v:7.1f}   off-axis "
              f"h={ang_h:+.1f}deg v={ang_v:+.1f}deg  (half-FOV h~{half_fov_h:.0f}deg)")

        in_frame = 0.0 <= u < img_w and 0.0 <= v < img_h
        if not in_frame:
            note(FAIL, f"projects OFF the {img_w}x{img_h} image -- the camera "
                       f"cannot see this fixture from this pose")

        if z > args.max_range:
            note(FAIL, f"depth {z:.2f} m exceeds max_range {args.max_range:.2f} m "
                       f"-- every observation would be REJECTED before association")

        spec = classes.get(lm.cls)
        if spec is None:
            note(FAIL, f"class {lm.cls!r} is not in {args.classes.name} -- "
                       f"detections would be silently skipped")
            continue

        W = spec.get("width_m")
        if not W:
            note(FAIL, f"class {lm.cls!r} has no width_m")
            continue

        w_px = intr.fx * float(W) / z
        est = estimate_range(BBox(cx=u, cy=v, w=w_px, h=w_px), intr,
                             real_width_m=float(W),
                             bbox_sigma_px=args.bbox_sigma_px)
        print(f"        expect  box {w_px:.0f} px wide (width_m={W})   "
              f"sigma_Z {est.sigma_z:.4f} m")

        if w_px < args.min_box_px:
            note(WARN, f"predicted box is only {w_px:.0f} px -- below "
                       f"{args.min_box_px:.0f} px the detector is unreliable and "
                       f"bbox noise dominates")
        if est.sigma_z > args.confirm_radius:
            z_max = math.sqrt(args.confirm_radius * intr.fx * float(W)
                              / args.bbox_sigma_px)
            note(WARN, f"sigma_Z {est.sigma_z:.3f} m EXCEEDS confirm_radius "
                       f"{args.confirm_radius:.2f} m -- this landmark can never be "
                       f"CONFIRMED, only FLAGGED. Move inside {z_max:.2f} m, or "
                       f"raise confirm_radius.")
        clean = (in_frame and z <= args.max_range
                 and w_px >= args.min_box_px
                 and est.sigma_z <= args.confirm_radius)
        if clean:
            print(f"        {OK}")

    # ---- cross-landmark: the association limitation --------------------- #
    print("\nAssociation (nearest-neighbour WITHIN CLASS)")
    print("-" * 78)
    by_class: dict = {}
    for lm in landmarks:
        by_class.setdefault(lm.cls, []).append(lm)
    for cls, group in sorted(by_class.items()):
        if len(group) == 1:
            print(f"  {cls:<15} 1 instance   {OK}")
            continue
        dists = [(float(np.linalg.norm(a.position - b.position)), a.id, b.id)
                 for i, a in enumerate(group) for b in group[i + 1:]]
        d, ida, idb = min(dists)
        print(f"  {cls:<15} {len(group)} instances, closest pair {d:.3f} m "
              f"({ida} <-> {idb})")
        note(WARN, f"association is nearest-neighbour within class, which is "
                   f"only valid with ONE instance of a class in view. A range "
                   f"error larger than {d / 2:.2f} m attaches an observation to "
                   f"the wrong {cls} and nothing can notice.")

    # ---- the line to paste --------------------------------------------- #
    print("\nLaunch line for this pose (angles in RADIANS, as the launch file expects)")
    print("-" * 78)
    print(f"  ros2 launch fixture_map_test tripod_test.launch.py \\\n"
          f"      cam_x:={args.cam_x} cam_y:={args.cam_y} cam_z:={args.cam_z} \\\n"
          f"      cam_yaw:={yaw:.6f} cam_pitch:={pitch:.6f} cam_roll:={roll:.6f}")

    print()
    if _fails:
        print(f"FAILED -- {_fails} error(s), {_warns} warning(s)")
    elif _warns:
        print(f"PASSED WITH WARNINGS -- {_warns} warning(s)")
    else:
        print("ALL CHECKS PASSED")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
