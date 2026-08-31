#!/usr/bin/env python3
"""Estimate 3D fixture positions from saved YOLO detections using pinhole
depth-from-known-size math, and check the result against the ZED's own
stereo depth 

For each detection:
    Z_pinhole = (fx * real_size_m) / apparent_size_px
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

apparent_size_px is chosen per-class via FIXTURE_DIMENSIONS in
fixture_config.py ("max" bbox dimension is the default -- see that file for
why it's recommended for roughly circular/symmetric fixtures).

Usage:
    python localize_fixtures_pinhole.py --query_dir query_capture --outfile results.csv
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from fixture_config import FIXTURE_DIMENSIONS, load_intrinsics, load_json


def apparent_size_px(bbox_xyxy, size_from):
    x1, y1, x2, y2 = bbox_xyxy
    w, h = x2 - x1, y2 - y1
    if size_from == "width":
        return w
    if size_from == "height":
        return h
    return max(w, h)  # "max"


def pinhole_depth(fx, real_size_m, size_px):
    if size_px <= 0:
        return None
    return (fx * real_size_m) / size_px


def backproject(u, v, z, fx, fy, cx, cy):
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return x, y, z


def transform_point(xyz_cam, T_cam_to_world):
    homog = np.array([xyz_cam[0], xyz_cam[1], xyz_cam[2], 1.0])
    world = np.array(T_cam_to_world) @ homog
    return world[:3].tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query_dir", required=True, help="output directory from capture_query_yolo.py")
    ap.add_argument("--outfile", default="localization_results.csv")
    args = ap.parse_args()

    query_dir = Path(args.query_dir)
    intrinsics = load_intrinsics(query_dir / "intrinsics.json")
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]

    det_files = sorted((query_dir / "detections").glob("frame_*.json"))
    if not det_files:
        raise SystemExit(f"No detection files found under {query_dir/'detections'}")

    rows = []
    unknown_classes = set()

    for det_path in det_files:
        frame_name = det_path.stem
        detections = load_json(det_path)

        pose_path = query_dir / "poses" / f"{frame_name}.json"
        T_cam_to_world = load_json(pose_path)["T_camera_to_world"] if pose_path.exists() else None

        for d in detections:
            cls = d["class_name"]
            if cls not in FIXTURE_DIMENSIONS:
                unknown_classes.add(cls)
                continue

            dim_cfg = FIXTURE_DIMENSIONS[cls]
            size_px = apparent_size_px(d["bbox_xyxy"], dim_cfg["size_from"])
            z_pinhole = pinhole_depth(fx, dim_cfg["real_size_m"], size_px)

            u, v = d["centroid_px"]
            row = {
                "frame": frame_name, "class_name": cls, "confidence": d["confidence"],
                "u_px": u, "v_px": v, "bbox_size_px": size_px,
                "z_pinhole_m": z_pinhole, "z_zed_gt_m": d.get("zed_depth_m"),
            }

            if z_pinhole is not None:
                x, y, z = backproject(u, v, z_pinhole, fx, fy, cx, cy)
                row.update({"x_cam_m": x, "y_cam_m": y, "z_cam_m": z})
                if T_cam_to_world is not None:
                    xw, yw, zw = transform_point((x, y, z), T_cam_to_world)
                    row.update({"x_world_m": xw, "y_world_m": yw, "z_world_m": zw})

            if row["z_zed_gt_m"] is not None and z_pinhole is not None:
                err = abs(z_pinhole - row["z_zed_gt_m"])
                row["abs_error_m"] = err
                row["pct_error"] = 100.0 * err / row["z_zed_gt_m"] if row["z_zed_gt_m"] > 0 else None

            rows.append(row)

    if unknown_classes:
        print(f"Warning: skipped detections for classes with no known dimension: "
              f"{sorted(unknown_classes)} -- add them to FIXTURE_DIMENSIONS in fixture_config.py")

    if not rows:
        raise SystemExit("No detections with a known fixture dimension were found -- nothing to write.")

    fieldnames = sorted({k for row in rows for k in row.keys()},
                         key=lambda k: (k != "frame", k != "class_name", k))
    with open(args.outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} detection(s) to {args.outfile}")

    # Summary accuracy stats per class, where ZED ground-truth depth was available.
    by_class = {}
    for row in rows:
        if row.get("abs_error_m") is not None:
            by_class.setdefault(row["class_name"], []).append(row)

    if by_class:
        print("\nPinhole vs. ZED stereo depth -- summary (meters):")
        for cls, cls_rows in sorted(by_class.items()):
            errs = np.array([r["abs_error_m"] for r in cls_rows])
            pct = np.array([r["pct_error"] for r in cls_rows if r["pct_error"] is not None])
            print(f"  {cls:15s} n={len(cls_rows):3d}  "
                  f"mean_err={errs.mean():.3f}  median_err={np.median(errs):.3f}  "
                  f"mean_pct_err={pct.mean():.1f}%")
    else:
        print("\nNo rows had both a pinhole estimate and a ZED ground-truth depth to compare.")


if __name__ == "__main__":
    main()