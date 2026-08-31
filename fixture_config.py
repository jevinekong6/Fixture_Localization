#!/usr/bin/env python3
"""Shared config for the ZED capture / YOLO / pinhole-localization test scripts.
"""
import json
from pathlib import Path

# --------------------------------------------------------------------------- #
# Known real-world size per YOLO class name.
#
# real_size_m: the physical dimension (meters) that corresponds to the pixel
#   measurement chosen by `size_from` below.
# size_from: which pixel measurement of the YOLO box to treat as the
#   projection of real_size_m.
#   - "max"    : max(bbox_width_px, bbox_height_px) -- recommended default for
#                roughly circular/symmetric fixtures (knobs, wheels, marman
#                ring), since it's the most stable under moderate rotation.
#   - "width"  : bbox width only -- use for fixtures where width is the
#                reliable/known dimension regardless of orientation.
#   - "height" : bbox height only.
#
# --------------------------------------------------------------------------- #
FIXTURE_DIMENSIONS = {
    "knob":       {"real_size_m": 0.06,  "size_from": "max"},
    "handle":     {"real_size_m": 0.08,  "size_from": "max"},
    "wheel":      {"real_size_m": 0.09,  "size_from": "max"},
    "marman_ring": {"real_size_m": 0.1, "size_from": "max"},
    "eva_handle": {"real_size_m": 0.09, "size_from": "max"},
    "oblique_knob": {"real_size_m": 0.063, "size_from": "max"},
    "rocket_nozzle": {"real_size_m": 0.07, "size_from": "max"},
    "snowflake": {"real_size_m": 0.13, "size_from": "max"},
}


def load_intrinsics(path):
    """Load {fx, fy, cx, cy, width, height} written by capture_map_zed.py or
    capture_query_yolo.py."""
    with open(path) as f:
        return json.load(f)


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)