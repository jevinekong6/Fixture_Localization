"""Depth-from-known-size geometry. ROS-free by design.

FRAME CONVENTION -- OpenCV / ROS *optical*, used everywhere in this module:

    +x  right across the image
    +y  DOWN the image
    +z  FORWARD, out of the lens along the optical axis

A fixture on the optical axis at range d back-projects to exactly (0, 0, d).
There is deliberately no rotation of any kind in here: converting optical
coordinates into the body frame or the map frame is tf2's job, driven by the
static transforms in the launch file. Hand-rolling that rotation in Python is
the single most reliable way to introduce a silent 90-degree error, so this
module does not offer the option.

The range model is the pinhole relation for an object of known real extent:

    Z = f * W_real / w_px

which is only as good as W_real. W_real must be the extent that the DETECTOR's
bounding box actually encloses, which is not always the catalog dimension of
the part -- see config/fixture_classes.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Intrinsics",
    "BBox",
    "RangeEstimate",
    "estimate_range",
    "backproject",
    "observe",
    "obliquity_proxy",
]


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics for a single rectified camera, in pixels."""

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_k(cls, k: Sequence[float]) -> "Intrinsics":
        """Build from a ``sensor_msgs/CameraInfo`` ``k``: a row-major 3x3
        matrix [fx 0 cx; 0 fy cy; 0 0 1] flattened to 9 elements."""
        k = list(k)
        if len(k) != 9:
            raise ValueError(f"CameraInfo.k must have 9 elements, got {len(k)}")
        if k[0] == 0.0 or k[4] == 0.0:
            raise ValueError("CameraInfo.k has zero focal length -- camera not calibrated?")
        return cls(fx=float(k[0]), fy=float(k[4]), cx=float(k[2]), cy=float(k[5]))


@dataclass(frozen=True)
class BBox:
    """An axis-aligned detection box in pixels.

    ``cx`` / ``cy`` are the box CENTRE, matching ``vision_msgs/BoundingBox2D``
    (whose ``center.position`` is also a centre, not a corner). Converting from
    a corner format (x1, y1, x2, y2) is the caller's job.
    """

    cx: float
    cy: float
    w: float
    h: float

    @property
    def aspect(self) -> float:
        """Width / height. Raises if the height is non-positive."""
        if self.h <= 0.0:
            raise ValueError(f"bbox height must be positive, got {self.h}")
        return self.w / self.h


@dataclass(frozen=True)
class RangeEstimate:
    """Range along the optical axis with its 1-sigma uncertainty, in metres."""

    z: float
    sigma_z: float
    used_dim: str  # "width" or "height" -- which measurement produced z


def estimate_range(
    bbox: BBox,
    intr: Intrinsics,
    real_width_m: Optional[float] = None,
    real_height_m: Optional[float] = None,
    bbox_sigma_px: float = 3.0,
    prefer: str = "width",
) -> RangeEstimate:
    """Estimate range from a bounding box of known real-world extent.

    ``Z = f * W_real / w_px``.

    Uncertainty is propagated analytically from the bounding-box pixel noise.
    Differentiating the range relation:

        dZ/dw = -f * W / w^2 = -Z^2 / (f * W)

    so, to first order,

        sigma_Z = (Z^2 / (f * W)) * sigma_w

    The Z-squared is the important part: range uncertainty grows with the
    SQUARE of range. Doubling the distance to a fixture quadruples the error
    bar even though the detector is behaving exactly as well as before. That is
    a property of depth-from-known-size, not a bug, and it is why ``max_range``
    exists in the registry.

    ``prefer`` picks which dimension to use when both real sizes are given; if
    only one is given, that one is used regardless of ``prefer``.

    Raises ValueError on a non-positive bbox dimension rather than returning
    inf, so a degenerate detection fails loudly at the source instead of
    poisoning the map with a landmark at infinity.
    """
    if prefer not in ("width", "height"):
        raise ValueError(f"prefer must be 'width' or 'height', got {prefer!r}")
    if real_width_m is None and real_height_m is None:
        raise ValueError("need at least one of real_width_m / real_height_m")

    use_width = (prefer == "width" and real_width_m is not None) or real_height_m is None
    if use_width:
        used_dim, f, real, px = "width", intr.fx, float(real_width_m), bbox.w
    else:
        used_dim, f, real, px = "height", intr.fy, float(real_height_m), bbox.h

    if px <= 0.0:
        raise ValueError(f"bbox {used_dim} must be positive, got {px}")
    if real <= 0.0:
        raise ValueError(f"real {used_dim} must be positive, got {real}")
    if f <= 0.0:
        raise ValueError(f"focal length must be positive, got {f}")
    if bbox_sigma_px < 0.0:
        raise ValueError(f"bbox_sigma_px must be non-negative, got {bbox_sigma_px}")

    z = f * real / px
    sigma_z = (z * z / (f * real)) * bbox_sigma_px
    return RangeEstimate(z=z, sigma_z=sigma_z, used_dim=used_dim)


def backproject(bbox: BBox, intr: Intrinsics, z: float) -> np.ndarray:
    """Back-project a bbox centre at range ``z`` into the camera optical frame.

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = Z

    Returns a numpy 3-vector in OPTICAL coordinates (+x right, +y down,
    +z forward). On-axis in, (0, 0, z) out.
    """
    if z <= 0.0:
        raise ValueError(f"range must be positive, got {z}")
    x = (bbox.cx - intr.cx) * z / intr.fx
    y = (bbox.cy - intr.cy) * z / intr.fy
    return np.array([x, y, float(z)], dtype=float)


def observe(
    bbox: BBox,
    intr: Intrinsics,
    real_width_m: Optional[float] = None,
    real_height_m: Optional[float] = None,
    bbox_sigma_px: float = 3.0,
    prefer: str = "width",
) -> Tuple[np.ndarray, RangeEstimate]:
    """Range-estimate and back-project in one call.

    Returns ``(point_optical, RangeEstimate)``. The point's sigma is the
    estimate's ``sigma_z``: this model gives POSITION ONLY, and the dominant
    error is along the ray, so the registry treats ``sigma_z`` as an isotropic
    stand-in for the full covariance.
    """
    est = estimate_range(
        bbox, intr,
        real_width_m=real_width_m,
        real_height_m=real_height_m,
        bbox_sigma_px=bbox_sigma_px,
        prefer=prefer,
    )
    return backproject(bbox, intr, est.z), est


def obliquity_proxy(bbox: BBox, nominal_aspect: float) -> float:
    """Observed aspect ratio divided by the fixture's nominal aspect ratio.

    1.0 means the box looks fronto-parallel. Below 1.0 the box is narrower
    than expected (the fixture is likely rotated away about a vertical axis);
    above 1.0 it is wider than expected.

    THIS IS A WARNING FLAG ONLY, NOT A CORRECTION. Nothing downstream divides
    by it or otherwise uses it to fix the range: a foreshortened box makes the
    fixture look smaller and therefore farther away, and the honest response is
    to distrust that observation, not to scale it by a number derived from the
    same corrupted box.

    It is MEANINGLESS FOR ROTATIONALLY SYMMETRIC FIXTURES. A knob or a marman
    ring viewed off-axis has an ambiguous projected aspect that depends on the
    tilt axis, so the ratio can sit at 1.0 while the fixture is badly oblique,
    or wander away from 1.0 while it is square-on. Read it as evidence only for
    fixtures with a clear, detectable long axis -- for anything round, the ray
    visualisation in RViz is a far better diagnostic.
    """
    if nominal_aspect <= 0.0:
        raise ValueError(f"nominal_aspect must be positive, got {nominal_aspect}")
    return bbox.aspect / float(nominal_aspect)
