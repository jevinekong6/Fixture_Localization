"""RViz markers for the fixture map.

DESIGN RULE: marker geometry must encode only what the estimator actually
knows. Depth-from-known-size gives POSITION ONLY -- a range along a ray, and
nothing whatsoever about which way the fixture is facing. Drawing a valve-wheel
mesh at an observed landmark would put a specific orientation on screen that no
measurement produced, and a reviewer looking at RViz would read that
orientation as a result. So observed landmarks are spheres: a sphere has no
orientation to misread, and its radius carries the one uncertainty we do have.

WHY THE RAYS MATTER
-------------------
The rays are the diagnostic that separates the two failure modes, which look
identical if you only look at the landmark markers:

  * The ray points AT the real fixture, but the marker sits too near or too far
    ALONG that ray -> the bearing is right and the range is wrong. That is the
    depth estimator, which in practice means ``width_m`` is wrong for that
    class. Bearing comes from (u - cx) / fx, which needs only calibration;
    range needs a physical size someone typed in by hand.

  * The ray points somewhere else entirely -> the bearing is wrong, and no
    amount of tuning ``width_m`` will fix it. That is a frame problem: the
    body->optical static transform, the measured tripod pose, or an axis
    convention.

Without the rays both cases present as "the marker is in the wrong place" and
people tune sizes for a day to fix a rotation.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from .fixture_registry import Landmark, Origin

__all__ = ["build_landmark_markers", "build_ray_markers"]

# ns names, kept stable so RViz display configs survive a rebuild.
NS_OBSERVED = "observed"
NS_CONFIRMED = "confirmed"
NS_LABELS = "labels"
NS_RAYS = "rays"

_ORANGE = ColorRGBA(r=1.0, g=0.55, b=0.0, a=0.55)
_GREEN = ColorRGBA(r=0.1, g=0.85, b=0.2, a=0.85)
_RED = ColorRGBA(r=0.95, g=0.1, b=0.1, a=0.9)
_GREY = ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.5)
_WHITE = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)

_CUBE_SIDE = 0.10   # apriori marker size: a fixed nominal, not an uncertainty
_LABEL_HEIGHT = 0.06
_LABEL_OFFSET_Z = 0.12
_MIN_SPHERE = 0.02  # keep a very confident landmark from vanishing


def _identity_pose(marker: Marker, position: Sequence[float]) -> None:
    marker.pose.position.x = float(position[0])
    marker.pose.position.y = float(position[1])
    marker.pose.position.z = float(position[2])
    # Identity orientation, stated explicitly: nothing here estimated a rotation.
    marker.pose.orientation.w = 1.0


def build_landmark_markers(
    landmarks: Iterable[Landmark],
    frame_id: str,
    stamp,
    confirm_radius: float = 0.15,
    lifetime_sec: float = 0.0,
) -> MarkerArray:
    """Spheres for observed landmarks, cubes for apriori ones, labels for both.

    ``lifetime_sec`` of 0 means the markers persist until replaced, which is
    what you want for a map: a landmark that stops being detected should stay
    on screen so you can see that it stopped being detected.
    """
    array = MarkerArray()
    life = Duration(sec=int(lifetime_sec), nanosec=int((lifetime_sec % 1.0) * 1e9))

    for i, lm in enumerate(landmarks):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.id = i
        marker.action = Marker.ADD
        marker.lifetime = life
        _identity_pose(marker, lm.position)

        if lm.origin is Origin.OBSERVED:
            # SPHERE, diameter 2*sigma. Position only -- a sphere asserts no
            # orientation, and its size is the honest error bar.
            marker.ns = NS_OBSERVED
            marker.type = Marker.SPHERE
            d = max(2.0 * float(lm.sigma), _MIN_SPHERE)
            marker.scale = Vector3(x=d, y=d, z=d)
            marker.color = _ORANGE
        else:
            # CUBE for a CAD entry: a fixed nominal size, deliberately not
            # scaled by sigma, because a surveyed position's uncertainty is not
            # what this display is about.
            #
            # MESH_RESOURCE becomes legitimate here ONLY once the apriori map
            # carries a full 6-DoF pose per landmark (position AND orientation
            # from the drawing). Until then a mesh would be drawn at an
            # arbitrary yaw and would be read as a measurement.
            marker.ns = NS_CONFIRMED
            marker.type = Marker.CUBE
            marker.scale = Vector3(x=_CUBE_SIDE, y=_CUBE_SIDE, z=_CUBE_SIDE)
            flagged = lm.max_residual > confirm_radius
            marker.color = _RED if flagged else _GREEN

        array.markers.append(marker)

        label = Marker()
        label.header.frame_id = frame_id
        label.header.stamp = stamp
        label.ns = NS_LABELS
        label.id = i
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.lifetime = life
        _identity_pose(label, lm.position)
        label.pose.position.z = float(lm.position[2]) + _LABEL_OFFSET_Z
        label.scale = Vector3(x=0.0, y=0.0, z=_LABEL_HEIGHT)
        label.color = _WHITE
        label.text = f"{lm.id}  n={lm.n_obs}  s={lm.sigma:.2f}m"
        if lm.origin is Origin.APRIORI and lm.max_residual > confirm_radius:
            label.text += f"  FLAG {lm.max_residual:.2f}m"
        array.markers.append(label)

    return array


def build_ray_markers(
    camera_position: Optional[Sequence[float]],
    points: Iterable[Sequence[float]],
    frame_id: str,
    stamp,
    lifetime_sec: float = 0.0,
) -> MarkerArray:
    """One LINE_LIST from the camera origin to each observed point.

    ``camera_position`` is the camera optical origin expressed in ``frame_id``
    -- i.e. the translation of the map<-optical transform. If it is None (no TF
    yet) an empty array comes back rather than rays from the map origin, which
    would be a lie that looks like data.
    """
    array = MarkerArray()

    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = NS_RAYS
    marker.id = 0
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.lifetime = Duration(sec=int(lifetime_sec),
                               nanosec=int((lifetime_sec % 1.0) * 1e9))
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.005  # line width, metres
    marker.color = _GREY

    pts: List[Point] = []
    if camera_position is not None:
        origin = Point(x=float(camera_position[0]),
                       y=float(camera_position[1]),
                       z=float(camera_position[2]))
        for p in points:
            pts.append(origin)
            pts.append(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))

    if pts:
        marker.points = pts
    else:
        # Nothing to draw: publish a DELETE so stale rays from the last frame
        # do not sit on screen pretending to be current.
        marker.action = Marker.DELETE

    array.markers.append(marker)
    return array
