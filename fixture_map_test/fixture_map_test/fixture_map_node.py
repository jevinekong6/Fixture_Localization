"""ROS2 node: YOLO detections -> map-frame fixture landmarks.

A thin wrapper. Every decision worth testing lives in fixture_geometry.py and
fixture_registry.py, which have no ROS imports; this file does message
plumbing, TF, patch saving and logging, and nothing else.

THE ONE THING THIS NODE MUST NOT DO is rotate anything itself. It back-projects
into the camera optical frame and hands the resulting point to tf2 to get into
the map frame. The body->optical rotation lives in the launch file as a static
transform. If you find yourself adding a rotation matrix to this file, stop:
that is how the 90-degree error gets in.

Pre-robot, tripod-stage testing. There is no robot and no pose estimation here
-- a static TF stands in for localization.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import MarkerArray

import message_filters
import tf2_ros
from tf2_geometry_msgs import do_transform_point  # registers PointStamped with tf2

from .fixture_geometry import BBox, Intrinsics, observe as observe_geometry, obliquity_proxy
from .fixture_registry import FixtureRegistry, Outcome
from .markers import build_landmark_markers, build_ray_markers

try:
    from cv_bridge import CvBridge
    import cv2
    _CV_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the install, not on logic
    CvBridge = None
    cv2 = None
    _CV_AVAILABLE = False


BEST_EFFORT_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    durability=QoSDurabilityPolicy.VOLATILE,
    depth=10,
)


class FixtureMapNode(Node):
    def __init__(self) -> None:
        super().__init__("fixture_map_node")

        self.declare_parameter("map_frame", "map")
        self.declare_parameter("optical_frame", "zed_left_camera_optical_frame")
        self.declare_parameter("class_config", "")
        self.declare_parameter("apriori_map", "")
        self.declare_parameter("output_map", "fixture_map_out.yaml")
        self.declare_parameter("patch_dir", "fixture_patches")
        self.declare_parameter("bbox_sigma_px", 3.0)
        self.declare_parameter("confirm_radius", 0.15)
        self.declare_parameter("flag_radius", 0.60)
        self.declare_parameter("assoc_radius", 0.30)
        self.declare_parameter("max_range", 6.0)
        self.declare_parameter("min_confidence", 0.40)
        self.declare_parameter("save_period", 10.0)
        self.declare_parameter("use_image", True)

        g = self.get_parameter
        self.map_frame = g("map_frame").value
        self.optical_frame = g("optical_frame").value
        self.output_map = g("output_map").value
        self.patch_dir = g("patch_dir").value
        self.bbox_sigma_px = float(g("bbox_sigma_px").value)
        self.min_confidence = float(g("min_confidence").value)
        self.obliquity_tol = 0.25  # warn past +/-25% off nominal aspect

        self.classes: Dict[str, dict] = self._load_class_config(g("class_config").value)

        self.registry = FixtureRegistry(
            confirm_radius=float(g("confirm_radius").value),
            flag_radius=float(g("flag_radius").value),
            assoc_radius=float(g("assoc_radius").value),
            max_range=float(g("max_range").value),
        )
        self.registry.frame_id = self.map_frame
        apriori = g("apriori_map").value
        if apriori:
            try:
                n = self.registry.load_apriori(apriori)
                self.get_logger().info(f"Loaded {n} apriori landmark(s) from {apriori}")
            except (FileNotFoundError, ValueError) as exc:
                self.get_logger().error(f"Could not load apriori map: {exc}")
        else:
            self.get_logger().warn("No apriori_map given -- every fixture will come out as NEW")

        os.makedirs(self.patch_dir, exist_ok=True)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.intrinsics: Optional[Intrinsics] = None
        self.create_subscription(
            CameraInfo, "~/camera_info", self._on_camera_info, BEST_EFFORT_QOS)

        # Image handling degrades gracefully: without cv_bridge we still map
        # fixtures, we just cannot save patches.
        self.bridge = CvBridge() if _CV_AVAILABLE else None
        self.use_image = bool(g("use_image").value) and _CV_AVAILABLE
        if bool(g("use_image").value) and not _CV_AVAILABLE:
            self.get_logger().warn(
                "cv_bridge/cv2 unavailable -- running detections-only, no patches saved")

        if self.use_image:
            det_sub = message_filters.Subscriber(
                self, Detection2DArray, "~/detections", qos_profile=BEST_EFFORT_QOS)
            img_sub = message_filters.Subscriber(
                self, Image, "~/image", qos_profile=BEST_EFFORT_QOS)
            self.sync = message_filters.ApproximateTimeSynchronizer(
                [det_sub, img_sub], queue_size=10, slop=0.05)
            self.sync.registerCallback(self._on_detections_and_image)
        else:
            self.create_subscription(
                Detection2DArray, "~/detections",
                lambda msg: self._process(msg, None), BEST_EFFORT_QOS)

        self.landmark_pub = self.create_publisher(MarkerArray, "~/landmarks", 1)
        self.ray_pub = self.create_publisher(MarkerArray, "~/rays", 1)

        save_period = float(g("save_period").value)
        if save_period > 0.0:
            self.create_timer(save_period, self._save_map)

        self._patch_seq = 0
        self.get_logger().info(
            f"fixture_map_node up: {self.optical_frame} -> {self.map_frame}, "
            f"{len(self.classes)} class(es) configured, "
            f"patches -> {self.patch_dir}, map -> {self.output_map}")

    # ------------------------------------------------------------------ #
    # Config
    # ------------------------------------------------------------------ #
    def _load_class_config(self, path: str) -> Dict[str, dict]:
        if not path:
            self.get_logger().error("class_config parameter is empty -- no detection can be sized")
            return {}
        try:
            with open(path, "r") as f:
                doc = yaml.safe_load(f) or {}
        except OSError as exc:
            self.get_logger().error(f"Could not read class_config {path}: {exc}")
            return {}
        classes = doc.get("classes") or {}
        if not classes:
            self.get_logger().error(f"No 'classes:' mapping in {path}")
        return {str(k): dict(v) for k, v in classes.items()}

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    def _on_camera_info(self, msg: CameraInfo) -> None:
        if self.intrinsics is None:
            try:
                self.intrinsics = Intrinsics.from_k(msg.k)
            except ValueError as exc:
                self.get_logger().error(f"Bad CameraInfo: {exc}")
                return
            self.get_logger().info(
                f"Intrinsics: fx={self.intrinsics.fx:.1f} fy={self.intrinsics.fy:.1f} "
                f"cx={self.intrinsics.cx:.1f} cy={self.intrinsics.cy:.1f}")

    def _on_detections_and_image(self, det_msg: Detection2DArray, img_msg: Image) -> None:
        self._process(det_msg, img_msg)

    def _process(self, det_msg: Detection2DArray, img_msg: Optional[Image]) -> None:
        if self.intrinsics is None:
            self.get_logger().warn("No CameraInfo yet -- skipping detections",
                                   throttle_duration_sec=5.0)
            return

        stamp = det_msg.header.stamp
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.optical_frame, rclpy.time.Time.from_msg(stamp),
                timeout=Duration(seconds=0.1))
        except tf2_ros.TransformException as exc:
            self.get_logger().warn(
                f"TF {self.map_frame} <- {self.optical_frame} unavailable: {exc}",
                throttle_duration_sec=2.0)
            return

        cam_origin = np.array([tf.transform.translation.x,
                               tf.transform.translation.y,
                               tf.transform.translation.z])

        cv_image = None
        if img_msg is not None and self.bridge is not None:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            except Exception as exc:  # cv_bridge raises its own error types
                self.get_logger().warn(f"cv_bridge failed, no patches this frame: {exc}",
                                       throttle_duration_sec=5.0)

        map_points = []
        for det in det_msg.detections:
            result = self._best_hypothesis(det)
            if result is None:
                continue
            class_id, score = result
            if score < self.min_confidence:
                continue
            spec = self.classes.get(class_id)
            if spec is None:
                self.get_logger().warn(
                    f"class_id {class_id!r} not in class_config -- add it, with the width the "
                    f"DETECTOR's box encloses", throttle_duration_sec=10.0)
                continue

            bbox = BBox(cx=float(det.bbox.center.position.x),
                        cy=float(det.bbox.center.position.y),
                        w=float(det.bbox.size_x),
                        h=float(det.bbox.size_y))
            try:
                p_opt, est = observe_geometry(
                    bbox, self.intrinsics,
                    real_width_m=spec.get("width_m"),
                    real_height_m=spec.get("height_m"),
                    bbox_sigma_px=self.bbox_sigma_px,
                    prefer=str(spec.get("prefer", "width")),
                )
            except ValueError as exc:
                self.get_logger().warn(f"Skipping {class_id} detection: {exc}",
                                       throttle_duration_sec=5.0)
                continue

            nominal = spec.get("nominal_aspect")
            if nominal:
                try:
                    ratio = obliquity_proxy(bbox, float(nominal))
                except ValueError:
                    ratio = None
                if ratio is not None and abs(ratio - 1.0) > self.obliquity_tol:
                    self.get_logger().warn(
                        f"{class_id}: obliquity proxy {ratio:.2f} is more than "
                        f"{self.obliquity_tol * 100:.0f}% off nominal -- box is foreshortened, "
                        f"range likely overestimated (warning only, no correction applied)",
                        throttle_duration_sec=5.0)

            p_map = self._to_map(p_opt, tf, stamp)
            if p_map is None:
                continue

            # The side-by-side log IS the debugging surface. Optical tells you
            # what the camera saw; map tells you what the transform did to it.
            # A wrong number in the first column is a size or a detector
            # problem; a right first column and a wrong second is a frame
            # problem. Printing only the map point hides which.
            self.get_logger().info(
                f"{class_id} p={score:.2f} bbox=({bbox.w:.0f}x{bbox.h:.0f}px) "
                f"| optical [{p_opt[0]:+.3f} {p_opt[1]:+.3f} {p_opt[2]:+.3f}] "
                f"| map [{p_map[0]:+.3f} {p_map[1]:+.3f} {p_map[2]:+.3f}] "
                f"| z={est.z:.3f}+/-{est.sigma_z:.3f}m via {est.used_dim}")

            patch = self._save_patch(cv_image, bbox, class_id)
            assoc = self.registry.observe(
                class_id, p_map, sigma=est.sigma_z, range_m=est.z, patch=patch)

            if assoc.outcome is Outcome.FLAGGED:
                self.get_logger().warn(f"FLAGGED {class_id}: {assoc.note}")
            elif assoc.outcome is Outcome.REJECTED:
                self.get_logger().info(f"REJECTED {class_id}: {assoc.note}")
            else:
                self.get_logger().info(f"{assoc.outcome.value.upper()} {class_id}: {assoc.note}")

            if assoc.outcome is not Outcome.REJECTED:
                map_points.append(p_map)

        self._publish(stamp, cam_origin, map_points)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _best_hypothesis(det):
        """Highest-scoring hypothesis as (class_id, score), or None."""
        if not det.results:
            return None
        best = max(det.results, key=lambda r: r.hypothesis.score)
        return str(best.hypothesis.class_id), float(best.hypothesis.score)

    def _to_map(self, p_opt: np.ndarray, tf, stamp) -> Optional[np.ndarray]:
        """Optical point -> map point, via tf2. No hand-rolled rotation."""
        ps = PointStamped()
        ps.header.frame_id = self.optical_frame
        ps.header.stamp = stamp
        ps.point.x = float(p_opt[0])
        ps.point.y = float(p_opt[1])
        ps.point.z = float(p_opt[2])
        try:
            out = do_transform_point(ps, tf)
        except Exception as exc:
            self.get_logger().warn(f"Transform of point failed: {exc}",
                                   throttle_duration_sec=5.0)
            return None
        return np.array([out.point.x, out.point.y, out.point.z])

    def _save_patch(self, cv_image, bbox: BBox, class_id: str, pad: int = 12) -> Optional[str]:
        if cv_image is None or cv2 is None:
            return None
        h, w = cv_image.shape[:2]
        x0 = max(0, int(bbox.cx - bbox.w / 2.0) - pad)
        y0 = max(0, int(bbox.cy - bbox.h / 2.0) - pad)
        x1 = min(w, int(bbox.cx + bbox.w / 2.0) + pad)
        y1 = min(h, int(bbox.cy + bbox.h / 2.0) + pad)
        if x1 <= x0 or y1 <= y0:
            return None
        self._patch_seq += 1
        name = f"{class_id}_{self._patch_seq:05d}.png"
        path = os.path.join(self.patch_dir, name)
        try:
            cv2.imwrite(path, cv_image[y0:y1, x0:x1])
        except Exception as exc:
            self.get_logger().warn(f"Could not write patch {path}: {exc}",
                                   throttle_duration_sec=5.0)
            return None
        return name

    def _publish(self, stamp, cam_origin, map_points) -> None:
        self.landmark_pub.publish(build_landmark_markers(
            self.registry.landmarks, self.map_frame, stamp,
            confirm_radius=self.registry.confirm_radius))
        self.ray_pub.publish(build_ray_markers(
            cam_origin, map_points, self.map_frame, stamp))

    def _save_map(self) -> None:
        try:
            self.registry.save(self.output_map)
        except OSError as exc:
            self.get_logger().error(f"Could not write {self.output_map}: {exc}")
            return
        flagged = self.registry.flagged()
        msg = f"Wrote {len(self.registry.landmarks)} landmark(s) to {self.output_map}"
        if flagged:
            msg += f" -- {len(flagged)} FLAGGED: " + ", ".join(
                f"{lm.id} ({lm.max_residual:.2f} m)" for lm in flagged)
        self.get_logger().info(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FixtureMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Write on the way out, so a Ctrl-C mid-run never loses the session.
        node._save_map()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
