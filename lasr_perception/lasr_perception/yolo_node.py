"""ROS 2 bridge: camera Image -> YOLO -> vision_msgs/Detection2DArray.

Detection ONLY. No depth, no 3D box, no geometry of any kind: all of that
lives downstream in fixture_map_test, which does it in the map frame with
uncertainty and an apriori map. This node's entire job is to put 2D boxes and
class names onto a topic with a correct timestamp.

WHY THIS FILE IMPORTS NOTHING FROM LASR-CV_App
----------------------------------------------
Deliberate, and each reason is separate:

* ``3D_run_live.py`` cannot be imported at all -- a module name may not start
  with a digit. calibrate_depth_collect.py already works around this by
  duplicating FIXTURE_DIMS by hand, with a comment saying to keep the copies
  in sync. Importing was never an option.

* ``bbox_3d_utils.py`` does not currently parse: it has unresolved git merge
  conflict markers (``<<<<<<< Updated upstream``) inside
  ``BackProjBBox3DEstimator.__init__``. Any import of it raises SyntaxError.

* ``config.py`` has IMPORT-TIME SIDE EFFECTS. Importing it runs os.makedirs on
  eight directories and OVERWRITES train/data.yaml. A ROS node that did that
  would silently rewrite training config every time you launched it. The model
  path is a ROS parameter here instead.

* The 3D estimation in bbox_3d_utils is not wanted here even if it worked.
  depth_from_box_size averages a width-derived and a height-derived range, and
  BackProjBBox3DEstimator then pushes the result back by half a dimension plus
  push_back_extra. fixture_map_test does its own range estimate with an
  analytic sigma and no push-back. Running both would apply two different depth
  models to the same box and silently disagree.

So this node depends on ultralytics and the weights file, nothing else.

THE TIMESTAMP CONTRACT -- THE THING MOST LIKELY TO BREAK
--------------------------------------------------------
The detection's ``header.stamp`` MUST be the stamp of the image the detection
came from, never ``now()``. Two things downstream depend on it:

  * fixture_map_test pairs detections with images using an
    ApproximateTimeSynchronizer with a 50 ms slop. Inference takes longer than
    that, so stamping at completion time means the pair never matches and no
    patches are ever saved -- or, without an image, nothing at all happens.
  * The TF lookup uses that stamp. Static transforms are timeless so it is
    harmless today, but the moment the camera is on a moving robot a late
    stamp is a real position error that scales with speed.

The inference worker below therefore carries the Header alongside the frame,
as one unit, all the way through. The latest-frame-wins pattern (drop stale
frames rather than queue them) is kept from 3D_run_live.py because it keeps
latency low -- but there it returned "the newest detections" decoupled from
any particular frame, which is exactly the bug this note is about.
"""
from __future__ import annotations

from pathlib import Path
import threading
import time
from typing import Optional, Tuple

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from cv_bridge import CvBridge

# Matches fixture_map_test's subscriber. A RELIABLE publisher will not connect
# to a BEST_EFFORT subscriber: the topic shows up in `ros2 topic list`, `ros2
# topic echo` works, and the node receives nothing. Check both ends with
# `ros2 topic info <topic> --verbose` before suspecting anything else.
BEST_EFFORT_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    durability=QoSDurabilityPolicy.VOLATILE,
    depth=10,
)


class _InferenceWorker(threading.Thread):
    """Latest-frame-wins inference thread. Carries the Header with the frame.

    Dropping stale frames instead of queueing them is what keeps latency
    bounded when inference is slower than the camera. The Header travelling
    with the frame is what keeps the result honest about which frame it came
    from.
    """

    def __init__(self, model, conf: float, iou: float, logger) -> None:
        super().__init__(daemon=True)
        self._model = model
        self._conf = conf
        self._iou = iou
        self._log = logger
        self._lock = threading.Lock()
        self._pending: Optional[Tuple] = None       # (frame, Header)
        self._result: Optional[Tuple] = None        # (detections, Header)
        self._result_seq = 0
        self._stop = False

    def submit(self, frame, header: Header) -> None:
        with self._lock:
            self._pending = (frame, header)

    def take(self):
        """Return (detections, header, seq) once per new result, else None.

        Sequence-gated so the caller publishes each inference exactly once
        rather than republishing the same boxes at camera rate under a
        succession of different stamps.
        """
        with self._lock:
            if self._result is None:
                return None
            dets, header = self._result
            self._result = None
            self._result_seq += 1
            return dets, header, self._result_seq

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        while not self._stop:
            with self._lock:
                job, self._pending = self._pending, None
            if job is None:
                time.sleep(0.002)
                continue
            frame, header = job
            try:
                results = self._model(frame, conf=self._conf, iou=self._iou, verbose=False)
                dets = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                    dets.append((
                        (x1, y1, x2, y2),
                        float(box.conf[0]),
                        # model.names maps the integer id to the training class
                        # STRING. fixture_map_test looks these up verbatim in
                        # fixture_classes.yaml, so the string is the contract --
                        # never str(int_id).
                        str(self._model.names[int(box.cls[0])]),
                    ))
            except Exception as exc:  # ultralytics raises its own types
                self._log.error(f"inference failed: {exc}")
                dets = []
            with self._lock:
                self._result = (dets, header)


class YoloDetectionNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_node")

        self.declare_parameter("weights", "")
        self.declare_parameter("conf_threshold", 0.25)
        self.declare_parameter("iou_threshold", 0.5)
        self.declare_parameter("class_config", "")
        self.declare_parameter("publish_annotated", False)

        weights = str(self.get_parameter("weights").value)
        if not weights:
            raise RuntimeError(
                "the 'weights' parameter is required -- point it at your "
                "best.pt (LASR-CV_App train/runs/<run>/weights/best.pt)")
        if not Path(weights).is_file():
            raise RuntimeError(f"weights file not found: {weights}")

        conf = float(self.get_parameter("conf_threshold").value)
        iou = float(self.get_parameter("iou_threshold").value)

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not importable from the interpreter running "
                "this node. ROS 2 uses the system python3; if your YOLO "
                "environment is a venv or conda env, either pip install "
                "ultralytics into the system python or recreate the venv with "
                "--system-site-packages."
            ) from exc

        self.get_logger().info(f"loading {weights} ...")
        model = YOLO(weights)
        self.class_names = [str(n) for n in model.names.values()]
        self.get_logger().info(
            f"model loaded, {len(self.class_names)} class(es): {self.class_names}")

        # Catch the single most common integration failure AT STARTUP rather
        # than as silence at run time. A class the detector emits but
        # fixture_map_test has no size for is skipped there with a throttled
        # warning, which presents as "my fixture never appears in RViz".
        self._check_class_config(str(self.get_parameter("class_config").value))

        self.bridge = CvBridge()
        self.worker = _InferenceWorker(model, conf, iou, self.get_logger())
        self.worker.start()

        self.pub = self.create_publisher(Detection2DArray, "~/detections", BEST_EFFORT_QOS)
        self.create_subscription(Image, "~/image", self._on_image, BEST_EFFORT_QOS)
        # Poll faster than any camera so a finished inference publishes promptly.
        self.create_timer(0.005, self._drain)

        self._last_seq = 0
        self._published = 0
        self.create_timer(5.0, self._heartbeat)
        self.get_logger().info(
            f"yolo_node up: conf={conf} iou={iou}, ~/image -> ~/detections")

    # ------------------------------------------------------------------ #
    def _check_class_config(self, path: str) -> None:
        """Warn about class-string mismatches against fixture_classes.yaml.

        Not fatal: a class with no size entry is a legitimate choice (the
        fixture_map_test config omits 'tensegrity' on purpose, because nobody
        has measured it and a guessed width_m would place a confident landmark
        in the wrong spot). This just makes the consequence visible up front.
        """
        if not path:
            self.get_logger().warn(
                "no class_config given -- skipping the class-name cross-check. "
                "Pass fixture_map_test's config/fixture_classes.yaml to catch "
                "class-string mismatches at startup.")
            return
        try:
            with open(path, "r") as f:
                doc = yaml.safe_load(f) or {}
        except OSError as exc:
            self.get_logger().error(f"could not read class_config {path}: {exc}")
            return

        configured = set((doc.get("classes") or {}).keys())
        emitted = set(self.class_names)
        unsized = sorted(emitted - configured)
        unused = sorted(configured - emitted)

        if unsized:
            self.get_logger().warn(
                f"classes the model can emit but {Path(path).name} has no size "
                f"for: {unsized} -- detections of these are SILENTLY SKIPPED "
                f"downstream")
        if unused:
            self.get_logger().warn(
                f"classes sized in {Path(path).name} that this model never "
                f"emits: {unused} -- dead config, or a class-name typo")
        if not unsized and not unused:
            self.get_logger().info(
                f"class names match {Path(path).name} exactly ({len(emitted)} classes)")

    # ------------------------------------------------------------------ #
    def _on_image(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"cv_bridge failed: {exc}", throttle_duration_sec=5.0)
            return
        # The Header goes in WITH the frame. See the module docstring.
        self.worker.submit(frame, msg.header)

    def _drain(self) -> None:
        got = self.worker.take()
        if got is None:
            return
        dets, header, seq = got
        if seq == self._last_seq:
            return
        self._last_seq = seq
        self.pub.publish(self._to_msg(dets, header))
        self._published += 1

    @staticmethod
    def _to_msg(dets, header: Header) -> Detection2DArray:
        """YOLO xyxy corners -> vision_msgs centre+size.

        vision_msgs/BoundingBox2D carries the box CENTRE and its size, not two
        corners. fixture_map_test reads det.bbox.center.position.x and
        det.bbox.size_x directly, so getting this wrong puts every landmark at
        the top-left of its box -- a bearing error that looks like a frame
        problem and will send you hunting through transforms.

        Field paths below are vision_msgs 4.x (Humble / Iron / Jazzy). Older
        vision_msgs put x/y directly on center, with no .position.
        """
        msg = Detection2DArray()
        msg.header = header                     # stamp AND frame_id, verbatim
        for (x1, y1, x2, y2), conf, cls_name in dets:
            det = Detection2D()
            det.header = header
            det.bbox.center.position.x = (x1 + x2) / 2.0
            det.bbox.center.position.y = (y1 + y2) / 2.0
            det.bbox.center.theta = 0.0
            det.bbox.size_x = abs(x2 - x1)
            det.bbox.size_y = abs(y2 - y1)

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = cls_name
            hyp.hypothesis.score = conf
            det.results.append(hyp)

            msg.detections.append(det)
        return msg

    def _heartbeat(self) -> None:
        if self._published == 0:
            self.get_logger().warn(
                "no detections published yet -- is ~/image remapped to a real "
                "topic, and is its publisher BEST_EFFORT?")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = YoloDetectionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.worker.stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
