"""Replay a saved capture directory into the ROS graph, as if it were a camera.

Without USB passthrough the ZED cannot be opened inside WSL, but nothing in
stage 1 needs a LIVE camera -- it needs frames and intrinsics. This node reads
a directory written by capture_query.py on the machine the camera works on and
publishes it onto the same topics the ZED wrapper would, so the whole pipeline
downstream runs unchanged:

    capture_player  ->  lasr_perception/yolo_node  ->  fixture_map_test  ->  RViz

Expects the layout capture_query.py already writes:

    <capture_dir>/
      intrinsics.json              {fx, fy, cx, cy, width, height}
      frames/frame_00000.png ...   left rectified image
      detections/frame_00000.json  optional -- only used with publish_detections

REPLAY IS BETTER THAN LIVE FOR TUNING, NOT JUST A FALLBACK
----------------------------------------------------------
The number you are about to spend real time on is width_m, and the honest way
to tune it is to change one value and see what moves. Against a live camera
every run differs in pose, lighting and detector jitter, so you are reading a
difference through noise. Against a recording the frames are byte-identical
every time and the ONLY thing that changed is the number you changed.

ON PUBLISHING SAVED DETECTIONS
------------------------------
publish_detections replays the boxes capture_query.py stored, instead of
running YOLO now. That isolates the two halves of the pipeline:

  * OFF (default): images only -> yolo_node runs inference in WSL. Tests the
    real chain, including the class-name contract and the timestamp handling.
  * ON: saved boxes -> yolo_node not needed at all. If the map is wrong with
    detections held fixed, the fault is in geometry, frames or config -- it
    cannot be the detector, because the detector did not run.

Turning it on is how you answer "is my map wrong because of the detector or
because of my numbers?" without changing anything else.

TIMESTAMPS
----------
Image, CameraInfo and Detection2DArray for a given frame are stamped
IDENTICALLY. fixture_map_test pairs detections with images through an
ApproximateTimeSynchronizer with a 50 ms slop, so a player that stamped them
independently would drop pairs and look like a sync bug in code that is fine.
"""
from __future__ import annotations

import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from cv_bridge import CvBridge
import cv2

# Matches the ZED wrapper and both consumers. A RELIABLE publisher will not
# connect to a BEST_EFFORT subscriber.
BEST_EFFORT_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    durability=QoSDurabilityPolicy.VOLATILE,
    depth=10,
)


class CapturePlayerNode(Node):
    def __init__(self) -> None:
        super().__init__("capture_player")

        self.declare_parameter("capture_dir", "")
        self.declare_parameter("rate_hz", 2.0)
        self.declare_parameter("loop", True)
        self.declare_parameter("publish_detections", False)
        self.declare_parameter("optical_frame", "zed_left_camera_optical_frame")

        g = self.get_parameter
        capture_dir = str(g("capture_dir").value)
        if not capture_dir:
            raise RuntimeError(
                "the 'capture_dir' parameter is required -- point it at a "
                "directory written by capture_query.py")
        self.dir = Path(capture_dir)
        if not self.dir.is_dir():
            raise RuntimeError(f"capture_dir is not a directory: {self.dir}")

        self.optical_frame = str(g("optical_frame").value)
        self.loop = bool(g("loop").value)
        self.publish_dets = bool(g("publish_detections").value)
        rate = float(g("rate_hz").value)

        self.camera_info = self._load_camera_info(self.dir / "intrinsics.json")
        self.frames = sorted((self.dir / "frames").glob("frame_*.png"))
        if not self.frames:
            raise RuntimeError(f"no frames/frame_*.png under {self.dir}")

        self.bridge = CvBridge()
        self.info_pub = self.create_publisher(CameraInfo, "~/camera_info", BEST_EFFORT_QOS)
        self.image_pub = self.create_publisher(Image, "~/image", BEST_EFFORT_QOS)
        self.det_pub = (
            self.create_publisher(Detection2DArray, "~/detections", BEST_EFFORT_QOS)
            if self.publish_dets else None)

        self.i = 0
        self.laps = 0
        self.create_timer(1.0 / max(rate, 0.01), self._tick)

        self.get_logger().info(
            f"capture_player: {len(self.frames)} frame(s) from {self.dir.name} "
            f"at {rate:.1f} Hz, loop={self.loop}, "
            f"detections={'REPLAYED from disk' if self.publish_dets else 'left to yolo_node'}")

    # ------------------------------------------------------------------ #
    def _load_camera_info(self, path: Path) -> CameraInfo:
        """intrinsics.json -> CameraInfo, with k laid out row-major.

        k is [fx 0 cx; 0 fy cy; 0 0 1] flattened. fixture_map_test reads k[0],
        k[4], k[2], k[5] and refuses a zero focal length, so a wrong layout
        here fails loudly rather than producing plausible wrong ranges.
        """
        if not path.is_file():
            raise RuntimeError(
                f"{path} not found. Generate it on the machine the camera "
                f"works on:  python dump_intrinsics.py --out intrinsics.json")
        d = json.loads(path.read_text())
        for key in ("fx", "fy", "cx", "cy"):
            if key not in d:
                raise RuntimeError(f"{path} has no {key!r}")

        msg = CameraInfo()
        msg.header.frame_id = str(self.get_parameter("optical_frame").value)
        msg.width = int(d.get("width") or 0)
        msg.height = int(d.get("height") or 0)
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]   # frames are already rectified
        fx, fy, cx, cy = float(d["fx"]), float(d["fy"]), float(d["cx"]), float(d["cy"])
        msg.k = [fx, 0.0, cx,
                 0.0, fy, cy,
                 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [fx, 0.0, cx, 0.0,
                 0.0, fy, cy, 0.0,
                 0.0, 0.0, 1.0, 0.0]
        self.get_logger().info(
            f"intrinsics from {path.name}: fx={fx:.1f} fy={fy:.1f} "
            f"cx={cx:.1f} cy={cy:.1f}  {msg.width}x{msg.height}")
        return msg

    def _load_detections(self, frame_stem: str, header) -> Detection2DArray:
        """capture_query.py's corner boxes -> vision_msgs centre+size."""
        msg = Detection2DArray()
        msg.header = header
        path = self.dir / "detections" / f"{frame_stem}.json"
        if not path.is_file():
            return msg
        for d in json.loads(path.read_text()):
            x1, y1, x2, y2 = (float(v) for v in d["bbox_xyxy"])
            det = Detection2D()
            det.header = header
            det.bbox.center.position.x = (x1 + x2) / 2.0
            det.bbox.center.position.y = (y1 + y2) / 2.0
            det.bbox.center.theta = 0.0
            det.bbox.size_x = abs(x2 - x1)
            det.bbox.size_y = abs(y2 - y1)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(d["class_name"])
            hyp.hypothesis.score = float(d.get("confidence", 1.0))
            det.results.append(hyp)
            msg.detections.append(det)
        return msg

    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        if self.i >= len(self.frames):
            if not self.loop:
                self.get_logger().info("end of capture -- shutting down")
                raise SystemExit(0)
            self.i = 0
            self.laps += 1
            self.get_logger().info(f"looped ({self.laps})")

        path = self.frames[self.i]
        self.i += 1

        cv_image = cv2.imread(str(path))
        if cv_image is None:
            self.get_logger().warn(f"could not read {path.name} -- skipping")
            return

        # One stamp, shared by every message for this frame.
        stamp = self.get_clock().now().to_msg()

        img_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = self.optical_frame

        self.camera_info.header.stamp = stamp
        self.camera_info.header.frame_id = self.optical_frame

        self.info_pub.publish(self.camera_info)
        self.image_pub.publish(img_msg)

        if self.det_pub is not None:
            dets = self._load_detections(path.stem, img_msg.header)
            self.det_pub.publish(dets)
            self.get_logger().info(
                f"{path.name}: replayed {len(dets.detections)} saved detection(s)",
                throttle_duration_sec=2.0)
        else:
            self.get_logger().info(f"{path.name}", throttle_duration_sec=2.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = CapturePlayerNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
