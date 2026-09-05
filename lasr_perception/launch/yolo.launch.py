"""Bring up the YOLO bridge alone.

To run the whole stage-1 stack, launch this and fixture_map_test's
tripod_test.launch.py together -- they are kept separate on purpose so the
detector can be swapped, or run without the mapper, or vice versa.

    ros2 launch lasr_perception yolo.launch.py \\
        weights:=/path/to/LASR-CV_App/train/runs/fixture_recognition_9/weights/best.pt

Pass class_config so the class-name cross-check runs at startup. A mismatch
there is the single most common integration failure and it is otherwise
silent -- the detection is skipped downstream with a throttled warning, which
looks like "my fixture never appears" rather than like an error.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    args = [
        DeclareLaunchArgument("weights", description="path to the trained best.pt"),
        DeclareLaunchArgument("image_topic",
                              default_value="/zed/zed_node/left/image_rect_color",
                              description="RECTIFIED image -- CameraInfo.k "
                                          "describes the rectified frame"),
        DeclareLaunchArgument("detections_topic", default_value="/yolo/detections"),
        DeclareLaunchArgument("conf_threshold", default_value="0.25"),
        DeclareLaunchArgument("iou_threshold", default_value="0.5"),
        DeclareLaunchArgument("class_config", default_value=PathJoinSubstitution(
            [FindPackageShare("fixture_map_test"), "config", "fixture_classes.yaml"])),
    ]

    lc = LaunchConfiguration
    node = Node(
        package="lasr_perception",
        executable="yolo_node",
        name="yolo_node",
        output="screen",
        parameters=[{
            "weights": lc("weights"),
            "conf_threshold": lc("conf_threshold"),
            "iou_threshold": lc("iou_threshold"),
            "class_config": lc("class_config"),
        }],
        remappings=[
            ("~/image", lc("image_topic")),
            ("~/detections", lc("detections_topic")),
        ],
    )
    return LaunchDescription(args + [node])
