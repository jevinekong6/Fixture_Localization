"""Tripod-stage bringup: two static transforms plus the mapping node.

There is no robot and no pose estimation in this stage. The camera does not
move, so a pair of static transforms IS the localization system, and the
accuracy of the whole test is bounded by how carefully you measured the tripod.

    map ---(measured tripod pose)---> zed_camera_link ---(fixed)---> optical
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

PKG = "fixture_map_test"


def generate_launch_description():
    pkg_share = FindPackageShare(PKG)

    args = [
        # --- measured tripod pose, map -> camera BODY origin -------------- #
        DeclareLaunchArgument("cam_x", default_value="0.0",
                              description="tripod x in the map frame, metres"),
        DeclareLaunchArgument("cam_y", default_value="0.0",
                              description="tripod y in the map frame, metres"),
        DeclareLaunchArgument("cam_z", default_value="1.20",
                              description="camera height above the map plane, metres"),
        DeclareLaunchArgument("cam_yaw", default_value="0.0",
                              description="tripod yaw, radians, right-handed about map +z"),
        DeclareLaunchArgument("cam_pitch", default_value="0.0",
                              description="tripod pitch, radians. Nose-down is positive"),
        DeclareLaunchArgument("cam_roll", default_value="0.0",
                              description="tripod roll, radians"),

        # --- frames ------------------------------------------------------- #
        DeclareLaunchArgument("map_frame", default_value="map"),
        DeclareLaunchArgument("body_frame", default_value="zed_camera_link"),
        DeclareLaunchArgument("optical_frame",
                              default_value="zed_left_camera_optical_frame"),
        DeclareLaunchArgument("publish_optical_tf", default_value="true",
                              description="set false if the ZED wrapper already "
                                          "publishes body -> optical"),

        # --- topics --------------------------------------------------------#
        DeclareLaunchArgument("detections_topic", default_value="/yolo/detections"),
        DeclareLaunchArgument("image_topic",
                              default_value="/zed/zed_node/left/image_rect_color"),
        DeclareLaunchArgument("camera_info_topic",
                              default_value="/zed/zed_node/left/camera_info"),

        # --- files and gates ------------------------------------------------#
        DeclareLaunchArgument("class_config", default_value=PathJoinSubstitution(
            [pkg_share, "config", "fixture_classes.yaml"])),
        DeclareLaunchArgument("apriori_map", default_value=PathJoinSubstitution(
            [pkg_share, "config", "apriori_map.yaml"])),
        DeclareLaunchArgument("output_map", default_value="fixture_map_out.yaml"),
        DeclareLaunchArgument("patch_dir", default_value="fixture_patches"),
        DeclareLaunchArgument("min_confidence", default_value="0.40"),
        DeclareLaunchArgument("max_range", default_value="6.0"),
        DeclareLaunchArgument("bbox_sigma_px", default_value="3.0"),
        DeclareLaunchArgument("confirm_radius", default_value="0.15"),
        DeclareLaunchArgument("flag_radius", default_value="0.60"),
        DeclareLaunchArgument("assoc_radius", default_value="0.30"),
        DeclareLaunchArgument("save_period", default_value="10.0"),
        DeclareLaunchArgument("use_image", default_value="true"),
    ]

    lc = LaunchConfiguration

    # ---------------------------------------------------------------------- #
    # 1. map -> camera BODY frame. THE measured number.
    #
    # This is the tripod pose you took with a tape measure and a level, and it
    # is the substitute for localization at this stage of the project. Every
    # landmark this package produces is rigidly offset by whatever error is in
    # these six values, and no amount of averaging observations will find that
    # error -- it is common to every observation. Measure to the camera BODY
    # origin (the ZED's own zed_camera_link datum, see the ZED drawing), not to
    # the front glass and not to the tripod head.
    #
    # static_transform_publisher argument order is:
    #     --x --y --z --yaw --pitch --roll --frame-id --child-frame-id
    # yaw-pitch-roll, in that order, applied as intrinsic Z-Y-X.
    # ---------------------------------------------------------------------- #
    tripod_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_to_camera_body",
        arguments=[
            "--x", lc("cam_x"), "--y", lc("cam_y"), "--z", lc("cam_z"),
            "--yaw", lc("cam_yaw"), "--pitch", lc("cam_pitch"), "--roll", lc("cam_roll"),
            "--frame-id", lc("map_frame"),
            "--child-frame-id", lc("body_frame"),
        ],
    )

    # ---------------------------------------------------------------------- #
    # 2. camera BODY -> camera OPTICAL. DO NOT EDIT THIS QUATERNION.
    #
    # (qx, qy, qz, qw) = (-0.5, 0.5, -0.5, 0.5) is the fixed ROS convention
    # rotation between a body frame (+x forward, +y left, +z up) and an optical
    # frame (+x right, +y down, +z forward). It is not a calibration, it is not
    # camera-specific, and it has no correct variant: it maps optical +z to body
    # +x, optical +y to body -z, and optical +x to body -y. scripts/check_frames.py
    # asserts exactly that. If landmarks come out rotated by 90 degrees, the
    # fault is somewhere else and changing these four numbers will only hide it
    # in one pose and expose it in the next.
    #
    # IF THE ZED WRAPPER IS RUNNING, IT ALREADY PUBLISHES THIS TRANSFORM.
    # Two publishers writing the same child frame produce a TF tree that flips
    # between them at whatever rate they happen to run, which looks like
    # intermittent noise and is nearly impossible to debug from RViz. In that
    # case drop this node -- launch with publish_optical_tf:=false -- rather
    # than competing for the frame.
    # ---------------------------------------------------------------------- #
    optical_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_body_to_optical",
        condition=IfCondition(lc("publish_optical_tf")),
        arguments=[
            "--x", "0.0", "--y", "0.0", "--z", "0.0",
            "--qx", "-0.5", "--qy", "0.5", "--qz", "-0.5", "--qw", "0.5",
            "--frame-id", lc("body_frame"),
            "--child-frame-id", lc("optical_frame"),
        ],
    )

    fixture_node = Node(
        package=PKG,
        executable="fixture_map_node",
        name="fixture_map_node",
        output="screen",
        parameters=[{
            "map_frame": lc("map_frame"),
            "optical_frame": lc("optical_frame"),
            "class_config": lc("class_config"),
            "apriori_map": lc("apriori_map"),
            "output_map": lc("output_map"),
            "patch_dir": lc("patch_dir"),
            "bbox_sigma_px": lc("bbox_sigma_px"),
            "confirm_radius": lc("confirm_radius"),
            "flag_radius": lc("flag_radius"),
            "assoc_radius": lc("assoc_radius"),
            "max_range": lc("max_range"),
            "min_confidence": lc("min_confidence"),
            "save_period": lc("save_period"),
            "use_image": lc("use_image"),
        }],
        remappings=[
            ("~/detections", lc("detections_topic")),
            ("~/image", lc("image_topic")),
            ("~/camera_info", lc("camera_info_topic")),
        ],
    )

    return LaunchDescription(args + [tripod_tf, optical_tf, fixture_node])
