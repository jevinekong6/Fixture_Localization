import pyzed.sl as sl

def main():
    # Initialize the camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Choose your resolution
    
    # Optional: Disable self-calibration if you need strict static factory values
    # init_params.camera_disable_self_calib = True

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED Mini camera.")
        return

    # Retrieve camera calibration parameters
    # Note: Pass sl.RESOLUTION.HD720 (or your target) to get rectified intrinsics for that format
    cam_info = zed.get_camera_information()
    calibration_params = cam_info.camera_configuration.calibration_parameters

    # 1. Left Camera Intrinsics
    left_intrinsics = calibration_params.left_cam
    print("--- Left Camera Intrinsics ---")
    print(f"Focal Length (fx, fy): ({left_intrinsics.fx}, {left_intrinsics.fy})")
    print(f"Principal Point (cx, cy): ({left_intrinsics.cx}, {left_intrinsics.cy})")
    print(f"Distortion Coefficients [k1, k2, p1, p2, k3]: {list(left_intrinsics.disto)}")

    # 2. Right Camera Intrinsics
    right_intrinsics = calibration_params.right_cam
    print("\n--- Right Camera Intrinsics ---")
    print(f"Focal Length (fx, fy): ({right_intrinsics.fx}, {right_intrinsics.fy})")
    print(f"Principal Point (cx, cy): ({right_intrinsics.cx}, {right_intrinsics.cy})")
    print(f"Distortion Coefficients [k1, k2, p1, p2, k3]: {list(right_intrinsics.disto)}")

    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()
