#!/usr/bin/env python3
import os
import json
import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def load_camera_intrinsics(intrinsics_file):
    """Load camera intrinsics from JSON file"""
    with open(intrinsics_file, 'r') as f:
        intrinsics = json.load(f)
    
    # Extract camera matrix and distortion coefficients
    camera_matrix = np.array(intrinsics['camera_matrix'])
    dist_coeffs = np.array(intrinsics['distortion_coefficients'])
    
    return camera_matrix, dist_coeffs


def load_robot_poses(robot_poses_file):
    """Load robot joint poses from text file"""
    poses = []
    with open(robot_poses_file, 'r') as f:
        for line in f:
            # Assuming each line contains joint values or transformation matrix
            # Format may vary based on actual data
            values = [float(val) for val in line.strip().split()]
            poses.append(values)
    
    return poses


def parse_robot_poses(poses):
    """Convert robot poses to transformation matrices"""
    transformation_matrices = []
    
    for pose in poses:
        # This parsing depends on the format of your robot poses
        # Assuming pose contains [x, y, z, qx, qy, qz, qw] or similar
        # You may need to adjust based on your actual data format
        
        if len(pose) == 7:  # Position and quaternion format
            pos = pose[:3]
            quat = pose[3:]
            
            # Create rotation matrix from quaternion
            rot_matrix = R.from_quat(quat).as_matrix()
            
            # Create 4x4 transformation matrix
            T = np.eye(4)
            T[:3, :3] = rot_matrix
            T[:3, 3] = pos
            
        elif len(pose) == 6:  # Position and Euler angles (XYZ) format
            pos = pose[:3]
            euler = pose[3:]
            
            # Create rotation matrix from Euler angles
            rot_matrix = R.from_euler('xyz', euler).as_matrix()
            
            # Create 4x4 transformation matrix
            T = np.eye(4)
            T[:3, :3] = rot_matrix
            T[:3, 3] = pos
            
        elif len(pose) == 16:  # Already a flattened 4x4 matrix
            T = np.array(pose).reshape(4, 4)
            
        else:
            raise ValueError(f"Unexpected pose format with {len(pose)} values")
        
        transformation_matrices.append(T)
    
    return transformation_matrices


def detect_chessboard(images_dir, camera_matrix, dist_coeffs, board_size=(12, 9), square_size=0.015):
    """Detect chessboard corners in all images and compute object poses"""
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    rvecs = []      # Rotation vectors
    tvecs = []      # Translation vectors
    image_names = []  # To keep track of successful detections
    
    # Get all image files
    image_files = sorted(glob(os.path.join(images_dir, '*.jpg'))) + \
                  sorted(glob(os.path.join(images_dir, '*.png')))
    
    for image_file in image_files:
        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Compute pose of the chessboard
            ret, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)
            
            # Store the data
            objpoints.append(objp)
            imgpoints.append(corners2)
            rvecs.append(rvec)
            tvecs.append(tvec)
            image_names.append(os.path.basename(image_file))
            
            # Draw and display the corners (optional for debugging)
            # img = cv2.drawChessboardCorners(img, board_size, corners2, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(500)
    
    # cv2.destroyAllWindows()
    
    # Convert rvecs and tvecs to transformation matrices
    camera_T_object = []
    for rvec, tvec in zip(rvecs, tvecs):
        R_matrix, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R_matrix
        T[:3, 3] = tvec.reshape(3)
        camera_T_object.append(T)
    
    return camera_T_object, image_names


def hand_eye_calibration(base_T_gripper, camera_T_object, calibration_method=cv2.CALIB_HAND_EYE_TSAI):
    """Perform hand-eye calibration
    
    Args:
        base_T_gripper: List of 4x4 transformation matrices from robot base to gripper
        camera_T_object: List of 4x4 transformation matrices from camera to calibration object
        calibration_method: Method to use for calibration
        
    Returns:
        gripper_T_camera: 4x4 transformation matrix from gripper to camera
    """
    # Extract rotation and translation components
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []
    
    for i in range(len(base_T_gripper)):
        R_gripper2base.append(base_T_gripper[i][:3, :3])
        t_gripper2base.append(base_T_gripper[i][:3, 3])
        
        R_target2cam.append(camera_T_object[i][:3, :3])
        t_target2cam.append(camera_T_object[i][:3, 3])
    
    # Convert lists to arrays
    R_gripper2base = np.array(R_gripper2base)
    t_gripper2base = np.array(t_gripper2base)
    R_target2cam = np.array(R_target2cam)
    t_target2cam = np.array(t_target2cam)
    
    # Perform hand-eye calibration
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=calibration_method
    )
    
    # Create transformation matrix from gripper to camera
    gripper_T_camera = np.eye(4)
    gripper_T_camera[:3, :3] = R_cam2gripper
    gripper_T_camera[:3, 3] = t_cam2gripper.reshape(3)
    
    return gripper_T_camera


def calculate_reprojection_error(base_T_gripper, camera_T_object, gripper_T_camera):
    """Calculate reprojection error for the hand-eye calibration"""
    errors = []
    
    for i in range(len(base_T_gripper)):
        # Calculate predicted camera_T_object
        predicted = np.dot(np.linalg.inv(base_T_gripper[i]), 
                           np.dot(np.linalg.inv(gripper_T_camera), np.eye(4)))
        
        # Calculate actual camera_T_object
        actual = camera_T_object[i]
        
        # Calculate error (using Frobenius norm of the difference)
        error = np.linalg.norm(predicted - actual, 'fro')
        errors.append(error)
    
    return errors


def main():
    # Paths to data files
    camera_intrinsics_file = 'configs/camera_intrinsics.json'
    robot_poses_file = 'data/robot_poses.txt'
    images_dir = 'data/camera_imgs'
    
    # Get absolute paths
    base_path = os.path.dirname(os.path.abspath(__file__))
    camera_intrinsics_file = os.path.join(base_path, camera_intrinsics_file)
    robot_poses_file = os.path.join(base_path, robot_poses_file)
    images_dir = os.path.join(base_path, images_dir)
    
    # Load camera intrinsics
    camera_matrix, dist_coeffs = load_camera_intrinsics(camera_intrinsics_file)
    print("Loaded camera intrinsics:")
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:", dist_coeffs)
    
    # Load robot poses
    robot_poses = load_robot_poses(robot_poses_file)
    print(f"Loaded {len(robot_poses)} robot poses")
    
    # Parse robot poses to transformation matrices
    base_T_gripper = parse_robot_poses(robot_poses)
    
    # Detect chessboard in images and compute poses
    # Adjust board_size and square_size based on your actual calibration pattern
    camera_T_object, valid_images = detect_chessboard(
        images_dir, camera_matrix, dist_coeffs, board_size=(9, 6), square_size=0.02
    )
    print(f"Detected chessboard in {len(camera_T_object)} images")
    
    if len(camera_T_object) != len(base_T_gripper):
        print("WARNING: Number of detected chessboards does not match number of robot poses")
        print(f"Using the first {min(len(camera_T_object), len(base_T_gripper))} valid poses")
        
        # Trim to the smaller size
        n = min(len(camera_T_object), len(base_T_gripper))
        camera_T_object = camera_T_object[:n]
        base_T_gripper = base_T_gripper[:n]
    
    if len(camera_T_object) < 4:
        print("ERROR: At least 4 poses are required for calibration")
        return
    
    # Perform hand-eye calibration
    print("Performing hand-eye calibration...")
    methods = {
        'TSAI': cv2.CALIB_HAND_EYE_TSAI,
        'PARK': cv2.CALIB_HAND_EYE_PARK,
        'HORAUD': cv2.CALIB_HAND_EYE_HORAUD,
        'ANDREFF': cv2.CALIB_HAND_EYE_ANDREFF,
        'DANIILIDIS': cv2.CALIB_HAND_EYE_DANIILIDIS
    }
    
    results = {}
    for method_name, method in methods.items():
        gripper_T_camera = hand_eye_calibration(base_T_gripper, camera_T_object, method)
        
        # Calculate reprojection error
        errors = calculate_reprojection_error(base_T_gripper, camera_T_object, gripper_T_camera)
        avg_error = np.mean(errors)
        
        results[method_name] = {
            'transform': gripper_T_camera,
            'error': avg_error
        }
        
        print(f"\nMethod: {method_name}")
        print(f"Average reprojection error: {avg_error:.6f}")
        print("Transformation matrix (gripper to camera):")
        print(gripper_T_camera)
    
    # Find the best method based on reprojection error
    best_method = min(results, key=lambda k: results[k]['error'])
    best_transform = results[best_method]['transform']
    
    print("\n---------------------------------------------")
    print(f"Best method: {best_method} with error {results[best_method]['error']:.6f}")
    print("Best transformation matrix (gripper to camera):")
    print(best_transform)
    
    # Extract rotation and translation components for better readability
    rotation = best_transform[:3, :3]
    translation = best_transform[:3, 3]
    
    # Convert rotation matrix to different representations
    euler_angles = R.from_matrix(rotation).as_euler('xyz', degrees=True)
    quaternion = R.from_matrix(rotation).as_quat()  # xyzw format
    
    print("\nRotation matrix:")
    print(rotation)
    print("\nTranslation vector (meters):")
    print(translation)
    print("\nEuler angles XYZ (degrees):")
    print(euler_angles)
    print("\nQuaternion (xyzw):")
    print(quaternion)
    
    # Save the calibration result
    result = {
        'gripper_T_camera': best_transform.tolist(),
        'rotation_matrix': rotation.tolist(),
        'translation_vector': translation.tolist(),
        'euler_angles_xyz_degrees': euler_angles.tolist(),
        'quaternion_xyzw': quaternion.tolist(),
        'calibration_method': best_method,
        'reprojection_error': float(results[best_method]['error']),
        'num_poses_used': len(camera_T_object)
    }
    
    result_file = os.path.join(base_path, 'data/hand_eye_calibration_result.json')
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"\nCalibration result saved to {result_file}")


if __name__ == "__main__":
    main()