
import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d

from collections import defaultdict
from device_manager import DeviceManager, save_frame, save_pcd, stream_frame
from cross_registration import calculate_cumulative_pointcloud
from calibration import PoseEstimation
from depth_projection import rgbd_projection

def stream():

	resolution_width = 640 # pixels
	resolution_height = 480 # pixels
	frame_rate = 30  # fps

	dispose_frames_for_stablisation = 30  # frames

	chessboard_width =12 # squares
	chessboard_height = 9 	# squares
	square_size = 0.03 # meters

	# chessboard_width = 6 # squares
	# chessboard_height = 9 	# squares
	# square_size = 0.023 # meters

	try:
		# Enable the streams from all the intel realsense devices
		rs_config = rs.config()
		rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
		rs_config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, frame_rate)
		rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)

		# Use the device manager class to enable the devices and get the frames
		device_manager = DeviceManager(rs.context(), rs_config)
		device_manager.enable_all_devices()
		
		# Allow some frames for the auto-exposure controller to stablise
		for frame in range(dispose_frames_for_stablisation):
			frames = device_manager.poll_frames()

		assert(len(device_manager._available_devices) > 0)

		# Get the intrinsics of the realsense device 
		intrinsics_devices = device_manager.get_device_intrinsics(frames)
		
				# Set the chessboard parameters for calibration 
		chessboard_params = [chessboard_height, chessboard_width, square_size] 
		
		# Estimate the pose of the chessboard in the world coordinate using the Kabsch Method
		calibrated_device_count = 0
		while calibrated_device_count < len(device_manager._available_devices):
			frames = device_manager.poll_frames()
			pose_estimator = PoseEstimation(frames, intrinsics_devices, chessboard_params)
			transformation_result_kabsch  = pose_estimator.perform_pose_estimation()
			object_point = pose_estimator.get_chessboard_corners_in3d()
			calibrated_device_count = 0
			for device_info in device_manager._available_devices:
				device = device_info[0]
				if not transformation_result_kabsch[device][0]:
					print("Place the chessboard on the plane where the object needs to be detected..")
				else:
					calibrated_device_count += 1

		# Save the transformation object for all devices in an array to use for measurements
		transformation_devices={}
		chessboard_points_cumulative_3d = np.array([-1,-1,-1]).transpose()
		for device_info in device_manager._available_devices:
			device = device_info[0]
			transformation_devices[device] = transformation_result_kabsch[device][1].inverse()
			# points3D = object_point[device][2][:,object_point[device][3]]
			# points3D = transformation_devices[device].apply_transformation(points3D)
			# chessboard_points_cumulative_3d = np.column_stack( (chessboard_points_cumulative_3d,points3D) )

		# Extract the bounds between which the object's dimensions are needed
		# It is necessary for this demo that the object's length and breath is smaller than that of the chessboard
		# chessboard_points_cumulative_3d = np.delete(chessboard_points_cumulative_3d, 0, 1)

		print("Calibration completed... \nPlace the box in the field of view of the devices...")

		# Enable the emitter of the devices
		device_manager.enable_emitter(True) #, disenable_device=['802212060162']

		# Load the JSON settings file in order to enable High Accuracy preset for the realsense
		device_manager.load_settings_json("./HighResHighAccuracyPreset.json")

		# Get the extrinsics of the device to be used later
		extrinsics_devices = device_manager.get_depth_to_color_extrinsics(frames)

		# Get the calibration info as a dictionary to help with display of the measurements onto the color image instead of infra red image
		calibration_info_devices = defaultdict(list)
		for calibration_info in (transformation_devices, intrinsics_devices, extrinsics_devices):
			for key, value in calibration_info.items():
				calibration_info_devices[key].append(value)
				print(key)
				print(value)

		# Continue acquisition until terminated with Ctrl+C by the user
		while True:

			# Get the frames from all the devices
			frames_devices = device_manager.poll_frames()

			stream_frame(frames_devices)

			# Calculate the pointcloud using the depth frames from all the devices
			merged_point_cloud, main_point_cloud = calculate_cumulative_pointcloud(frames_devices, calibration_info_devices)

			color_image, depth_image = rgbd_projection(merged_point_cloud, dim = (640, 480))
			
			cv2.imshow('Color image', color_image/255)
			cv2.imshow('Depth image', depth_image/255)
			cv2.waitKey(1)

	except KeyboardInterrupt:

		print("The program was interupted by the user. Closing the program...")

	finally:

		cv2.imwrite('merged_color.png', color_image)
		cv2.imwrite('merged_depth.png', depth_image)		
		save_frame(frames_devices)
		save_pcd(merged_point_cloud, main_point_cloud)
		print('frame saved')

		device_manager.disable_streams()
		cv2.destroyAllWindows()

if __name__ == "__main__":
	stream()
