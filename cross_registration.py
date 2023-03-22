
import numpy as np
import cv2
import pyrealsense2 as rs


def convert_depth_frame_to_pointcloud(depth_image, camera_intrinsics):

	[height, width] = depth_image.shape

	nx = np.linspace(0, width-1, width)
	ny = np.linspace(0, height-1, height)
	u, v = np.meshgrid(nx, ny)
	x = (u.flatten() - camera_intrinsics.ppx)/camera_intrinsics.fx
	y = (v.flatten() - camera_intrinsics.ppy)/camera_intrinsics.fy
		
	z = depth_image.flatten() / 1000
	x = np.multiply(x,z)
	y = np.multiply(y,z)

	u = u.flatten()
	v = v.flatten()

	return x, y, z

def convert_pointcloud_to_depth_frame(point_cloud, camera_intrinsics):

	x = point_cloud[0,:]
	y = point_cloud[1,:]
	z = point_cloud[2,:]

	u = camera_intrinsics.fx*x + camera_intrinsics.ppx*z
	v = camera_intrinsics.fy*y + camera_intrinsics.ppy*z
	
	u = np.divide(u,z)
	v = np.divide(v,z)
	
	return u, v, z
	
def calculate_cumulative_pointcloud(frames_devices, calibration_info_devices):

	point_cloud_cumulative = np.array([-1, -1, -1, -1, -1, -1]).transpose()

	for (device_info, frame) in frames_devices.items() :

		device = device_info[0]

		depth_image = np.asarray(frame[rs.stream.depth].get_data())
		cv2.imwrite(str(device) + '_depth.png', depth_image/2000*255)

		point_cloud = convert_depth_frame_to_pointcloud(depth_image, calibration_info_devices[device][1][rs.stream.depth] )

		color_cloud = np.reshape(np.asarray(frame[rs.stream.color].get_data()), (640*480, 3))
		color_cloud = color_cloud.T

		point_cloud = np.asanyarray(point_cloud)
		
		point_cloud = calibration_info_devices[device][0].apply_transformation(point_cloud)
		point_cloud = np.row_stack((point_cloud, color_cloud))

		
		# Set desired field of view
		# point_cloud = point_cloud[:,point_cloud[0,:] > -0.2]
		# point_cloud = point_cloud[:,point_cloud[0,:] < 0.6]
		# point_cloud = point_cloud[:,point_cloud[1,:] > -0.15]
		# point_cloud = point_cloud[:,point_cloud[1,:] < 0.3]

		point_cloud = point_cloud + 0.2

		point_cloud_cumulative = np.column_stack(( point_cloud_cumulative, point_cloud ))

		np.save(str(device) + '_pointcloud.npy', point_cloud) 

	point_cloud_cumulative = np.delete(point_cloud_cumulative, 0, 1)
	np.save('cumulative_pointcloud.npy', point_cloud_cumulative) 

	return point_cloud_cumulative, point_cloud
