
import numpy as np
import cv2
import skimage.measure


def preprocess_pcd(point_cloud, scale):

    point_cloud = point_cloud * scale
    point_cloud = point_cloud.astype(int)
    
    x_0 = np.min(point_cloud[0,:])
    y_0 = np.min(point_cloud[1,:])

    x_1 = np.max(point_cloud[0,:])
    y_1 = np.max(point_cloud[1,:])

    width = x_1 - x_0
    height = y_1 - y_0

    point_cloud_norm = np.zeros(point_cloud.shape).astype(int)

    point_cloud_norm[0,:] = point_cloud[0,:] - x_0
    point_cloud_norm[1,:] = point_cloud[1,:] - y_0
    point_cloud_norm[2,:] = point_cloud[2,:]

    point_cloud_norm = point_cloud_norm.T

    point_cloud_norm = point_cloud_norm[point_cloud_norm[:, 2].argsort()]

    return point_cloud_norm, width, height, scale


def localize(point_cloud, scale = 4e3):

    point_cloud, width, height, scale = preprocess_pcd(point_cloud, scale)

    u = point_cloud[:,0]
    v = point_cloud[:,1]
    depth = point_cloud[:,2]

    mesh = np.zeros((height + 1, width + 1)).astype(int)

    for _, (x, y, z) in enumerate(zip(u, v, depth)):
        mesh[y, x] = z

    mesh = mesh/scale

    return mesh


def max_pool(mat, block_size = (10, 10)):

    image = skimage.measure.block_reduce(mat, block_size, np.max)

    return image


def generate_depth(depth_image, dim):

    depth_image = depth_image * 255
    depth_image = cv2.resize(depth_image, dim, interpolation = cv2.INTER_AREA)

    return depth_image


def generate_rgb(blue_image, green_image, red_image, dim):

    color_image = np.dstack((red_image, green_image, blue_image))
    color_image = cv2.resize(color_image, dim, interpolation = cv2.INTER_AREA)

    return color_image


def rgbd_projection(pcd, dim = (640, 480)):

    pixel_cloud  = localize(pcd[:3, :])

    red_cloud = localize(pcd[[0,1,3], :])
    green_cloud = localize(pcd[[0,1,4], :])
    blue_cloud = localize(pcd[[0,1,5], :])

    depth_image = max_pool(pixel_cloud)
    
    red_image = max_pool(red_cloud, block_size = (9, 9))
    green_image = max_pool(green_cloud, block_size = (9, 9))
    blue_image = max_pool(blue_cloud, block_size = (9, 9))

    color_image = generate_rgb(blue_image, green_image, red_image, dim)
    depth_image = generate_depth(depth_image, dim)

    color_image = np.flipud(color_image)
    depth_image = np.flipud(depth_image)

    return color_image, depth_image



def pixel_quantization(pointcloud, dim = (480, 640)):

    # x = pointcloud[:0, :]
    # y = pointcloud[:1, :]

    # x = np.round(x)
    # y = np.round(y)

    # _, index = np.unique([x, y], axis = 1, return_index=True)

    # depth_pixel = pointcloud[index]

    u = np.round(pointcloud[0, :]).astype(int)
    v = np.round(pointcloud[1, :]).astype(int)
    depth = pointcloud[2, :] * 1000
    depth = depth.astype(int)

    mesh = np.zeros(dim).astype(int)

    for _, (x, y, z) in enumerate(zip(u, v, depth)):

        if y > 479 or x > 639:
            continue

        mesh[y, x] = z

    return mesh



