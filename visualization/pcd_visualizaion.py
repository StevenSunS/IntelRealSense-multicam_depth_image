
import numpy as np
import open3d as o3d

print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud("merged.ply")
o3d.visualization.draw_geometries([pcd])

pcd = o3d.io.read_point_cloud("main.ply")
o3d.visualization.draw_geometries([pcd])