# Intel RealSense D400s Imaging with Multiple Camera Setting
 
# Overview
The project is a custom implementation of Intel RealSense Python Wrapper to achieve synchronized imaging of overlapped field of view using multiple camera setup. The output of the sample code can be cross-registered point cloud in `npy` or `ply` format, or a RGB-D image pair.

## Example





## Notes
Multiple camera imaging also supports Framos cameras D415e. However, Intel RealSense device manager only supports connection over USB, Framos SDK provides support for GigE connection. Download SDK via [https://www.framos.com/en/industrial-depth-cameras#downloads].


# Requirements

- Install Python3
- In virtual environment, run `pip install opencv-python numpy pyrealsense2 open3d`

If module `pyrealsense2` is cannot be found via `pip`, consider [https://github.com/IntelRealSense/librealsense/issues/5777#issuecomment-582480988].


# Workflow

1. Calibration

Place Charuco board inside field of view. The Charuco board used in the project is 9x12. To change a Charuco board size, open `calibration.py` and access Charuco definition in funtion `cv_find_charuco`.

![alt text](https://github.com/StevenSunS/IntelRealSense-multicam_depth_image/blob/main/calibration_instruction.png)

2. Start the Program

3. RGB-D Data Live Stream

4. Save RGB-D Image Pair and Point Cloud Data upon Exit


# Reference
Box dimensions calculation using multiple realsense camera: [https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python/examples/box_dimensioner_multicam]
