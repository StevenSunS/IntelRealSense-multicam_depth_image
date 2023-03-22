# Intel RealSense D400s Imaging with Multiple Camera Setting
 
# Overview



## Notes
Multiple camera imaging also supports Framos cameras D415e. However, Intel RealSense device manager only supports connection over USB, Framos SDK provides support for GigE connection. Download SDK via [https://www.framos.com/en/industrial-depth-cameras#downloads].


# Requirements

- Install Python3
- In virtual environment, run `pip install opencv-python numpy pyrealsense2 open3d`

If module `pyrealsense2` is cannot be found via `pip`, consider [https://github.com/IntelRealSense/librealsense/issues/5777#issuecomment-582480988].

# Reference
Box dimensions calculation using multiple realsense camera: [https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python/examples/box_dimensioner_multicam]
