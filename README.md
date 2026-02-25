# Feature-Based Image Alignment & Panorama Generation Project - Intro to Computer Graphics Course
This project implements a classical feature-based image alignment & panorama generation pipeline in Python, developed as part of the *"Introduction to Graphics, Image Processing & Computer Vision"* course at TAU’s School of Computer Science (2025–2026), during the final year of our studies. The project pipeline constructs panoramic images from a sequence of overlapping frames extracted from video by estimating geometric transformations directly from image content.

To align consecutive image frames, the pipeline was implemented from scratch, covering feature detection, descriptor construction, robust image matching, homography estimation, and backward image warping.

More specifically, as a group of 2, we developed the project pipeline incrementally, focusing on the following core components:

- Harris Corner Detection for identifying repeatable interest points.
- Patch-Based Feature Descriptors with normalization for illumination robustness.
- Descriptor Matching using similarity scoring and mutual consistency checks.
- RANSAC-Based Robust Homography Estimation for rejecting outlier correspondences.
- Homography Accumulation into a common reference frame.
- Backward Image Warping.
- Strip-Based Panorama Construction.

Our design mirrors traditional panorama stitching approaches used in early vision systems and highlights the geometric foundations behind stitching tools.

## Project Structure

The core image alignment and panorama generation pipeline is implemented in:

- `panorama_pipeline.py`: Contains the full feature-based image alignment pipeline, including Harris corner detection, descriptor construction, feature matching, RANSAC-based homography estimation, homography accumulation, backward warping, and panorama construction.

- `utils.py`: Supporting image processing utilities and helper functions.

- `videos/`: Example video inputs used for panorama generation.

- `out/`: Generated panoramic images.

