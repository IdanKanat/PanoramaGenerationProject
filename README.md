# Feature-Based Image Alignment & Panorama Generation Project - Intro to Computer Graphics Course
This project implements a classical feature-based image alignment & panorama generation pipeline in Python, developed as part of the *"Introduction to Graphics, Image Processing & Computer Vision"* course at TAU’s School of Computer Science (2025–2026), during the final year of our studies. The goal of the project is to construct panoramic images from a sequence of overlapping frames obtained from a video, by estimating geometric transformations directly from image content. 

To align consecutive image frames, the pipeline was implemented from scratch, covering feature detection, descriptor construction, robust image matching, homography estimation, and backward image warping.

More specifically, as a group of 2, we developed the project pipeline incrementally, focusing on the following core components:

- Harris Corner Detection for identifying repeatable interest points.
- Patch-Based Feature Descriptors with normalization for illumination robustness.
- Descriptor Matching using similarity scoring and mutual consistency checks.
- Robust RANSAC-Based Homography Estimation for rejecting outlier correspondences.
- Homography Accumulation into a common reference frame.
- Backward Image Warping.
- Strip-Based Panorama Construction.

Our design mirrors traditional panorama stitching approaches used in early vision systems and highlights the geometric foundations behind stitching tools.

## Project Structure

The repository is organized in a modular, object-oriented manner, with each
component of the ray tracing pipeline implemented in a dedicated module:

- `panorama_pipeline.py`: Core image alignment & panorama generation loop, including feature detection, description, matching, homography estimation & accumulation, and backward image warping.

- `utils.py`: .

- `videos/`: video inputs for panorama generation pipeline.

- `Examples/`: Example scenes and corresponding rendered outputs used for testing and visualization.

