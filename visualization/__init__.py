"""HandFlow visualization module: pytorch3d Phong rendering + video I/O.

- renderer_p3d.py: PhongRenderer — overlay (perspective, OpenCV intrinsics) + orthographic
                   trajectory view (topdown | side), both SoftPhongShader-shaded.
- video_io.py: read video frames (cv2) + encode mp4 (imageio/ffmpeg)
"""
