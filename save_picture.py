import pyrealsense2 as rs
import numpy as np
import cv2
import datetime

WIDTH = 640
HEIGHT = 480
FPS = 30


def main():
    count = 0
    # stream(Depth/Color) setting
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    # Start streaming
    pipeline = rs.pipeline()
    pipeline.start(config)

    # Generate align object
    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        while True:
            # Wait for frames(Color/Depth)
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            # depth_frame = aligned_frames.get_depth_frame()
            # if not depth_frame or not color_frame:
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            # depth_image = np.asanyarray(depth_frame.get_data())
            # Convert depth image into colormap
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)

            # Show images
            color_image_s = cv2.resize(color_image, (WIDTH, HEIGHT))
            # depth_colormap_s = cv2.resize(depth_colormap, (WIDTH, HEIGHT))
            # images = np.hstack((color_image_s, depth_colormap_s))
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('RealSense', images)
            cv2.imshow('RealSense', color_image_s)

            INTERVAL = 10
            key = cv2.waitKey(INTERVAL) & 0xff
            if key == 27:  # End with ESC
                cv2.destroyAllWindows()
                break
            elif key == ord('s'):
                now = datetime.datetime.now()
                cv2.imwrite(now.strftime('%Y%m%d_%H%M%S') + '.jpg', color_image_s)
                print("save", count)
                count += 1
            elif key == ord("t"):
                print(datetime.datetime.now())

    finally:
        # Stop streaming
        pipeline.stop()


if __name__ == '__main__':
    main()
