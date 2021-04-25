import pyrealsense2 as rs
import numpy as np
import cv2

WIDTH = 640
HEIGHT = 480
FPS = 30
# file name which you want to open
FILE = './data/stairs.bag'


def main():
    # stream(Depth/Color) setting
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.rgb8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_device_from_file(FILE)

    # Start streaming
    pipeline = rs.pipeline()
    pipeline.start(config)

    try:
        while True:
            # Wait for frames(Color/Depth)
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)
            color_image = np.asanyarray(color_frame.get_data())
            # Show images
            color_image_s = cv2.resize(color_image, (WIDTH, HEIGHT))
            depth_colormap_s = cv2.resize(depth_colormap, (WIDTH, HEIGHT))
            images = np.hstack((color_image_s, depth_colormap_s))
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)

            INTERVAL = 10
            if cv2.waitKey(INTERVAL) & 0xff == 27:  # End with ESC
                cv2.destroyAllWindows()
                break

    finally:
        # Stop streaming
        pipeline.stop()


if __name__ == '__main__':
    main()
