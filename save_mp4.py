import pyrealsense2 as rs
import numpy as np
import cv2

WIDTH = 640
HEIGHT = 480
FPS = 30


def main():
    # stream(Depth/Color) setting
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    # Start streaming
    pipeline = rs.pipeline()
    pipeline.start(config)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter('data/video.mp4', fourcc, FPS, (WIDTH, HEIGHT))

    try:
        while True:
            # Wait for frames(Color)
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            # Show images
            color_image_s = cv2.resize(color_image, (WIDTH, HEIGHT))
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image_s)
            video.write(color_image_s)

            INTERVAL = 10
            if cv2.waitKey(INTERVAL) & 0xff == 27:  # End with ESC
                cv2.destroyAllWindows()
                break

    finally:
        # Stop streaming
        pipeline.stop()


if __name__ == '__main__':
    main()
