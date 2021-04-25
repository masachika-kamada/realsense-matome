import pyrealsense2 as rs
import numpy as np
import cv2

WIDTH = 640
HEIGHT = 480
FPS = 30


def white_detect(img):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_white_min = np.array([0, 0, 100])
    hsv_white_max = np.array([179, 45, 255])
    mask_image = cv2.inRange(hsv_image, hsv_white_min, hsv_white_max)
    nLabels, labelImage, stats, centroids = cv2.connectedComponentsWithStats(mask_image)
    max_area = 0
    max_label = 0
    for i in range(1, nLabels):
        if stats[i][4] > max_area:
            max_area = stats[i][4]
            max_label = i

    # cv2.imshow("gray", mask_image)
    # img_mask2 = np.zeros(mask_image.shape[0:3], np.uint8)
    # img_mask2[labelImage == max_label, ] = 255
    # cv2.imshow("images", img_mask2)
    # print(int(centroids[max_label][0]))
    return centroids[max_label]


def main():
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
            aligned_depth_frame = aligned_frames.get_depth_frame()
            depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            # Convert depth image into colormap
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)

            # coordinate setting
            # x = int(WIDTH / 2)
            # y = int(HEIGHT / 2)
            coord = white_detect(color_image)
            cv2.drawMarker(color_image, (int(coord[0]), int(coord[1])), (0, 0, 255))
            depth = aligned_depth_frame.get_distance(int(coord[0]), int(coord[1]))
            depth_point_in_meters_camera_coords = rs.rs2_deproject_pixel_to_point(
                depth_intrin, [int(coord[0]), int(coord[1])], depth)
            print(depth_point_in_meters_camera_coords)

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
