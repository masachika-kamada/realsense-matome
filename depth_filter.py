import pyrealsense2 as rs
import numpy as np
import cv2

WIDTH = 640
HEIGHT = 480
FPS = 60
THRESHOLD = 0.5  # これより遠い距離の画素を無視する
BG_PATH = "dst.jpg"  # 背景画像のパス


def main():
    align = rs.align(rs.stream.color)
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    max_dist = THRESHOLD/depth_scale

    bg_image = cv2.imread(BG_PATH)
    bg_image = bg_image[0:480, 0:640]

    # Skip 5 first frames to give the Auto-Exposure time to adjust
    for x in range(5):
        pipeline.wait_for_frames()

    try:
        while True:
            # フレーム取得
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not depth_frame or not color_frame:
                continue

            # RGB画像
            color_image = np.asanyarray(color_frame.get_data())

            # 深度画像
            colorizer = rs.colorizer()
            depth_color_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            # decimation = rs.decimation_filter()
            # decimation.set_option(rs.option.filter_magnitude, 4)
            # decimated_depth = decimation.process(depth_frame)
            # depth_filter = np.asanyarray(colorizer.colorize(decimated_depth).get_data())
            spatial = rs.spatial_filter()
            filtered_depth = spatial.process(depth_frame)
            depth_filter = np.asanyarray(colorizer.colorize(filtered_depth).get_data())

            # 指定距離以上を無視した深度画像
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_filtered_image = (depth_image < max_dist) * depth_image
            depth_gray_filtered_image = (depth_filtered_image * 255. / max_dist).reshape((HEIGHT, WIDTH)).astype(np.uint8)

            # 指定距離以上を無視したRGB画像
            color_filtered_image = (depth_filtered_image.reshape((HEIGHT, WIDTH, 1)) > 0) * color_image

            # 背景合成
            background_masked_image = (depth_filtered_image.reshape((HEIGHT, WIDTH, 1)) == 0) * bg_image
            composite_image = background_masked_image + color_filtered_image

            # 表示
            # cv2.namedWindow('depth_filter', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('depth_filter', composite_image)
            # cv2.imshow("gray", depth_gray_filtered_image)
            cv2.imshow("depth", depth_color_image)
            cv2.imshow("filter", depth_filter)

            if cv2.waitKey(10) & 0xff == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
