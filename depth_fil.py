import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
print("Environment Ready")

# Setup:
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file("data/realsense.bag")
profile = pipe.start(cfg)

# Skip 5 first frames to give the Auto-Exposure time to adjust
for x in range(5):
    pipe.wait_for_frames()

# Store next frameset for later processing:
frameset = pipe.wait_for_frames()
depth_frame = frameset.get_depth_frame()

# Cleanup:
pipe.stop()
print("Frames Captured")

colorizer = rs.colorizer()
colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

plt.rcParams["axes.grid"] = False
plt.rcParams['figure.figsize'] = [8, 4]
plt.imshow(colorized_depth)
plt.show()

'''decimation filter'''  # 間引きフィルター
# 低分解能での基本的な深度欠落の補間
decimation = rs.decimation_filter()
decimated_depth = decimation.process(depth_frame)
colorized_depth = np.asanyarray(colorizer.colorize(decimated_depth).get_data())
plt.imshow(colorized_depth)
plt.show()
# オプションの調整
decimation.set_option(rs.option.filter_magnitude, 4)
decimated_depth = decimation.process(depth_frame)
colorized_depth = np.asanyarray(colorizer.colorize(decimated_depth).get_data())
plt.imshow(colorized_depth)
plt.show()

'''spatial filter'''  # 空間フィルター
# 高速でフィルタリングが可能
spatial = rs.spatial_filter()
filtered_depth = spatial.process(depth_frame)
colorized_depth = np.asanyarray(colorizer.colorize(filtered_depth).get_data())
plt.imshow(colorized_depth)
plt.show()

spatial.set_option(rs.option.filter_magnitude, 5)
spatial.set_option(rs.option.filter_smooth_alpha, 1)
spatial.set_option(rs.option.filter_smooth_delta, 50)
filtered_depth = spatial.process(depth_frame)
colorized_depth = np.asanyarray(colorizer.colorize(filtered_depth).get_data())
plt.imshow(colorized_depth)
plt.show()
# hole filling
spatial.set_option(rs.option.holes_fill, 3)
filtered_depth = spatial.process(depth_frame)
colorized_depth = np.asanyarray(colorizer.colorize(filtered_depth).get_data())
plt.imshow(colorized_depth)
plt.show()

'''temporal filter'''  # 時間フィルター
# 基本的な時間的平滑化と穴埋め、複数の連続するフレームに対して
profile = pipe.start(cfg)

frames = []
for x in range(10):
    frameset = pipe.wait_for_frames()
    frames.append(frameset.get_depth_frame())

pipe.stop()
print("Frames Captured")

temporal = rs.temporal_filter()
for x in range(10):
    temp_filtered = temporal.process(frames[x])
colorized_depth = np.asanyarray(colorizer.colorize(temp_filtered).get_data())
plt.imshow(colorized_depth)
plt.show()
# フィルターオプションの変更で結果の微調整可能
# 一時的なフィルタリングでは、平滑化と動きの間にトレードオフあり

'''hole filling'''
hole_filling = rs.hole_filling_filter()
filled_depth = hole_filling.process(depth_frame)
colorized_depth = np.asanyarray(colorizer.colorize(filled_depth).get_data())
plt.imshow(colorized_depth)
plt.show()

'''putting everything together'''
# 以上のフィルターを順に適用すると最適に機能する
depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)

for x in range(10):
    frame = frames[x]
    frame = decimation.process(frame)
    frame = depth_to_disparity.process(frame)
    frame = spatial.process(frame)
    frame = temporal.process(frame)
    frame = disparity_to_depth.process(frame)
    frame = hole_filling.process(frame)

colorized_depth = np.asanyarray(colorizer.colorize(frame).get_data())
plt.imshow(colorized_depth)
plt.show()
