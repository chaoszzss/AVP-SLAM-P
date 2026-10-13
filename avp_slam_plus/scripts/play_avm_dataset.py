#!/usr/bin/env python3
"""
Simple player for AVM-SLAM dataset (four RGB cameras).
- Reads <dataset_root>/Camera*/Data/data.csv
- Publishes to /camera{0-3}/color/image_raw with recorded timestamps (ms -> sec)
- ~play_rate scales playback speed
Falls back to Pillow if cv2 is unavailable.
"""

from __future__ import print_function
import os
import sys
import rospy
from sensor_msgs.msg import Image

try:
    import cv2  # Prefer OpenCV if available
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

try:
    import numpy as np
except ImportError:
    np = None


def _load_camera_csv(camera_dir):
    csv_path = os.path.join(camera_dir, "Data", "data.csv")
    img_dir = os.path.join(camera_dir, "Data", "Image")
    if not os.path.exists(csv_path):
        rospy.logerr("CSV not found: %s", csv_path)
        return []

    samples = []
    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ts_str, img_name = line.split(",")
                ts = float(ts_str) / 1000.0  # ms -> sec
            except ValueError:
                rospy.logwarn("Skip malformed line in %s: %s", csv_path, line)
                continue

            img_path = os.path.join(img_dir, img_name)
            samples.append((ts, img_path))

    if not samples:
        rospy.logerr("No samples loaded from %s", csv_path)
    return samples


def main():
    rospy.init_node("play_avm_dataset")
    dataset_root = rospy.get_param("~dataset_root", "/home/chaoszzss/My_files/SLAM_files/Datasets/AVM-SLAM_Dataset/data1")
    play_rate = rospy.get_param("~play_rate", 1.0)
    start_index = rospy.get_param("~start_index", 0)
    use_csv_time = rospy.get_param("~use_csv_time", False)
    fixed_fps = rospy.get_param("~fixed_fps", 10.0)  # 默认 10Hz

    if not _HAS_CV2 and PILImage is None:
        rospy.logerr("cv2 and Pillow are both missing; cannot read images. Install python3-opencv or python3-pil. Python: %s", sys.executable)
        return
    camera_samples = []
    for cam_id in range(4):
        cam_dir = os.path.join(dataset_root, "Camera{}".format(cam_id))
        samples = _load_camera_csv(cam_dir)
        if not samples:
            return
        camera_samples.append(samples)

    lengths = [len(s) for s in camera_samples]
    min_len = min(lengths)
    if start_index >= min_len:
        rospy.logerr("start_index %d exceeds smallest sequence length %d", start_index, min_len)
        return

    # Basic sync check
    warn_desync = False
    for idx in range(start_index, min_len):
        stamps = [camera_samples[c][idx][0] for c in range(4)]
        if max(stamps) - min(stamps) > 1e-3:
            warn_desync = True
            break
    if warn_desync:
        rospy.logwarn("Camera timestamps are not perfectly aligned; minor drops may occur.")

    pubs = [
        rospy.Publisher("/camera{}/color/image_raw".format(i), Image, queue_size=1)
        for i in range(4)
    ]

    rospy.loginfo("Starting playback from %s (start index %d, rate x%.2f)", dataset_root, start_index, play_rate)

    idx = start_index
    # 基于 CSV 时间或固定 FPS 生成时间戳
    if use_csv_time:
        base_time = None
    else:
        base_time = rospy.Time.now()
        step = 1.0 / max(1e-6, fixed_fps)

    while not rospy.is_shutdown() and idx < min_len:
        csv_ts = camera_samples[0][idx][0]
        if use_csv_time:
            target_stamp = rospy.Time.from_sec(csv_ts)
        else:
            target_stamp = base_time + rospy.Duration.from_sec(idx * step)

        for cam_id in range(4):
            ts, img_path = camera_samples[cam_id][idx]
            if _HAS_CV2:
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    rospy.logwarn("Failed to read image: %s", img_path)
                    continue
                height, width, _ = img.shape
                data_bytes = img.tobytes()
            else:
                if PILImage is None:
                    rospy.logwarn("No reader available, skip %s", img_path)
                    continue
                img = PILImage.open(img_path).convert("RGB")
                if np is not None:
                    arr = np.array(img)[:, :, ::-1]  # RGB -> BGR
                    data_bytes = arr.tobytes()
                    height, width = arr.shape[0], arr.shape[1]
                else:
                    rgb_bytes = img.tobytes()
                    width, height = img.size
                    # Manual RGB -> BGR
                    bgr = bytearray(len(rgb_bytes))
                    for i in range(0, len(rgb_bytes), 3):
                        r = rgb_bytes[i]
                        g = rgb_bytes[i + 1]
                        b = rgb_bytes[i + 2]
                        bgr[i] = b
                        bgr[i + 1] = g
                        bgr[i + 2] = r
                    data_bytes = bytes(bgr)

            msg = Image()
            msg.header.stamp = target_stamp
            msg.header.frame_id = "camera{}_link".format(cam_id)
            msg.height = height
            msg.width = width
            msg.encoding = "bgr8"
            msg.is_bigendian = 0
            msg.step = width * 3
            msg.data = data_bytes
            pubs[cam_id].publish(msg)

        if idx + 1 < min_len:
            if use_csv_time:
                dt = camera_samples[0][idx + 1][0] - csv_ts
                # 数据集时间戳是毫秒，差值可能很大，做一个上限
                dt = min(dt, 0.5)  # 最长 0.5s
            else:
                dt = step
            dt = max(0.0, dt) / max(1e-6, play_rate)
            rospy.sleep(dt)
        idx += 1

    rospy.loginfo("Playback finished")


if __name__ == "__main__":
    main()
