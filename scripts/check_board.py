#!/usr/bin/env python3
"""Check actual ROS, camera and NPU readiness on the documented RK3588 board."""
import argparse
import os
import platform
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    if platform.machine() != "aarch64":
        raise SystemExit("请在 RK3588 aarch64 板卡上执行，不是在训练主机执行")
    if not args.model.is_file():
        raise SystemExit("模型不存在，请先传输: {}".format(args.model))
    try:
        import cv2
        import numpy
        import rospy
        from rknnlite.api import RKNNLite
    except ImportError as error:
        raise SystemExit("Python 依赖缺失: {}".format(error))
    print("Python {} / NumPy {} / OpenCV {}".format(sys.version.split()[0], numpy.__version__, cv2.__version__))
    try:
        rospy.get_master().getPid()
    except Exception as error:
        raise SystemExit("ROS Master 不可达，请在另一终端启动 roscore: {}".format(error))
    print("ROS Master OK")
    runtime = RKNNLite()
    try:
        result = runtime.load_rknn(str(args.model.resolve()))
        if result != 0:
            raise SystemExit("RKNN 模型加载失败，返回码 {}".format(result))
        result = runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        if result != 0:
            raise SystemExit("NPU 初始化失败；核对内核驱动、Runtime 和模型版本，返回码 {}".format(result))
        print("RKNN 模型加载和真实 NPU 初始化 OK")
    finally:
        runtime.release()
    cap = cv2.VideoCapture(args.camera)
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise SystemExit("摄像头 {} 取帧失败；检查索引、权限、占用".format(args.camera))
        print("摄像头 {} OK: {}".format(args.camera, frame.shape))
    finally:
        cap.release()
    if not os.environ.get("DISPLAY"):
        print("未检测到 DISPLAY：请用 README 的 xvfb-run 无显示器命令启动")
    print("预检通过，可以启动 v888_basic.py")


if __name__ == "__main__":
    main()
