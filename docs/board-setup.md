# 香橙派 5 Max：系统、ROS、NPU 与摄像头准备

本流程针对用户提供的板卡配置：**Orange Pi 5 Max / RK3588 / Ubuntu 20.04.6 aarch64**。板卡使用系统 Python 3.8，与主机的 Conda Python 3.9 分开。

## 1. 系统镜像与 NPU 驱动

已有可用 Ubuntu 20.04.6 时不必重刷。全新板卡先按 [香橙派 5 Max 官方 Wiki](http://www.orangepi.org/orangepiwiki/index.php/Orange_Pi_5_Max) 的镜像下载和烧录说明准备存储卡或系统盘，再完成首次开机。必须选择 **5 Max** 对应的镜像，不要使用 5 / 5 Plus 的镜像。

教程中的 Ubuntu 20.04.6 是用户实际使用的系统，不代表官网当前一定提供这一版本；若取得的是 Ubuntu 22.04/24.04，不能直接执行本篇 Noetic 安装步骤。使用 20.04.6 时，镜像供应方需要提供适配 5 Max 的内核和 RKNPU 驱动。

在板卡终端确认：

```bash
tr -d '\0' < /proc/device-tree/model
uname -m
cat /etc/os-release
uname -r
sudo dmesg | grep -iE 'rknpu|npu'
sudo cat /sys/kernel/debug/rknpu/version
```

预期架构为 `aarch64`、系统为 Ubuntu 20.04，启动日志中应有 NPU 驱动初始化记录。最后一个版本节点可能因 debugfs 未启用而不存在，不能只凭它判断驱动缺失。

**复制 `librknnrt.so` 只安装用户态运行库，不会安装内核驱动。** 若没有 NPU 初始化记录或后面的 `init_runtime()` 失败，先核对镜像供应方的驱动支持及日志，再按该系统对应的 BSP 更新内核/驱动；不要把其他内核编译的 `.ko` 直接复制进来。最终以本篇后面的真实 NPU 初始化结果为准。

## 2. 安装板卡基础工具

在板卡执行：

```bash
sudo apt update
sudo apt install -y git curl ca-certificates software-properties-common \
  python3-pip python3-venv openssh-server v4l-utils xvfb xauth \
  libgl1 libglib2.0-0 libxcb-xinerama0
sudo systemctl enable --now ssh
/usr/bin/python3 --version
```

本篇后续统一用 `/usr/bin/python3`，预期为 Python 3.8；不要在板卡激活主机的 `deeplearning` 环境。

## 3. 安装 ROS1 Noetic

先确认系统确实为 Ubuntu 20.04（focal）：

```bash
. /etc/os-release
test "$ID" = ubuntu && test "$VERSION_ID" = 20.04
```

上条检查成功后，安装官方 ROS 软件源配置包和 Noetic 基础环境：

```bash
sudo add-apt-repository -y universe
curl --fail --location --retry 3 \
  https://github.com/ros-infrastructure/ros-apt-source/releases/download/1.3.0/ros-apt-source_1.3.0.focal_all.deb \
  -o /tmp/ros-apt-source.deb
sudo dpkg -i /tmp/ros-apt-source.deb
sudo apt update
sudo apt install -y ros-noetic-ros-base
source /opt/ros/noetic/setup.bash
grep -qxF 'source /opt/ros/noetic/setup.bash' "$HOME/.bashrc" || \
  echo 'source /opt/ros/noetic/setup.bash' >> "$HOME/.bashrc"
/usr/bin/python3 -c 'import rospy; print("rospy OK")'
```

这里使用已核对支持 focal/ROS1 的 `ros-apt-source` 配置密钥和软件源。若网络无法访问 GitHub 或 ROS 软件源，先处理下载/网络错误，不能跳过失败步骤继续运行。

## 4. 安装 Lite2 与 Runtime

回到 [README 第 5 步](../README.md#board-install)，直接使用克隆后 `rknn_api/` 下的 Lite2 wheel 和 Linux Runtime，安装匹配 Python 的包，并复制对应的 `lib/bin`，无需另外克隆 SDK。Python 3.8 的 NumPy 使用 `1.24.4`；不要直接安装只支持更新 Python 的 NumPy 版本。

安装后至少检查：

```bash
/usr/bin/python3 -c 'import numpy, cv2, rospy; from rknnlite.api import RKNNLite; print("板卡 Python 依赖 OK")'
ldconfig -p | grep librknnrt
```

成功导入 Lite2 还不等于 NPU 已可用。收到主机传来的模型后，按 README 执行 `scripts/check_board.py`，实际加载模型并初始化 NPU。

## 5. 摄像头与图形显示

本流程默认 **USB UVC 摄像头**。CSI 摄像头需要按 5 Max 系统的相机驱动、设备树和采集方式单独配置，不能保证直接对应摄像头索引 0。

```bash
v4l2-ctl --list-devices
ls -l /dev/video*
id -nG
```

如果当前用户没有视频设备权限：

```bash
sudo usermod -aG video "$USER"
```

修改用户组后退出登录再登录。确认实际能取帧，而不只是设备节点存在：

```bash
/usr/bin/python3 - <<'PY'
import cv2
cap = cv2.VideoCapture(0)
ok, frame = cap.read()
cap.release()
if not ok:
    raise SystemExit("摄像头 0 取帧失败；检查设备索引、占用和权限")
print("摄像头 OK:", frame.shape)
PY
```

若实际摄像头索引不是 0，README 的配置脚本可用 `--camera` 指定。`v888_basic.py` 优先尝试 MJPG/1920×1080，实际分辨率由摄像头协商决定。

- **本地桌面：** 在板卡已登录的图形桌面终端运行，确认 `echo "$DISPLAY"` 非空。
- **只有 SSH/没有显示器：** 用 README 中的 `xvfb-run` 命令提供虚拟显示，原脚本可以继续使用 OpenCV 窗口，但 SSH 终端不会显示实时画面；通过 ROS 参数检查识别结果。按 `Ctrl+C` 结束。

不要只把脚本的 `Flags` 改成 False 就认为已支持无显示环境，原代码还有无条件创建窗口的语句。
