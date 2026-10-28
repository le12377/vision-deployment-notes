# 我的视觉训练与 RKNN 部署流程

按我的七步流程整理。**首次先用现成 YOLOv8n 跑通推理，再换成自己的训练模型。** 主机负责训练/转换，板卡负责推理；不要混用两边的 Python 环境。

| 位置 | 本流程配置 |
| --- | --- |
| 主机（已检测） | Ubuntu 20.04.6 x86_64，RTX 4070 Laptop 8 GB |
| 主机 Python / PyTorch | Conda `deeplearning`，Python 3.9.18，torch 2.4.0 / torchvision 0.19.0，CUDA 12.1 |
| 主机工具 | Ultralytics 8.4.120，RKNN-Toolkit2 2.3.2，NumPy 1.26.4，ONNX 1.17.0 |
| 板卡（用户提供） | 香橙派 5 Max，RK3588，Ubuntu 20.04.6 aarch64 |
| 板卡 Python / ROS | 系统 Python 3.8，ROS1 Noetic，Lite2 / Runtime 2.3.2 |

本仓库带入了我的完整训练包、`train.py`、`data.yaml`、`convert/` 和输出修改项目，并收录本机提供的 [rknn_api](rknn_api) 与 [rknn_model_zoo](rknn_model_zoo) 的源码、安装包、运行库及示例数据，**不上传模型权重或导出的模型文件**。主机和板卡各克隆本仓库即可取得这些依赖，无需再单独克隆板端 SDK 或 Model Zoo；两个目录的非模型文件约 1.5 GiB，首次克隆需要相应空间。以下命令以 Bash 为例，项目均放在 `$HOME/vision-deployment-notes`。

## 1. 主机：安装 Miniconda 和基础视觉环境

在**主机**安装基础工具，再下载适用于这套 Ubuntu 20.04 配置的 Miniconda 安装器：

```bash
sudo apt update
sudo apt install -y git curl ca-certificates bzip2 build-essential \
  python3-pip openssh-client libgl1 libglib2.0-0
curl --fail --location --retry 3 \
  https://repo.anaconda.com/miniconda/Miniconda3-py39_24.7.1-0-Linux-x86_64.sh \
  -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda init bash
conda create -y -n deeplearning python=3.9.18
conda activate deeplearning
python -m pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

已有同名 Miniconda/环境时直接激活，不要重复安装或覆盖。以后新开主机终端先运行 `conda activate deeplearning`。`deeplearning` 是我的习惯名称，可以修改。

检查 NVIDIA 驱动和 PyTorch：

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

我的输出对应 torch 2.4.0、CUDA 12.1、True。全新主机没有驱动时，可在 Ubuntu“软件和更新 → 附加驱动”安装推荐 NVIDIA 驱动并重启，再检查；下面的模型导出使用 CPU，不要求先训练。

## 2. 主机：准备训练项目，先取得一个可用模型

```bash
cd "$HOME"
git clone https://github.com/le12377/vision-deployment-notes.git
export VISION_ROOT="$HOME/vision-deployment-notes"
cd "$VISION_ROOT"
python -m pip install -r requirements-vision.txt
python -m pip install -e ./train/ultralytics
mkdir -p "$VISION_ROOT/models"
curl --fail --location --retry 3 \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt \
  -o "$VISION_ROOT/models/yolov8n.pt"
```

首次使用这份 **80 类 COCO 的 YOLOv8n** 继续第 3 步，不需要先准备数据集。

训练自己的目标时，使用 [train/ultralytics/train.py](train/ultralytics/train.py) 和 [data.yaml](train/ultralytics/data.yaml)。修改数据集路径、类别和参数，再执行：

```bash
cd "$VISION_ROOT/train/ultralytics"
python train.py
```

我的默认参数是 YOLOv8、640 输入、200 epochs、batch 16、SGD、GPU 0。当前 `data.yaml` 是自用 7 类数据配置，并不包含实际图像/标注。训练完成后把实际 `best.pt` 复制到 `models/`，第 3 步的 `--weights` 改为该文件即可；类别会自动随模型导出。

[train/yolov5](train/yolov5) 也带入了本机源码，按其 `requirements.txt` 安装依赖，运行 `train.py --data` 指定自己的数据配置。自用 `detect.py`、`detect_ocr.py`、`convert.py` 均保留；OCR 额外需要 PaddleOCR/PaddlePaddle。

## 3. 主机：用 convert 优化源码导出 PT → ONNX

[convert/](convert) 来自我的本机优化项目，包含 v5、v7、v8、v10、v11。这里的“9 通道”指 **9 个输出张量**，输入仍是 RGB 三通道。

```bash
export VISION_ROOT="$HOME/vision-deployment-notes"
cd "$VISION_ROOT"
python scripts/export_yolov8.py --weights "$VISION_ROOT/models/yolov8n.pt" --imgsz 640
```

脚本实际调用 `convert/yolov8_p-o` 的优化版导出，使用 `format=rknn` 产生 **ONNX**，并核对 9 输出；它不使用普通 Ultralytics 的 ONNX 导出替代优化项目。输出为：

```text
models/yolov8n.onnx   # RKNN 友好输出结构
models/yolov8n.json   # 自动记录类别、输入尺寸和输出数量
```

也可进入优化项目，修改 `ultralytics/cfg/default.yaml` 的模型路径/输入尺寸、保留 `format: rknn`，执行 `PYTHONPATH=. python ultralytics/engine/exporter.py`。首次按上面的命令走即可。

## 4. 主机：用 Model Zoo 导出 ONNX → RKNN，并传到板卡

```bash
cd "$VISION_ROOT"
git clone --branch v2.3.2 --depth 1 https://github.com/airockchip/rknn-toolkit2.git
python -m pip install rknn-toolkit2/rknn-toolkit2/packages/x86_64/rknn_toolkit2-2.3.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
cd "$VISION_ROOT/rknn_model_zoo/examples/yolov8/python"
python convert.py "$VISION_ROOT/models/yolov8n.onnx" rk3588 fp "$VISION_ROOT/models/yolov8n.rknn"
test -s "$VISION_ROOT/models/yolov8n.rknn"
```

Model Zoo 直接使用仓库内这份本机修改版源码；主机 Toolkit2 安装包仍从官方 v2.3.2 取得，因为 `rknn_api` 里包含的是板端 Lite2 和 Runtime。首次使用非量化 `fp`，不需要校准数据。INT8 是后续选项：把 `fp` 改为 `i8`，并将 `DATASET_PATH` 改成实际校准图像列表。

先在板卡完成第 5 步，让 SSH 可用、项目目录建立起来。再回到**主机**传输，下面提示输入的是板卡登录用户名和 IP，不是 GitHub 账号：

```bash
read -rp '板卡登录用户名: ' BOARD_USER
read -rp '板卡 IP（板卡运行 hostname -I 查看）: ' BOARD_IP
ssh "${BOARD_USER}@${BOARD_IP}" 'mkdir -p "$HOME/vision-deployment-notes/models"'
scp "$VISION_ROOT/models/yolov8n.rknn" "$VISION_ROOT/models/yolov8n.json" \
  "${BOARD_USER}@${BOARD_IP}:vision-deployment-notes/models/"
sha256sum "$VISION_ROOT/models/yolov8n.rknn"
```

板卡上对同一文件执行 `sha256sum`，两个结果应一致。

<a id="board-install"></a>

## 5. 板卡：准备系统、ROS、Lite2 和 lib/bin

**先完成 [香橙派 5 Max 系统、驱动、ROS 和摄像头准备](docs/board-setup.md)**，其中给出了基础工具、ROS Noetic 安装和 NPU 驱动检查命令。本流程以已有匹配 RKNPU 驱动的 Ubuntu 20.04.6 镜像为前提；用户态 `.so` 不能替代内核驱动。

然后在**板卡**执行；后续统一用系统 `/usr/bin/python3`：

```bash
cd "$HOME"
git clone https://github.com/le12377/vision-deployment-notes.git
export VISION_ROOT="$HOME/vision-deployment-notes"
cd "$VISION_ROOT"
mkdir -p models
cd "$VISION_ROOT/rknn_api"
/usr/bin/python3 -m pip install --user numpy==1.24.4 opencv-python==4.11.0.86
PY_TAG=$(/usr/bin/python3 -c 'import sys; print("cp%d%d" % sys.version_info[:2])')
/usr/bin/python3 -m pip install --user rknn-toolkit-lite2/packages/*-"${PY_TAG}"-*.whl
sudo cp rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/librknnrt.so
sudo cp rknpu2/runtime/Linux/rknn_server/aarch64/usr/bin/rknn_server /usr/bin/rknn_server
sudo chmod +x /usr/bin/rknn_server
sudo ldconfig
/usr/bin/python3 -c 'import cv2, numpy, rospy; from rknnlite.api import RKNNLite; print("依赖导入 OK")'
```

Lite2 wheel、`librknnrt.so` 和 `rknn_server` 都已随 `rknn_api/` 提供，板卡无需再下载完整 SDK。系统 Python 3.8 选择 `cp38`；上面的命令按实际 Python 自动匹配，不能安装主机的 x86_64 wheel。完成后回到主机执行第 4 步的传输命令。

## 6. 板卡：启动 ROS，检查设备，再运行 v888_basic.py

取得指定的推理源码并固定到已核对的提交：

```bash
cd "$HOME"
git clone https://github.com/zdongdong12311-a11y/rknn_yolov8-ros1.git
cd "$HOME/rknn_yolov8-ros1"
git checkout 69e6ace08d606600563c66027e5840d837aac102
export VISION_ROOT="$HOME/vision-deployment-notes"
/usr/bin/python3 "$VISION_ROOT/scripts/configure_v888.py" \
  --script "$HOME/rknn_yolov8-ros1/multi_threaded-python-v8_ros_pub/v888_basic.py" \
  --model "$VISION_ROOT/models/yolov8n.rknn" \
  --metadata "$VISION_ROOT/models/yolov8n.json" --camera 0
```

实际文件名是 `v888_basic.py`。配置脚本根据随模型传来的 JSON 自动设置模型绝对路径、`IMG_SIZE` 和 `CLASSES`，首次保留 `.py.bak` 备份；摄像头不是 0 时改 `--camera`。

**板卡终端 A：启动 ROS Master，保持运行。**

```bash
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11311
roscore
```

**板卡终端 B：预检并推理。**

```bash
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11311
export VISION_ROOT="$HOME/vision-deployment-notes"
/usr/bin/python3 "$VISION_ROOT/scripts/check_board.py" --model "$VISION_ROOT/models/yolov8n.rknn" --camera 0
cd "$HOME/rknn_yolov8-ros1/multi_threaded-python-v8_ros_pub"
/usr/bin/python3 v888_basic.py
```

预检应依次显示 ROS Master、模型加载/NPU 初始化和摄像头取帧成功。有图形桌面时，在板卡桌面终端运行最后一条命令；它会打开实时检测窗口。

**只有 SSH、没有显示器：** 预检通过后，用这条命令替代上面的最后一条：

```bash
xvfb-run -a -s '-screen 0 1920x1080x24' /usr/bin/python3 v888_basic.py
```

Xvfb 提供虚拟显示，因此不需要改原脚本的 OpenCV 窗口代码。SSH 不会显示实时画面，另开板卡终端查看识别结果：

```bash
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11311
rosnode list
rosparam get /cls_ids
rosparam get /scores
```

参数只在识别到目标后创建；YOLOv8n 可先对准人或椅子测试。原脚本在未出现新检测时可能保留旧参数，所以参数存在不代表当前帧一定检测成功。停止用 `Ctrl+C`。

这份后处理读取 YOLOv8 的 DFL 框回归及类别张量，可处理结构匹配的 6/9 输出，不适用于 YOLOv5 的 3 输出。

## 7. 可选：用我的项目修改 3 / 6 / 9 输出

[YOLO-RKNN9-Export](YOLO-RKNN9-Export) 是我修改 ONNX 输出结构的项目。主机激活 `deeplearning` 后执行：

```bash
export VISION_ROOT="$HOME/vision-deployment-notes"
cd "$VISION_ROOT/YOLO-RKNN9-Export"
python -m pip install -r requirements.txt
cd src/yolo26_rknn9
python3 export.py "$VISION_ROOT/models/yolov8n.onnx" 6
python3 export.py "$VISION_ROOT/models/yolov8n.onnx" 9
```

命令格式是 `python3 export.py 模型路径 3/6/9`，输入支持 PT 或 ONNX。3 输出用于 YOLOv5 风格 anchor 模型，YOLOv8 等使用 6 或 9 输出；输出文件带 `_rknn3.onnx`、`_rknn6.onnx`、`_rknn9.onnx` 后缀。首次按前六步的 9 输出路线跑通，之后再改结构，并同步匹配转换和后处理。

---

主机环境来自实际检测，板卡型号/系统由用户提供。本次已在主机实际跑通 YOLOv8n 的 PT → 640 输入/9 输出 ONNX → RK3588 FP RKNN，配置脚本也已用上游推理源码验证。安装说明中的板卡命令没有在远程板卡执行；板端驱动、摄像头和完整推理需在目标板卡按预检实际确认。源码保留各项目原许可证。
