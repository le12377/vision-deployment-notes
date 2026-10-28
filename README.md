# 我的视觉训练与 RKNN 部署流程

按本机配置整理，流程为：训练 → RKNN 优化导出 ONNX → Model Zoo 转 RKNN → 板卡推理。

本机实测配置（2026-10-28）：

| 项目 | 配置 |
| --- | --- |
| 系统 / GPU | Ubuntu 20.04.6 x86_64 / RTX 4070 Laptop 8 GB |
| Conda / 视觉环境 | Miniconda，Conda 25.9.1 / `deeplearning` |
| Python / PyTorch / torchvision | 3.9.18 / 2.4.0 / 0.19.0 |
| PyTorch 使用的 CUDA | 12.1，`torch.cuda.is_available()` 为 True |
| Ultralytics / RKNN-Toolkit2 | 8.4.120 / 2.3.2 |
| NumPy / OpenCV / ONNX | 1.26.4 / 4.11.0.86 / 1.17.0 |

本仓库带入了我的训练、转换和输出修改源码，保留完整 `ultralytics/` 包、`train.py`、`data.yaml` 及自用检测/导出入口；不带数据集、模型权重和训练结果。

## 1. 下载 Miniconda，创建 deeplearning 环境

从 [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) 下载 Linux x86_64 安装包。`deeplearning` 是我习惯使用的基础视觉环境名，可以自行修改。

```bash
conda create -n deeplearning python=3.9.18
conda activate deeplearning
python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
```

后续主机命令均在此环境中执行；第 3、4、7 步从本仓库根目录开始，板卡命令单独说明。

## 2. 准备 Ultralytics、YOLOv5 和训练配置

下载本仓库，使用带入的本机源码：

```bash
git clone https://github.com/le12377/vision-deployment-notes.git
cd vision-deployment-notes
python -m pip install -r requirements-vision.txt
python -m pip install -e ./train/ultralytics
```

[train/ultralytics](train/ultralytics) 来自本机 `/home/lsf/桌面/视觉/yolo/train/ultralytics`，已带 [train.py](train/ultralytics/train.py) 和 [data.yaml](train/ultralytics/data.yaml)。修改数据集路径、类别及训练参数后运行：

```bash
cd train/ultralytics
python train.py
```

我的 `train.py` 默认 YOLOv8、640 输入、200 epochs、batch 16、SGD、GPU 0；数据配置目前为 7 类，需要改成自己的数据。预训练 `yolov8n.pt` 自行下载或使用框架自动下载。

自用的 `detect.py`、`detect_ocr.py`、`convert.py` 也已保留；运行前修改各自的模型和图像路径，OCR 入口另外需要 PaddleOCR/PaddlePaddle。

YOLOv5 源码在 `train/yolov5/`，使用时安装其依赖，并指定自己的数据配置：

```bash
cd train/yolov5
python -m pip install -r requirements.txt
python train.py --weights yolov5s.pt --data ../ultralytics/data.yaml --imgsz 640 --epochs 200 --batch-size 16 --device 0
```

上面两段 `cd` 均以仓库根目录为起点；每次切换训练项目前先回到仓库根目录。也可自行克隆 [ultralytics](https://github.com/ultralytics/ultralytics.git) 或 [yolov5](https://github.com/ultralytics/yolov5.git)，再带入自己的配置。

## 3. 用 convert 中的 RKNN 优化项目，将 PT 转 ONNX

[convert/](convert) 直接整理自本机 `/home/lsf/桌面/视觉/yolo/convert`，包含 YOLOv5、v7、v8、v10、v11 对应的优化导出代码。这里的“9 通道”指 **9 个输出张量**，输入仍是 RGB 三通道。

以 YOLOv8 为例，在仓库根目录执行：

```bash
cd convert/yolov8_p-o
```

修改 `ultralytics/cfg/default.yaml` 中的 `model` 为训练好的 PT 路径，`imgsz` 为模型输入尺寸，保留 `format: rknn`，然后运行：

```bash
PYTHONPATH=. python ultralytics/engine/exporter.py
```

这个优化项目中的 `format: rknn` 导出的是适配 RKNN 的 **ONNX**，实际 `.rknn` 转换在第 4 步进行。`PYTHONPATH=.` 保证使用当前优化版源码。

YOLOv11 使用 `convert/yolov11_p-o/` 中同样的配置和命令。YOLOv5、v7 则进入各自目录，按其 `README_rkopt.md` 使用 `export.py --rknpu`；v10 按 `RKNN_README_CN.md` 导出。

## 4. 用 RKNN Model Zoo，将 ONNX 转 RKNN

主机安装与本机相同的 Toolkit2 2.3.2，Python 3.9 x86_64 wheel：

```bash
git clone --branch v2.3.2 --depth 1 https://github.com/airockchip/rknn-toolkit2.git
python -m pip install rknn-toolkit2/rknn-toolkit2/packages/x86_64/rknn_toolkit2-2.3.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

克隆 [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo.git)，以 RK3588、YOLOv8 为例：

```bash
git clone https://github.com/airockchip/rknn_model_zoo.git
cd rknn_model_zoo/examples/yolov8/python
python convert.py /path/to/best.onnx rk3588 fp /path/to/best.rknn
```

需要 INT8 时把 `fp` 改为 `i8`，并修改 `convert.py` 的 `DATASET_PATH` 为自己的校准图像列表。不同板卡修改 `rk3588`，使用对应模型目录的转换脚本。

## 5. 在板卡安装 Lite2，并复制 lib / bin

板卡下载同一份 SDK；与本机 `/home/lsf/桌面/视觉/rknn/rknn-toolkit2-master` 对应的是 Toolkit2 2.3.2：

```bash
git clone --branch v2.3.2 --depth 1 https://github.com/airockchip/rknn-toolkit2.git
cd rknn-toolkit2
python3 --version
uname -m
```

进入 `rknn-toolkit-lite2/packages` 选择与**板卡实际 Python**匹配的 wheel。下面自动选择 Python 标签，适用于 SDK 中提供的 aarch64 wheel：

```bash
PY_TAG=$(python3 -c 'import sys; print("cp%d%d" % sys.version_info[:2])')
python3 -m pip install rknn-toolkit-lite2/packages/*-"${PY_TAG}"-*.whl
```

如 Python 3.8 对应 `cp38`、3.9 对应 `cp39`；板卡不是主机，不能直接按主机 Python 3.9 选包。

仍在 SDK 根目录，以 aarch64 板卡为例，按我的配置把两个 API 复制到 `lib` 和 `bin`：

```bash
sudo cp rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/librknnrt.so
sudo cp rknpu2/runtime/Linux/rknn_server/aarch64/usr/bin/rknn_server /usr/bin/rknn_server
sudo chmod +x /usr/bin/rknn_server
sudo ldconfig
python3 -c 'from rknnlite.api import RKNNLite; print("Lite2 OK")'
```

其他架构选择对应目录，系统若使用不同库路径则修改目标位置。以上安装用户态依赖；板卡镜像中需已有匹配的 RKNPU 内核驱动。

## 6. 使用 v888_basic.py 推理 YOLOv8

在板卡克隆你指定的 [rknn_yolov8-ros1](https://github.com/zdongdong12311-a11y/rknn_yolov8-ros1)：

```bash
git clone https://github.com/zdongdong12311-a11y/rknn_yolov8-ros1.git
cd rknn_yolov8-ros1/multi_threaded-python-v8_ros_pub
```

实际文件名是 `v888_basic.py`。修改其中 `MODEL_PATH` 为第 4 步生成的 RKNN，`IMG_SIZE` 与导出尺寸一致，`CLASSES` 与训练类别顺序一致，摄像头默认 `VideoCapture(0)`。

```bash
source /opt/ros/noetic/setup.bash
python3 -m pip install numpy opencv-python
python3 v888_basic.py
```

板卡需已有 ROS1 和 `rospy`。这份脚本按每个尺度的框回归、类别张量读取，可处理结构匹配的 YOLOv8 6/9 输出，不适用于 YOLOv5 的 3 输出。

## 7. 用我的 YOLO-RKNN9-Export 修改 3 / 6 / 9 输出

源码带在 [YOLO-RKNN9-Export](YOLO-RKNN9-Export)，来自本机 `/home/lsf/桌面/视觉/yolo/YOLO-RKNN9-Export`。在仓库根目录执行：

```bash
cd YOLO-RKNN9-Export
python -m pip install -r requirements.txt
cd src/yolo26_rknn9
python3 export.py /path/to/model.onnx 9
```

用法为 `python3 export.py 模型路径 输出数`，支持输入 PT 或 ONNX。示例：

```bash
python3 export.py /path/to/yolov5.onnx 3
python3 export.py /path/to/yolov8.onnx 6
python3 export.py /path/to/yolov8.onnx 9
```

当前实现中，3 输出用于 YOLOv5 风格 anchor 检测头；YOLOv8 等使用 6 或 9 输出。输出文件默认带 `_rknn3.onnx`、`_rknn6.onnx` 或 `_rknn9.onnx` 后缀，再按第 4 步转 RKNN；本流程优先使用 9 输出，改格式后核对第 6 步的后处理。

---

源码保留各项目原许可证。数据、权重和板卡 SDK 不打包，按上述路径自行准备；本次只检测主机配置、整理源码和核对命令，没有重新训练或在板卡上执行部署。
