# RKNN 部署：版本、cp38 依赖与板端运行

本文展开 [原始笔记](../reference/deeplearn.txt) 中 `rknpu`、`/usr/lib`、`/usr/bin` 和 `model-zoo cp38依赖` 的含义。部署前需要确认具体开发板与 SDK 版本，笔记本身尚未记录这些值。

## 1. 区分训练、转换与板端推理

| 组件 | 主要职责 | 使用位置 |
| --- | --- | --- |
| PyTorch / Ultralytics | 训练、验证、导出模型 | 训练环境 |
| RKNN-Toolkit2 | 加载模型、构建及导出 RKNN，支持相应调试接口 | 与版本支持列表匹配的转换主机 |
| RKNN Runtime | 加载 RKNN，执行推理的用户态运行库 | 目标设备或对应 SDK 环境 |
| RKNN-Toolkit-Lite2 | 支持设备上的轻量 Python 推理接口 | 对应平台的板端 Python 环境 |
| RKNPU 驱动 | 对接 NPU 硬件的内核组件 | 板端系统镜像/BSP |
| RKNN Model Zoo | 特定模型的转换、预处理和后处理示例 | 转换及推理参考工程 |

不同 SoC、SDK 代际和操作系统的支持范围不同，不能认为所有 Rockchip 芯片都使用同一套组件。本文用 RK3588/Toolkit2 的例子解释流程，最终以所选版本官方支持表为准。

## 2. cp38 到底是什么意思？

`cp38` 是 Python wheel 常见的兼容标签，表示 CPython 3.8 对应的解释器/ABI 条件；`cp39` 对应 CPython 3.9。还需核对 wheel 文件名中的其他标签：

| 标签或属性 | 需要核对什么 |
| --- | --- |
| Python 标签，如 `cp38` | 当前解释器是否为对应版本 |
| ABI 标签 | wheel 是否支持当前解释器 ABI；`abi3` 等需按包声明解释 |
| 平台标签 | Linux、Windows 或具体 manylinux 条件是否匹配 |
| CPU 架构 | `x86_64` 与 `aarch64` 等不能直接混装 |

普通的 `cp38-cp38` wheel 不能直接当作 `cp39` 包安装。原笔记的训练环境为 Python 3.9，若所选转换 SDK 提供的是 Python 3.8 wheel，应按其支持条件另建转换环境，而不是强行混入训练环境。

例如，仅在该 SDK 确实要求并支持 Python 3.8 时：

```bash
conda create -n rknn-convert python=3.8
conda activate rknn-convert
python --version
uname -m
python -m pip debug --verbose
```

`pip debug` 可显示当前解释器支持的 wheel 标签。安装时选择实际发布包中的匹配 wheel；文件名、依赖列表和版本以对应 Toolkit2 发布说明为准，本文不填入未经确认的 wheel 文件名。

本机 Model Zoo 副本中存在 `docs/requirements_cp38.txt`，这是 Python 3.8 相关依赖提示的一个具体对应。不能仅凭文件名认为所有当前版本都应安装同一个依赖文件；先核对所用 Model Zoo 提交、Toolkit2 版本和 Python 支持条件。

## 3. /usr/lib、/usr/bin 与 sudo cp 的含义

原笔记写到：

```text
rknpu:
     /usr/lib  sudo cp 源文件路径 目标路径
     /usr/bin
```

它表达的是库与程序的复制提示，没有指定应复制哪个文件，也没有说明 SDK 版本和目标系统。

| 项目 | 常见作用 | 实际部署时要确认 |
| --- | --- | --- |
| 用户态 `.so`，例如 `librknnrt.so` | RKNN 运行库 | 与驱动、模型工具链的兼容性，以及程序实际加载路径 |
| 可执行程序，例如部分 SDK 的 `rknn_server` | 相应设备连接或调试服务 | 该 SDK 和运行模式是否需要此程序 |
| `/usr/lib`、架构专用库目录或程序自带 `lib/` | 动态库的可能位置 | 对应系统/程序的搜索路径 |
| `/usr/bin` 或 SDK 指定目录 | 可执行文件的可能位置 | 安装说明、权限及服务配置 |

`sudo cp` 只是复制，不会自动解决 ABI、驱动或服务兼容问题。具体库和程序的位置应按板卡系统/BSP 或 SDK 文档确认；未经确认，不把笔记里的两个目录扩展成通用覆盖命令。复制用户态库也不会升级内核驱动。

## 4. Model Zoo 的转换流程

### 4.1 先选匹配的模型输出

Rockchip YOLOv8 示例说明其优化模型与原始模型输出不同，后处理依赖这些输出结构。因此需要一起匹配：

- 导出模型的代码版本和输出节点。
- 输入尺寸、RGB/BGR、归一化和 letterbox 方式。
- RKNN 构建版本及目标 SoC。
- 类别数、类别顺序、框解码、DFL 和 NMS 实现。

单纯把文件重命名成 `.onnx` 或 `.rknn` 不会完成转换。

### 4.2 获取示例并转换

以下接口已与 Rockchip 上游的 YOLOv8 示例说明核对。命令假设在受支持的转换环境中安装好了 Toolkit2，且模型已经按该示例要求准备；具体模型文件不包含在本仓库中。

```bash
git clone https://github.com/airockchip/rknn_model_zoo.git
cd rknn_model_zoo/examples/yolov8/python
```

先以非量化模式检查流程，显式指定输出路径：

```bash
python convert.py ../model/yolov8n.onnx rk3588 fp ../model/yolov8_fp.rknn
```

参数依次为 ONNX 路径、目标平台、数据类型选项、输出 RKNN 路径。`rk3588` 是示例平台，不应照填到其他 SoC；其他版本的参数可能变化，应核对所选提交中的 `convert.py`。

需要 INT8 时，在确认校准数据路径与内容后，例如：

```bash
python convert.py ../model/yolov8n.onnx rk3588 i8 ../model/yolov8_int8.rknn
```

校准数据应覆盖实际相机图像与部署场景；先对照脚本中的数据集路径再运行。其他平台可能使用不同量化类型，不能把 `i8` 视为所有 SoC 的通用选项。

## 5. 推理接口与 Lite2 要分开核对

Model Zoo 上游 YOLOv8 Python 示例记录的 RKNN 命令形式为：

```bash
python yolov8.py --model_path ../model/yolov8_fp.rknn --target rk3588 --img_show
```

这展示的是该 Model Zoo Python 示例接口，不能仅凭命令中的 `--target` 就认定它是独立板端 Lite2 程序。实际应检查所用版本内部采用的 API、设备连接方式和 Runtime 要求；纯板端 Lite2 程序按对应 SDK 示例运行。

Toolkit2 Python 常见导入为 `from rknn.api import RKNN`，Lite2 常见导入为 `from rknnlite.api import RKNNLite`，二者环境及运行方式应分别配置。C/C++ 程序还需与板端 Runtime 动态库和编译架构匹配。

`--img_show` 需要图形显示条件，无显示设备时需按示例支持方式保存结果或关闭显示，而不是将窗口启动失败当作 NPU 推理失败。

## 6. 怎么判断部署是否正确？

先用同一组固定图像，分别对照训练模型、ONNX 和 RKNN 输出：

| 检查 | 观察内容 |
| --- | --- |
| 输入 | 分辨率、颜色通道、归一化、padding 是否一致 |
| 模型输出 | 张量形状、类型、类别数、输出节点是否与后处理匹配 |
| 检测结果 | 框是否偏移、类别是否错位、分数是否异常 |
| 量化结果 | 与非量化结果相比的精度变化，以及实际场景误检漏检 |
| 性能 | 图像读取、预处理、NPU 推理、后处理分别耗时；整体 FPS 不能直接等同纯 NPU FPS |

常见现象与排查方向：

| 现象 | 优先核对 |
| --- | --- |
| `not a supported wheel` | Python/ABI、CPU 架构和系统标签 |
| `No module named rknn` 或 `rknnlite` | 是否激活正确环境，是否安装了对应 API 包 |
| `.so` 加载失败 | 文件架构、加载路径、依赖库与 Runtime 版本 |
| 无法初始化 NPU | 设备/驱动、权限、Runtime 和 SDK 兼容性，或连接方式 |
| 有框但位置不对 | letterbox 还原、坐标尺度、解码与模型输出 |
| INT8 精度明显下降 | 校准图像、输入预处理与量化配置 |

本仓库只整理笔记和操作条件，尚无特定开发板上的安装记录、精度数据或性能测量。

## 7. 官方参考

- [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo)
- [YOLOv8 示例：模型差异、转换及运行接口](https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolov8/README.md)
- [RKNN-Toolkit2 及发布说明](https://github.com/airockchip/rknn-toolkit2)
- [Python wheel 平台兼容标签](https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/)

模型和依赖的版本选择以对应发布文档为准；正文保留了与原始短笔记的区别，补充说明不代表原文件已经记录过这些配置。
