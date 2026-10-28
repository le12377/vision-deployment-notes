# YOLOv8, YOLOv26, YOLO11, YOLOX and YOLOv5-style RKNN Output Export

This is a standalone exporter for YOLOv8, YOLOv26, YOLO11, YOLOX and YOLOv5-style
anchor ONNX models running on RK3588. It preserves the RGB input and replaces
the terminal post-process output with decoded single-output, YOLOv5 three-head,
or six/nine RKNN-friendly tensors.
The `export.py` default remains the nine-output contract used by the supplied RKNPU
references:

| Per-scale order | Tensor | Shape |
| --- | --- | --- |
| 1 | raw bounding-box regression | `[1, 4 or 64, H, W]` |
| 2 | sigmoid class probabilities | `[1, nc, H, W]` |
| 3 | clipped sum of class probabilities | `[1, 1, H, W]` |

The three detection scales produce nine graph outputs in this exact order:
`bbox_0, scores_0, score_sum_0, ..., bbox_2, scores_2, score_sum_2`.
When the six-output contract is selected, `score_sum_*` is omitted and the
order is `bbox_0, scores_0, ..., bbox_2, scores_2`.

For YOLOv5-style anchor models, the optional three-output contract keeps one
activated detection head per scale in NCHW form:
`[N, anchors*(5+nc), H, W]`. This is the format consumed by the standard
YOLOv5 RKNN post-processing code. The three-output contract is only available
for YOLOv5-style anchor heads; YOLOv8, YOLOv11, YOLOv26, and YOLOX use the
six- or nine-output contracts.

For YOLOv5-style anchor heads, those outputs keep the anchor dimension:
`bbox` is `[N, anchors, H, W, 4]`, `scores` is `[N, anchors, H, W, nc]`, and
`score_sum` is `[N, anchors, H, W, 1]`.

The exporter recognizes YOLOv26's three `one2one_cv2` and `one2one_cv3`
branches, plus YOLOv8/YOLO11 `cv2` and `cv3` branches. It prunes the embedded
terminal post-process from the ONNX graph before RKNN conversion. It also
falls back to YOLOv5-style anchor heads, where the six/nine outputs are 5D tensors in
`[N, anchors, H, W, channels]` form and `score_sum` carries objectness when
the nine-output contract is selected. YOLOv5 exports that flatten each head
before sigmoid to `[N, anchors, H*W, channels]` are also recognized, and the
exporter restores the static `[N, anchors, H, W, channels]` layout in the
generated graph. The three-output contract converts these heads to NCHW.

YOLOX decoupled heads are recognized from `head.reg_preds.*`,
`head.obj_preds.*`, and `head.cls_preds.*` branches. For YOLOX, `bbox_*` is raw
4-channel regression, `scores_*` is sigmoid class probability, and `score_sum_*`
is sigmoid objectness when the nine-output contract is selected.

## Important

"Nine-output" does not mean a nine-channel image input. The source model
remains `[N, 3, H, W]`, and RKNN normalization must therefore contain three
mean/std values. Changing an RGB model to a nine-channel input requires a
different first convolution and training or weight migration; this exporter
does not do that.

The resulting RKNN model requires CPU-side decoding and NMS. YOLOv26 uses
4-channel direct box regression while YOLO11 uses 64-channel DFL regression,
so the deployment-side decoder must match the source model family. YOLOv5-style
anchor exports use the same three-output-per-scale layout with explicit anchor
dimensions.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
# Install rknn-toolkit2 separately when building the .rknn model.
```

## YOLO26 单输出：onnx_decoded.py（GB 盒子已上板验证）

运行 `onnx_decoded.py`，默认保留 YOLO26 的坐标解码和全部类别概率，移除末尾
TopK/类别选择，将 `x1,y1,x2,y2` 转为 `中心X,中心Y,宽,高`。
输出排列为 `[N,4+nc,候选数]`。这与此次 GB 盒子成功使用的模型格式一致。

仅生成 ONNX（默认行为）：

```bash
python3 src/yolo26_rknn9/onnx_decoded.py \
  artifacts/6AY_v26m_1007_320x320.onnx
```

同时生成 RK3588 FP16 RKNN：

```bash
python3 src/yolo26_rknn9/onnx_decoded.py \
  artifacts/6AY_v26m_1007_320x320.onnx \
  --rknn-output artifacts/6AY_v26m_1007_320x320_decoded_xywh.rknn
```

也支持在 `src/yolo26_rknn9` 目录运行：

```bash
python3 onnx_decoded.py ../../artifacts/6AY_v26m_1007_320x320.onnx
```

只提供源文件时，默认生成 `6AY_v26m_1007_320x320_decoded_xywh.onnx`。
第二个位置参数可指定 ONNX 输出路径；`--rknn-output` 可额外指定 RKNN 输出路径。
默认采用本次验证过的 XYWH/BCN 格式；ONNX 保持 FP32，RKNN 编译为 RK3588 FP16。
无需传输出数量 `1`，也无需额外传 `--box-format xywh`。
模块形式 `PYTHONPATH=src python -m yolo26_rknn9.onnx_decoded` 同样支持。

输入必须是保留解码分支的原始 end-to-end YOLO26 ONNX；不能使用已经转换的
RKNN9 ONNX 或 `.rknn`。6AY 的输入为 `[1,3,320,320]`，输出为 `[1,11,2100]`，
类别顺序为 `body, head, teammate, ai, tool, down, fyb`。
软件配置保持 YOLO26、320×320、7 类，阈值可沿用已验证的 0.65。
输入尺寸读取 ONNX 本身。

单输出转换在 `onnx_decoded.py` 中完成；`export.py` 保持原有的 3/6/9 输出入口。

## Export From An Existing ONNX Model: Three / Six / Nine Outputs

Create and validate only the nine-output ONNX model:

```bash
PYTHONPATH=src python -m yolo26_rknn9.export \
  --weights /path/to/yolov26.onnx \
  --onnx-only
```

This writes `/path/to/yolov26_rknn9.onnx` unless `--onnx-output` is supplied.
Select the six-output contract either positionally or with `--output-count`:

```bash
cd src/yolo26_rknn9
python3 export.py ../../artifacts/model.onnx 6

PYTHONPATH=src python -m yolo26_rknn9.export \
  --weights /path/to/model.onnx \
  --output-count 6
```

Without the trailing `6` or `--output-count 6`, the exporter writes the
default nine-output model. Six-output defaults use the `_rknn6.onnx` suffix.

For a YOLOv5 model, request the traditional three detection-head outputs:

```bash
cd src/yolo26_rknn9
python3 export.py ../../artifacts/yolov5.onnx 3
```

Three-output models use the `_rknn3.onnx` suffix and expose
`yolov5_output_0`, `yolov5_output_1`, and `yolov5_output_2`.

## Export ONNX To EP-Style `reg/cls` ONNX

For OBS/plugin builds that expect the EP-style six-output contract, use:

```bash
cd src/yolo26_rknn9
python3 onnx_regcls.py ../../artifacts/model.onnx
```

This writes `../../artifacts/model_regcls.onnx` by default. You can also pass
the destination explicitly:

```bash
python3 onnx_regcls.py ../../artifacts/model.onnx ../../artifacts/model_ep.onnx
```

The output order is exactly:
`reg1, cls1, reg2, cls2, reg3, cls3`.

For YOLOv8/YOLO11 64-channel DFL heads, the script adds the DFL projection
inside the ONNX graph, so `reg*` becomes `[N, 1, 4, H*W]`. `cls*` keeps the
raw class logits `[N, nc, H, W]` without sigmoid, matching the EP RKNN-style
contract.

The DFL projection is implemented in an opset-12-safe layout: the 16 DFL bins
are moved to the last axis before `Softmax`, then projected with a fixed
`0..15` convolution. This avoids the older ONNX `Softmax` axis behavior from
normalizing across the flattened box/grid dimensions and collapsing `reg*`
values toward zero.

## Export And Convert To RKNN

```bash
PYTHONPATH=src python -m yolo26_rknn9.export \
  --weights /path/to/yolov26.pt \
  --imgsz 416 \
  --quantize \
  --calib-data /path/to/calibration-images \
  --target-platform rk3588
```

`--calib-data` accepts either an image directory or an RKNN dataset-list text
file. Images selected from a directory are sorted and sampled with a fixed
seed, so the generated calibration list is reproducible.

## Quantization Controls

When `--quantize` is present, the exporter exposes the RKNN Toolkit2 2.3.2
quantization controls. Defaults are `w8a8`, `normal`, and `channel`.

```bash
PYTHONPATH=src python -m yolo26_rknn9.export \
  --weights /path/to/yolov26.onnx \
  --quantize \
  --calib-data /path/to/calibration-images \
  --quantized-dtype w8a8 \
  --quantized-algorithm mmse \
  --quantized-method channel
```

Toolkit2 exposes `w8a8`, `w8a16`, `w16a16i`, `w16a16i_dfp`, and `w4a16` in
its generic API. For the RK3588 target used by this project, Toolkit2 2.3.2
accepts `w8a8`, `w16a16i`, and `w16a16i_dfp`; the CLI rejects the other two
before building. Supported algorithms are `normal`, `mmse`, `kl_divergence`,
and `gdq`. The method can be `layer`, `channel`, or a group size such as
`group32` through `group256` in multiples of 32. Actual precision and
performance still depend on the calibration data and model operators.

## Advanced Decoded Output and Diagnostics

`onnx_decoded.py` defaults to the verified GB XYWH/BCN contract above.
Explicit options allow source-format coordinates, alternate layouts and
diagnostic models.

For software that rejects both `[1, 300, 6]` end-to-end detections and raw
six/nine-output heads, expose the existing decoded tensor before TopK:

```bash
PYTHONPATH=src python -m yolo26_rknn9.onnx_decoded \
  artifacts/6AY_v26m_1007_320x320.onnx \
  artifacts/6AY_v26m_1007_320x320_decoded.onnx \
  --box-format source \
  --rknn-output artifacts/6AY_v26m_1007_320x320_decoded_fp16.rknn
```

Omit `--rknn-output` to produce only ONNX. This separate entry point leaves
the existing three/six/nine-output exporter unchanged. It requires the original
end-to-end ONNX, not an already converted nine-output or RKNN model.

For software that reads one candidate per row, pass `--layout bnc` and use
distinct output filenames. This emits `[1, 2100, 11]` for 6AY, preserving the
same coordinates and probabilities. It is a layout compatibility alternative;
it does not change confidence thresholds, apply NMS, or convert coordinates.
Do not assume the target application's layout from successful initialization
alone: reading BCN data as BNC can interpret box coordinates as class scores.

If neither layout has been verified in the target software, avoid treating a
layout guess as a fix. `--diagnostic-zero-scores` creates a control model with
unchanged box coordinates, layout and precision, but all class probabilities
forced to zero. At a positive confidence threshold it must produce no
detections. This model is for diagnosing application output handling only;
it intentionally disables recognition and is not a fixed detector.

For a consumer that expects center-based `xywh` coordinates, use
`--box-format xywh` (the default). This explicitly assumes the source coordinates are
`xyxy`, as verified for the supplied 6AY model. It computes
`cx=(x1+x2)/2`, `cy=(y1+y2)/2`, `w=x2-x1`, `h=y2-y1` in input pixels,
and keeps every class probability unchanged. Output layout remains BCN by
default. Pass `--box-format source` for consumers that need source coordinates.

```bash
PYTHONPATH=src python -m yolo26_rknn9.onnx_decoded \
  artifacts/6AY_v26m_1007_320x320.onnx \
  artifacts/6AY_v26m_1007_320x320_xywh_fp16.onnx \
  --box-format xywh \
  --rknn-output artifacts/6AY_v26m_1007_320x320_xywh_fp16.rknn
```

With `--diagnostic-fixed-box --box-format xywh`, the known box is emitted
as `[160,160,160,192]` for a 320x320 input. A center-based consumer should
draw the same central rectangle as source corners `[80,64,240,256]`.

After a successful zero-score test, `--diagnostic-fixed-box` checks one known
nonzero candidate. It forces all class scores to zero except class 0 at the
middle point, which is exactly `0.875`, and replaces that candidate's box with
`xyxy = [0.25*width, 0.2*height, 0.75*width, 0.8*height]`. Other candidates
retain their live boxes and zero scores. For 320x320 input the visible box
should be `[80, 64, 240, 256]`, inside the inference region. Allow class 0 in
any application class filter. This is another diagnostic that disables live
recognition; it cannot be combined with `--diagnostic-zero-scores`.

The 6AY output is `output0: [1, 11, 2100]`, with four decoded coordinates in
input pixels and seven sigmoid class probabilities. `onnx_decoded.py` converts
to center-based `xywh` by default; `--box-format source` preserves source
`xyxy`. Class order:
`body, head, teammate, ai, tool, down, fyb`. The transformation preserves the
original one-to-one head, coordinate decode, RGB input and model weights,
and removes terminal TopK/class selection. No NMS is embedded in this graph.
RKNN conversion targets RK3588 using FP16 with normalization `pixel / 255`
and does not use calibration quantization. The user confirmed the XYWH FP16
6AY model works correctly on the GB third-generation box on 2026-10-19.
Other consumers and models still require their own compatibility checks.

## Validate

```bash
PYTHONPATH=src pytest -q
```

The tests verify the existing output contracts, decoded-coordinate conversion,
unchanged class probabilities, diagnostic models, `onnx_decoded.py` default
export in script/module forms, optional RKNN compilation and source-file protection.
