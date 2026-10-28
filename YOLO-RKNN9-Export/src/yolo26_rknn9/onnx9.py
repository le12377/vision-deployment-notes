"""Convert YOLOv8, YOLOv26, YOLO11 or YOLOX ONNX graphs to RKNN-friendly output contracts."""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union, cast

import onnx
from onnx import TensorProto, checker, helper, shape_inference


class NineOutputError(ValueError):
    """Raised when an ONNX model does not match a supported YOLO detection head."""


@dataclass(frozen=True)
class TensorDetails:
    """ONNX tensor metadata needed to create a graph output."""

    name: str
    element_type: int
    shape: tuple[object, ...]


@dataclass(frozen=True)
class HeadScale:
    """Raw one-to-one box and class tensors for one YOLOv26 detection scale."""

    index: int
    bbox: TensorDetails
    classes: TensorDetails


@dataclass(frozen=True)
class AnchorHeadScale:
    """Raw anchor-based detection tensor for one legacy YOLO scale."""

    index: int
    raw: TensorDetails
    anchor_count: int
    class_count: int
    grid_height: int
    grid_width: int
    raw_layout: Literal["nchw", "nahwc", "na_hw_c"]
    activated: TensorDetails


@dataclass(frozen=True)
class YoloXHeadScale:
    """Raw YOLOX box, objectness and class tensors for one detection scale."""

    index: int
    bbox: TensorDetails
    objectness: TensorDetails
    classes: TensorDetails


DetectionScale = Union[HeadScale, AnchorHeadScale, YoloXHeadScale]
OutputCount = Literal[3, 6, 9]
AnchorCandidate = tuple[TensorDetails, TensorDetails, int, int, int, str]

# The zy box accepts the same legacy output names as the known-good
# ``TF_v8s_2c_320--i8.rknn`` model.  The names are only a compatibility
# surface; tensor order and tensor shapes remain the normal RKNN9 contract.
ZY_LEGACY_OUTPUT_NAMES = (
    "318",
    "onnx::ReduceSum_326",
    "331",
    "338",
    "onnx::ReduceSum_346",
    "350",
    "357",
    "onnx::ReduceSum_365",
    "369",
)

# 这个正则表达式已经兼容 YOLOv8 (cv2.0) 和 YOLOv11/YOLOv26 (one2one_cv2.0) 的命名
_FINAL_HEAD_CONV = re.compile(r"/(?:one2one_)?(cv[23])\.(\d+)/(?:one2one_)?\1\.\2\.2/Conv$")

# Older Ultralytics/PyTorch ONNX exports rename graph nodes to Conv_*.
# The detection-head identity is still present in the convolution weight input.
_LEGACY_FINAL_HEAD_WEIGHT = re.compile(
    r"^model\.\d+\.(cv[23])\.(\d+)\.2(?:\.conv)?\.weight$"
)

_YOLOX_HEAD_CONV = re.compile(r"^/head/(reg_preds|obj_preds|cls_preds)\.(\d+)/Conv$")


def _infer_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Return shape-enriched model metadata without changing the caller's model."""

    try:
        return shape_inference.infer_shapes(model, strict_mode=True)
    except Exception as exc:
        raise NineOutputError(f"ONNX shape inference failed: {exc}") from exc


def _value_details(model: onnx.ModelProto) -> dict[str, TensorDetails]:
    details: dict[str, TensorDetails] = {}
    values = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    for value in values:
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        shape: list[object] = []
        for dimension in tensor_type.shape.dim:
            if dimension.HasField("dim_value"):
                shape.append(dimension.dim_value)
            elif dimension.HasField("dim_param"):
                shape.append(dimension.dim_param)
            else:
                shape.append(None)
        details[value.name] = TensorDetails(value.name, tensor_type.elem_type, tuple(shape))
    return details


def _image_input(model: onnx.ModelProto) -> TensorDetails:
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    inputs = [value for value in model.graph.input if value.name not in initializer_names]
    if len(inputs) != 1:
        raise NineOutputError(f"expected one image input, found {len(inputs)}")
    details = _value_details(model).get(inputs[0].name)
    if details is None or len(details.shape) != 4:
        raise NineOutputError("the image input must have a four-dimensional NCHW shape")
    if details.shape[1] != 3:
        raise NineOutputError(
            f"expected an RGB [N, 3, H, W] input, found {list(details.shape)}. "
            "Nine-output export does not change the model input channels."
        )
    return details


def _find_head_scales(model: onnx.ModelProto) -> list[HeadScale]:
    details = _value_details(model)
    branches: dict[int, dict[str, TensorDetails]] = {}

    for node in model.graph.node:
        match = _FINAL_HEAD_CONV.search(node.name)
        if match is None or node.op_type != "Conv" or len(node.output) != 1:
            continue

        branch, scale_text = match.groups()
        output = details.get(node.output[0])
        if output is None or len(output.shape) != 4:
            raise NineOutputError(f"missing static BCHW shape for {node.name}")
        branches.setdefault(int(scale_text), {})[branch] = output

    # Keep the current node-name matcher as the primary path. A number of
    # older YOLOv8 exports use generic Conv_* node names, while retaining the
    # original model.*.cv2/cv3.<scale>.2.weight names on Conv inputs.
    if not branches:
        for node in model.graph.node:
            if node.op_type != "Conv" or len(node.output) != 1 or len(node.input) < 2:
                continue
            match = _LEGACY_FINAL_HEAD_WEIGHT.fullmatch(node.input[1])
            if match is None:
                continue
            branch, scale_text = match.groups()
            output = details.get(node.output[0])
            if output is None or len(output.shape) != 4:
                raise NineOutputError(f"missing static BCHW shape for {node.name}")
            branches.setdefault(int(scale_text), {})[branch] = output

    scales: list[HeadScale] = []
    for index in sorted(branches):
        tensors = branches[index]
        if set(tensors) != {"cv2", "cv3"}:
            raise NineOutputError(f"incomplete one-to-one detection branch at scale {index}")
        bbox, classes = tensors["cv2"], tensors["cv3"]
        if bbox.shape[1] not in (4, 64):
            raise NineOutputError(
                f"scale {index} bbox output must have 4 (YOLOv26) or 64 (YOLOv8/YOLO11 DFL) channels, "
                f"found {bbox.shape[1]}"
            )
        if not isinstance(classes.shape[1], int) or classes.shape[1] < 1:
            raise NineOutputError(f"scale {index} class output must have at least one channel")
        if bbox.shape[0] != classes.shape[0] or bbox.shape[2:] != classes.shape[2:]:
            raise NineOutputError(f"scale {index} bbox and class outputs have incompatible shapes")
        scales.append(HeadScale(index=index, bbox=bbox, classes=classes))

    if len(scales) != 3:
        raise NineOutputError(
            "expected exactly three detection scales (YOLOv8/v11/v26), "
            f"found {len(scales)}"
        )
    bbox_channel_counts = {scale.bbox.shape[1] for scale in scales}
    if len(bbox_channel_counts) != 1:
        raise NineOutputError("all detection scales must use the same bbox regression channel count")
    return scales


def _find_anchor_head_scales(model: onnx.ModelProto) -> list[AnchorHeadScale]:
    details = _value_details(model)
    candidates: list[AnchorCandidate] = []

    for node in model.graph.node:
        if node.op_type != "Sigmoid" or len(node.input) != 1:
            continue
        raw = details.get(node.input[0])
        activated = details.get(node.output[0]) if node.output else None
        if raw is None or activated is None or len(raw.shape) not in (4, 5):
            continue
        if not all(isinstance(dimension, int) for dimension in raw.shape):
            raise NineOutputError(f"missing static anchor-head shape for {node.name}")
        if len(raw.shape) == 5:
            if raw.shape[1] < 1 or raw.shape[4] < 6:
                continue
            candidates.append(
                (raw, activated, int(raw.shape[1]), int(raw.shape[2]), int(raw.shape[3]), "nahwc")
            )
            continue

        # Some YOLOv5 exports transpose each head to [N, anchors, H*W, 5+nc]
        # before applying sigmoid and then decode those flattened predictions.
        # Only accept the distinctive layout: a small anchor axis, a larger
        # flattened grid axis, and the channel axis in the final position.
        _batch, dim1, dim2, dim3 = map(int, raw.shape)
        if 1 <= dim1 <= 9 and dim2 > dim3 >= 6:
            grid_size = math.isqrt(dim2)
            if grid_size * grid_size != dim2:
                raise NineOutputError(
                    f"flattened anchor-head grid must be square, found {dim2} points in {node.name}"
                )
            candidates.append((raw, activated, dim1, grid_size, grid_size, "na_hw_c"))
            continue

        # Standard YOLOv5 ONNX exports keep the final head as NCHW
        # [N, anchors*(5+nc), H, W] and apply sigmoid directly to it.
        if dim2 != dim3 or dim1 < 6:
            continue
        for anchor_count in range(1, 10):
            if dim1 % anchor_count:
                continue
            channels_per_anchor = dim1 // anchor_count
            if channels_per_anchor >= 6:
                candidates.append(
                    (
                        raw,
                        activated,
                        anchor_count,
                        dim2,
                        dim3,
                        "nchw",
                    )
                )

    if not candidates:
        raise NineOutputError(
            "expected exactly three detection scales (YOLOv5-style anchor head), "
            "found 0"
        )

    groups: dict[tuple[int, int], list[AnchorCandidate]] = {}
    for candidate in candidates:
        raw, _activated, anchor_count, grid_height, grid_width, layout = candidate
        channels = int(raw.shape[4] if layout == "nahwc" else raw.shape[3] if layout == "na_hw_c" else raw.shape[1])
        class_count = channels - 5 if layout != "nchw" else channels // anchor_count - 5
        if class_count < 1:
            continue
        groups.setdefault((anchor_count, class_count), []).append(candidate)

    viable_groups = []
    for (anchor_count, class_count), group in groups.items():
        by_area: dict[int, AnchorCandidate] = {}
        for candidate in group:
            area = candidate[3] * candidate[4]
            by_area[area] = candidate
        if len(by_area) >= 3:
            viable_groups.append((anchor_count, class_count, list(by_area.values())))

    if not viable_groups:
        raise NineOutputError(
            "expected exactly three detection scales (YOLOv5-style anchor head), "
            f"found {len(candidates)} compatible head tensors"
        )

    # Prefer the conventional three-anchor interpretation when NCHW channel
    # counts admit multiple factorizations (for example, 24 = 3 * 8).
    viable_groups.sort(
        key=lambda item: (
            item[0] == 3,
            len(item[2]),
            max(candidate[3] * candidate[4] for candidate in item[2]),
        ),
        reverse=True,
    )
    anchor_count, class_count, selected = viable_groups[0]
    selected.sort(key=lambda item: item[3] * item[4], reverse=True)
    selected = selected[:3]

    scales: list[AnchorHeadScale] = []
    for index, (raw, activated, _, grid_height, grid_width, layout) in enumerate(selected):
        scales.append(
            AnchorHeadScale(
                index=index,
                raw=raw,
                anchor_count=anchor_count,
                class_count=class_count,
                grid_height=grid_height,
                grid_width=grid_width,
                raw_layout=cast(Literal["nchw", "nahwc", "na_hw_c"], layout),
                activated=activated,
            )
        )
    return scales


def _find_yolox_head_scales(model: onnx.ModelProto) -> list[YoloXHeadScale]:
    details = _value_details(model)
    branches: dict[int, dict[str, TensorDetails]] = {}

    for node in model.graph.node:
        match = _YOLOX_HEAD_CONV.fullmatch(node.name)
        if match is None or node.op_type != "Conv" or len(node.output) != 1:
            continue
        branch, scale_text = match.groups()
        output = details.get(node.output[0])
        if output is None or len(output.shape) != 4:
            raise NineOutputError(f"missing static BCHW shape for {node.name}")
        branches.setdefault(int(scale_text), {})[branch] = output

    scales: list[YoloXHeadScale] = []
    for index in sorted(branches):
        tensors = branches[index]
        if set(tensors) != {"reg_preds", "obj_preds", "cls_preds"}:
            raise NineOutputError(f"incomplete YOLOX detection branch at scale {index}")
        bbox, objectness, classes = tensors["reg_preds"], tensors["obj_preds"], tensors["cls_preds"]
        if bbox.shape[1] != 4:
            raise NineOutputError(f"YOLOX scale {index} bbox output must have 4 channels, found {bbox.shape[1]}")
        if objectness.shape[1] != 1:
            raise NineOutputError(
                f"YOLOX scale {index} objectness output must have 1 channel, found {objectness.shape[1]}"
            )
        if not isinstance(classes.shape[1], int) or classes.shape[1] < 1:
            raise NineOutputError(f"YOLOX scale {index} class output must have at least one channel")
        if bbox.shape[0] != classes.shape[0] or bbox.shape[2:] != classes.shape[2:]:
            raise NineOutputError(f"YOLOX scale {index} bbox and class outputs have incompatible shapes")
        if bbox.shape[0] != objectness.shape[0] or bbox.shape[2:] != objectness.shape[2:]:
            raise NineOutputError(f"YOLOX scale {index} bbox and objectness outputs have incompatible shapes")
        scales.append(YoloXHeadScale(index=index, bbox=bbox, objectness=objectness, classes=classes))

    if len(scales) != 3:
        raise NineOutputError(
            "expected exactly three detection scales (YOLOX head), "
            f"found {len(scales)}"
        )
    return scales


def _default_opset(model: onnx.ModelProto) -> int:
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return opset.version
    raise NineOutputError("the ONNX model does not declare a default-domain opset")


def _prune_to_graph_outputs(model: onnx.ModelProto) -> None:
    """Remove nodes and initializers that do not contribute to a graph output."""

    nodes = list(model.graph.node)
    producer: dict[str, int] = {}
    for index, node in enumerate(nodes):
        for output_name in node.output:
            if output_name:
                producer[output_name] = index

    required_nodes: set[int] = set()

    def visit(tensor_name: str) -> None:
        index = producer.get(tensor_name)
        if index is None or index in required_nodes:
            return
        required_nodes.add(index)
        for input_name in nodes[index].input:
            if input_name:
                visit(input_name)

    for output in model.graph.output:
        visit(output.name)

    retained_nodes = [node for index, node in enumerate(nodes) if index in required_nodes]
    required_inputs = {input_name for node in retained_nodes for input_name in node.input if input_name}
    retained_initializers = [
        initializer for initializer in model.graph.initializer if initializer.name in required_inputs
    ]

    model.graph.ClearField("node")
    model.graph.node.extend(retained_nodes)
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(retained_initializers)
    model.graph.ClearField("value_info")


def _value_info(name: str, details: TensorDetails) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, details.element_type, list(details.shape))


def _validate_output_count(output_count: int) -> OutputCount:
    if output_count not in (3, 6, 9):
        raise NineOutputError(f"output count must be 3, 6, or 9, found {output_count}")
    return cast(OutputCount, output_count)


def build_output_model(
    model: onnx.ModelProto,
    output_count: int = 9,
    *,
    legacy_output_names: bool = False,
) -> tuple[onnx.ModelProto, list[TensorDetails]]:
    """Create a pruned YOLOv8, YOLOv26 or YOLO11 ONNX model with two or three outputs per scale.

    Six-output order is bbox and sigmoid class scores for scale indexes 0, 1,
    and 2. Nine-output order additionally appends clipped class-score sum after
    every score tensor. YOLOv26 contributes 4-channel direct box regression
    while YOLOv8 and YOLO11 contribute 64-channel DFL regression. The raw
    bbox tensors are intentionally not decoded in the graph because the
    post-process is not RKNN quantization friendly.
    """

    output_count = _validate_output_count(output_count)
    converted = _infer_shapes(copy.deepcopy(model))
    _image_input(converted)
    head_mode = "cv"
    try:
        scales: list[DetectionScale] = _find_head_scales(converted)
    except NineOutputError as exc:
        if "found 0" not in str(exc):
            raise
        try:
            scales = _find_yolox_head_scales(converted)
            head_mode = "yolox"
        except NineOutputError as yolox_exc:
            if "found 0" not in str(yolox_exc):
                raise
            scales = _find_anchor_head_scales(converted)
            head_mode = "anchor"
    if output_count == 3 and head_mode != "anchor":
        raise NineOutputError("the 3-output contract is only supported for YOLOv5-style anchor heads")
    if legacy_output_names and (output_count != 9 or head_mode != "cv"):
        raise NineOutputError("legacy zy output names require a nine-output YOLOv8/YOLO11-style head")
    if legacy_output_names and len(scales) != 3:
        raise NineOutputError("legacy zy output names require exactly three detection scales")
    opset = _default_opset(converted)
    if opset < 11:
        raise NineOutputError(f"ONNX opset {opset} is unsupported; use opset 11 or newer")

    graph = converted.graph
    existing_names = {
        value.name
        for value in list(graph.input) + list(graph.output) + list(graph.value_info) + list(graph.initializer)
    }
    existing_names.update(output for node in graph.node for output in node.output)
    new_nodes: list[onnx.NodeProto] = []
    new_outputs: list[onnx.ValueInfoProto] = []
    output_details: list[TensorDetails] = []

    min_name = "rknn9_clip_min"
    max_name = "rknn9_clip_max"
    sum_axes_name = "rknn9_score_sum_axes"
    if output_count == 9 and head_mode == "cv":
        reserved_names = {min_name, max_name}
        if opset >= 13:
            reserved_names.add(sum_axes_name)
        conflicts = reserved_names & existing_names
        if conflicts:
            raise NineOutputError(f"reserved RKNN9 initializer names already exist in the source graph: {sorted(conflicts)}")
        existing_names.update(reserved_names)
        graph.initializer.extend(
            [
                helper.make_tensor(min_name, TensorProto.FLOAT, [], [0.0]),
                helper.make_tensor(max_name, TensorProto.FLOAT, [], [1.0]),
            ]
        )
        if opset >= 13:
            graph.initializer.extend([helper.make_tensor(sum_axes_name, TensorProto.INT64, [1], [1])])

    for scale in scales:
        if legacy_output_names:
            name_offset = scale.index * 3
            bbox_name, scores_name, sum_name = ZY_LEGACY_OUTPUT_NAMES[name_offset : name_offset + 3]
        else:
            bbox_name = f"bbox_{scale.index}"
            scores_name = f"scores_{scale.index}"
            sum_name = f"score_sum_{scale.index}"
        if output_count == 3:
            assert isinstance(scale, AnchorHeadScale)
            output_name = f"yolov5_output_{scale.index}"
            if output_name in existing_names:
                raise NineOutputError(f"source graph already uses output name: {output_name}")
            existing_names.add(output_name)
            output_shape = (
                int(scale.activated.shape[0]),
                scale.anchor_count * (5 + scale.class_count),
                scale.grid_height,
                scale.grid_width,
            )
            if scale.raw_layout == "nchw":
                new_nodes.append(
                    helper.make_node(
                        "Identity",
                        [scale.activated.name],
                        [output_name],
                        name=f"RKNN3/output_{scale.index}",
                    )
                )
            else:
                reshape_name = f"rknn3_anchor_{scale.index}_reshape"
                transpose_name = f"rknn3_anchor_{scale.index}_transpose"
                shape_name = f"rknn3_anchor_{scale.index}_shape"
                nchw_shape_name = f"rknn3_anchor_{scale.index}_nchw_shape"
                conflicts = {
                    reshape_name,
                    transpose_name,
                    shape_name,
                    nchw_shape_name,
                } & existing_names
                if conflicts:
                    raise NineOutputError(f"reserved RKNN3 names already exist in the source graph: {sorted(conflicts)}")
                existing_names.update({reshape_name, transpose_name, shape_name, nchw_shape_name})
                graph.initializer.extend(
                    [
                        helper.make_tensor(
                            shape_name,
                            TensorProto.INT64,
                            [5],
                            [
                                int(scale.activated.shape[0]),
                                scale.anchor_count,
                                scale.grid_height,
                                scale.grid_width,
                                5 + scale.class_count,
                            ],
                        ),
                        helper.make_tensor(
                            nchw_shape_name,
                            TensorProto.INT64,
                            [4],
                            list(output_shape),
                        ),
                    ]
                )
                new_nodes.extend(
                    [
                        helper.make_node(
                            "Reshape",
                            [scale.activated.name, shape_name],
                            [reshape_name],
                            name=f"RKNN3/reshape_{scale.index}",
                        ),
                        helper.make_node(
                            "Transpose",
                            [reshape_name],
                            [transpose_name],
                            name=f"RKNN3/transpose_{scale.index}",
                            perm=[0, 1, 4, 2, 3],
                        ),
                        helper.make_node(
                            "Reshape",
                            [transpose_name, nchw_shape_name],
                            [output_name],
                            name=f"RKNN3/output_{scale.index}",
                        ),
                    ]
                )
            details = TensorDetails(output_name, scale.activated.element_type, output_shape)
            new_outputs.append(_value_info(details.name, details))
            output_details.append(details)
            continue

        if head_mode == "cv":
            requested_names = [bbox_name, scores_name]
            if output_count == 9:
                requested_names.extend([sum_name, f"score_sum_raw_{scale.index}"])
        elif head_mode == "anchor":
            requested_names = [
                bbox_name,
                scores_name,
                f"rknn9_anchor_{scale.index}_bbox",
                f"rknn9_anchor_{scale.index}_cls",
            ]
            if output_count == 9:
                requested_names.extend([sum_name, f"rknn9_anchor_{scale.index}_obj"])
            if len(scale.raw.shape) == 4:
                requested_names.append(f"rknn9_anchor_{scale.index}_raw")
        else:
            requested_names = [bbox_name, scores_name]
            if output_count == 9:
                requested_names.append(sum_name)
        conflicts = set(requested_names) & existing_names
        if conflicts:
            raise NineOutputError(f"source graph already uses output names: {sorted(conflicts)}")
        existing_names.update(requested_names)

        if head_mode == "cv":
            assert isinstance(scale, HeadScale)
            sum_raw_name = f"score_sum_raw_{scale.index}"
            reduce_sum_inputs = [scores_name]
            reduce_sum_attributes = {"keepdims": 1}
            if opset >= 13:
                reduce_sum_inputs.append(sum_axes_name)
            else:
                reduce_sum_attributes["axes"] = [1]
            new_nodes.extend(
                [
                    helper.make_node("Identity", [scale.bbox.name], [bbox_name], name=f"RKNN9/bbox_{scale.index}"),
                    helper.make_node("Sigmoid", [scale.classes.name], [scores_name], name=f"RKNN9/scores_{scale.index}"),
                ]
            )
            if output_count == 9:
                new_nodes.extend(
                    [
                        helper.make_node(
                            "ReduceSum",
                            reduce_sum_inputs,
                            [sum_raw_name],
                            name=f"RKNN9/score_sum_{scale.index}",
                            **reduce_sum_attributes,
                        ),
                        helper.make_node(
                            "Clip",
                            [sum_raw_name, min_name, max_name],
                            [sum_name],
                            name=f"RKNN9/clip_score_sum_{scale.index}",
                        ),
                    ]
                )
            bbox_details = TensorDetails(bbox_name, scale.bbox.element_type, scale.bbox.shape)
            scores_details = TensorDetails(scores_name, scale.classes.element_type, scale.classes.shape)
            sum_shape = (scale.classes.shape[0], 1, scale.classes.shape[2], scale.classes.shape[3])
            sum_details = TensorDetails(sum_name, scale.classes.element_type, sum_shape)
        elif head_mode == "yolox":
            assert isinstance(scale, YoloXHeadScale)
            new_nodes.extend(
                [
                    helper.make_node("Identity", [scale.bbox.name], [bbox_name], name=f"RKNN9/bbox_{scale.index}"),
                    helper.make_node("Sigmoid", [scale.classes.name], [scores_name], name=f"RKNN9/scores_{scale.index}"),
                ]
            )
            if output_count == 9:
                new_nodes.append(
                    helper.make_node(
                        "Sigmoid",
                        [scale.objectness.name],
                        [sum_name],
                        name=f"RKNN9/score_sum_{scale.index}",
                    )
                )
            bbox_details = TensorDetails(bbox_name, scale.bbox.element_type, scale.bbox.shape)
            scores_details = TensorDetails(scores_name, scale.classes.element_type, scale.classes.shape)
            sum_details = TensorDetails(sum_name, scale.objectness.element_type, scale.objectness.shape)
        else:
            assert isinstance(scale, AnchorHeadScale)
            slice_prefix = f"rknn9_anchor_{scale.index}"
            raw_input_name = scale.raw.name
            raw_shape_name = f"{slice_prefix}_raw_shape"
            if len(scale.raw.shape) == 4:
                raw_input_name = f"{slice_prefix}_raw"
            starts_bbox = f"{slice_prefix}_bbox_starts"
            ends_bbox = f"{slice_prefix}_bbox_ends"
            starts_obj = f"{slice_prefix}_obj_starts"
            ends_obj = f"{slice_prefix}_obj_ends"
            starts_cls = f"{slice_prefix}_cls_starts"
            ends_cls = f"{slice_prefix}_cls_ends"
            axes_name = f"{slice_prefix}_axes"
            steps_name = f"{slice_prefix}_steps"
            init_names = {
                starts_bbox,
                ends_bbox,
                starts_cls,
                ends_cls,
                axes_name,
                steps_name,
            }
            if scale.raw_layout == "na_hw_c":
                init_names.add(raw_shape_name)
            if scale.raw_layout == "nchw":
                init_names.update({raw_shape_name, f"{slice_prefix}_nchw", f"{slice_prefix}_transpose"})
            if output_count == 9:
                init_names.update({starts_obj, ends_obj})
            if init_names & existing_names:
                raise NineOutputError("reserved RKNN9 initializer names already exist in the source graph")
            existing_names.update(init_names)
            initializers = [
                helper.make_tensor(starts_bbox, TensorProto.INT64, [1], [0]),
                helper.make_tensor(ends_bbox, TensorProto.INT64, [1], [4]),
                helper.make_tensor(starts_cls, TensorProto.INT64, [1], [5]),
                helper.make_tensor(ends_cls, TensorProto.INT64, [1], [5 + scale.class_count]),
                helper.make_tensor(axes_name, TensorProto.INT64, [1], [4]),
                helper.make_tensor(steps_name, TensorProto.INT64, [1], [1]),
            ]
            if scale.raw_layout == "na_hw_c":
                initializers.append(
                    helper.make_tensor(
                        raw_shape_name,
                        TensorProto.INT64,
                        [5],
                        [
                            int(scale.raw.shape[0]),
                            scale.anchor_count,
                            scale.grid_height,
                            scale.grid_width,
                            5 + scale.class_count,
                        ],
                    )
                )
            elif scale.raw_layout == "nchw":
                initializers.extend(
                    [
                        helper.make_tensor(
                            raw_shape_name,
                            TensorProto.INT64,
                            [5],
                            [
                                int(scale.raw.shape[0]),
                                scale.anchor_count,
                                5 + scale.class_count,
                                scale.grid_height,
                                scale.grid_width,
                            ],
                        ),
                    ]
                )
            graph.initializer.extend(initializers)
            if output_count == 9:
                graph.initializer.extend(
                    [
                        helper.make_tensor(starts_obj, TensorProto.INT64, [1], [4]),
                        helper.make_tensor(ends_obj, TensorProto.INT64, [1], [5]),
                    ]
                )
            bbox_slice = f"{slice_prefix}_bbox"
            obj_slice = f"{slice_prefix}_obj"
            cls_slice = f"{slice_prefix}_cls"
            if scale.raw_layout == "na_hw_c":
                new_nodes.append(
                    helper.make_node(
                        "Reshape",
                        [scale.raw.name, raw_shape_name],
                        [raw_input_name],
                        name=f"RKNN9/anchor_reshape_{scale.index}",
                    )
                )
            elif scale.raw_layout == "nchw":
                nchw_name = f"{slice_prefix}_nchw"
                transpose_name = f"{slice_prefix}_transpose"
                new_nodes.extend(
                    [
                        helper.make_node(
                            "Reshape",
                            [scale.raw.name, raw_shape_name],
                            [nchw_name],
                            name=f"RKNN9/anchor_reshape_{scale.index}",
                        ),
                        helper.make_node(
                            "Transpose",
                            [nchw_name],
                            [raw_input_name],
                            name=f"RKNN9/anchor_transpose_{scale.index}",
                            perm=[0, 1, 3, 4, 2],
                        ),
                    ]
                )
            new_nodes.extend(
                [
                    helper.make_node(
                        "Slice",
                        [raw_input_name, starts_bbox, ends_bbox, axes_name, steps_name],
                        [bbox_slice],
                        name=f"RKNN9/bbox_slice_{scale.index}",
                    ),
                    helper.make_node(
                        "Slice",
                        [raw_input_name, starts_cls, ends_cls, axes_name, steps_name],
                        [cls_slice],
                        name=f"RKNN9/class_slice_{scale.index}",
                    ),
                    helper.make_node("Identity", [bbox_slice], [bbox_name], name=f"RKNN9/bbox_{scale.index}"),
                    helper.make_node("Sigmoid", [cls_slice], [scores_name], name=f"RKNN9/scores_{scale.index}"),
                ]
            )
            if output_count == 9:
                new_nodes.extend(
                    [
                        helper.make_node(
                            "Slice",
                            [raw_input_name, starts_obj, ends_obj, axes_name, steps_name],
                            [obj_slice],
                            name=f"RKNN9/objectness_slice_{scale.index}",
                        ),
                        helper.make_node("Sigmoid", [obj_slice], [sum_name], name=f"RKNN9/score_sum_{scale.index}"),
                    ]
                )
            canonical_shape = (
                int(scale.raw.shape[0]),
                scale.anchor_count,
                scale.grid_height,
                scale.grid_width,
                5 + scale.class_count,
            )
            bbox_details = TensorDetails(
                bbox_name,
                scale.raw.element_type,
                canonical_shape[:-1] + (4,),
            )
            scores_details = TensorDetails(
                scores_name,
                scale.raw.element_type,
                canonical_shape[:-1] + (scale.class_count,),
            )
            sum_details = TensorDetails(
                sum_name,
                scale.raw.element_type,
                canonical_shape[:-1] + (1,),
            )
        scale_details = [bbox_details, scores_details]
        if output_count == 9:
            scale_details.append(sum_details)
        for details in scale_details:
            new_outputs.append(_value_info(details.name, details))
            output_details.append(details)

    graph.node.extend(new_nodes)
    graph.ClearField("output")
    graph.output.extend(new_outputs)
    _prune_to_graph_outputs(converted)
    checker.check_model(converted)
    return converted, output_details


def build_nine_output_model(model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[TensorDetails]]:
    """Create a pruned YOLO model with the legacy nine-output contract."""

    return build_output_model(model, output_count=9)


def build_three_output_model(model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[TensorDetails]]:
    """Create the three activated NCHW detection-head outputs for YOLOv5."""

    return build_output_model(model, output_count=3)


def _output_count_label(output_count: int) -> str:
    return {3: "three-output", 6: "six-output", 9: "nine-output"}[output_count]


def export_output_onnx(
    source_path: str | Path,
    destination_path: str | Path,
    output_count: int = 9,
    *,
    legacy_output_names: bool = False,
) -> list[TensorDetails]:
    """Transform and save a supported YOLO ONNX model, returning output contracts."""

    output_count = _validate_output_count(output_count)
    source = Path(source_path)
    destination = Path(destination_path)
    if source.suffix.lower() != ".onnx":
        raise NineOutputError(f"expected an .onnx source model, received {source}")
    if not source.is_file():
        raise NineOutputError(f"ONNX source model does not exist: {source}")
    if destination.suffix.lower() != ".onnx":
        raise NineOutputError(f"{_output_count_label(output_count)} destination must use the .onnx suffix: {destination}")

    transformed, outputs = build_output_model(
        onnx.load_model(source),
        output_count=output_count,
        legacy_output_names=legacy_output_names,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(transformed, destination)
    return outputs


def export_nine_output_onnx(source_path: str | Path, destination_path: str | Path) -> list[TensorDetails]:
    """Transform and save a supported YOLO ONNX model, returning nine output contracts."""

    return export_output_onnx(source_path, destination_path, output_count=9)


def export_three_output_onnx(source_path: str | Path, destination_path: str | Path) -> list[TensorDetails]:
    """Transform and save a YOLOv5 ONNX model, returning three NCHW head contracts."""

    return export_output_onnx(source_path, destination_path, output_count=3)


def validate_output_model(
    model_path: str | Path,
    output_count: int = 9,
    *,
    legacy_output_names: bool = False,
) -> list[TensorDetails]:
    """Validate an exported three-, six-, or nine-output model and return outputs in runtime order."""

    output_count = _validate_output_count(output_count)
    model = onnx.load_model(Path(model_path))
    checker.check_model(model)
    _image_input(model)
    outputs = _value_details(model)
    if output_count == 3:
        expected_names = [f"yolov5_output_{scale}" for scale in range(3)]
        actual_names = [output.name for output in model.graph.output]
        if actual_names != expected_names:
            raise NineOutputError(f"unexpected three-output order: {actual_names}")
        result = []
        for scale, name in enumerate(expected_names):
            details = outputs.get(name)
            if details is None or len(details.shape) != 4:
                raise NineOutputError(f"invalid YOLOv5 output at scale {scale}")
            if not all(isinstance(dimension, int) for dimension in details.shape):
                raise NineOutputError(f"YOLOv5 output shape at scale {scale} must be static")
            if details.shape[1] < 6 or details.shape[2] < 1 or details.shape[3] < 1:
                raise NineOutputError(f"invalid YOLOv5 output shape at scale {scale}: {list(details.shape)}")
            result.append(details)
        return result

    if legacy_output_names:
        if output_count != 9:
            raise NineOutputError("legacy zy output names require the nine-output contract")
        expected_names = list(ZY_LEGACY_OUTPUT_NAMES)
    else:
        expected_names = [
            name
            for scale in range(3)
            for name in (
                (f"bbox_{scale}", f"scores_{scale}", f"score_sum_{scale}")
                if output_count == 9
                else (f"bbox_{scale}", f"scores_{scale}")
            )
        ]
    actual_names = [output.name for output in model.graph.output]
    if actual_names != expected_names:
        raise NineOutputError(f"unexpected {_output_count_label(output_count)} order: {actual_names}")

    result: list[TensorDetails] = []
    for scale in range(3):
        if legacy_output_names:
            bbox_name, scores_name, sum_name = ZY_LEGACY_OUTPUT_NAMES[scale * 3 : scale * 3 + 3]
        else:
            bbox_name, scores_name, sum_name = (
                f"bbox_{scale}",
                f"scores_{scale}",
                f"score_sum_{scale}",
            )
        bbox = outputs.get(bbox_name)
        scores = outputs.get(scores_name)
        score_sum = outputs.get(sum_name) if output_count == 9 else None
        if bbox is None or scores is None or (output_count == 9 and score_sum is None):
            raise NineOutputError(f"missing output metadata at scale {scale}")
        rank = len(bbox.shape)
        if rank not in (4, 5) or len(scores.shape) != rank or (score_sum is not None and len(score_sum.shape) != rank):
            raise NineOutputError(f"invalid channel contract at scale {scale}")
        if rank == 4:
            if bbox.shape[1] not in (4, 64) or (score_sum is not None and score_sum.shape[1] != 1):
                raise NineOutputError(f"invalid channel contract at scale {scale}")
            if bbox.shape[0] != scores.shape[0] or bbox.shape[2:] != scores.shape[2:]:
                raise NineOutputError(f"bbox and score shape mismatch at scale {scale}")
            if score_sum is not None and score_sum.shape != (scores.shape[0], 1, scores.shape[2], scores.shape[3]):
                raise NineOutputError(f"score-sum shape mismatch at scale {scale}")
        else:
            if bbox.shape[-1] != 4 or (score_sum is not None and score_sum.shape[-1] != 1):
                raise NineOutputError(f"invalid anchor-head channel contract at scale {scale}")
            if bbox.shape[:4] != scores.shape[:4] or (score_sum is not None and bbox.shape[:4] != score_sum.shape[:4]):
                raise NineOutputError(f"bbox and score shape mismatch at scale {scale}")
        result.extend((bbox, scores))
        if score_sum is not None:
            result.append(score_sum)
    return result


def validate_nine_output_model(model_path: str | Path) -> list[TensorDetails]:
    """Validate an exported nine-output model and return outputs in runtime order."""

    return validate_output_model(model_path, output_count=9)


def validate_three_output_model(model_path: str | Path) -> list[TensorDetails]:
    """Validate an exported YOLOv5 three-output model."""

    return validate_output_model(model_path, output_count=3)
