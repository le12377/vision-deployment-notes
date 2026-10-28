"""Convert YOLO ONNX detection heads to EP-style ``reg*/cls*`` outputs."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper

try:
    from .onnx9 import (
        HeadScale,
        NineOutputError,
        TensorDetails,
        _default_opset,
        _find_head_scales,
        _image_input,
        _infer_shapes,
        _prune_to_graph_outputs,
        _value_details,
        _value_info,
    )
except ImportError:  # pragma: no cover - supports running scripts from this directory.
    from onnx9 import (  # type: ignore[no-redef]
        HeadScale,
        NineOutputError,
        TensorDetails,
        _default_opset,
        _find_head_scales,
        _image_input,
        _infer_shapes,
        _prune_to_graph_outputs,
        _value_details,
        _value_info,
    )


REG_MAX = 16


def _require_static_4d(details: TensorDetails, name: str) -> tuple[int, int, int, int]:
    if len(details.shape) != 4 or not all(isinstance(dimension, int) for dimension in details.shape):
        raise NineOutputError(f"{name} must have a static four-dimensional NCHW shape, found {list(details.shape)}")
    return tuple(int(dimension) for dimension in details.shape)


def _projection_dtype(element_type: int) -> np.dtype:
    if element_type == TensorProto.FLOAT16:
        return np.dtype(np.float16)
    if element_type == TensorProto.FLOAT:
        return np.dtype(np.float32)
    raise NineOutputError(f"DFL projection requires float bbox tensors, found ONNX element type {element_type}")


def _ensure_names_available(existing_names: set[str], names: set[str]) -> None:
    conflicts = existing_names & names
    if conflicts:
        raise NineOutputError(f"source graph already uses reserved EP reg/cls names: {sorted(conflicts)}")
    existing_names.update(names)


def _direct_regression_nodes(
    scale: HeadScale, reg_name: str, output_index: int
) -> tuple[list[onnx.NodeProto], list[onnx.TensorProto]]:
    batch, channels, height, width = _require_static_4d(scale.bbox, f"bbox scale {scale.index}")
    if channels != 4:
        raise NineOutputError(f"direct regression expects 4 bbox channels, found {channels}")

    shape_name = f"EP6/reg{output_index}_shape"
    return (
        [
            helper.make_node(
                "Reshape",
                [scale.bbox.name, shape_name],
                [reg_name],
                name=f"EP6/reg{output_index}",
            )
        ],
        [helper.make_tensor(shape_name, TensorProto.INT64, [4], [batch, 1, 4, height * width])],
    )


def _dfl_regression_nodes(
    scale: HeadScale, reg_name: str, output_index: int
) -> tuple[list[onnx.NodeProto], list[onnx.TensorProto]]:
    batch, channels, height, width = _require_static_4d(scale.bbox, f"bbox scale {scale.index}")
    if channels != 4 * REG_MAX:
        raise NineOutputError(f"DFL regression expects {4 * REG_MAX} bbox channels, found {channels}")

    reshape_name = f"EP6/dfl_reshape_{output_index}"
    bins_last_name = f"EP6/dfl_bins_last_{output_index}"
    softmax_name = f"EP6/dfl_softmax_{output_index}"
    project_input_name = f"EP6/dfl_project_input_{output_index}"
    shape_name = f"EP6/dfl_shape_{output_index}"
    weight_name = f"EP6/dfl_project_{output_index}.weight"

    projection = np.arange(REG_MAX, dtype=_projection_dtype(scale.bbox.element_type)).reshape(1, REG_MAX, 1, 1)
    return (
        [
            helper.make_node(
                "Reshape",
                [scale.bbox.name, shape_name],
                [reshape_name],
                name=f"EP6/dfl_reshape_{output_index}",
            ),
            helper.make_node(
                "Transpose",
                [reshape_name],
                [bins_last_name],
                name=f"EP6/dfl_bins_last_{output_index}",
                perm=[0, 1, 3, 2],
            ),
            helper.make_node(
                "Softmax",
                [bins_last_name],
                [softmax_name],
                name=f"EP6/dfl_softmax_{output_index}",
                axis=3,
            ),
            helper.make_node(
                "Transpose",
                [softmax_name],
                [project_input_name],
                name=f"EP6/dfl_project_input_{output_index}",
                perm=[0, 3, 1, 2],
            ),
            helper.make_node(
                "Conv",
                [project_input_name, weight_name],
                [reg_name],
                name=f"EP6/dfl_project_{output_index}",
            ),
        ],
        [
            helper.make_tensor(shape_name, TensorProto.INT64, [4], [batch, 4, REG_MAX, height * width]),
            numpy_helper.from_array(projection, weight_name),
        ],
    )


def build_regcls_model(model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[TensorDetails]]:
    """Create an EP-style six-output ONNX model.

    Output order is ``reg1, cls1, reg2, cls2, reg3, cls3``.
    ``reg*`` is shaped ``[N, 1, 4, H*W]``. ``cls*`` keeps the raw detection-head
    logits shaped ``[N, nc, H, W]`` without sigmoid.
    """

    converted = _infer_shapes(copy.deepcopy(model))
    _image_input(converted)
    scales = _find_head_scales(converted)
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
    new_initializers: list[onnx.TensorProto] = []
    new_outputs: list[onnx.ValueInfoProto] = []
    output_details: list[TensorDetails] = []

    for runtime_index, scale in enumerate(scales, start=1):
        reg_name = f"reg{runtime_index}"
        cls_name = f"cls{runtime_index}"
        _ensure_names_available(
            existing_names,
            {
                reg_name,
                cls_name,
                f"EP6/reg{runtime_index}_shape",
                f"EP6/dfl_reshape_{runtime_index}",
                f"EP6/dfl_bins_last_{runtime_index}",
                f"EP6/dfl_softmax_{runtime_index}",
                f"EP6/dfl_project_input_{runtime_index}",
                f"EP6/dfl_shape_{runtime_index}",
                f"EP6/dfl_project_{runtime_index}.weight",
            },
        )

        batch, bbox_channels, height, width = _require_static_4d(scale.bbox, f"bbox scale {scale.index}")
        if bbox_channels == 4:
            reg_nodes, reg_initializers = _direct_regression_nodes(scale, reg_name, runtime_index)
        elif bbox_channels == 4 * REG_MAX:
            reg_nodes, reg_initializers = _dfl_regression_nodes(scale, reg_name, runtime_index)
        else:
            raise NineOutputError(
                f"scale {scale.index} bbox output must have 4 or {4 * REG_MAX} channels, found {bbox_channels}"
            )

        cls_batch, cls_channels, cls_height, cls_width = _require_static_4d(scale.classes, f"class scale {scale.index}")
        if (cls_batch, cls_height, cls_width) != (batch, height, width):
            raise NineOutputError(f"scale {scale.index} bbox and class outputs have incompatible shapes")

        new_nodes.extend(reg_nodes)
        new_nodes.append(helper.make_node("Identity", [scale.classes.name], [cls_name], name=f"EP6/cls{runtime_index}"))
        new_initializers.extend(reg_initializers)

        reg_details = TensorDetails(reg_name, scale.bbox.element_type, (batch, 1, 4, height * width))
        cls_details = TensorDetails(cls_name, scale.classes.element_type, (cls_batch, cls_channels, cls_height, cls_width))
        for details in (reg_details, cls_details):
            new_outputs.append(_value_info(details.name, details))
            output_details.append(details)

    graph.initializer.extend(new_initializers)
    graph.node.extend(new_nodes)
    graph.ClearField("output")
    graph.output.extend(new_outputs)
    _prune_to_graph_outputs(converted)
    checker.check_model(converted)
    return converted, output_details


def export_regcls_onnx(source_path: str | Path, destination_path: str | Path) -> list[TensorDetails]:
    """Transform and save an EP-style ``reg*/cls*`` ONNX model."""

    source = Path(source_path)
    destination = Path(destination_path)
    if source.suffix.lower() != ".onnx":
        raise NineOutputError(f"expected an .onnx source model, received {source}")
    if not source.is_file():
        raise NineOutputError(f"ONNX source model does not exist: {source}")
    if destination.suffix.lower() != ".onnx":
        raise NineOutputError(f"reg/cls destination must use the .onnx suffix: {destination}")

    transformed, outputs = build_regcls_model(onnx.load_model(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(transformed, destination)
    return outputs


def validate_regcls_model(model_path: str | Path) -> list[TensorDetails]:
    """Validate an EP-style ``reg*/cls*`` model and return outputs in runtime order."""

    model = onnx.load_model(Path(model_path))
    checker.check_model(model)
    _image_input(model)
    outputs = _value_details(model)
    expected_names = [name for scale in range(1, 4) for name in (f"reg{scale}", f"cls{scale}")]
    actual_names = [output.name for output in model.graph.output]
    if actual_names != expected_names:
        raise NineOutputError(f"unexpected reg/cls output order: {actual_names}")

    result: list[TensorDetails] = []
    for scale in range(1, 4):
        reg = outputs.get(f"reg{scale}")
        cls = outputs.get(f"cls{scale}")
        if reg is None or cls is None:
            raise NineOutputError(f"missing reg/cls output metadata at scale {scale}")
        if len(reg.shape) != 4 or len(cls.shape) != 4:
            raise NineOutputError(f"invalid reg/cls rank at scale {scale}")
        if reg.shape[:3] != (cls.shape[0], 1, 4):
            raise NineOutputError(f"invalid reg shape at scale {scale}: {list(reg.shape)}")
        if not isinstance(cls.shape[2], int) or not isinstance(cls.shape[3], int):
            raise NineOutputError(f"class output shape at scale {scale} must be static")
        if reg.shape[3] != cls.shape[2] * cls.shape[3]:
            raise NineOutputError(f"reg grid count does not match class map at scale {scale}")
        result.extend((reg, cls))
    return result
