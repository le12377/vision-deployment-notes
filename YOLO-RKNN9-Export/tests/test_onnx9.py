"""Regression tests for the YOLOv26 RKNN nine-output ONNX transform."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, checker, helper, numpy_helper

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yolo26_rknn9.onnx9 import (
    NineOutputError,
    build_nine_output_model,
    build_output_model,
    export_nine_output_onnx,
    export_output_onnx,
    validate_nine_output_model,
    validate_output_model,
)
from yolo26_rknn9.regcls import build_regcls_model, export_regcls_onnx, validate_regcls_model
from yolo26_rknn9.export import parse_args, validate_quantized_dtype


def make_yolo_like_model(
    scale_count: int = 3,
    bbox_channels: int = 4,
    one2one: bool = True,
    legacy_names: bool = False,
    opset: int = 12,
) -> onnx.ModelProto:
    """Create a supported three-scale YOLOv26 or YOLO11 detection head."""

    image = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 16, 16])
    output = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, bbox_channels, 16, 16])
    nodes = []
    initializers = []
    bbox_outputs = []
    for scale in range(scale_count):
        bbox_weight = (
            f"model.22.cv2.{scale}.2.weight"
            if legacy_names
            else f"bbox_weight_{scale}"
        )
        bbox_bias = f"bbox_bias_{scale}"
        cls_weight = (
            f"model.22.cv3.{scale}.2.weight"
            if legacy_names
            else f"cls_weight_{scale}"
        )
        cls_bias = f"cls_bias_{scale}"
        bbox_output = f"bbox_raw_{scale}"
        cls_output = f"cls_raw_{scale}"
        bbox_outputs.append(bbox_output)
        initializers.extend(
            [
                numpy_helper.from_array(np.ones((bbox_channels, 3, 1, 1), dtype=np.float32), bbox_weight),
                numpy_helper.from_array(np.zeros(bbox_channels, dtype=np.float32), bbox_bias),
                numpy_helper.from_array(np.ones((6, 3, 1, 1), dtype=np.float32), cls_weight),
                numpy_helper.from_array(np.zeros(6, dtype=np.float32), cls_bias),
            ]
        )
        prefix = "one2one_" if one2one else ""
        node_names = (
            (f"Conv_{scale * 2}", f"Conv_{scale * 2 + 1}")
            if legacy_names
            else (
                f"/model.23/{prefix}cv2.{scale}/{prefix}cv2.{scale}.2/Conv",
                f"/model.23/{prefix}cv3.{scale}/{prefix}cv3.{scale}.2/Conv",
            )
        )
        nodes.extend(
            [
                helper.make_node(
                    "Conv",
                    ["images", bbox_weight, bbox_bias],
                    [bbox_output],
                    name=node_names[0],
                ),
                helper.make_node(
                    "Conv",
                    ["images", cls_weight, cls_bias],
                    [cls_output],
                    name=node_names[1],
                ),
            ]
        )
    nodes.append(helper.make_node("Identity", [bbox_outputs[0]], ["output0"], name="terminal_postprocess"))
    graph = helper.make_graph(nodes, "v26_like", [image], [output], initializer=initializers)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", opset)])


def make_anchor_yolo_like_model(
    scale_count: int = 3,
    anchor_count: int = 3,
    class_count: int = 2,
    flattened: bool = False,
    nchw: bool = False,
) -> onnx.ModelProto:
    """Create a supported three-scale YOLOv5-style anchor head."""

    image = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 320, 320])
    sizes = (40, 20, 10)[:scale_count]
    output_size = sum(anchor_count * size * size for size in sizes)
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, output_size, 5 + class_count])
    nodes = []
    initializers = []
    flattened_outputs = []

    for scale, size in enumerate(sizes):
        pooled = f"pool_{scale}"
        conv_weight = f"detect_weight_{scale}"
        conv_bias = f"detect_bias_{scale}"
        conv_output = f"detect_conv_output_{scale}"
        reshape_output = f"detect_reshape_output_{scale}"
        transpose_output = f"detect_transpose_output_{scale}"
        sigmoid_output = f"detect_sigmoid_output_{scale}"
        flat_output = f"detect_flat_output_{scale}"
        reshape_shape = f"detect_reshape_shape_{scale}"
        flat_shape = f"detect_flat_shape_{scale}"

        channels = anchor_count * (5 + class_count)
        step = 320 // size
        initializers.extend(
            [
                numpy_helper.from_array(np.ones((channels, 3, 1, 1), dtype=np.float32), conv_weight),
                numpy_helper.from_array(np.zeros(channels, dtype=np.float32), conv_bias),
                numpy_helper.from_array(
                    np.array(
                        [1, anchor_count, 5 + class_count, -1]
                        if flattened
                        else [1, anchor_count, 5 + class_count, size, size],
                        dtype=np.int64,
                    ),
                    reshape_shape,
                ),
                numpy_helper.from_array(np.array([1, -1, 5 + class_count], dtype=np.int64), flat_shape),
            ]
        )
        nodes.extend(
            [
                helper.make_node(
                    "AveragePool",
                    ["images"],
                    [pooled],
                    name=f"pool_{scale}",
                    kernel_shape=[step, step],
                    strides=[step, step],
                ),
                helper.make_node(
                    "Conv",
                    [pooled, conv_weight, conv_bias],
                    [conv_output],
                    name=f"detect_conv_{scale}",
                ),
            ]
        )
        if nchw:
            nodes.append(
                helper.make_node(
                    "Sigmoid",
                    [conv_output],
                    [sigmoid_output],
                    name=f"detect_sigmoid_{scale}",
                )
            )
        else:
            nodes.extend(
                [
                    helper.make_node(
                        "Reshape",
                        [conv_output, reshape_shape],
                        [reshape_output],
                        name=f"detect_reshape_{scale}",
                    ),
                    helper.make_node(
                        "Transpose",
                        [reshape_output],
                        [transpose_output],
                        name=f"detect_transpose_{scale}",
                        perm=[0, 1, 3, 2] if flattened else [0, 1, 3, 4, 2],
                    ),
                    helper.make_node(
                        "Sigmoid",
                        [transpose_output],
                        [sigmoid_output],
                        name=f"detect_sigmoid_{scale}",
                    ),
                    helper.make_node(
                        "Reshape",
                        [sigmoid_output, flat_shape],
                        [flat_output],
                        name=f"detect_flatten_{scale}",
                    ),
                ]
            )
        flattened_outputs.append(flat_output)

    if nchw:
        graph_outputs = [
            helper.make_tensor_value_info(
                f"detect_sigmoid_{scale}",
                TensorProto.FLOAT,
                [1, anchor_count * (5 + class_count), size, size],
            )
            for scale, size in enumerate(sizes)
        ]
    else:
        nodes.append(helper.make_node("Concat", flattened_outputs, ["output"], name="final_concat", axis=1))
        graph_outputs = [output]
    graph = helper.make_graph(nodes, "anchor_like", [image], graph_outputs, initializer=initializers)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 12)])


def make_yolox_like_model(class_count: int = 2) -> onnx.ModelProto:
    """Create a supported three-scale YOLOX-style decoupled head."""

    image = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])
    output_size = 16 * 16 + 8 * 8 + 4 * 4
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, output_size, 5 + class_count])
    nodes = []
    initializers = []
    flattened_outputs = []

    reshape_shape = "yolox_reshape_shape"
    initializers.append(numpy_helper.from_array(np.array([1, 5 + class_count, -1], dtype=np.int64), reshape_shape))

    for scale, size in enumerate((16, 8, 4)):
        pooled = f"yolox_pool_{scale}"
        step = 32 // size
        nodes.append(
            helper.make_node(
                "AveragePool",
                ["images"],
                [pooled],
                name=f"yolox_pool_{scale}",
                kernel_shape=[step, step],
                strides=[step, step],
            )
        )

        branch_outputs = []
        for branch, channels in (("reg_preds", 4), ("obj_preds", 1), ("cls_preds", class_count)):
            weight = f"head.{branch}.{scale}.weight"
            bias = f"head.{branch}.{scale}.bias"
            conv_output = f"/{branch}.{scale}/Conv_output_0"
            initializers.extend(
                [
                    numpy_helper.from_array(np.ones((channels, 3, 1, 1), dtype=np.float32), weight),
                    numpy_helper.from_array(np.zeros(channels, dtype=np.float32), bias),
                ]
            )
            nodes.append(
                helper.make_node(
                    "Conv",
                    [pooled, weight, bias],
                    [conv_output],
                    name=f"/head/{branch}.{scale}/Conv",
                )
            )
            if branch == "reg_preds":
                branch_outputs.append(conv_output)
            else:
                sigmoid_output = f"/{branch}.{scale}/Sigmoid_output_0"
                nodes.append(
                    helper.make_node(
                        "Sigmoid",
                        [conv_output],
                        [sigmoid_output],
                        name=f"/head/{branch}.{scale}/Sigmoid",
                    )
                )
                branch_outputs.append(sigmoid_output)

        concat_output = f"/head/Concat_{scale}_output_0"
        reshape_output = f"/head/Reshape_{scale}_output_0"
        nodes.extend(
            [
                helper.make_node("Concat", branch_outputs, [concat_output], name=f"/head/Concat_{scale}", axis=1),
                helper.make_node("Reshape", [concat_output, reshape_shape], [reshape_output], name=f"/head/Reshape_{scale}"),
            ]
        )
        flattened_outputs.append(reshape_output)

    concat_all = "/head/Concat_all_output_0"
    nodes.extend(
        [
            helper.make_node("Concat", flattened_outputs, [concat_all], name="/head/Concat_all", axis=2),
            helper.make_node("Transpose", [concat_all], ["output"], name="/head/Transpose", perm=[0, 2, 1]),
        ]
    )
    graph = helper.make_graph(nodes, "yolox_like", [image], [output], initializer=initializers)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 15)])


def test_build_nine_output_model_prunes_terminal_postprocess() -> None:
    transformed, outputs = build_nine_output_model(make_yolo_like_model())

    checker.check_model(transformed)
    assert [output.name for output in transformed.graph.output] == [
        "bbox_0",
        "scores_0",
        "score_sum_0",
        "bbox_1",
        "scores_1",
        "score_sum_1",
        "bbox_2",
        "scores_2",
        "score_sum_2",
    ]
    assert [details.shape for details in outputs] == [
        (1, 4, 16, 16),
        (1, 6, 16, 16),
        (1, 1, 16, 16),
    ] * 3
    assert "terminal_postprocess" not in {node.name for node in transformed.graph.node}


def test_exported_model_passes_nine_output_validation(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    destination = tmp_path / "source_rknn9.onnx"
    onnx.save_model(make_yolo_like_model(), source)

    export_nine_output_onnx(source, destination)
    validated = validate_nine_output_model(destination)

    assert destination.is_file()
    assert len(validated) == 9
    assert validated[0].shape[1] == 4
    assert validated[1].shape[1] == 6
    assert validated[2].shape[1] == 1


def test_exported_model_passes_six_output_validation(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    destination = tmp_path / "source_rknn6.onnx"
    onnx.save_model(make_yolo_like_model(), source)

    export_output_onnx(source, destination, output_count=6)
    validated = validate_output_model(destination, output_count=6)

    assert destination.is_file()
    assert [output.name for output in onnx.load_model(destination).graph.output] == [
        "bbox_0",
        "scores_0",
        "bbox_1",
        "scores_1",
        "bbox_2",
        "scores_2",
    ]
    assert len(validated) == 6
    assert validated[0].shape[1] == 4
    assert validated[1].shape[1] == 6


def test_transform_requires_exactly_three_detection_scales() -> None:
    with pytest.raises(NineOutputError, match="exactly three"):
        build_nine_output_model(make_yolo_like_model(scale_count=2))


def test_yolo11_head_uses_64_channel_bbox_outputs() -> None:
    transformed, outputs = build_nine_output_model(make_yolo_like_model(bbox_channels=64, one2one=False))

    checker.check_model(transformed)
    assert len(outputs) == 9
    assert [details.shape[1] for details in outputs[::3]] == [64, 64, 64]


def test_legacy_yolov8_head_names_are_detected_from_weight_inputs() -> None:
    transformed, outputs = build_nine_output_model(
        make_yolo_like_model(bbox_channels=64, one2one=False, legacy_names=True)
    )

    checker.check_model(transformed)
    assert len(outputs) == 9
    assert [details.shape[1] for details in outputs[::3]] == [64, 64, 64]


def test_modern_opset_uses_reduce_sum_axes_input() -> None:
    transformed, outputs = build_nine_output_model(make_yolo_like_model(bbox_channels=64, one2one=False, opset=19))

    checker.check_model(transformed)
    assert len(outputs) == 9
    reduce_sum_nodes = [node for node in transformed.graph.node if node.op_type == "ReduceSum"]
    assert reduce_sum_nodes
    assert all(len(node.input) == 2 for node in reduce_sum_nodes)
    assert "rknn9_score_sum_axes" in {initializer.name for initializer in transformed.graph.initializer}


def test_zy_compatibility_uses_legacy_tf_output_names() -> None:
    transformed, outputs = build_output_model(
        make_yolo_like_model(bbox_channels=64),
        output_count=9,
        legacy_output_names=True,
    )

    checker.check_model(transformed)
    assert [output.name for output in transformed.graph.output] == [
        "318",
        "onnx::ReduceSum_326",
        "331",
        "338",
        "onnx::ReduceSum_346",
        "350",
        "357",
        "onnx::ReduceSum_365",
        "369",
    ]
    assert [details.shape for details in outputs[::3]] == [(1, 64, 16, 16)] * 3


def test_yolov5_anchor_head_is_detected_and_split() -> None:
    transformed, outputs = build_nine_output_model(make_anchor_yolo_like_model())

    checker.check_model(transformed)
    assert [output.name for output in transformed.graph.output] == [
        "bbox_0",
        "scores_0",
        "score_sum_0",
        "bbox_1",
        "scores_1",
        "score_sum_1",
        "bbox_2",
        "scores_2",
        "score_sum_2",
    ]
    assert [details.shape for details in outputs] == [
        (1, 3, 40, 40, 4),
        (1, 3, 40, 40, 2),
        (1, 3, 40, 40, 1),
        (1, 3, 20, 20, 4),
        (1, 3, 20, 20, 2),
        (1, 3, 20, 20, 1),
        (1, 3, 10, 10, 4),
        (1, 3, 10, 10, 2),
        (1, 3, 10, 10, 1),
    ]


def test_yolov5_anchor_head_supports_six_outputs() -> None:
    transformed, outputs = build_output_model(make_anchor_yolo_like_model(), output_count=6)

    checker.check_model(transformed)
    assert [output.name for output in transformed.graph.output] == [
        "bbox_0",
        "scores_0",
        "bbox_1",
        "scores_1",
        "bbox_2",
        "scores_2",
    ]
    assert [details.shape for details in outputs] == [
        (1, 3, 40, 40, 4),
        (1, 3, 40, 40, 2),
        (1, 3, 20, 20, 4),
        (1, 3, 20, 20, 2),
        (1, 3, 10, 10, 4),
        (1, 3, 10, 10, 2),
    ]


def test_yolov5_flattened_anchor_head_is_reshaped_and_split() -> None:
    transformed, outputs = build_nine_output_model(make_anchor_yolo_like_model(flattened=True))

    checker.check_model(transformed)
    assert [details.shape for details in outputs] == [
        (1, 3, 40, 40, 4),
        (1, 3, 40, 40, 2),
        (1, 3, 40, 40, 1),
        (1, 3, 20, 20, 4),
        (1, 3, 20, 20, 2),
        (1, 3, 20, 20, 1),
        (1, 3, 10, 10, 4),
        (1, 3, 10, 10, 2),
        (1, 3, 10, 10, 1),
    ]
    assert {node.name for node in transformed.graph.node} >= {
        "RKNN9/anchor_reshape_0",
        "RKNN9/anchor_reshape_1",
        "RKNN9/anchor_reshape_2",
    }


def test_yolov5_three_output_contract_uses_nchw_heads() -> None:
    transformed, outputs = build_output_model(
        make_anchor_yolo_like_model(nchw=True),
        output_count=3,
    )

    checker.check_model(transformed)
    assert [output.name for output in transformed.graph.output] == [
        "yolov5_output_0",
        "yolov5_output_1",
        "yolov5_output_2",
    ]
    assert [details.shape for details in outputs] == [
        (1, 21, 40, 40),
        (1, 21, 20, 20),
        (1, 21, 10, 10),
    ]


def test_yolov5_nchw_heads_can_still_build_six_outputs() -> None:
    transformed, outputs = build_output_model(
        make_anchor_yolo_like_model(nchw=True),
        output_count=6,
    )

    checker.check_model(transformed)
    assert [details.shape for details in outputs] == [
        (1, 3, 40, 40, 4),
        (1, 3, 40, 40, 2),
        (1, 3, 20, 20, 4),
        (1, 3, 20, 20, 2),
        (1, 3, 10, 10, 4),
        (1, 3, 10, 10, 2),
    ]


def test_three_output_contract_rejects_non_yolov5_heads() -> None:
    with pytest.raises(NineOutputError, match="only supported for YOLOv5"):
        build_output_model(make_yolo_like_model(), output_count=3)


def test_yolox_head_is_detected_and_split() -> None:
    transformed, outputs = build_output_model(make_yolox_like_model(), output_count=6)

    checker.check_model(transformed)
    assert [output.name for output in transformed.graph.output] == [
        "bbox_0",
        "scores_0",
        "bbox_1",
        "scores_1",
        "bbox_2",
        "scores_2",
    ]
    assert [details.shape for details in outputs] == [
        (1, 4, 16, 16),
        (1, 2, 16, 16),
        (1, 4, 8, 8),
        (1, 2, 8, 8),
        (1, 4, 4, 4),
        (1, 2, 4, 4),
    ]
    assert "/head/Transpose" not in {node.name for node in transformed.graph.node}


def test_yolox_head_exports_objectness_as_score_sum() -> None:
    transformed, outputs = build_output_model(make_yolox_like_model(), output_count=9)

    checker.check_model(transformed)
    assert [output.name for output in transformed.graph.output] == [
        "bbox_0",
        "scores_0",
        "score_sum_0",
        "bbox_1",
        "scores_1",
        "score_sum_1",
        "bbox_2",
        "scores_2",
        "score_sum_2",
    ]
    assert [details.shape for details in outputs] == [
        (1, 4, 16, 16),
        (1, 2, 16, 16),
        (1, 1, 16, 16),
        (1, 4, 8, 8),
        (1, 2, 8, 8),
        (1, 1, 8, 8),
        (1, 4, 4, 4),
        (1, 2, 4, 4),
        (1, 1, 4, 4),
    ]


def test_regcls_export_projects_dfl_outputs_to_ep_contract(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    destination = tmp_path / "source_regcls.onnx"
    onnx.save_model(make_yolo_like_model(bbox_channels=64, one2one=False), source)

    export_regcls_onnx(source, destination)
    validated = validate_regcls_model(destination)

    assert [output.name for output in onnx.load_model(destination).graph.output] == [
        "reg1",
        "cls1",
        "reg2",
        "cls2",
        "reg3",
        "cls3",
    ]
    assert [details.shape for details in validated] == [
        (1, 1, 4, 256),
        (1, 6, 16, 16),
        (1, 1, 4, 256),
        (1, 6, 16, 16),
        (1, 1, 4, 256),
        (1, 6, 16, 16),
    ]
    node_names = {node.name for node in onnx.load_model(destination).graph.node}
    assert "EP6/dfl_project_1" in node_names
    assert "terminal_postprocess" not in node_names


def test_regcls_dfl_projection_matches_numpy_for_opset12(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    destination = tmp_path / "source_regcls.onnx"
    onnx.save_model(make_yolo_like_model(bbox_channels=64, one2one=False, opset=12), source)
    export_regcls_onnx(source, destination)

    import onnxruntime as ort

    raw_model = onnx.load_model(source)
    regcls_model = onnx.load_model(destination)
    raw_model.graph.output.append(helper.make_tensor_value_info("bbox_raw_0", TensorProto.FLOAT, [1, 64, 16, 16]))
    raw_model.ir_version = min(raw_model.ir_version, 9)
    regcls_model.ir_version = min(regcls_model.ir_version, 9)
    input_tensor = np.random.default_rng(0).normal(size=(1, 3, 16, 16)).astype(np.float32)

    raw_session = ort.InferenceSession(raw_model.SerializeToString(), providers=["CPUExecutionProvider"])
    regcls_session = ort.InferenceSession(regcls_model.SerializeToString(), providers=["CPUExecutionProvider"])
    raw_bbox = raw_session.run(["bbox_raw_0"], {"images": input_tensor})[0]
    actual_reg = regcls_session.run(["reg1"], {"images": input_tensor})[0]

    dfl = raw_bbox.reshape(1, 4, 16, -1)
    dfl = np.exp(dfl - np.max(dfl, axis=2, keepdims=True))
    dfl = dfl / np.sum(dfl, axis=2, keepdims=True)
    expected_reg = np.sum(dfl * np.arange(16, dtype=np.float32).reshape(1, 1, 16, 1), axis=2, keepdims=True)
    expected_reg = np.transpose(expected_reg, (0, 2, 1, 3))

    np.testing.assert_allclose(actual_reg, expected_reg, rtol=1e-5, atol=1e-5)


def test_regcls_export_reshapes_direct_bbox_outputs() -> None:
    transformed, outputs = build_regcls_model(make_yolo_like_model(bbox_channels=4, one2one=True))

    checker.check_model(transformed)
    assert [output.name for output in transformed.graph.output] == [
        "reg1",
        "cls1",
        "reg2",
        "cls2",
        "reg3",
        "cls3",
    ]
    assert [details.shape for details in outputs] == [
        (1, 1, 4, 256),
        (1, 6, 16, 16),
        (1, 1, 4, 256),
        (1, 6, 16, 16),
        (1, 1, 4, 256),
        (1, 6, 16, 16),
    ]


def test_quantization_cli_accepts_rknn_configuration() -> None:
    args = parse_args(
        [
            "--weights",
            "model.onnx",
            "--quantize",
            "--quantized-dtype",
            "w16a16i",
            "--quantized-algorithm",
            "mmse",
            "--quantized-method",
            "group32",
        ]
    )

    assert args.quantize is True
    assert args.quantized_dtype == "w16a16i"
    assert args.quantized_algorithm == "mmse"
    assert args.quantized_method == "group32"


def test_cli_defaults_to_nine_outputs() -> None:
    args = parse_args(["model.onnx"])

    assert args.weights == Path("model.onnx")
    assert args.output_count == 9


def test_cli_accepts_positional_six_output_contract() -> None:
    args = parse_args(["model.onnx", "6"])

    assert args.weights == Path("model.onnx")
    assert args.output_count == 6


def test_cli_accepts_positional_three_output_contract() -> None:
    args = parse_args(["model.onnx", "3"])

    assert args.weights == Path("model.onnx")
    assert args.output_count == 3


def test_cli_rejects_conflicting_output_contracts() -> None:
    with pytest.raises(SystemExit):
        parse_args(["model.onnx", "6", "--output-count", "9"])


def test_rk3588_rejects_unsupported_quantized_dtype() -> None:
    with pytest.raises(ValueError, match="not supported by RK3588"):
        validate_quantized_dtype("rk3588", "w8a16")
