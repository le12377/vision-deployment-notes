"""Export decoded YOLO26 XYWH/class scores for GB boxes; optionally compile RKNN."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import onnx
from onnx import checker, helper, numpy_helper

if __package__ in (None, ""):
    from onnx9 import NineOutputError, _image_input, _infer_shapes, _prune_to_graph_outputs, _value_details
else:
    from .onnx9 import NineOutputError, _image_input, _infer_shapes, _prune_to_graph_outputs, _value_details


def build_decoded_model(
    model: onnx.ModelProto, *, layout: str = "bcn", diagnostic_zero_scores: bool = False,
    diagnostic_fixed_box: bool = False, box_format: str = "xywh",
) -> onnx.ModelProto:
    """Keep the existing decode, defaulting to center-based XYWH in BCN layout.

    This targets end-to-end YOLO26 graphs with a decoded box/Sigmoid Concat
    immediately before the terminal Transpose. Source boxes are assumed to be
    xyxy; use box_format="source" to preserve them. It does not reconstruct scores
    from class IDs or reinterpret raw per-scale regression as decoded boxes.
    """
    if layout not in ("bcn", "bnc"):
        raise NineOutputError("decoded output layout must be bcn or bnc")
    if box_format not in ("source", "xywh"):
        raise NineOutputError("box format must be source or xywh")
    if diagnostic_zero_scores and diagnostic_fixed_box:
        raise NineOutputError("select only one diagnostic mode")
    converted = _infer_shapes(copy.deepcopy(model))
    image = _image_input(converted)
    metadata = {item.key: item.value for item in converted.metadata_props}
    if metadata.get("end2end", "").lower() != "true":
        raise NineOutputError("expected an end2end=True YOLO26 source ONNX")
    if len(converted.graph.output) != 1:
        raise NineOutputError("expected a single end-to-end source output")
    details = _value_details(converted)
    producers = {name: node for node in converted.graph.node for name in node.output}
    candidates = []
    for node in converted.graph.node:
        if node.op_type != "Transpose" or not node.input:
            continue
        concat = producers.get(node.input[0])
        if concat is None or concat.op_type != "Concat" or len(concat.input) != 2:
            continue
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in concat.attribute}
        if attrs.get("axis") != 1:
            continue
        boxes = details.get(concat.input[0])
        scores = details.get(concat.input[1])
        score_node = producers.get(concat.input[1])
        if boxes is None or scores is None or score_node is None:
            continue
        if len(boxes.shape) != 3 or len(scores.shape) != 3:
            continue
        if (
            boxes.shape[1] == 4
            and boxes.shape[0::2] == scores.shape[0::2]
            and isinstance(scores.shape[1], int)
            and scores.shape[1] > 0
            and score_node.op_type == "Sigmoid"
        ):
            candidates.append(details[concat.output[0]])
    if len(candidates) != 1:
        raise NineOutputError(f"expected one decoded boxes/scores tensor, found {len(candidates)}")

    decoded = candidates[0]
    converted.graph.ClearField("output")
    converted.graph.output.append(
        helper.make_tensor_value_info(decoded.name, decoded.element_type, list(decoded.shape))
    )
    _prune_to_graph_outputs(converted)
    output_input = decoded.name
    if diagnostic_zero_scores or diagnostic_fixed_box:
        # Suppress live class scores while preserving the output shape/type.
        # Fixed-box mode then injects exactly one known candidate.
        mask_name = "DiagnosticZeroScoresMask"
        masked_name = "DiagnosticZeroScoresOutput"
        bias_name = "DiagnosticFixedBoxBias"
        fixed_name = "DiagnosticFixedBoxOutput"
        existing_names = {
            name for node in converted.graph.node for name in (*node.input, *node.output)
        }
        if {mask_name, masked_name, bias_name, fixed_name} & existing_names:
            raise NineOutputError("reserved diagnostic tensor names already exist")
        channels = int(decoded.shape[1])
        if diagnostic_fixed_box:
            if not all(isinstance(dim, int) and dim > 0 for dim in (*image.shape[2:], decoded.shape[2])):
                raise NineOutputError("fixed-box diagnostic requires static positive image and point sizes")
            if decoded.element_type != onnx.TensorProto.FLOAT:
                raise NineOutputError("fixed-box diagnostic requires float32 source tensors")
            points = int(decoded.shape[2])
            index = points // 2
            height, width = image.shape[2:]
            mask = np.zeros((1, channels, points), dtype=np.float32)
            mask[:, :4] = 1
            mask[:, :4, index] = 0
            bias = np.zeros_like(mask)
            bias[0, :4, index] = [width * 0.25, height * 0.2, width * 0.75, height * 0.8]
            bias[0, 4, index] = 0.875
            converted.graph.initializer.extend(
                [numpy_helper.from_array(mask, mask_name), numpy_helper.from_array(bias, bias_name)]
            )
        else:
            converted.graph.initializer.append(helper.make_tensor(
                mask_name, decoded.element_type, [1, channels, 1],
                [1.0] * 4 + [0.0] * (channels - 4),
            ))
        converted.graph.node.append(
            helper.make_node("Mul", [decoded.name, mask_name], [masked_name], name="DiagnosticZeroScores")
        )
        output_input = masked_name
        metadata["diagnostic"] = "ZERO CLASS SCORES; expected no detections; not a production detector"
        if diagnostic_fixed_box:
            converted.graph.node.append(
                helper.make_node("Add", [masked_name, bias_name], [fixed_name], name="DiagnosticFixedBox")
            )
            output_input = fixed_name
            metadata["diagnostic"] = (
                "FIXED BOX ONLY; class 0 probability 0.875; xyxy at 25%,20%,75%,80% of input; "
                "not a production detector"
            )
    if box_format == "xywh":
        # The 6AY source decodes two corner points. GB's fixed-box result is
        # consistent with a consumer expecting center-x, center-y, width, height.
        prefix = "DecodedXYWH"
        existing_names = {
            name for node in converted.graph.node for name in (*node.input, *node.output)
        } | {item.name for item in converted.graph.initializer}
        if any(name.startswith(prefix) for name in existing_names):
            raise NineOutputError("reserved XYWH tensor names already exist")
        for suffix, indices in (
            ("LowerIndices", [0, 1]), ("UpperIndices", [2, 3]),
            ("ScoreIndices", list(range(4, int(decoded.shape[1])))),
        ):
            converted.graph.initializer.append(
                helper.make_tensor(prefix + suffix, onnx.TensorProto.INT64, [len(indices)], indices)
            )
        converted.graph.initializer.append(
            helper.make_tensor(prefix + "Half", decoded.element_type, [1], [0.5])
        )
        for suffix in ("Lower", "Upper", "Score"):
            converted.graph.node.append(helper.make_node(
                "Gather", [output_input, prefix + suffix + "Indices"], [prefix + suffix],
                name=prefix + suffix, axis=1,
            ))
        converted.graph.node.extend([
            helper.make_node("Add", [prefix + "Lower", prefix + "Upper"], [prefix + "Sum"], name=prefix + "Sum"),
            helper.make_node("Mul", [prefix + "Sum", prefix + "Half"], [prefix + "Center"], name=prefix + "Center"),
            helper.make_node("Sub", [prefix + "Upper", prefix + "Lower"], [prefix + "Size"], name=prefix + "Size"),
            helper.make_node(
                "Concat", [prefix + "Center", prefix + "Size", prefix + "Score"],
                [prefix + "Output"], name=prefix + "Output", axis=1,
            ),
        ])
        output_input = prefix + "Output"
        metadata["box_format"] = "cx, cy, width, height in input pixels; converted from source xyxy"
    output_shape = list(decoded.shape)
    if layout == "bnc":
        output_shape = [decoded.shape[0], decoded.shape[2], decoded.shape[1]]
        output_node = helper.make_node(
            "Transpose", [output_input], ["output0"], name="DecodedOutput", perm=[0, 2, 1]
        )
    else:
        output_node = helper.make_node("Identity", [output_input], ["output0"], name="DecodedOutput")
    converted.graph.node.append(output_node)
    converted.graph.ClearField("output")
    converted.graph.output.append(
        helper.make_tensor_value_info("output0", decoded.element_type, output_shape)
    )
    metadata["end2end"] = "False"
    coordinates = "source coordinate format" if box_format == "source" else "cx, cy, width, height"
    metadata["output_contract"] = f"decoded boxes in {coordinates} + sigmoid class scores; {layout.upper()}; no TopK/NMS"
    helper.set_model_props(converted, metadata)
    converted = _infer_shapes(converted)
    checker.check_model(converted, full_check=True)
    return converted


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", nargs="?", type=Path, help="output ONNX; defaults to SOURCE_decoded_xywh.onnx")
    parser.add_argument("--layout", choices=("bcn", "bnc"), default="bcn", help="bcn: [batch,channels,points]; bnc: [batch,points,channels]")
    parser.add_argument("--rknn-output", type=Path, help="also build an RK3588 FP16 RKNN model")
    parser.add_argument(
        "--box-format", choices=("source", "xywh"), default="xywh",
        help="xywh (default): source xyxy to center-x, center-y, width, height; source: preserve coordinates",
    )
    diagnostic = parser.add_mutually_exclusive_group()
    diagnostic.add_argument(
        "--diagnostic-zero-scores", action="store_true",
        help="diagnostic only: force all class probabilities to zero; expected no detections",
    )
    diagnostic.add_argument(
        "--diagnostic-fixed-box", action="store_true",
        help="diagnostic only: emit a single class-0 box with probability 0.875",
    )
    args = parser.parse_args(argv)
    if args.destination is None:
        suffix = "_decoded_xywh" if args.box_format == "xywh" else "_decoded"
        if args.layout == "bnc":
            suffix += "_bnc"
        args.destination = args.source.with_name(f"{args.source.stem}{suffix}.onnx")
    if args.source.resolve() == args.destination.resolve():
        parser.error("destination must differ from the source model")
    if args.rknn_output is not None and args.rknn_output.suffix.lower() != ".rknn":
        parser.error("--rknn-output must have the .rknn extension")
    model = build_decoded_model(
        onnx.load(str(args.source)), layout=args.layout,
        diagnostic_zero_scores=args.diagnostic_zero_scores,
        diagnostic_fixed_box=args.diagnostic_fixed_box,
        box_format=args.box_format,
    )
    onnx.save(model, str(args.destination))
    output = model.graph.output[0]
    print(args.destination, [dim.dim_value for dim in output.type.tensor_type.shape.dim])
    if args.diagnostic_zero_scores:
        print("DIAGNOSTIC ONLY: all class scores are zero; this model must not detect objects.")
    if args.diagnostic_fixed_box:
        print("DIAGNOSTIC ONLY: one fixed class-0 box with probability 0.875; no live recognition.")
    if args.rknn_output is not None:
        if __package__ in (None, ""):
            from export import convert_to_rknn
        else:
            from .export import convert_to_rknn

        convert_to_rknn(
            args.destination, args.rknn_output, "rk3588", False, None,
            "w8a8", "normal", "channel", "float16",
        )
        print(f"RK3588 FP16 export completed: {args.rknn_output}")


if __name__ == "__main__":
    main()
