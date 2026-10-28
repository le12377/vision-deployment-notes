"""Verify that decoded export preserves predictions while removing selection."""

import sys
import subprocess
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yolo26_rknn9.onnx9 import NineOutputError
from yolo26_rknn9.onnx_decoded import build_decoded_model
from yolo26_rknn9 import export, onnx_decoded


def make_end2end_model():
    boxes = np.array([[[1, 2], [3, 4], [5, 6], [7, 8]]], dtype=np.float32)
    logits = np.array([[[-2, 2], [1, -1]]], dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["box_values"], ["boxes"]),
            helper.make_node("Sigmoid", ["logits"], ["scores"]),
            helper.make_node("Concat", ["boxes", "scores"], ["decoded"], axis=1),
            helper.make_node("Transpose", ["decoded"], ["rows"], perm=[0, 2, 1]),
            helper.make_node("TopK", ["rows", "k"], ["output0", "indices"], axis=1),
        ],
        "synthetic_end2end",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 2, 2])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 1, 6])],
        initializer=[
            numpy_helper.from_array(boxes, "box_values"),
            numpy_helper.from_array(logits, "logits"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "k"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 12)])
    model.ir_version = 8
    helper.set_model_props(model, {"end2end": "True", "names": "{0: 'body', 1: 'head'}"})
    return model, boxes, logits


@pytest.mark.parametrize("layout", ["bcn", "bnc"])
@pytest.mark.parametrize("diagnostic_zero_scores", [False, True])
def test_preserves_decoded_boxes_and_every_class_score(layout, diagnostic_zero_scores):
    ort = pytest.importorskip("onnxruntime")
    source, boxes, logits = make_end2end_model()
    original = source.SerializeToString()
    converted = build_decoded_model(
        source, layout=layout, diagnostic_zero_scores=diagnostic_zero_scores, box_format="source"
    )
    session = ort.InferenceSession(converted.SerializeToString(), providers=["CPUExecutionProvider"])
    result = session.run(None, {"images": np.zeros((1, 3, 2, 2), dtype=np.float32)})[0]
    assert result.shape == ((1, 6, 2) if layout == "bcn" else (1, 2, 6))
    if layout == "bnc":
        result = result.transpose(0, 2, 1)
    np.testing.assert_array_equal(result[:, :4], boxes)
    if diagnostic_zero_scores:
        np.testing.assert_array_equal(result[:, 4:], np.zeros_like(logits))
        assert np.count_nonzero(result[:, 4:] > 0.65) == 0
    else:
        np.testing.assert_allclose(result[:, 4:], 1 / (1 + np.exp(-logits)), rtol=1e-6)
    assert result.shape == (1, 6, 2)
    assert len(converted.graph.output) == 1
    assert not any(node.op_type in ("TopK", "NonMaxSuppression") for node in converted.graph.node)
    assert "k" not in {item.name for item in converted.graph.initializer}
    assert {item.key: item.value for item in converted.metadata_props}["end2end"] == "False"
    assert source.SerializeToString() == original


def test_rejects_non_end2end_source():
    source, _, _ = make_end2end_model()
    helper.set_model_props(source, {"end2end": "False"})
    with pytest.raises(NineOutputError, match="end2end=True"):
        build_decoded_model(source)


def test_rejects_unactivated_class_scores():
    source, _, _ = make_end2end_model()
    source.graph.node[1].op_type = "Identity"
    with pytest.raises(NineOutputError, match="found 0"):
        build_decoded_model(source)


def test_rejects_multiple_source_outputs():
    source, _, _ = make_end2end_model()
    source.graph.output.append(helper.make_tensor_value_info("rows", TensorProto.FLOAT, [1, 2, 6]))
    with pytest.raises(NineOutputError, match="single end-to-end"):
        build_decoded_model(source)


def test_rejects_unknown_layout():
    source, _, _ = make_end2end_model()
    with pytest.raises(NineOutputError, match="bcn or bnc"):
        build_decoded_model(source, layout="unknown")


@pytest.mark.parametrize("layout", ["bcn", "bnc"])
def test_fixed_box_diagnostic_has_exactly_one_known_score_and_box(layout):
    ort = pytest.importorskip("onnxruntime")
    source, boxes, logits = make_end2end_model()
    original = source.SerializeToString()
    converted = build_decoded_model(source, layout=layout, diagnostic_fixed_box=True, box_format="source")
    session = ort.InferenceSession(converted.SerializeToString(), providers=["CPUExecutionProvider"])
    result = session.run(None, {"images": np.zeros((1, 3, 2, 2), dtype=np.float32)})[0]
    assert result.shape == ((1, 6, 2) if layout == "bcn" else (1, 2, 6))
    if layout == "bnc":
        result = result.transpose(0, 2, 1)
    np.testing.assert_array_equal(result[:, :4, 0], boxes[:, :, 0])
    np.testing.assert_array_equal(result[0, :4, 1], np.array([0.5, 0.4, 1.5, 1.6], np.float32))
    expected_scores = np.zeros_like(logits)
    expected_scores[0, 0, 1] = 0.875
    np.testing.assert_array_equal(result[:, 4:], expected_scores)
    assert np.count_nonzero(result[0, 4:].max(axis=0) > 0.65) == 1
    assert source.SerializeToString() == original


def test_rejects_conflicting_diagnostics():
    source, _, _ = make_end2end_model()
    with pytest.raises(NineOutputError, match="one diagnostic mode"):
        build_decoded_model(source, diagnostic_zero_scores=True, diagnostic_fixed_box=True)


@pytest.mark.parametrize("layout", ["bcn", "bnc"])
@pytest.mark.parametrize("diagnostic_fixed_box", [False, True])
def test_xywh_conversion_preserves_scores_and_box_geometry(layout, diagnostic_fixed_box):
    ort = pytest.importorskip("onnxruntime")
    source, _, _ = make_end2end_model()
    original = source.SerializeToString()
    reference = build_decoded_model(source, diagnostic_fixed_box=diagnostic_fixed_box, box_format="source")
    converted = build_decoded_model(
        source, layout=layout, diagnostic_fixed_box=diagnostic_fixed_box, box_format="xywh"
    )
    data = {"images": np.zeros((1, 3, 2, 2), dtype=np.float32)}
    providers = ["CPUExecutionProvider"]
    before = ort.InferenceSession(reference.SerializeToString(), providers=providers).run(None, data)[0]
    after = ort.InferenceSession(converted.SerializeToString(), providers=providers).run(None, data)[0]
    assert after.shape == ((1, 6, 2) if layout == "bcn" else (1, 2, 6))
    if layout == "bnc":
        after = after.transpose(0, 2, 1)
    np.testing.assert_array_equal(after[:, 4:], before[:, 4:])
    np.testing.assert_allclose(after[:, :2], (before[:, :2] + before[:, 2:4]) / 2)
    np.testing.assert_allclose(after[:, 2:4], before[:, 2:4] - before[:, :2])
    # A center-based consumer reconstructs the original corner geometry.
    np.testing.assert_allclose(after[:, :2] - after[:, 2:4] / 2, before[:, :2])
    np.testing.assert_allclose(after[:, :2] + after[:, 2:4] / 2, before[:, 2:4])
    if diagnostic_fixed_box:
        np.testing.assert_allclose(after[0, :4, 1], [1, 1, 1, 1.2])
        assert after[0, 4, 1] == 0.875
    assert source.SerializeToString() == original


def test_rejects_unknown_box_format():
    source, _, _ = make_end2end_model()
    with pytest.raises(NineOutputError, match="source or xywh"):
        build_decoded_model(source, box_format="unknown")


@pytest.mark.parametrize("direct_script", [False, True])
def test_decoded_cli_defaults_to_gb_xywh_with_automatic_output_name(tmp_path, direct_script):
    source = tmp_path / "model.onnx"
    model, _, _ = make_end2end_model()
    onnx.save(model, source)
    original = source.read_bytes()
    if direct_script:
        subprocess.run(
            [sys.executable, str(Path(onnx_decoded.__file__)), str(source)],
            check=True, capture_output=True, text=True,
        )
    else:
        onnx_decoded.main([str(source)])
    result = onnx.load(tmp_path / "model_decoded_xywh.onnx")
    expected = build_decoded_model(model, box_format="xywh")
    assert result.SerializeToString() == expected.SerializeToString()
    assert build_decoded_model(model).SerializeToString() == expected.SerializeToString()
    assert source.read_bytes() == original
    assert not any(item.key == "diagnostic" for item in result.metadata_props)


def test_decoded_cli_can_explicitly_preserve_source_coordinates(tmp_path):
    source = tmp_path / "model.onnx"
    model, _, _ = make_end2end_model()
    onnx.save(model, source)
    onnx_decoded.main([str(source), "--box-format", "source"])
    result = onnx.load(tmp_path / "model_decoded.onnx")
    assert result.SerializeToString() == build_decoded_model(model, box_format="source").SerializeToString()


def test_decoded_cli_compiles_fp16_with_rknn_output(tmp_path, monkeypatch):
    source = tmp_path / "model.onnx"
    model, _, _ = make_end2end_model()
    onnx.save(model, source)
    destination = tmp_path / "model.rknn"
    calls = []
    monkeypatch.setattr(export, "convert_to_rknn", lambda *args: calls.append(args))
    onnx_decoded.main([str(source), "--rknn-output", str(destination)])
    assert len(calls) == 1
    onnx_path, rknn_path, platform, quantize, dataset, *_settings, precision = calls[0]
    assert onnx_path == tmp_path / "model_decoded_xywh.onnx"
    assert onnx.load(onnx_path).SerializeToString() == build_decoded_model(model).SerializeToString()
    assert rknn_path == destination and platform == "rk3588"
    assert quantize is False and dataset is None and precision == "float16"


def test_decoded_cli_preserves_source_when_destination_matches(tmp_path):
    source = tmp_path / "model.onnx"
    model, _, _ = make_end2end_model()
    onnx.save(model, source)
    original = source.read_bytes()
    with pytest.raises(SystemExit):
        onnx_decoded.main([str(source), str(source)])
    assert source.read_bytes() == original


def test_main_export_remains_a_three_six_nine_output_entry_point():
    assert export.parse_args(["model.onnx"]).output_count == 9
    with pytest.raises(SystemExit):
        export.parse_args(["model.onnx", "1"])
