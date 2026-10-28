"""CLI for exporting YOLOv8, YOLOv26, YOLO11, YOLOX and YOLOv5-style models to RKNN output contracts."""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    # Support `python export.py ...` when running from this package directory.
    from onnx9 import NineOutputError, export_output_onnx, validate_output_model
else:
    from .onnx9 import NineOutputError, export_output_onnx, validate_output_model


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png"}
QUANTIZED_DTYPES = ("w8a8", "w8a16", "w16a16i", "w16a16i_dfp", "w4a16")
QUANTIZED_ALGORITHMS = ("normal", "mmse", "kl_divergence", "gdq")
FLOAT_DTYPES = ("float16",)
RK3588_QUANTIZED_DTYPES = ("w8a8", "w16a16i", "w16a16i_dfp")
_GROUP_QUANTIZATION = re.compile(r"group(\d+)$")


def export_onnx_from_pt(weights: Path, imgsz: int) -> Path:
    """Export a supported YOLO PyTorch checkpoint to a static ONNX graph."""

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("install ultralytics to export a .pt checkpoint") from exc

    model = YOLO(str(weights))
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=True,
        opset=12,
        dynamic=False,
    )
    return Path(exported)


def prepare_calibration_dataset(source: Path, destination: Path, sample_count: int) -> Path:
    """Return an RKNN dataset list, generating a reproducible one from an image directory."""

    if source.is_file():
        if source.suffix.lower() != ".txt":
            raise ValueError("--calib-data file must be an RKNN dataset-list .txt file")
        return source
    if not source.is_dir():
        raise ValueError(f"calibration data path does not exist: {source}")

    images = sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"no calibration images found under {source}")
    selected = random.Random(0).sample(images, min(sample_count, len(images)))
    destination.write_text("".join(f"{path}\n" for path in selected), encoding="utf-8")
    return destination


def convert_to_rknn(
    onnx_path: Path,
    rknn_path: Path,
    target_platform: str,
    quantize: bool,
    calibration_dataset: Path | None,
    quantized_dtype: str,
    quantized_algorithm: str,
    quantized_method: str,
    float_dtype: str,
) -> None:
    """Build an RKNN model from a validated nine-output ONNX graph."""

    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise RuntimeError(
            "rknn-toolkit2 is required for .rknn conversion and must be installed on an x86 Linux host"
        ) from exc

    rknn = RKNN(verbose=True)
    try:
        result = rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            quantized_dtype=quantized_dtype,
            quantized_algorithm=quantized_algorithm,
            quantized_method=quantized_method,
            float_dtype=float_dtype,
            target_platform=target_platform,
            optimization_level=3,
        )
        if result not in (None, 0):
            raise RuntimeError(f"RKNN config failed with status {result}")

        result = rknn.load_onnx(model=str(onnx_path))
        if result != 0:
            raise RuntimeError(f"RKNN failed to load ONNX with status {result}")

        if quantize:
            if calibration_dataset is None:
                raise ValueError("--quantize requires --calib-data")
            result = rknn.build(do_quantization=True, dataset=str(calibration_dataset))
        else:
            result = rknn.build(do_quantization=False)
        if result != 0:
            raise RuntimeError(f"RKNN build failed with status {result}")

        result = rknn.export_rknn(str(rknn_path))
        if result != 0:
            raise RuntimeError(f"RKNN export failed with status {result}")
    finally:
        rknn.release()


def _default_output_path(source: Path, output_count: int) -> Path:
    return source.with_name(f"{source.stem}_rknn{output_count}.onnx")


def _print_outputs(onnx_path: Path, output_count: int, *, legacy_output_names: bool = False) -> None:
    outputs = validate_output_model(
        onnx_path,
        output_count=output_count,
        legacy_output_names=legacy_output_names,
    )
    print(f"Validated {output_count}-output ONNX: {onnx_path}")
    for output in outputs:
        print(f"  {output.name}: {list(output.shape)}")


def _quantized_method(value: str) -> str:
    if value in {"layer", "channel"}:
        return value
    match = _GROUP_QUANTIZATION.fullmatch(value)
    if match is not None:
        group_size = int(match.group(1))
        if 32 <= group_size <= 256 and group_size % 32 == 0:
            return value
    raise argparse.ArgumentTypeError(
        "must be layer, channel, or groupSIZE where SIZE is 32..256 in multiples of 32"
    )


def validate_quantized_dtype(target_platform: str, quantized_dtype: str) -> None:
    """Fail early when a requested quantized dtype is invalid for a known target."""

    if target_platform.lower() == "rk3588" and quantized_dtype not in RK3588_QUANTIZED_DTYPES:
        supported = ", ".join(RK3588_QUANTIZED_DTYPES)
        raise ValueError(
            f"{quantized_dtype} is not supported by RK3588 in RKNN Toolkit2; use one of: {supported}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a YOLOv8, YOLOv26, YOLO11, YOLOX or YOLOv5-style model to RKNN 3/6/9-output ONNX/RKNN"
    )
    parser.add_argument("weights_positional", nargs="?", type=Path, help="YOLOv8, YOLOv26, YOLO11, YOLOX or YOLOv5-style .pt or .onnx model")
    parser.add_argument(
        "output_count_positional",
        nargs="?",
        type=int,
        choices=(3, 6, 9),
        help="output contract: 3 emits one YOLOv5 head per scale; 6 emits bbox/scores; 9 also emits score_sum (default: 9)",
    )
    parser.add_argument("--weights", type=Path, help="YOLOv8, YOLOv26, YOLO11, YOLOX or YOLOv5-style .pt or .onnx model")
    parser.add_argument("--imgsz", default=416, type=int, help="static input size when --weights is .pt")
    parser.add_argument("--onnx-output", type=Path, help="output ONNX destination")
    parser.add_argument(
        "--output-count",
        type=int,
        choices=(3, 6, 9),
        default=None,
        help="output contract: 3 emits one YOLOv5 head per scale; 6 emits bbox/scores; 9 also emits score_sum (default: 9)",
    )
    parser.add_argument("--output", type=Path, help="RKNN destination")
    parser.add_argument(
        "--zy-compat",
        action="store_true",
        help="use the legacy nine-output names accepted by the known-good zy/TF YOLOv8 model",
    )
    parser.add_argument("--onnx-only", default=True, action="store_true", help="create and validate ONNX without RKNN conversion")
    parser.add_argument("--target-platform", default="rk3588", help="RKNN target platform")
    parser.add_argument("--quantize", action="store_true", help="enable RKNN calibration quantization")
    parser.add_argument(
        "--quantized-dtype",
        choices=QUANTIZED_DTYPES,
        default="w8a8",
        help="RKNN quantized data type (default: w8a8)",
    )
    parser.add_argument(
        "--quantized-algorithm",
        choices=QUANTIZED_ALGORITHMS,
        default="normal",
        help="RKNN calibration algorithm (default: normal)",
    )
    parser.add_argument(
        "--quantized-method",
        type=_quantized_method,
        default="channel",
        help="RKNN quantization granularity: layer, channel, or group32..group256",
    )
    parser.add_argument(
        "--float-dtype",
        choices=FLOAT_DTYPES,
        default="float16",
        help="RKNN non-quantized precision (default: float16)",
    )
    parser.add_argument("--calib-data", type=Path, help="calibration image directory or RKNN dataset-list text file")
    parser.add_argument("--calib-samples", default=200, type=int, help="maximum images selected from a directory")
    args = parser.parse_args(argv)
    if args.weights is not None and args.weights_positional is not None:
        parser.error("provide the model either positionally or with --weights, not both")
    args.weights = args.weights or args.weights_positional
    if args.weights is None:
        parser.error("the model path is required (use --weights MODEL or provide MODEL positionally)")
    if (
        args.output_count_positional is not None
        and args.output_count is not None
        and args.output_count_positional != args.output_count
    ):
        parser.error("positional output count and --output-count must match when both are provided")
    args.output_count = args.output_count or args.output_count_positional or 9
    return args


def main() -> int:
    args = parse_args()
    weights = args.weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"model does not exist: {weights}")

    suffix = weights.suffix.lower()
    if suffix == ".pt":
        source_onnx = export_onnx_from_pt(weights, args.imgsz)
    elif suffix == ".onnx":
        source_onnx = weights
    else:
        raise ValueError("--weights must point to a .pt or .onnx model")

    if args.quantize:
        validate_quantized_dtype(args.target_platform, args.quantized_dtype)

    output_onnx = (args.onnx_output or _default_output_path(source_onnx, args.output_count)).resolve()
    export_output_onnx(
        source_onnx,
        output_onnx,
        output_count=args.output_count,
        legacy_output_names=args.zy_compat,
    )
    _print_outputs(output_onnx, args.output_count, legacy_output_names=args.zy_compat)
    if args.onnx_only:
        return 0

    calibration_dataset = None
    if args.quantize:
        if args.calib_data is None:
            raise ValueError("--quantize requires --calib-data")
        calibration_dataset = prepare_calibration_dataset(
            args.calib_data.resolve(), output_onnx.with_name("calibration.txt"), args.calib_samples
        )

    rknn_path = (args.output or output_onnx.with_suffix(".rknn")).resolve()
    convert_to_rknn(
        output_onnx,
        rknn_path,
        args.target_platform,
        args.quantize,
        calibration_dataset,
        args.quantized_dtype,
        args.quantized_algorithm,
        args.quantized_method,
        args.float_dtype,
    )
    print(f"RKNN export completed: {rknn_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, NineOutputError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
