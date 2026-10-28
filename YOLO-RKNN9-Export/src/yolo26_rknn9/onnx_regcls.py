"""Command line wrapper for ONNX -> EP-style reg/cls ONNX conversion."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

try:
    from .regcls import export_regcls_onnx, validate_regcls_model
except ImportError:  # pragma: no cover - supports `python3 onnx_regcls.py ...`.
    from regcls import export_regcls_onnx, validate_regcls_model  # type: ignore[no-redef]


def _default_output_path(source: Path) -> Path:
    return source.with_name(f"{source.stem}_regcls.onnx")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a YOLOv8/YOLO11/YOLOv26 ONNX model to EP-style reg1/cls1/reg2/cls2/reg3/cls3 outputs"
    )
    parser.add_argument("source", type=Path, help="source YOLO .onnx model")
    parser.add_argument("destination", nargs="?", type=Path, help="destination .onnx path; default: SOURCE_regcls.onnx")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    destination = (args.destination or _default_output_path(source)).resolve()
    export_regcls_onnx(source, destination)
    outputs = validate_regcls_model(destination)

    print(f"Validated reg/cls ONNX: {destination}")
    for output in outputs:
        print(f"  {output.name}: {list(output.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
