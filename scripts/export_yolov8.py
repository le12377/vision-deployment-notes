#!/usr/bin/env python3
"""Export through the bundled RKNN-optimized YOLOv8 source and save labels."""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error("PT 权重不存在，请先完成下载或训练")
    if args.imgsz <= 0 or args.imgsz % 32:
        parser.error("imgsz 必须是正的 32 倍数")
    optimized = Path(__file__).resolve().parents[1] / "convert/yolov8_p-o"
    sys.path.insert(0, str(optimized))
    import onnx
    from ultralytics import YOLO

    model = YOLO(str(args.weights.resolve()))
    output = Path(model.export(format="rknn", imgsz=args.imgsz, device="cpu"))
    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    if len(graph.graph.output) != 9:
        raise RuntimeError("优化导出结果不是预期的 9 输出，请检查模型类型")
    names = [model.names[index] for index in range(len(model.names))]
    metadata = output.with_suffix(".json")
    metadata.write_text(json.dumps({"imgsz": args.imgsz, "names": names, "output_count": 9},
                                   ensure_ascii=False, indent=2) + "\n")
    print("ONNX:", output)
    print("模型类别/尺寸:", metadata)


if __name__ == "__main__":
    main()
