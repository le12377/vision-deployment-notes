#!/usr/bin/env python3
"""Configure the downloaded v888 script from the exported model metadata."""
import argparse
import ast
import json
from pathlib import Path


def configure(script, model, metadata, camera):
    script, model, metadata = map(Path, (script, model, metadata))
    if not model.is_file():
        raise ValueError("RKNN 模型不存在: {}".format(model))
    config = json.loads(metadata.read_text())
    names = config["names"]
    size = config["imgsz"]
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise ValueError("模型 names 必须是非空字符串列表")
    if not isinstance(size, int) or size <= 0:
        raise ValueError("模型 imgsz 必须是正整数")
    if config["output_count"] not in (6, 9):
        raise ValueError("v888 YOLOv8 后处理需要 6 或 9 输出")
    source = script.read_text()
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    replacements = []
    values = {"MODEL_PATH": str(model.resolve()), "IMG_SIZE": (size, size), "CLASSES": tuple(names)}
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in values:
                found.add(target.id)
                replacements.append((node.lineno - 1, node.end_lineno,
                                     "{} = {!r}\n".format(target.id, values[target.id])))
    if found != set(values):
        raise ValueError("上游脚本的配置结构已变化，请核对 MODEL_PATH/IMG_SIZE/CLASSES")
    cameras = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Attribute) and node.func.attr == "VideoCapture"
               and isinstance(node.func.value, ast.Name) and node.func.value.id == "cv2"]
    if len(cameras) != 1 or cameras[0].lineno != cameras[0].end_lineno:
        raise ValueError("上游摄像头入口已变化，请人工核对")
    call = cameras[0]
    line = lines[call.lineno - 1]
    encoded = line.encode("utf-8")
    camera_line = (encoded[:call.col_offset] + "cv2.VideoCapture({})".format(camera).encode()
                   + encoded[call.end_col_offset:]).decode("utf-8")
    replacements.append((call.lineno - 1, call.end_lineno, camera_line))
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = [replacement]
    updated = "".join(lines)
    ast.parse(updated)
    backup = script.with_suffix(script.suffix + ".bak")
    if not backup.exists():
        backup.write_text(source)
    script.write_text(updated)
    print("已配置 {}：{} 类，{}×{}，摄像头 {}".format(script.name, len(names), size, size, camera))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    configure(args.script, args.model, args.metadata, args.camera)


if __name__ == "__main__":
    main()
