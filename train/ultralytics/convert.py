from ultralytics import YOLO

model = YOLO("val_v11s_320.pt")

onnx_path = model.export(
    format="onnx",
    imgsz=320,
    batch=1,
    dynamic=False,
    simplify=True,
    opset=12,
    nms=False,
    half=False,
)

print(f"HALCON-compatible candidate: {onnx_path}")
