# YOLOv8 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv8 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov8n.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov8n.pt                 # PyTorch
                                 yolov8n.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov8n_openvino_model     # OpenVINO
                                 yolov8n.engine             # TensorRT
                                 yolov8n.mlmodel            # CoreML (macOS-only)
                                 yolov8n_saved_model        # TensorFlow SavedModel
                                 yolov8n.pb                 # TensorFlow GraphDef
                                 yolov8n.tflite             # TensorFlow Lite
                                 yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov8n_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import cv2
import torch
import numpy as np
from ultralytics import YOLO

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv8 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class Annotator:
    """YOLOv8 Annotator for drawing bounding boxes and labels on images"""

    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width
        self.pil = pil

        # Colors for different classes
        self.colors = self.generate_colors(80)

        if self.pil:
            from PIL import ImageDraw, ImageFont
            self.draw = ImageDraw.Draw(im)
            self.font = ImageFont.truetype(font, font_size or max(round(max(self.im.size) / 40), 12))
        else:
            self.font = cv2.FONT_HERSHEY_SIMPLEX
            self.font_scale = font_size or max(self.im.shape[0] / 1000, 0.5)
            self.thickness = max(self.lw - 1, 1)

    def generate_colors(self, n):
        """Generate distinct colors for n classes"""
        return [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                for _ in range(n)]

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        """Add one bounding box to image with label"""
        if self.pil:
            from PIL import ImageDraw
            self.draw = ImageDraw.Draw(self.im)
            self.draw.rectangle(box, width=self.lw, outline=color)
            if label:
                w, h = self.font.getsize(label)
                outside = box[1] - h >= 0
                self.draw.rectangle(
                    (box[0], box[1] - h if outside else box[1],
                     box[0] + w + 1, box[1] + 1 if outside else box[1] + h + 1),
                    fill=color
                )
                self.draw.text(
                    (box[0], box[1] - h if outside else box[1]),
                    label, fill=txt_color, font=self.font
                )
        else:
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                w, h = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)[0]
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)
                cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                            self.font, self.font_scale, txt_color, self.thickness)

    def result(self):
        """Return annotated image"""
        return self.im


def load_model(weights, device=''):
    """Load YOLOv8 model"""
    device = select_device(device)
    # 加上 task='detect' 可以消除那个 WARNING 警告
    model = YOLO(weights, task='detect')
    
    # 只有当模型是 PyTorch (.pt) 时，才需要 .to(device)
    # ONNX, OpenVINO 等导出格式会自动在 predict 时处理设备
    if Path(weights).suffix == '.pt':
        model.to(device)
        
    return model, device


def select_device(device='', batch_size=0):
    """Select device for inference"""
    if device and device.lower() != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'

    cuda = device.lower() != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    return device


@torch.no_grad()
def run(
        weights=ROOT / 'yolov8n.pt',  # model path
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        imgsz=640,  # inference size
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'dng', 'webp', 'mpo']
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)

    # Directories
    if not nosave:
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    else:
        save_dir = None

    # Load model
    model, device = load_model(weights, device)

    # Dataloader (for webcam/video)
    if webcam:
        view_img = check_imshow()
        cap = cv2.VideoCapture(0 if source.isnumeric() else source)
        if not cap.isOpened():
            raise Exception(f"Failed to open video source: {source}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Video writer
        if save_img:
            save_path = str(save_dir / 'webcam_output.mp4')
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        else:
            vid_writer = None

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % vid_stride != 0:
                continue

            # Run inference
            results = model.predict(
                source=frame,
                imgsz=imgsz,
                conf=conf_thres,
                iou=iou_thres,
                max_det=max_det,
                device=device,
                augment=augment,
                verbose=False,
                classes=classes
            )

            # Process results
            for r in results:
                im0 = frame.copy()
                annotator = Annotator(im0, line_width=line_thickness)

                if r.boxes is not None and len(r.boxes) > 0:
                    # Print results
                    s = ''
                    for c in r.boxes.cls.unique():
                        n = (r.boxes.cls == c).sum()
                        class_name = model.names[int(c)]
                        s += f"{n} {class_name}{'s' * (n > 1)}, "

                    # Draw bounding boxes
                    for box in r.boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())

                        # Create label
                        label = ''
                        if not hide_labels:
                            label = model.names[cls]
                            if not hide_conf:
                                label += f' {conf:.2f}'

                        # Draw box
                        color = annotator.colors[cls % len(annotator.colors)]
                        annotator.box_label([x1, y1, x2, y2], label, color)

                # Display results
                im0 = annotator.result()
                if view_img:
                    cv2.imshow('YOLOv8 Detection', im0)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Save results
                if save_img and vid_writer is not None:
                    vid_writer.write(im0)

        cap.release()
        if vid_writer is not None:
            vid_writer.release()
        cv2.destroyAllWindows()

    else:
        # For image and video files
        results = model.predict(
            source=source,
            stream=True,
            imgsz=imgsz,
            conf=conf_thres,
            iou=iou_thres,
            max_det=max_det,
            device=device,
            augment=augment,
            save=save_img,
            save_txt=save_txt,
            save_conf=save_conf,
            project=project,
            name=name,
            exist_ok=exist_ok,
            line_width=line_thickness,
            show_labels=True,
            show_conf=True,
            classes=classes,
            vid_stride=vid_stride
        )

        # Print results
        for r in results:
            print(f"Detected {len(r.boxes) if r.boxes is not None else 0} objects")


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """Increment file or directory path"""
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'
            if not os.path.exists(p):
                path = Path(p)
                break
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


def check_imshow():
    """Check if environment supports image display"""
    try:
        cv2.namedWindow('test')
        cv2.destroyWindow('test')
        return True
    except Exception:
        return False


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / './yolov8n.pt', help='model path')
    #parser.add_argument('--source', type=str, default='http://admin:admin@192.168.238.16:8081', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--source', type=str, default='/home/lsf/桌面/视觉/yolo/train/ultralytics/images/3.jpg', help='')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz = opt.imgsz[0] if len(opt.imgsz) == 1 else opt.imgsz  # expand if single value
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
