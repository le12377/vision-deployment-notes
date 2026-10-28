# YOLOv8 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv8 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.
With PaddleOCR integration for license plate text recognition.
"""

import argparse
import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_enable_onednn"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", "/tmp/paddlex_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
import platform
import sys
from pathlib import Path

import cv2
import torch
import numpy as np
from ultralytics import YOLO

# ======================== OCR 相关导入 ========================
try:
    from paddleocr import PaddleOCR
    PADDLE_OCR_AVAILABLE = True
except ImportError:
    PADDLE_OCR_AVAILABLE = False
    print("[WARNING] PaddleOCR 未安装，无法进行文字识别。")
    print("请安装: pip install paddleocr==2.7.0.3 paddlepaddle")

from PIL import Image, ImageDraw, ImageFont

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# ======================== OCR 工具函数 ========================
_ocr_engine = None
_ocr_api_version = 2
_font_cache = {}            # 字体缓存

def get_ocr_engine(lang='ch', use_gpu=False):
    """初始化 PaddleOCR 引擎，兼容 PaddleOCR 2.x / 3.x 参数"""
    global _ocr_engine, _ocr_api_version

    if _ocr_engine is None:
        try:
            # PaddleOCR 3.x 参数
            # 注意：OCR 强制使用 CPU + paddle 引擎，避免 PaddleOCR 3.x 在 GPU/OneDNN
            # 静态推理下出现 ConvertPirAttribute2RuntimeAttribute 报错。
            _ocr_engine = PaddleOCR(
                lang=lang,
                device='cpu',
                engine='paddle',
                use_textline_orientation=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                enable_mkldnn=False,
                cpu_threads=1,
            )
            _ocr_api_version = 3
        except Exception as e_new:
            try:
                # PaddleOCR 2.x 参数
                _ocr_engine = PaddleOCR(
                    use_angle_cls=False,
                    lang=lang,
                    use_gpu=use_gpu,
                    enable_mkldnn=False,
                    cpu_threads=1,
                    show_log=True
                )
                _ocr_api_version = 2
            except Exception as e_old:
                print(f"[ERROR] PaddleOCR 初始化失败: {e_old}")
                print(f"[DEBUG] PaddleOCR 3.x 初始化错误: {e_new}")
                return None

        print(f"[INFO] PaddleOCR 引擎初始化完成 (lang={lang}, device=cpu)")

    return _ocr_engine


def find_chinese_font(size=24):
    """查找系统中可用的中文字体"""
    if size in _font_cache:
        return _font_cache[size]

    candidate_fonts = [
        'C:/Windows/Fonts/simhei.ttf',
        'C:/Windows/Fonts/simsun.ttc',
        'C:/Windows/Fonts/msyh.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/System/Library/Fonts/PingFang.ttc',
        '/System/Library/Fonts/STHeiti Medium.ttc',
    ]
    for fp in candidate_fonts:
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, size)
            _font_cache[size] = font
            return font

    print("[WARNING] 未找到中文字体，使用默认字体（可能无法显示中文）")
    font = ImageFont.load_default()
    _font_cache[size] = font
    return font


def draw_texts_with_pil(img, texts_info):
    """使用 PIL 在图像上绘制文字（支持中文）"""
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    for x, y, text, color_bgr, font_size in texts_info:
        font = find_chinese_font(font_size)
        rgb_color = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        draw.text((x, y), text, fill=rgb_color, font=font)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def preprocess_plate_crop(crop_img):
    """对车牌裁剪区域进行预处理，提升 OCR 识别率"""
    if crop_img is None or crop_img.size == 0:
        return None

    h, w = crop_img.shape[:2]
    min_height = 32
    min_width = 100
    if h < min_height or w < min_width:
        scale = max(min_height / h if h < min_height else 1.0,
                    min_width / w if w < min_width else 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        crop_img = cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return crop_img


def recognize_plate_text(crop_img, ocr_engine):
    """使用 PaddleOCR 识别车牌裁剪区域中的文字"""
    if crop_img is None or crop_img.size == 0:
        return '', 0.0

    processed = preprocess_plate_crop(crop_img)
    if processed is None:
        return '', 0.0

    try:
        if _ocr_api_version >= 3:
            # PaddleOCR 3.x 推荐调用方式
            result = ocr_engine.predict(processed)
        else:
            # PaddleOCR 2.x 版本调用方式
            result = ocr_engine.ocr(processed, cls=False)

        if not result or not result[0]:
            return '', 0.0

        all_texts = []
        best_text = ''
        best_conf = 0.0

        # PaddleOCR 3.x 返回 OCRResult/dict 风格
        first_result = result[0]
        if isinstance(first_result, dict) or hasattr(first_result, 'get'):
            rec_texts = first_result.get('rec_texts', []) or first_result.get('text', [])
            rec_scores = first_result.get('rec_scores', []) or first_result.get('scores', [])
            for text, conf in zip(rec_texts, rec_scores):
                all_texts.append(text)
                if conf > best_conf:
                    best_text = text
                    best_conf = conf

            combined_text = ''.join(all_texts).replace(' ', '').replace('·', '')
            if len(combined_text) >= len(best_text):
                return combined_text, best_conf
            else:
                return best_text, best_conf

        # 2.7.x 的返回格式: [框坐标, (文本, 置信度)]
        for line in result[0]:
            text = line[1][0]
            conf = line[1][1]
            all_texts.append(text)
            if conf > best_conf:
                best_text = text
                best_conf = conf

        # 拼接所有文本并去除空格和特殊符号
        combined_text = ''.join(all_texts).replace(' ', '').replace('·', '')

        if len(combined_text) >= len(best_text):
            return combined_text, best_conf
        else:
            return best_text, best_conf

    except Exception as e:
        print(f"[OCR ERROR] {e}")
        return '', 0.0


# ======================== Annotator 类 ========================

class Annotator:
    """YOLOv8 Annotator for drawing bounding boxes and labels on images"""

    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)
        self.pil = pil
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
        """Generate distinct colors for n classes (固定种子保证颜色一致)"""
        np.random.seed(42)
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


# ======================== 工具函数 ========================

def load_model(weights, device=''):
    """Load YOLOv8 model"""
    device = select_device(device)
    model = YOLO(weights)
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


# ======================== 主推理函数 ========================

@torch.no_grad()
def run(
        weights=ROOT / 'yolov8n.pt',
        source=ROOT / 'data/images',
        imgsz=640,
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device='',
        view_img=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,
        update=False,
        project=ROOT / 'runs/detect',
        name='exp',
        exist_ok=False,
        line_thickness=3,
        hide_labels=False,
        hide_conf=False,
        half=False,
        dnn=False,
        vid_stride=1,
        # ===== 新增 OCR 参数 =====
        use_ocr=True,  # 是否启用 OCR 文字识别
        ocr_lang='ch',  # OCR 语言: 'ch' 中文, 'en' 英文
        save_ocr_txt=True,  # 是否将 OCR 结果保存到 txt
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')
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

    # ===== 初始化 OCR 引擎 =====
    ocr_engine = None
    if use_ocr:
        if PADDLE_OCR_AVAILABLE:
            ocr_engine = get_ocr_engine(lang=ocr_lang, use_gpu=(device.type == 'cuda'))
        else:
            print("[WARNING] PaddleOCR 不可用，跳过文字识别。")
            use_ocr = False

    # ===== 判断源类型 =====
    is_webcam = source.isnumeric()
    is_video_file = Path(source).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', 'ts']
    is_stream = is_url and not is_file
    needs_video_writer = is_webcam or is_video_file or is_stream

    # 获取视频 fps（用于 VideoWriter）
    fps = 30
    if is_video_file:
        cap_temp = cv2.VideoCapture(source)
        if cap_temp.isOpened():
            fps = cap_temp.get(cv2.CAP_PROP_FPS) or 30
        cap_temp.release()

    # ===== 使用 stream=True 统一处理所有源 =====
    results = model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf_thres,
        iou=iou_thres,
        max_det=max_det,
        device=device,
        augment=augment,
        stream=True,
        verbose=False,
        classes=classes,
        vid_stride=vid_stride,
    )

    # VideoWriter 延迟初始化
    vid_writer = None
    all_ocr_results = []  # 保存所有 OCR 结果用于汇总
    frame_idx = 0

    for r in results:
        frame_idx += 1
        im0 = r.orig_img.copy()
        annotator = Annotator(im0, line_width=line_thickness)

        # 存储待绘制的 OCR 文本 (x, y, text, color_bgr, font_size)
        ocr_texts_to_draw = []
        # 存储当前帧的 OCR 结果
        frame_ocr_results = []

        if r.boxes is not None and len(r.boxes) > 0:
            # 打印检测信息
            s = ''
            for c in r.boxes.cls.unique():
                n = (r.boxes.cls == c).sum()
                class_name = model.names[int(c)]
                s += f"{n} {class_name}{'s' * (n > 1)}, "

            # 遍历每个检测框
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())

                # ===== OCR 识别 =====
                ocr_text = ''
                ocr_conf = 0.0
                if use_ocr and ocr_engine:
                    # 裁剪车牌区域
                    crop = r.orig_img[int(y1):int(y2), int(x1):int(x2)]
                    ocr_text, ocr_conf = recognize_plate_text(crop, ocr_engine)

                # 创建检测框标签
                label = ''
                if not hide_labels:
                    label = model.names[cls]
                    if not hide_conf:
                        label += f' {conf:.2f}'

                # 绘制检测框
                color = annotator.colors[cls % len(annotator.colors)]
                annotator.box_label([x1, y1, x2, y2], label, color)

                # ===== 准备 OCR 文本绘制 =====
                if ocr_text:
                    ocr_display = ocr_text
                    if ocr_conf > 0 and not hide_conf:
                        ocr_display += f' ({ocr_conf:.2f})'

                    # 文字位置：框下方
                    text_y = int(y2) + 3
                    text_x = int(x1)

                    # 字体大小根据框高度自适应
                    box_h = int(y2 - y1)
                    font_size = max(min(int(box_h * 0.4), 36), 16)

                    ocr_texts_to_draw.append(
                        (text_x, text_y, ocr_display, (0, 255, 0), font_size)
                    )

                    frame_ocr_results.append({
                        'frame': frame_idx,
                        'class': model.names[cls],
                        'conf': float(conf),
                        'box': [float(x1), float(y1), float(x2), float(y2)],
                        'ocr_text': ocr_text,
                        'ocr_conf': float(ocr_conf),
                    })

                    print(f"  [OCR] 车牌识别: '{ocr_text}'  置信度: {ocr_conf:.3f}")

            print(f"Frame {frame_idx}: {s}")

        # 获取标注后的图像
        im0 = annotator.result()

        # ===== 使用 PIL 绘制 OCR 文本（支持中文）=====
        if ocr_texts_to_draw:
            im0 = draw_texts_with_pil(im0, ocr_texts_to_draw)

        # 保存 OCR 结果到汇总列表
        if frame_ocr_results:
            all_ocr_results.extend(frame_ocr_results)

        # ===== 显示 =====
        if view_img:
            cv2.imshow('YOLOv8 Detection + OCR', im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # ===== 保存 =====
        if save_img:
            if needs_video_writer:
                # 视频写入（延迟初始化）
                if vid_writer is None:
                    h, w = im0.shape[:2]
                    if is_webcam:
                        save_name = 'webcam_output.mp4'
                    else:
                        save_name = Path(source).name
                    save_path = str(save_dir / save_name)
                    vid_writer = cv2.VideoWriter(
                        save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
                    )
                    print(f"[INFO] 视频保存到: {save_path}")
                vid_writer.write(im0)
            else:
                # 图片保存
                if hasattr(r, 'path') and r.path:
                    save_name = Path(r.path).name
                else:
                    save_name = f'result_{frame_idx}.jpg'
                save_path = str(save_dir / save_name)
                cv2.imwrite(save_path, im0)
                print(f"[INFO] 图片保存到: {save_path}")

        # ===== 保存检测标签 txt =====
        if save_txt and r.boxes is not None:
            if hasattr(r, 'path') and r.path:
                txt_name = Path(r.path).stem + '.txt'
            else:
                txt_name = f'frame_{frame_idx}.txt'
            txt_path = str(save_dir / 'labels' / txt_name)
            with open(txt_path, 'w', encoding='utf-8') as f:
                for box in r.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    bconf = box.conf[0].cpu().numpy()
                    bcls = int(box.cls[0].cpu().numpy())
                    ih, iw = r.orig_img.shape[:2]
                    xc = (bx1 + bx2) / 2 / iw
                    yc = (by1 + by2) / 2 / ih
                    bw = (bx2 - bx1) / iw
                    bh = (by2 - by1) / ih
                    f.write(f'{bcls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}')
                    if save_conf:
                        f.write(f' {bconf:.6f}')
                    f.write('\n')

        # ===== 保存裁剪图 =====
        if save_crop and r.boxes is not None:
            crop_dir = save_dir / 'crops'
            crop_dir.mkdir(parents=True, exist_ok=True)
            for idx, box in enumerate(r.boxes):
                bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                bcls = int(box.cls[0].cpu().numpy())
                crop_img = r.orig_img[int(by1):int(by2), int(bx1):int(bx2)]
                crop_name = f'{model.names[bcls]}_frame{frame_idx}_{idx}.jpg'
                cv2.imwrite(str(crop_dir / crop_name), crop_img)

    # 释放 VideoWriter
    if vid_writer is not None:
        vid_writer.release()
    if view_img:
        cv2.destroyAllWindows()

    # ===== 保存 OCR 汇总结果 =====
    if save_ocr_txt and all_ocr_results and save_dir:
        ocr_txt_path = save_dir / 'ocr_results.txt'
        with open(ocr_txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("车牌 OCR 识别结果汇总\n")
            f.write("=" * 60 + "\n\n")
            for item in all_ocr_results:
                f.write(f"帧 {item['frame']:>4d}  |  "
                        f"类别: {item['class']}  "
                        f"检测置信度: {item['conf']:.3f}  |  "
                        f"车牌: {item['ocr_text']}  "
                        f"OCR置信度: {item['ocr_conf']:.3f}\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"共识别 {len(all_ocr_results)} 个车牌\n")
        print(f"\n[INFO] OCR 结果汇总保存到: {ocr_txt_path}")

    # 打印汇总
    if all_ocr_results:
        print("\n" + "=" * 60)
        print("车牌识别结果汇总:")
        print("=" * 60)
        for item in all_ocr_results:
            print(f"  帧{item['frame']:>4d}  车牌: {item['ocr_text']}  "
                  f"(OCR置信度: {item['ocr_conf']:.3f})")
        print("=" * 60)
        print(f"共识别 {len(all_ocr_results)} 个车牌\n")

    if save_dir:
        print(f"\n[INFO] 所有结果保存在: {save_dir}")


# ======================== 命令行参数 ========================

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / './car.pt', help='model path')
    parser.add_argument('--source', type=str,
                        default='./1.jpg',
                        help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
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

    # ===== 新增 OCR 相关参数 =====
    parser.add_argument('--use-ocr', action='store_true', default=True, help='enable OCR text recognition')
    parser.add_argument('--no-ocr', dest='use_ocr', action='store_false', help='disable OCR text recognition')
    parser.add_argument('--ocr-lang', type=str, default='ch', help='OCR language: ch (Chinese), en (English)')
    parser.add_argument('--no-ocr-txt', dest='save_ocr_txt', action='store_false', default=True,
                        help='do not save OCR results to txt')

    opt = parser.parse_args()
    opt.imgsz = opt.imgsz[0] if len(opt.imgsz) == 1 else opt.imgsz
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
