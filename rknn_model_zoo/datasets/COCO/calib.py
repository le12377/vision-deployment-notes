import os
import glob
import cv2
import numpy as np
from pathlib import Path

def letterbox(img, new_shape=(416, 416), color=(114, 114, 114)):
    """
    YOLO 标准的 Letterbox 预处理：等比缩放图片，并用灰色填充多余部分。
    """
    shape = img.shape[:2]  # 当前图片的高和宽 [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 计算缩放比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # 计算缩放后的未填充尺寸
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # (width, height)
    
    # 计算宽高需要填充的像素数
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    
    # 将填充量平分到上下/左右两边
    dw /= 2
    dh /= 2

    # 缩放图片
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # 计算上下左右的填充像素
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    # 填充边缘
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img

def process_dataset(src_dir, dst_dir, target_size=416):
    """
    读取目录下的所有图片，进行 Letterbox 处理后保存，并生成 calibration.txt。
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    # 如果目标文件夹不存在则创建
    dst_path.mkdir(parents=True, exist_ok=True)

    # 支持的图片格式
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_files = []
    for ext in extensions:
        # 递归查找当前目录及所有子目录，并用 set 去重
        image_files.extend(glob.glob(str(src_path / '**' / ext), recursive=True))
    
    image_files = list(set(image_files))  # 去重，防止重复处理

    if not image_files:
        print(f"错误: 在 {src_dir} 中没有找到任何图片！")
        return

    print(f"找到了 {len(image_files)} 张不重复的图片，开始预处理 (尺寸 -> {target_size}x{target_size})...")

    # 用于保存处理后的图片绝对路径，供 RKNN 使用
    valid_processed_paths = []

    for img_path in image_files:
        try:
            # 读取图片
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告: 无法读取图片 {img_path}，跳过。")
                continue

            # 执行 Letterbox
            img_processed = letterbox(img, new_shape=(target_size, target_size))

            # 构造保存路径 (保持原文件名，防止重名加上前缀)
            img_name = Path(img_path).name
            save_path = dst_path / f"calib_{img_name}"
            
            # 保存图片
            cv2.imwrite(str(save_path), img_processed)
            valid_processed_paths.append(str(save_path.resolve()))
            
        except Exception as e:
            print(f"处理 {img_path} 时出错: {e}")

    # 生成 calibration.txt 文件
    calib_txt_path = dst_path / "calibration.txt"
    with open(calib_txt_path, 'w', encoding='utf-8') as f:
        for path in valid_processed_paths:
            f.write(f"{path}\n")

    print("-" * 40)
    print(f"处理完成！共成功处理 {len(valid_processed_paths)} 张图片。")
    print(f"处理后的图片保存在: {dst_path.resolve()}")
    print(f"RKNN 量化数据集列表保存在: {calib_txt_path.resolve()}")

if __name__ == "__main__":
    # ============ 在这里修改你的配置 ============
    # 原始图片所在文件夹（里面可以是各种不同尺寸的图片）
    SOURCE_DIR = "./subset8" 
    
    # 处理后图片保存的文件夹
    DEST_DIR = "./image8"
    
    # 目标尺寸（必须与你导出 ONNX 时使用的 imgsz 一致）
    TARGET_SIZE = 416
    # ==========================================

    process_dataset(SOURCE_DIR, DEST_DIR, TARGET_SIZE)
