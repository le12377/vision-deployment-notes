import os
import re

def generate_sorted_image_list_simple():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "quantify", "data1")
    dataset_path = os.path.join(current_dir, "dataset.txt")

    if not os.path.exists(data_dir):
        print(f"错误：目录 {data_dir} 不存在")
        return

    # 获取所有图片文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = [f for f in os.listdir(data_dir) 
                  if f.lower().endswith(image_extensions) and os.path.isfile(os.path.join(data_dir, f))]

    if len(image_files) == 0:
        print("没有找到图片文件！")
        return

    # 按文件名排序
    image_files.sort()

    # 写入文件 - 关键修正：使用正确的路径格式
    with open(dataset_path, 'w', encoding='utf-8') as f:
        for filename in image_files:
            # 正确的相对路径格式
            relative_path = f"quantify/data1/{filename}"
            f.write(relative_path + '\n')

    print(f"成功生成 dataset.txt，包含 {len(image_files)} 个图片")
    print("前5个路径：")
    for filename in image_files[:5]:
        print(f"  quantify/data1/{filename}")

# 最简单的版本
if __name__ == "__main__":
    generate_sorted_image_list_simple()
