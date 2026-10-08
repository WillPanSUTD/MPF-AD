import os
import json
from PIL import Image
from tqdm import tqdm

# ====================== 【修改这里！】你的配置 ======================
# 图片文件夹路径
IMG_DIR = r"/path/to/LUT-AD/Train_Data/YOLO/Phong/test/images"  # 训练集图片
# YOLO标签文件夹路径（txt文件）
LABEL_DIR = r"/path/to/LUT-AD/Train_Data/YOLO/Phong/test/labels"  # 对应标签
# 输出COCO格式JSON路径
OUTPUT_JSON = r"/path/to/LUT-AD/Train_Data/COCO/annotations/test.json"
# 你的类别名称（顺序必须和YOLO标签的类别ID完全一致！）
CLASSES = ["Inveracious soldering", "Pinhole", "Pit","Burst","Fish-scale welding","Bump"]  # 按你的实际类别修改
# =================================================================

def yolo2coco(img_dir, label_dir, output_json, classes):
    # COCO格式基础结构
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 1. 写入类别信息
    for idx, cls in enumerate(classes):
        coco["categories"].append({
            "id": idx,
            "name": cls,
            "supercategory": "none"
        })

    img_id = 0  # 图片ID
    ann_id = 0  # 标注ID

    # 2. 遍历所有图片
    for img_name in tqdm(os.listdir(img_dir)):
        if img_name.lower().endswith(('jpg', 'jpeg', 'png', 'bmp')):
            # 读取图片尺寸
            img_path = os.path.join(img_dir, img_name)
            try:
                img = Image.open(img_path)
                img_w, img_h = img.size
            except Exception as e:
                print(f"跳过损坏图片: {img_name}, 错误: {e}")
                continue

            # 添加图片信息到COCO
            coco["images"].append({
                "id": img_id,
                "file_name": img_name,
                "width": img_w,
                "height": img_h
            })

            # 读取对应YOLO标签
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(label_dir, label_name)

            # 如果标签文件不存在，直接跳过
            if not os.path.exists(label_path):
                img_id += 1
                continue

            # 解析YOLO标签
            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # YOLO格式: class_id x_center y_center width height (归一化)
                cls_id, x_c, y_c, w, h = [float(i) for i in line.split()]
                cls_id = int(cls_id)

                # 转像素坐标（COCO bbox格式: [x_min, y_min, width, height]）
                x_min = (x_c - w / 2) * img_w
                y_min = (y_c - h / 2) * img_h
                bbox_w = w * img_w
                bbox_h = h * img_h

                # 计算标注面积
                area = bbox_w * bbox_h

                # 添加标注信息
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [x_min, y_min, bbox_w, bbox_h],
                    "area": area,
                    "iscrowd": 0,  # 0=普通目标，1= crowd
                    "ignore": 0
                })
                ann_id += 1

            img_id += 1

    # 3. 保存JSON文件
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=4)

    print(f"\n✅ 转换完成！COCO标签已保存至: {output_json}")
    print(f"📊 统计: 图片={img_id}张 | 标注={ann_id}个 | 类别={len(classes)}类")

if __name__ == '__main__':
    yolo2coco(IMG_DIR, LABEL_DIR, OUTPUT_JSON, CLASSES)