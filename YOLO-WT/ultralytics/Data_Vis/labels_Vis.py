import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 无白边全屏显示
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.bottom'] = False
plt.rcParams['ytick.left'] = False
plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['ytick.labelleft'] = False
plt.rcParams['figure.subplot.top'] = 1.0
plt.rcParams['figure.subplot.bottom'] = 0.0
plt.rcParams['figure.subplot.left'] = 0.0
plt.rcParams['figure.subplot.right'] = 1.0

plt.switch_backend('TkAgg')

# ================= 路径 =================
IMG_DIR = r"/path/to/LUT-AD/EXP/EXP3/DetectionResults/GT/image"
LAB_DIR = r"/path/to/LUT-AD/EXP/EXP3/DetectionResults/GT/label"
# ========================================

# 类别名称
CLASS_NAMES = ["Inveracious solding", "Pinhole", "Pit", "Burst","Fish-scale welding","Bump"]

# YOLO 风格配色
COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]

# 获取所有图片
img_files = [f for f in os.listdir(IMG_DIR) if f.endswith(('.png', '.jpg', '.bmp'))]
np.random.shuffle(img_files)

print("===== Check Labels =====")
print("Press Enter -> next")
print("Type q + Enter -> quit")

for i, fname in enumerate(img_files):
    img_path = os.path.join(IMG_DIR, fname)
    lab_path = os.path.join(LAB_DIR, os.path.splitext(fname)[0] + '.txt')

    # 读图
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # 画标签
    if os.path.exists(lab_path):
        with open(lab_path) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            cls, xc, yc, bw, bh = map(float, line.split())
            cls = int(cls)

            xc = xc * w
            yc = yc * h
            bw = bw * w
            bh = bh * h

            x1 = int(xc - bw / 2)
            y1 = int(yc - bh / 2)
            x2 = int(xc + bw / 2)
            y2 = int(yc + bh / 2)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            color = COLORS[cls % len(COLORS)]
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)

            class_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"cls{cls}"
            label = f"{class_name}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9
            thickness = 2
            text_w, text_h = cv2.getTextSize(label, font, font_scale, thickness)[0]

            cv2.rectangle(img_rgb, (x1, y1 - text_h - 4), (x1 + text_w + 2, y1), color, -1)
            cv2.putText(img_rgb, label, (x1 + 1, y1 - 3), font, font_scale, (255, 255, 255), thickness)

    # ====================== 关键：无白边显示 ======================
    fig = plt.figure(figsize=(8, 8), dpi=100)
    ax = fig.add_subplot(111)
    ax.imshow(img_rgb)
    ax.set_position([0, 0, 1, 1])  # 占满整个窗口
    plt.subplots_adjust(0, 0, 1, 1)
    plt.margins(0, 0)
    plt.show()
    # =============================================================

    cmd = input("Next? (Enter=next, q=quit): ")
    if cmd.lower() == 'q':
        print("Exit")
        break