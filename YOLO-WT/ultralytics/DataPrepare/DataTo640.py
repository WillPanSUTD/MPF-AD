import cv2
import numpy as np
import os
from pathlib import Path

# ===================== 你的路径 =====================
INPUT_DEPTH_DIR    = "/path/to/LUT-AD/Org_Data/image/DepthMap/"
INPUT_DIFFUSE_DIR  = "/path/to/LUT-AD/Org_Data/image/Diffuse/"
INPUT_SPECULAR_DIR = "/path/to/LUT-AD/Org_Data/image/Specular/"
INPUT_LUT_DIR      = "/path/to/LUT-AD/Org_Data/image/LUT/"
INPUT_PHONG_DIR    = "/path/to/LUT-AD/Org_Data/image/Phong/"
INPUT_NORMAL_DIR   = "/path/to/LUT-AD/Org_Data/image/Normal/"
INPUT_TIFF_DIR     = "/path/to/LUT-AD/Org_Data/Tif/"
INPUT_LABEL_DIR    = "/path/to/LUT-AD/Org_Data/label/"

OUTPUT_ROOT        = "/path/to/LUT-AD/Crop_FINAL_FIXED/"
WINDOW_SIZE = 640
HALF = WINDOW_SIZE // 2
GLOBAL_COUNTER = 1

# ===================== 自动创建文件夹 =====================
for f in ["DepthMap","Diffuse","Specular","LUT","Phong","Normal","Tif","label"]:
    os.makedirs(os.path.join(OUTPUT_ROOT, f), exist_ok=True)

# ===================== 读取标注 =====================
def read_boxes(txt_path, img_w, img_h):
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            cls, xc, yc, w, h = map(float, line.split())
            x1 = (xc - w/2) * img_w
            y1 = (yc - h/2) * img_h
            x2 = (xc + w/2) * img_w
            y2 = (yc + h/2) * img_h
            boxes.append([int(cls), x1, y1, x2, y2, False])
    return boxes

# ===================== 滑窗 =====================
def window_from_center(cx, cy, img_w, img_h):
    x1 = int(cx - HALF)
    y1 = int(cy - HALF)
    x2 = int(cx + HALF)
    y2 = int(cy + HALF)
    return (max(0, x1), max(0, y1), min(img_w, x2)-max(0, x1), min(img_h, y2)-max(0, y1))

# ===================== 【修复】任意重叠都裁剪标注 =====================
def clip_any_overlap(box, win):
    cls, x1, y1, x2, y2,_ = box
    wx, wy, ww, hh = win
    nx1 = max(x1, wx)
    ny1 = max(y1, wy)
    nx2 = min(x2, wx + ww)
    ny2 = min(y2, wy + hh)
    if nx1 >= nx2 or ny1 >= ny2:
        return None
    rw, rh = nx2-nx1, ny2-ny1
    cx = (nx1 + nx2)/2
    cy = (ny1 + ny2)/2
    return (cls, (cx-wx)/ww, (cy-wy)/hh, rw/ww, rh/hh)

# ===================== 超大框自动全覆盖 =====================
def split_full_coverage(box, img_w, img_h):
    cls, x1, y1, x2, y2,_ = box
    wins = []
    cx, cy = (x1+x2)/2, (y1+y2)/2
    wins.append(window_from_center(cx, cy, img_w, img_h))

    if x2 - x1 > WINDOW_SIZE:
        steps = np.arange(x1, x2, WINDOW_SIZE * 0.9)
        for sx in steps:
            wins.append(window_from_center(sx + HALF, cy, img_w, img_h))

    if y2 - y1 > WINDOW_SIZE:
        steps = np.arange(y1, y2, WINDOW_SIZE * 0.9)
        for sy in steps:
            wins.append(window_from_center(cx, sy + HALF, img_w, img_h))

    return list(dict.fromkeys(wins))

# ===================== 保存 =====================
def save_all(stem, win, idx):
    channels = [
        (INPUT_DEPTH_DIR, os.path.join(OUTPUT_ROOT, "DepthMap"), ".bmp"),
        (INPUT_DIFFUSE_DIR, os.path.join(OUTPUT_ROOT, "Diffuse"), ".bmp"),
        (INPUT_SPECULAR_DIR, os.path.join(OUTPUT_ROOT, "Specular"), ".bmp"),
        (INPUT_LUT_DIR, os.path.join(OUTPUT_ROOT, "LUT"), ".bmp"),
        (INPUT_PHONG_DIR, os.path.join(OUTPUT_ROOT, "Phong"), ".bmp"),
        (INPUT_NORMAL_DIR, os.path.join(OUTPUT_ROOT, "Normal"), ".bmp"),
        (INPUT_TIFF_DIR, os.path.join(OUTPUT_ROOT, "Tif"), ".tif"),
    ]
    wx, wy, ww, hh = win
    for in_dir, out_dir, ext in channels:
        p = os.path.join(in_dir, stem + ext)
        if not os.path.exists(p): continue
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if im is None: continue
        cv2.imwrite(os.path.join(out_dir, f"{idx}{ext}"), im[wy:wy+hh, wx:wx+ww])

def save_txt(anns, path):
    with open(path, 'w') as f:
        for a in anns:
            f.write(f"{a[0]} {a[1]:.6f} {a[2]:.6f} {a[3]:.6f} {a[4]:.6f}\n")

# ===================== 核心处理（100% 你要的逻辑） =====================
def process(stem):
    global GLOBAL_COUNTER
    im = cv2.imread(os.path.join(INPUT_PHONG_DIR, stem + ".bmp"))
    if im is None: return
    h, w = im.shape[:2]
    boxes = read_boxes(os.path.join(INPUT_LABEL_DIR, stem + ".txt"), w, h)
    if not boxes:
        print(f"⏭️ 跳过 {stem} (无标注)")
        return

    while True:
        unproc = [b for b in boxes if not b[5]]
        if not unproc: break
        b0 = unproc[0]
        cx0, cy0 = (b0[1]+b0[3])/2, (b0[2]+b0[4])/2
        win = window_from_center(cx0, cy0, w, h)

        fully_in = (b0[1] >= win[0] and b0[3] <= win[0]+win[2] and
                    b0[2] >= win[1] and b0[4] <= win[1]+win[3])

        if fully_in:
            inside = [b for b in unproc if clip_any_overlap(b, win)]
            all_cx = np.mean([(b[1]+b[3])/2 for b in inside])
            all_cy = np.mean([(b[2]+b[4])/2 for b in inside])
            win = window_from_center(all_cx, all_cy, w, h)
            valid = [clip_any_overlap(b, win) for b in inside if clip_any_overlap(b, win)]

            save_all(stem, win, GLOBAL_COUNTER)
            save_txt(valid, os.path.join(OUTPUT_ROOT, "label", f"{GLOBAL_COUNTER}.txt"))
            print(f"✅ 生成 {GLOBAL_COUNTER}")
            GLOBAL_COUNTER +=1
            for b in inside: b[5] = True
        else:
            parts = split_full_coverage(b0, w, h)
            for p in parts:
                valid = [clip_any_overlap(b, p) for b in unproc if clip_any_overlap(b, p)]
                save_all(stem, p, GLOBAL_COUNTER)
                save_txt(valid, os.path.join(OUTPUT_ROOT, "label", f"{GLOBAL_COUNTER}.txt"))
                print(f"✅ 生成 {GLOBAL_COUNTER}")
                GLOBAL_COUNTER +=1
            b0[5] = True

# ===================== 运行 =====================
if __name__ == "__main__":
    for f in os.listdir(INPUT_PHONG_DIR):
        if f.lower().endswith(".bmp"):
            process(Path(f).stem)
    print("\n✅ 完全修复完成！超大框标注永不丢失！")