import numpy as np
import cv2
import os
import warnings
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau

warnings.filterwarnings('ignore')
eps = 1e-8

# ================================
# 配置
# ================================
CONFIG = {
    'patch_size': 16,
    'stride': 4,
    'render_types': ['diff', 'spec', 'full', 'lut'],
}

# ================================
# Patch 工具
# ================================
def create_patch_grid(h, w, patch_size, stride):
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patches.append((x, y, x + patch_size, y + patch_size))
    return patches


def extract_patch_data(img, patch_coords):
    x1, y1, x2, y2 = patch_coords
    return img[y1:y2, x1:x2]


# ================================
# 四邻域法向量
# ================================
def compute_normal_map(depth_tif):
    depth = depth_tif.astype(np.float32)
    dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    dz = np.ones_like(depth)
    norm = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2) + eps
    return np.dstack([dx / norm, dy / norm, dz / norm])


# ================================
# 法向量方向差异（你效果最好的原版）
# ================================
def compute_patch_normal_dir_diff(patch_normal):
    flat = patch_normal.reshape(-1, 3)
    if len(flat) < 2:
        return 0.0

    mean_n = np.mean(flat, axis=0)
    mean_n /= np.linalg.norm(mean_n) + eps
    total = 0.0
    for v in flat:
        v /= np.linalg.norm(v) + eps
        cos = np.clip(np.dot(v, mean_n), -1, 1)
        total += 1.0 - cos
    return total / len(flat)


def generate_normal_diff(depth_img, patch_coords):
    normal = compute_normal_map(depth_img)
    diffs = []
    for coord in patch_coords:
        patch = extract_patch_data(normal, coord)
        diffs.append(compute_patch_normal_dir_diff(patch))
    return np.array(diffs)


# ================================
# 图像对比度
# ================================
def patch_contrast(patch):
    maxv = np.max(patch)
    minv = np.min(patch)
    if maxv - minv < eps:
        return 0.0
    return (maxv - minv) / (maxv + minv + eps)


def generate_contrast(img, patch_coords):
    img = img.astype(np.float32)
    cont = []
    for coord in patch_coords:
        patch = extract_patch_data(img, coord)
        cont.append(patch_contrast(patch))
    return np.array(cont)


# ================================
# 单文件分析（已去掉所有星号）
# ================================
def analyze_file(img_dirs, fname):
    base = os.path.splitext(fname)[0]
    depth_path = os.path.join(img_dirs['depth'], fname)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    if depth is None:
        print(f"❌ {fname} 读取失败")
        return None

    h, w = depth.shape
    patches = create_patch_grid(h, w, CONFIG['patch_size'], CONFIG['stride'])
    normal_diff = generate_normal_diff(depth, patches)

    contrast_dict = {}
    for tp in ['diff', 'spec', 'full', 'lut']:
        p = os.path.join(img_dirs[tp], base + '.bmp')
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is not None:
            if im.shape != (h, w):
                im = cv2.resize(im, (w, h))
            contrast_dict[tp] = generate_contrast(im, patches)
        else:
            contrast_dict[tp] = np.zeros_like(normal_diff)

    print(f"\n📄 {fname}")
    print("-" * 90)
    print(f"{'Type':<6} {'Pearson':<15} {'Spearman':<15} {'Kendall':<15}")
    print("-" * 90)

    rows = []
    for tp in CONFIG['render_types']:
        nd = normal_diff
        ct = contrast_dict[tp]

        mask = (nd > 0) & (ct > 0)
        if np.sum(mask) < 2:
            pe, sp, ke = 0.0, 0.0, 0.0
        else:
            pe, _ = pearsonr(nd[mask], ct[mask])
            sp, _ = spearmanr(nd[mask], ct[mask])
            ke, _ = kendalltau(nd[mask], ct[mask])
            pe = np.nan_to_num(pe)
            sp = np.nan_to_num(sp)
            ke = np.nan_to_num(ke)

        print(f"{tp:<6} {pe:<15.4f} {sp:<15.4f} {ke:<15.4f}")
        rows.append({
            'file': fname, 'type': tp,
            'pearson': round(pe, 4), 'spearman': round(sp, 4), 'kendall': round(ke, 4),
        })
    return rows


# ================================
# 批量运行
# ================================
def batch_run(IMG_DIRS, csv_path):
    depth_files = sorted([f for f in os.listdir(IMG_DIRS['depth']) if f.endswith('.tif')])
    all_rows = []

    print("=" * 70)
    print(f"开始处理 {len(depth_files)} 张图")
    print("=" * 70)

    for f in depth_files:
        r = analyze_file(IMG_DIRS, f)
        if r:
            all_rows.extend(r)

    final_df = pd.DataFrame(all_rows)
    final_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print("\n📊 【最终结果：均值 ± std】")
    print("-" * 90)
    for tp in CONFIG['render_types']:
        sub = final_df[final_df['type'] == tp]

        m_p = sub['pearson'].mean()
        m_s = sub['spearman'].mean()
        m_k = sub['kendall'].mean()

        std_p = sub['pearson'].std()
        std_s = sub['spearman'].std()
        std_k = sub['kendall'].std()

        print(f"{tp:<6} Pearson={m_p:.4f}±{std_p:.4f} | Spearman={m_s:.4f}±{std_s:.4f} | Kendall={m_k:.4f}±{std_k:.4f}")

    print("\n✅ 结果已保存到:", csv_path)


# ================================
# 运行
# ================================
if __name__ == '__main__':
    IMG_DIRS = {
        'depth': '/path/to/LUT-AD/Crop_Data/Tif',
        'diff': '/path/to/LUT-AD/Crop_Data/image/Diffuse',
        'spec': '/path/to/LUT-AD/Crop_Data/image/Specular',
        'full': '/path/to/LUT-AD/Crop_Data/image/Phong',
        'lut': '/path/to/LUT-AD/Crop_Data/image/LUT'
    }

    batch_run(IMG_DIRS, csv_path='/path/to/LUT-AD/correlation_result16_4.csv')