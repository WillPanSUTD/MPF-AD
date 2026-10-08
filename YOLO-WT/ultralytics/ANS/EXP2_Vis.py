import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
from statsmodels.nonparametric.smoothers_lowess import lowess

# ================================
# 仅开放LOWESS拟合参数（仅此修改！）
# ================================
LOWESS_FRAC = 0.1  # 平滑系数
LOWESS_IT = 1  # 迭代次数
LOWESS_DELTA = 0.0  # 加速参数

# ================================
# 全局学术绘图配置（你的原版）
# ================================
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 1.5
eps = 1e-8

# ================================
# 你的原版配置 + 所有函数（完全没动！）
# ================================
CONFIG = {
    'patch_size': 16,
    'stride': 4,
    'render_types': ['diff', 'spec', 'full', 'lut'],
}


def create_patch_grid(h, w, patch_size, stride):
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patches.append((x, y, x + patch_size, y + patch_size))
    return patches


def extract_patch_data(img, patch_coords):
    x1, y1, x2, y2 = patch_coords
    return img[y1:y2, x1:x2]


def compute_normal_map(depth_tif):
    depth = depth_tif.astype(np.float32)
    dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    dz = np.ones_like(depth)
    norm = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2) + eps
    return np.dstack([dx / norm, dy / norm, dz / norm])


def compute_patch_normal_dir_diff(patch_normal):
    flat = patch_normal.reshape(-1, 3)
    if len(flat) < 2:
        return 0.0
    mean_n = np.mean(flat, axis=0)
    mean_n /= np.linalg.norm(mean_n) + eps
    total = 0.0
    for v in flat:
        v /= np.linalg.norm(v) + eps
        cos = np.clip(np.dot(v, mean_n, ), -1, 1)
        total += 1.0 - cos
    return total / len(flat)


def generate_normal_diff_map(depth_img, patches):
    normal = compute_normal_map(depth_img)
    diffs = []
    for coord in patches:
        patch = extract_patch_data(normal, coord)
        diffs.append(compute_patch_normal_dir_diff(patch))
    h, w = depth_img.shape
    ph = (h - CONFIG['patch_size']) // CONFIG['stride'] + 1
    pw = (w - CONFIG['patch_size']) // CONFIG['stride'] + 1
    return np.array(diffs).reshape(ph, pw)


def patch_contrast(patch):
    maxv = np.max(patch)
    minv = np.min(patch)
    if maxv - minv < eps:
        return 0.0
    return (maxv - minv) / (maxv + minv)


def generate_contrast_map(img, patches):
    img = img.astype(np.float32)
    cont = []
    for coord in patches:
        patch = extract_patch_data(img, coord)
        cont.append(patch_contrast(patch))
    h, w = img.shape
    ph = (h - CONFIG['patch_size']) // CONFIG['stride'] + 1
    pw = (w - CONFIG['patch_size']) // CONFIG['stride'] + 1
    return np.array(cont).reshape(ph, pw)


# ================================
# 你的原版主函数（仅修改LOWESS调用处）
# ================================
def visualize_single_image(depth_path, diff_path, spec_path, full_path, lut_path):
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    diff = cv2.imread(diff_path, cv2.IMREAD_GRAYSCALE)
    spec = cv2.imread(spec_path, cv2.IMREAD_GRAYSCALE)
    full = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    lut = cv2.imread(lut_path, cv2.IMREAD_GRAYSCALE)

    if depth is None:
        print("❌ 深度图读取失败")
        return
    h, w = depth.shape

    patches = create_patch_grid(h, w, CONFIG['patch_size'], CONFIG['stride'])
    nd_map = generate_normal_diff_map(depth, patches)
    dc_map = generate_contrast_map(diff, patches)
    sc_map = generate_contrast_map(spec, patches)
    fc_map = generate_contrast_map(full, patches)
    lc_map = generate_contrast_map(lut, patches)

    nd = nd_map.flatten()
    dc = dc_map.flatten()
    sc = sc_map.flatten()
    fc = fc_map.flatten()
    lc = lc_map.flatten()

    results = {}
    name_map = {'diff': 'Diffuse', 'spec': 'Specular', 'full': 'Full', 'lut': 'LUT'}
    for tp, ct in zip(['diff', 'spec', 'full', 'lut'], [dc, sc, fc, lc]):
        mask = (nd > 0) & (ct > 0)
        if np.sum(mask) < 2:
            pe, sp, ke = 0.0, 0.0, 0.0
        else:
            pe, _ = pearsonr(nd[mask], ct[mask])
            sp, _ = spearmanr(nd[mask], ct[mask])
            ke, _ = kendalltau(nd[mask], ct[mask])
        results[tp] = (pe, sp, ke)

    print("\n📊 相关性系数结果")
    print("-" * 80)
    print(f"{'Type':<10} {'Pearson':<15} {'Spearman':<15} {'Kendall':<15}")
    print("-" * 80)
    for tp in CONFIG['render_types']:
        pe, sp, ke = results[tp]
        print(f"{name_map[tp]:<10} {pe:<15.4f} {sp:<15.4f} {ke:<15.4f}")

    heatmap_data = [
        (nd_map, 'Normal Vector Direction Variance'),
        (dc_map, 'Diffuse Contrast'),
        (sc_map, 'Specular Contrast'),
        (fc_map, 'Full Render Contrast'),
        (lc_map, 'LUT Contrast')
    ]
    for data, title in heatmap_data:
        plt.figure(figsize=(8, 8))
        plt.imshow(data, cmap='viridis')
        plt.axis('off')
        plt.colorbar(shrink=0.8)
        plt.tight_layout()
        plt.show()

    scatter_data = [
        (dc, 'Diffuse', results['diff']),
        (sc, 'Specular', results['spec']),
        (fc, 'Full', results['full']),
        (lc, 'LUT', results['lut'])
    ]
    for y_data, title, (pe, sp, ke) in scatter_data:
        plt.figure(figsize=(5, 5))

        plt.scatter(nd, y_data, s=10, c='#8B0000', alpha=1, marker='o', edgecolors='none')

        # ================================
        # 仅这里使用开放的参数（仅此修改！）
        # ================================
        lowess_fit = lowess(y_data, nd, frac=LOWESS_FRAC, it=LOWESS_IT, delta=LOWESS_DELTA)

        plt.plot(lowess_fit[:, 0], lowess_fit[:, 1],
                 c='#00008B', linewidth=2.5, linestyle='-', label='LOWESS Fit')

        plt.xlabel('Normal Vector Direction Dispersion', fontsize=12)
        plt.ylabel('Michelson Contrast', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.2, linewidth=0.5)
        plt.legend(frameon=False, fontsize=10)

        plt.tight_layout()
        plt.show()


# ================================
# 你的原版运行代码
# ================================
if __name__ == '__main__':
    depth_path = "/path/to/LUT-AD/Crop_Data/Tif/130.tif"
    diff_path = "/path/to/LUT-AD/Crop_Data/image/Diffuse/130.bmp"
    spec_path = "/path/to/LUT-AD/Crop_Data/image/Specular/130.bmp"
    full_path = "/path/to/LUT-AD/Crop_Data/image/Phong/130.bmp"
    lut_path = "/path/to/LUT-AD/Crop_Data/image/LUT/130.bmp"

    visualize_single_image(depth_path, diff_path, spec_path, full_path, lut_path)