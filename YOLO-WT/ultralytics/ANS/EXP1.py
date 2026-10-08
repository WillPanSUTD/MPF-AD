import cv2
import numpy as np
import os

eps = 1e-8
zero_threshold = 0.001

# ---------------------------
# 工具：读取 YOLO 标签
# ---------------------------
def get_bboxes(label_path, img_w, img_h):
    bboxes = []
    try:
        if not os.path.exists(label_path):
            return bboxes
        with open(label_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, xc, yc, w, h = parts
            xc = float(xc) * img_w
            yc = float(yc) * img_h
            w = float(w) * img_w
            h = float(h) * img_h
            x1 = int(np.clip(xc - w / 2, 0, img_w - 1))
            y1 = int(np.clip(yc - h / 2, 0, img_h - 1))
            x2 = int(np.clip(xc + w / 2, 0, img_w - 1))
            y2 = int(np.clip(yc + h / 2, 0, img_h - 1))
            if (x2 - x1) * (y2 - y1) >= 4:
                bboxes.append((x1, y1, x2, y2))
    except Exception as e:
        print(f"读取标签失败：{e}")
    return bboxes


# ---------------------------
# 梯度计算
# ---------------------------
def calc_robust_grad(im):
    im_float = im.astype(np.float32) / 255.0
    grad_x = cv2.Sobel(im_float, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(im_float, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(np.square(grad_x) + np.square(grad_y))
    grad_mean = np.mean(grad_mag)
    grad_mean = grad_mean if grad_mean > eps else 1.0
    grad_mag = grad_mag / grad_mean
    grad_mag = np.clip(grad_mag, 0, 10)
    return grad_mag


# ---------------------------
# 指标计算（已删除 ΔC）
# ---------------------------
def compute_metrics(img, depth_img, defect_bboxes, single_defect_box):
    x1, y1, x2, y2 = single_defect_box
    defect_h, defect_w = y2 - y1, x2 - x1
    if defect_h * defect_w < 4:
        return 0.0, 0.0, 0.0

    img_float = img.astype(np.float32) / 255.0
    defect_region = img_float[y1:y2, x1:x2]
    mu_d = np.mean(defect_region)

    bg_mask = np.ones_like(img_float, dtype=bool)
    bg_mask[img_float < 0.01] = False
    for (bx1, by1, bx2, by2) in defect_bboxes:
        bg_mask[by1:by2, bx1:bx2] = False
    bg_pixels = img_float[bg_mask].flatten()

    if len(bg_pixels) < 10:
        mu_b, sigma_b = 0.5, 0.1
    else:
        mu_b = np.mean(bg_pixels)
        sigma_b_raw = np.std(bg_pixels)
        bg_filtered = bg_pixels[np.abs(bg_pixels - mu_b) < 2 * sigma_b_raw]
        sigma_b = np.std(bg_filtered) if len(bg_filtered) > 0 else sigma_b_raw

    delta = abs(mu_d - mu_b)
    snr = delta / (sigma_b + eps)
    snr = np.clip(snr, 0, 20)

    grad_img = calc_robust_grad(img)
    grad_depth = calc_robust_grad(depth_img)
    g_img = np.median(grad_img[y1:y2, x1:x2])
    g_depth = np.median(grad_depth[y1:y2, x1:x2])
    G = g_img / (g_depth + eps)
    G = np.clip(G, 0.001, 100)

    return delta, snr, G


# ---------------------------
# 评估主函数
# ---------------------------
def evaluate(img_dirs, label_dir, prefix=""):
    # ✅ 顺序：diff → spec → lut → normal → Phong(ours)
    clean_dataset_metrics = {
        'diff': [], 'spec': [], 'lut': [], 'normal': [], 'phong': []
    }
    total_img_count = 0
    valid_img_count = 0
    clean_img_count = 0
    excluded_img_list = []

    files = [f for f in os.listdir(img_dirs['depth']) if f.endswith('.bmp')]
    total_img_count = len(files)

    for f in files:
        depth = cv2.imread(os.path.join(img_dirs['depth'], f), cv2.IMREAD_GRAYSCALE)
        diff = cv2.imread(os.path.join(img_dirs['diff'], f), cv2.IMREAD_GRAYSCALE)
        spec = cv2.imread(os.path.join(img_dirs['spec'], f), cv2.IMREAD_GRAYSCALE)
        lut = cv2.imread(os.path.join(img_dirs['lut'], f), cv2.IMREAD_GRAYSCALE)
        normal = cv2.imread(os.path.join(img_dirs['normal'], f), cv2.IMREAD_GRAYSCALE)
        phong = cv2.imread(os.path.join(img_dirs['phong'], f), cv2.IMREAD_GRAYSCALE)

        if any([img is None for img in [depth, diff, spec, lut, normal, phong]]):
            print(f"{prefix}跳过 {f}：图片加载失败")
            continue

        h, w = depth.shape[:2]
        defect_bboxes = get_bboxes(os.path.join(label_dir, f.replace('.bmp', '.txt')), w, h)
        if not defect_bboxes:
            print(f"{prefix}跳过 {f}：无有效缺陷框")
            continue

        valid_img_count += 1
        diff_metrics = []
        spec_metrics = []
        lut_metrics = []
        normal_metrics = []  # ✅ 新增
        phong_metrics = []
        img_has_zero = False

        for single_box in defect_bboxes:
            d1, s1, g1 = compute_metrics(diff, depth, defect_bboxes, single_box)
            d2, s2, g2 = compute_metrics(spec, depth, defect_bboxes, single_box)
            d3, s3, g3 = compute_metrics(lut, depth, defect_bboxes, single_box)
            d4, s4, g4 = compute_metrics(normal, depth, defect_bboxes, single_box)  # ✅ 新增
            d5, s5, g5 = compute_metrics(phong, depth, defect_bboxes, single_box)

            if d1 < zero_threshold or g1 < zero_threshold: img_has_zero = True
            if d2 < zero_threshold or g2 < zero_threshold: img_has_zero = True
            if d3 < zero_threshold or g3 < zero_threshold: img_has_zero = True
            if d4 < zero_threshold or g4 < zero_threshold: img_has_zero = True  # ✅ 新增
            if d5 < zero_threshold or g5 < zero_threshold: img_has_zero = True

            if d1 >= zero_threshold and g1 >= zero_threshold: diff_metrics.append((d1, s1, g1))
            if d2 >= zero_threshold and g2 >= zero_threshold: spec_metrics.append((d2, s2, g2))
            if d3 >= zero_threshold and g3 >= zero_threshold: lut_metrics.append((d3, s3, g3))
            if d4 >= zero_threshold and g4 >= zero_threshold: normal_metrics.append((d4, s4, g4))  # ✅ 新增
            if d5 >= zero_threshold and g5 >= zero_threshold: phong_metrics.append((d5, s5, g5))

        if img_has_zero:
            excluded_img_list.append(f)
            print(f"{prefix}❌ 剔除 {f}：含0值指标")
            continue

        if any([diff_metrics, spec_metrics, lut_metrics, normal_metrics, phong_metrics]):
            clean_img_count += 1
            if diff_metrics: clean_dataset_metrics['diff'].append((np.mean([x[0] for x in diff_metrics]), np.mean([x[1] for x in diff_metrics]), np.mean([x[2] for x in diff_metrics])))
            if spec_metrics: clean_dataset_metrics['spec'].append((np.mean([x[0] for x in spec_metrics]), np.mean([x[1] for x in spec_metrics]), np.mean([x[2] for x in spec_metrics])))
            if lut_metrics: clean_dataset_metrics['lut'].append((np.mean([x[0] for x in lut_metrics]), np.mean([x[1] for x in lut_metrics]), np.mean([x[2] for x in lut_metrics])))
            if normal_metrics: clean_dataset_metrics['normal'].append((np.mean([x[0] for x in normal_metrics]), np.mean([x[1] for x in normal_metrics]), np.mean([x[2] for x in normal_metrics])))  # ✅ 新增
            if phong_metrics: clean_dataset_metrics['phong'].append((np.mean([x[0] for x in phong_metrics]), np.mean([x[1] for x in phong_metrics]), np.mean([x[2] for x in phong_metrics])))

        # 输出顺序
        print(f"{prefix}=== {f} 单图平均指标 ===")
        if diff_metrics:
            avg_d = np.mean([x[0] for x in diff_metrics])
            avg_s = np.mean([x[1] for x in diff_metrics])
            avg_g = np.mean([x[2] for x in diff_metrics])
            print(f"{prefix}diffuse     | ΔI={avg_d:.3f} SNR={avg_s:.3f} G={avg_g:.3f}")
        if spec_metrics:
            avg_d = np.mean([x[0] for x in spec_metrics])
            avg_s = np.mean([x[1] for x in spec_metrics])
            avg_g = np.mean([x[2] for x in spec_metrics])
            print(f"{prefix}specular    | ΔI={avg_d:.3f} SNR={avg_s:.3f} G={avg_g:.3f}")
        if lut_metrics:
            avg_d = np.mean([x[0] for x in lut_metrics])
            avg_s = np.mean([x[1] for x in lut_metrics])
            avg_g = np.mean([x[2] for x in lut_metrics])
            print(f"{prefix}pseudocolor | ΔI={avg_d:.3f} SNR={avg_s:.3f} G={avg_g:.3f}")
        if normal_metrics:  # ✅ 新增
            avg_d = np.mean([x[0] for x in normal_metrics])
            avg_s = np.mean([x[1] for x in normal_metrics])
            avg_g = np.mean([x[2] for x in normal_metrics])
            print(f"{prefix}normal      | ΔI={avg_d:.3f} SNR={avg_s:.3f} G={avg_g:.3f}")
        if phong_metrics:
            avg_d = np.mean([x[0] for x in phong_metrics])
            avg_s = np.mean([x[1] for x in phong_metrics])
            avg_g = np.mean([x[2] for x in phong_metrics])
            print(f"{prefix}Phong (ours)| ΔI={avg_d:.3f} SNR={avg_s:.3f} G={avg_g:.3f}")
        print()

    print("=" * 80)
    print(f"{prefix}=== 数据集筛选统计 ===")
    print(f"{prefix}原始图片总数：{total_img_count}")
    print(f"{prefix}有效图片数：{valid_img_count}")
    print(f"{prefix}被剔除图片数：{len(excluded_img_list)}")
    print(f"{prefix}纯有效图片数：{clean_img_count}")
    print("=" * 80)

    print(f"{prefix}=== 剔除0值后 - 数据集全局平均指标 ===")
    print("=" * 80)
    mode_names = {
        'diff': 'diffuse',
        'spec': 'specular',
        'lut': 'pseudocolor',
        'normal': 'normal',
        'phong': 'Phong (ours)'
    }
    for mode in ['diff', 'spec', 'lut', 'normal', 'phong']:
        if not clean_dataset_metrics[mode]:
            print(f"{prefix}{mode_names[mode]:<11} | 无有效数据")
            continue
        total_d = np.mean([x[0] for x in clean_dataset_metrics[mode]])
        total_s = np.mean([x[1] for x in clean_dataset_metrics[mode]])
        total_g = np.mean([x[2] for x in clean_dataset_metrics[mode]])
        print(f"{prefix}{mode_names[mode]:<11} | ΔI={total_d:.3f} SNR={total_s:.3f} G={total_g:.3f}")
    print("=" * 80)


# ---------------------------
# 主程序
# ---------------------------
if __name__ == '__main__':
    print("=" * 50)
    print("【640裁剪图】评估开始（已添加 normal 输出）")
    print("=" * 50)
    evaluate({
        'depth': '/path/to/LUT-AD/Crop_Data/image/DepthMap',
        'diff': '/path/to/LUT-AD/Crop_Data/image/Diffuse',
        'spec': '/path/to/LUT-AD/Crop_Data/image/Specular',
        'lut': '/path/to/LUT-AD/Crop_Data/image/LUT',
        'normal': '/path/to/LUT-AD/Crop_Data/image/Normal',
        'phong': '/path/to/LUT-AD/Crop_Data/image/Phong'
    }, '/path/to/LUT-AD/Crop_Data/label', prefix="【640裁剪】")