import os
import pandas as pd
import matplotlib.pyplot as plt

# 设置后端和字体
plt.switch_backend('TkAgg')
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# ====================== 【6种方法路径】填写你的6个CSV路径 ======================
results_files = [
    'Phong_Exp/train/DepthMap-250-16-640-SGD/results.csv',
    'Phong_Exp/train/LUT-250-16-640-SGD/results.csv',
    'Phong_Exp/train/Normal-250-16-640-SGD/results.csv',
    'Phong_Exp/train/Diffuse-250-16-640-SGD/results.csv',
    'Phong_Exp/train/Specular-250-16-640-SGD/results.csv',
    'Phong_Exp/train/Phong-250-16-640-SGD/results.csv',
]

# ====================== 6种方法名称（固定顺序） ======================
custom_labels = [
    'Depth Map',
    'Pseudo-color Image',
    'Normal Map',
    'Diffuse Component',
    'Specular Component',
    'Rendered Image(Ours)',
]

# ====================== 6种颜色（论文美观配色） ======================
custom_colors = [
    '#1f77b4',   # 蓝
    '#ff7f0e',   # 橙
    '#2ca02c',   # 绿
    '#9467bd',   # 紫
    '#8c564b',   # 棕
    '#d62728',   # 红（Ours）
]


def moving_average(data, window_size=5):
    window_size = min(window_size, len(data))
    smoothed_data = data.rolling(window=window_size, center=True, min_periods=1).mean()
    return smoothed_data


def plot_metric_comparison(metric_key, metric_label, custom_labels, custom_colors, smooth_window=5):
    plt.figure(figsize=(10, 7))

    for file_path, custom_label, color in zip(results_files, custom_labels, custom_colors):
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue

        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()

        if 'epoch' not in df.columns or metric_key not in df.columns:
            print(f"列缺失: {file_path}")
            continue

        smoothed_data = moving_average(df[metric_key], window_size=smooth_window)
        plt.plot(df['epoch'], smoothed_data, label=f'{custom_label}', color=color, linewidth=2)

    plt.title(f'{metric_label}', fontsize=18, fontweight='bold')
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel(metric_label, fontsize=18)

    plt.legend(
        loc='lower right',
        frameon=True,
        fontsize=14,
        fancybox=True
    )

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    metrics = [
        ('metrics/precision(B)', 'Precision'),
        ('metrics/recall(B)', 'Recall'),
        ('metrics/mAP50(B)', 'mAP@50'),
        ('metrics/mAP50-95(B)', 'mAP@50-95')
    ]

    SMOOTH_WINDOW = 5

    for metric, label in metrics:
        plot_metric_comparison(metric, label, custom_labels, custom_colors, smooth_window=SMOOTH_WINDOW)