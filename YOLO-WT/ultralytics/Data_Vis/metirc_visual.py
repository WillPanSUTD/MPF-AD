import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('TkAgg')

# ---------------------------
# 全局配置（论文级样式）
# ---------------------------
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 160
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0

# ---------------------------
# 基础数据（已删除ΔC，新增Normal Map）
# ---------------------------
# ✅ 方法名称（新增 Normal Map）
methods = [
    'PseudoColor Image',
    'Normal Map',
    'Diffuse component',
    'Specular component',
    'Rendered Image(Ours)'
]

# ✅ 指标数据
metrics_data = {
    'ΔI (Mean Intensity Contrast)': [0.064, 0.038, 0.056, 0.049, 0.051],   # 加了Normal数值
    'SNR (Signal to Noise Ratio)': [0.568, 0.682, 0.577, 1.557, 0.771],    # 加了Normal数值
    'ΔG (Edge Gradient Gain)': [0.750, 3.624, 3.389, 3.407, 4.130],        # 加了Normal数值
}

# ✅ 扩展为5个颜色（对应5个方法）
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']


# ---------------------------
# 绘制单张图表的通用函数
# ---------------------------
def plot_single_metric(metric_name, values):
    fig, ax = plt.subplots(figsize=(8, 7))
    bars = ax.bar(methods, values, color=colors, edgecolor='black', linewidth=0.8, alpha=0.8)

    ax.set_title(metric_name, fontsize=16, fontweight='bold', pad=15)
    ax.tick_params(axis='x', rotation=15, labelsize=15)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_ylim(0, max(values) * 1.1)
    plt.tight_layout()
    plt.show()


# ---------------------------
# 依次绘制 3 张图（ΔI / SNR / ΔG）
# ---------------------------
plot_single_metric('ΔI (Mean Intensity Contrast)', metrics_data['ΔI (Mean Intensity Contrast)'])
plot_single_metric('SNR (Signal to Noise Ratio)', metrics_data['SNR (Signal to Noise Ratio)'])
plot_single_metric('ΔG (Edge Gradient Gain)', metrics_data['ΔG (Edge Gradient Gain)'])