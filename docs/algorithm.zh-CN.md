# 算法详解

本文档对仓库中两个阶段给出自洽的数学描述。Stage 1 是 `Depth-Normal_Rendering/`
下的深度-法向引导光照渲染器；Stage 2 是 `YOLO-WT/ultralytics/` 下的 YOLO-WaveletTiny
检测器。

[ [English](algorithm.md) ]

---

## Stage 1 — 深度-法向引导光照渲染

渲染器输入单通道 32-bit float 深度图 $D(x, y) \in \mathbb{R}$
（哨兵值 `< -1e30` 表示无效像素），输出 BGR 图像
$C'(x, y) \in [0, 255]^3$，把全局 + 局部几何信息编码为光度对比。

### 1.1 基于统计先验的深度→伪彩色（LUT）

实现：[`LUT.cpp`](../Depth-Normal_Rendering/LUT.cpp)

**目标**：把焊缝表面的全局深度层级稳定地嵌入 2D 彩色图，同时保留缺陷造成的微小相对深度差。

**自适应 bin 宽**：基于深度有效值的四分位距 $\mathrm{IQR} = Q_3 - Q_1$ 的 Freedman–Diaconis 规则：

$$
h \;=\; \frac{2\,\mathrm{IQR}}{N^{1/3}},
\qquad
N_k \;=\; \left\lceil \frac{d_{\max} - d_{\min}}{h} \right\rceil.
$$

当 $\mathrm{IQR} < 10^{-6}$（深度图近似为平面），退化到 $N_k = 1$；
$Q_1$、$Q_3$ 在排序后的有效值向量上线性插值得到。

**直方图与累计分布**：

$$
H(k) \;=\; \sum_{x,y}
\mathbb{I}\!\bigl(\,D(x, y) \in [d_{\min} + (k-1)h,\; d_{\min} + kh]\bigr),
\quad
\mathrm{CDF}(k) \;=\; \frac{1}{N}\sum_{a=1}^{k} H(a).
$$

**查找表**：一次性构造 256 项 BGR 渐变 LUT（蓝→绿→红，分段线性）：

$$
\mathrm{LUT}[i] \;=\;
\begin{cases}
\bigl(\,255(1-2t),\; 255 \cdot 2t,\; 0\bigr) & t < 0.5,\\[2pt]
\bigl(\,0,\; 255(1-2(t - 0.5)),\; 255\cdot 2(t-0.5)\bigr) & t \ge 0.5,
\end{cases}
\qquad t = i/255.
$$

**逐像素映射**：实现上，工程代码先把有效深度做 min–max 归一化到 $[0,255]$ 再查表；
这等价于对均匀分 bin 的直方图取
$i(x,y) = \lfloor \mathrm{CDF}(k(x,y)) \cdot 255 \rfloor$。两种实现得到相同的输出：

$$
C(x, y) \;=\; \mathrm{LUT}\bigl[i(x, y)\bigr]\;\in\; [0,255]^3,
$$

无效像素置黑。

**为何要这样做**：亚毫米级的缺陷高度差很容易被相邻整数值吃掉；
数据自适应 bin 宽保留它们的相对顺序。同时确定性 LUT（不学习）让编码方式跨样本可解释。

### 1.2 表面法向估计

实现：[`Phong.cpp::computeNormals`](../Depth-Normal_Rendering/Phong.cpp)

对每个有效像素 $P(x, y) = (x, y, d(x, y))$，从 8-邻域（或 4-邻域，可切换）取 3D 点。
为了避免像素偏移（每步 1 px）和深度量纲（毫米）之间的尺度失配，对 $z$ 自动缩放：

$$
s \;=\; \frac{512}{d_{\max} - d_{\min}},
\qquad
\bigl(\Delta x,\; \Delta y,\; s \cdot \Delta d\bigr),
$$

让三个坐标分量量级一致。法向估计采用以下回退策略：

| 邻域有效点数 | 方法 |
|--------------|------|
| ≥ 3          | 局部 3D 邻域 PCA：协方差矩阵的最小特征值对应特征向量取作法向 |
| = 2          | 两个相对向量的叉乘 $\vec{v}_1 \times \vec{v}_2$ |
| ≤ 1          | 取邻域内"深度差最大"的有效像素的法向（搜索半径 1） |

法向最后归一化并强制朝向相机（$n_z \ge 0$）。给*每个*有效像素都得到稳定法向（包括边界像素）非常重要，因为
Phong 计算依赖 $\vec{N} \cdot \vec{L}$，对法向的奇异点很敏感。

### 1.3 Phong 光照

实现：[`Phong.cpp::renderPhong`](../Depth-Normal_Rendering/Phong.cpp)

给定逐像素单位法向 $\vec{N}(x, y)$、归一化光源方向 $\vec{L}$、归一化视点方向 $\vec{V}$，计算 Phong 三分量：

$$
\boxed{
\begin{aligned}
I_a &= w_a I_l, \\
I_d &= w_d I_l \,\max(\vec{N}\cdot\vec{L},\,0), \\
I_s &= w_s I_l \,\max(\vec{R}\cdot\vec{V},\,0)^{\,n},
\end{aligned}}
\qquad
\vec{R} \;=\; 2(\vec{N}\!\cdot\!\vec{L})\,\vec{N} - \vec{L}.
$$

合成像素强度截断到 $[0, 1]$：

$$
I(x, y) \;=\; \min\bigl(1,\; I_a + I_d + I_s\bigr),
$$

并以此对伪彩色图作乘性调制：

$$
\boxed{\;C'(x, y) \;=\; I(x, y) \cdot C(x, y)\;}.
$$

无效像素（无法向，或外部传入掩码标注）置黑。
由于 $I$ 显式依赖 $\vec{N}\cdot\vec{L}$，微小缺陷处的法向小扰动会放大成显著的光度差——
这正是论文所依赖的"几何到光度的非线性增强"。

**论文使用的默认参数**：

| 数学符号 | 代码变量 | 取值 |
|----------|----------|------|
| $\vec{L}$ | `light`        | `(500, 150, 1500)` |
| $\vec{V}$ | `view`         | `(500, 150, 1500)` |
| $I_l$     | `lightIntensity` | `1.3` |
| $w_a$     | `ambient`      | `0.1` |
| $w_d$     | `kd`           | `0.5` |
| $w_s$     | `ks`           | `0.3` |
| $n$       | `shine`        | `32`  |

---

## Stage 2 — YOLO-WaveletTiny

YOLO-WT 是基于 YOLOv11n 的衍生网络，在两个位置加入了小波感知模块；
完整 backbone/neck 结构见
[`cfg/models/YOLO-WT/YOLO-WT.yaml`](../YOLO-WT/ultralytics/cfg/models/YOLO-WT/YOLO-WT.yaml)。
scale `n` 配置下参数量 **2.47 M**、计算量 **5.7 GFLOPs**。

### 2.1 WDSConv —— 小波深度可分离卷积

实现：[`AddModules/WDSConv.py`](../YOLO-WT/ultralytics/nn/AddModules/WDSConv.py)；
在 P3/8 与 P4/16 两次降采样位置使用。

WDSConv 保留深度可分离形式（depthwise → pointwise），但 depthwise 核被换成
3 级 Haar 小波块（`WTConv`）。

**单级分解**：4 个正交 2×2 Haar 滤波器：

$$
f_{LL} = \tfrac{1}{2}\!\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix},
\quad
f_{LH} = \tfrac{1}{2}\!\begin{pmatrix} 1 & -1 \\ 1 & -1 \end{pmatrix},
\quad
f_{HL} = \tfrac{1}{2}\!\begin{pmatrix} 1 & 1 \\ -1 & -1 \end{pmatrix},
\quad
f_{HH} = \tfrac{1}{2}\!\begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}.
$$

小波变换实现为带步长的分组卷积（`groups = C`，stride 2）：

$$
\bigl[X_{LL}, X_{LH}, X_{HL}, X_{HH}\bigr] \;=\; \mathrm{WT}(X).
$$

每个子带再过一次 $K\times K$ depthwise 卷积，随后乘上一个可学习的逐通道尺度
（`_ScaleModule`，高频子带初值 $0.15$）。重建利用 Haar 基的正交性，
直接用分组转置卷积实现逆小波变换（IWT）。整个流程递归 3 级，
让感受野按几何级数扩张但参数量保持很低。

WDSConv 把小波分支与并行的"基础 depthwise 卷积"分支相加，再过一次 $1{\times}1$ pointwise：

$$
Y \;=\; \mathrm{PWConv}\bigl(\,\sigma_{\text{base}}(W_{\text{base}} * X) + \mathrm{IWT}(\cdot)\,\bigr).
$$

效果上：同一卷积同时扩大感受野（通过小波分解）和分离高低频特征——
高频子带恰好承载裂纹、针孔、凹坑边缘这些缺陷特征。

### 2.2 IWUpsample —— 逆小波上采样

实现：[`AddModules/IWUpSample.py`](../YOLO-WT/ultralytics/nn/AddModules/IWUpSample.py)；
替换 YOLOv11n neck 中的两次 `nn.Upsample`。

常规双线性 / 最近邻上采样会糊掉 WDSConv 刚刚分离出的高频成分；
IWUpsample 用逆小波变换替代，把高频还原回更高分辨率。

输入 $X \in \mathbb{R}^{B \times C \times H \times W}$，通道维已编码了 $J$ 级小波子带
（$C = 4^J \cdot C_0$）。`_split_coeffs` 把它切成

$$
y_L \in \mathbb{R}^{B\times C_0 \times H\times W},\qquad
\{y_H^{(j)}\}_{j=1}^{J},\quad y_H^{(j)}\in\mathbb{R}^{B\times C_0\times 3\times H\times W},
$$

即 1 个低频分量 + 每级 3 个高频子带。然后递归

$$
\mathit{LL}_{j} \;=\; \mathrm{IWT}\!\bigl(\mathit{LL}_{j+1},\,\mathit{LH}_{j+1},\,\mathit{HL}_{j+1},\,\mathit{HH}_{j+1}\bigr),
$$

每级让空间分辨率 ×2，直到恢复目标尺寸。最后用一层 $1{\times}1$ Conv-BN-ReLU
把通道数对齐到 YAML 指定的值。

由于 Haar IWT 正是 WDSConv 中所用 WT 的伴随算子，
这个上采样在小波意义下保持信息——微小缺陷纹理能不被低通平滑地传到检测头。

### 2.3 整体连接

YOLO-WT 的 YAML 在 backbone 的两次 stride-2 降采样位（P3/8、P4/16）插入 WDSConv，
neck 中的两次上采样替换为 IWUpsample（通道 256 → 128 → 输出）；
其他模块（`C3k2`、`SPPF`、`C2PSA`、3 个 `Detect` 头）继承 YOLOv11n。
训练用 ultralytics 默认的 SGD、250 epoch、batch 16、输入 640、不加预训练权重，
配置和 baseline 完全一致，以保证模态消融与模块消融的可比性。

---

## 参考

Finder, S. E., Amoyal, R., Treister, E., & Freifeld, O.
*Wavelet Convolutions for Large Receptive Fields.* ECCV 2024.
([arXiv:2407.05848](https://arxiv.org/abs/2407.05848))，
WDSConv 改造的小波卷积思路源于此文。
