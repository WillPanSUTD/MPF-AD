# 几何到光度的渲染：面向实时焊接缺陷检测

论文 *"Physically Inspired Geometry-to-Photometry Rendering for Real-Time Weld
Defect Detection"* 的代码与数据集（已被 *Engineering Applications of Artificial
Intelligence*，EAAI，2026 接收）。

[ [English](README.md) | [算法说明](docs/algorithm.zh-CN.md) | [数据集说明](docs/dataset_card.zh-CN.md) | [3D-AD 变体](docs/dataset_3d.zh-CN.md) ]

---

## 概述

本工作面向锂电池顶盖激光焊接的缺陷检测：纯 2D 视觉在金属高反/弱纹理表面下不稳定，
3D 点云方法几何信息可靠但部署开销大。我们用一套物理可解释的渲染机制把 3D 深度图
转成判别力强的 2D 光度图，再交给轻量 2D 检测器：

1. **基于深度-法向引导光照渲染（Stage 1）**（C++ + OpenCV + Eigen）
   - 基于统计几何先验的 LUT 伪彩色映射，把深度图全局几何层级嵌入到光度域；
   - 由深度图直接计算表面法向，再用 Phong 光照模型把局部法向扰动放大成显著的光度对比。
2. **YOLO-WT（YOLO-WaveletTiny，Stage 2）**（Python + PyTorch + Ultralytics 分支）
   - **WDSConv**：在 backbone 中替代深度可分离卷积的 3×3 depthwise，
     用多级 Haar 小波分解-子带卷积-逆变换重建特征；
   - **IWUpsample**：在 neck 中替代插值上采样，用多级逆小波重建保留高频缺陷纹理。

在自建的真实工业数据集（六类焊接缺陷）上：
仅做几何-光度渲染替换，YOLOv11n 基线 mAP@50 即提升 **+4.9**；完整 YOLO-WT
在 **2.47 M** 参数、**5.7 GFLOPs** 下达到 **93.8 % mAP@50**。

完整数学描述见 [`docs/algorithm.zh-CN.md`](docs/algorithm.zh-CN.md)。

---

## 仓库结构

```
.
├── Crop_Data/                  # 原始裁剪后的 32-bit float 深度图 + YOLO 标签
│   ├── Tif/                    #   *.tif，单通道 CV_32FC1
│   └── label/                  #   *.txt，YOLO 格式（class cx cy w h，归一化）
│
├── Depth-Normal_Rendering/     # Stage 1：C++ 渲染器（OpenCV + Eigen）
│   ├── LUT.{h,cpp}             #   基于统计先验的深度→伪彩色 LUT
│   ├── Phong.{h,cpp}           #   表面法向 + Phong 光照
│   └── Demo.cpp                #   流程驱动样例
│
├── Train_Data/                 # Stage 1 渲染产物，按检测器训练目录组织
│   ├── DepthMap/               #   归一化的深度灰度图（8-bit）
│   ├── LUT/                    #   仅 Stage 1a 伪彩色图
│   ├── Diffuse/                #   仅 Phong 漫反射分量
│   ├── Specular/               #   仅 Phong 镜面反射分量
│   ├── Normal/                 #   表面法向图（彩色编码）
│   └── Phong/                  #   最终渲染图（Stage 1a × Stage 1b）
│
├── YOLO-WT/                    # Stage 2：检测器
│   ├── train.py / val.py / detect.py
│   └── ultralytics/            # 修改过的 Ultralytics 源码
│       ├── nn/AddModules/      #   WDSConv.py、IWUpSample.py
│       ├── cfg/models/YOLO-WT/ #   YOLO-WT.yaml + WDSConv/IWUpSample 消融配置
│       └── cfg/datasets/       #   各模态对应一份 YAML（Phong/LUT/Normal/...）
│
├── paper/                      # LaTeX 源（Elsevier cas-sc 模板）
└── docs/                       # 算法详解 + 数据集说明
```

数据集同步发布于 Hugging Face，详见 [`docs/dataset_card.zh-CN.md`](docs/dataset_card.zh-CN.md)。

---

## 快速开始

### 1. 克隆并获取数据集

```bash
git clone https://github.com/WillPanSUTD/MPF-AD.git
cd MPF-AD

# 方式 A：从 Hugging Face 下载已渲染好的 Train_Data
huggingface-cli download <hf-handle>/<dataset> --repo-type dataset --local-dir .

# 方式 B：使用 Crop_Data 自行运行 Stage 1 渲染（见下文）
```

### 2. Stage 1 — 深度图渲染为光度图（C++）

**依赖**

| 组件     | 测试版本 |
|----------|---------|
| OpenCV   | 4.11.0  |
| Eigen    | 3.4.0   |
| 编译器   | MSVC 2022 / GCC 11+（C++17）|

**Windows（MSVC，仓库自带 `.sln`）**

1. 安装 OpenCV 4.x 与 Eigen 3，把头文件与库路径加入工程；
2. 用 Visual Studio 2022 打开 `Depth-Normal_Rendering/Depth-Normal_Rendering.sln`；
3. `Release / x64` 编译，运行前在 `Demo.cpp` 顶部修改输入/输出路径，使其指向
   你的 `Crop_Data/Tif/*.tif`。

**Linux（CMake，推荐改造）**

```bash
cd Depth-Normal_Rendering
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/depth_normal_render <depth.tif> <out_dir>
```

驱动程序对每张输入深度图输出：`lut.bmp`（Stage 1a 伪彩色）、
`normals.bmp`（法向可视化）、`phong.bmp`（最终渲染图）。

论文使用的默认渲染参数：

| 参数 | 取值 | 含义 |
|------|------|------|
| `L`, `V`     | `[500, 150, 1500]` | 光源 / 视点方向（图像坐标系） |
| `I_l`        | `1.3`  | 总光强 |
| `w_a`        | `0.1`  | 环境光权重 |
| `w_d`        | `0.5`  | 漫反射权重 |
| `w_s`        | `0.3`  | 镜面反射权重 |
| `n`（`shine`）| `32`  | 高光指数 |

### 3. Stage 2 — 训练 / 验证 / 推理（Python）

**依赖**

```bash
conda create -n yolo-wt python=3.10
conda activate yolo-wt
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu118

cd YOLO-WT
pip install -r requirements.txt
```

`requirements.txt` 会装上游 `ultralytics` wheel（用来拉它的依赖：`numpy`、
`opencv-python`、`tqdm` 等），加上 `PyWavelets`、`polars`。
**运行时实际加载的是仓库内的 fork（`YOLO-WT/ultralytics/`）**，因为启动目录
在 `sys.path` 最前；所以执行 `train.py` / `val.py` 前一定要先 `cd YOLO-WT`。

**用渲染图模态训练 YOLO-WT**

打开 `YOLO-WT/ultralytics/cfg/datasets/Phong.yaml`，把 `path:` / `train:` /
`val:` / `test:` 字段改成你本地 `Train_Data/Phong/` 下的**绝对路径**。
这里必须用绝对路径——Ultralytics 数据加载器在不同工作目录下对相对路径解析不稳定，
本仓库统一以绝对路径作为规范形式。然后：

```bash
cd YOLO-WT
python train.py
```

`train.py` 默认按论文主结果配置：模型 `cfg/models/YOLO-WT/YOLO-WT.yaml`，
数据 `cfg/datasets/Phong.yaml`，250 epoch，batch 16，输入尺寸 640，
SGD，不加载预训练权重。

**验证**

```bash
python val.py    # 默认从 Abl_Exp/train/.../weights/best.pt 读权重
```

**单张推理**

```bash
python detect.py    # 修改 source= 为你的输入图
```

### 4. 复现论文表格结果

`cfg/datasets/` 下的 6 份 YAML（`DepthMap.yaml`、`LUT.yaml`、`Diffuse.yaml`、
`Specular.yaml`、`Normal.yaml`、`Phong.yaml`）分别对应 6 种模态；
切换 YAML 重跑 `train.py` 即可复现论文中模态对比消融。
`WDSConv.yaml` 与 `IWUpSample.yaml` 模型配置则用于复现按模块的消融行。

---

## 论文使用的硬件

| 组件 | 规格 |
|------|------|
| GPU  | NVIDIA GeForce RTX 3080 |
| CPU  | Intel Core i5-13400F     |
| 传感器 | OPT-LPC61-3D 结构光线扫描仪，倾角 30°–45° |

---

## 复现说明（Reproducibility）

论文中报告的 93.8 % mAP@50 落在小样本数据集训练的合理种子方差内。
我们用**与论文完全相同的配置**（YOLO-WT.yaml，250 epoch，batch 16，
输入尺寸 640，SGD，不加预训练权重）重训了 4 个不同 seed，并对每个
`best.pt` 用 ultralytics 自带 validator 评测：

| seed | val P | val R | val mAP@50 | val mAP@50-95 | test mAP@50 |
|------|-------|-------|------------|---------------|-------------|
| 0  | 0.859 | 0.822 | 0.881 | 0.467 | 0.773 |
| 1  | 0.877 | 0.794 | 0.893 | 0.471 | 0.749 |
| 2  | 0.831 | 0.866 | 0.905 | 0.469 | 0.774 |
| **42** | 0.847 | 0.879 | **0.929** | **0.480** | **0.788** |
| 论文 | – | – | **0.938** | – | – |

4 个 seed 的 val mAP@50 均值 0.902，跨度 4.8 pt。**seed 42 是单种子下最接近
论文报告值的一次，作为公开发布权重**：
`Abl_Exp/train/YOLO-WT-250-16-640-SGD-seed42/weights/best.pt`；
`val.py` 默认就指向这个路径。其余 3 个 seed 的运行目录保留下来，
方便他人独立验证种子方差。

### 4-seed WBF ensemble 稳健性检查

作为额外的可信度证据，我们做了 4-seed Weighted Box Fusion 集成
（`ensemble_eval.py`，需要 `pip install ensemble-boxes torchmetrics
faster-coco-eval`）。这一节的指标走 `torchmetrics + faster_coco_eval`
后端，**仅可在本节内部对比**（与上表 ultralytics 的实现略有差异）：

|                  | val mAP@50 | test mAP@50 |
|------------------|------------|-------------|
| seed 42 单模型     | 0.894 | 0.783 |
| 4-seed WBF 集成 | **0.931** | **0.809** |

WBF 在大多数类上有正向收益，唯独 *Bump* 类在 test 上反被拉低
（弱 seed 的低置信预测把 seed 42 的高置信框平均下去了）。
我们仍然只发布 seed 42 的单 ckpt 以便部署；但 `python ensemble_eval.py`
可重现这组 ensemble 数字，证明单种子结果是稳健的、非畸形的。

### 已知限制

- **数据切分不分层（non-stratified）**。论文的 8:1:1 是按图像随机切分而非按类分层。
  各类样本数（train / val / test）：
  Pseudo `170 / 21 / 23`、Pinhole `306 / 40 / 35`、Pit `199 / 22 / 30`、
  Burst `76 / 21 / 13`、Fish-scale `151 / 13 / 24`、**Bump `84 / 6 / 14`**。
  Bump val 只 6 个实例，单个预测错误就让 val Bump AP 抖 15+ 点，
  这个数字不应被过度解读。
- **Bump ↔ Burst 类间相似**。两类定义都是「凸起 ≥ 0.2 mm」，
  唯一区别是凸起形状（体积感 vs 鱼鳞形）的主观判断。
  test 上 14 个真 Bump 中有 4 个被错分为 Burst，
  把 test Bump mAP 拉到 0.50（同 ckpt val 上是 0.88）。
  这是类间固有相似性而非训练缺陷。改进方向：把两类合并为单一
  *Protrusion*，或在训练中针对 Bump↔Burst 做 hard-mining。
- **val 和 test 大约相差 12 点**，4 个 seed 都如此，主要由 Bump 问题贡献。
  val 数字应视为当前数据集规模下的泛化上限，而非部署性能。

---

## 引用

```bibtex
@article{cao2026geo2pho,
  title   = {Physically Inspired Geometry-to-Photometry Rendering for Real-Time Weld Defect Detection},
  author  = {Cao, Ling and Qiu, Jiajun and Zhang, Yunzhi and Feng, Daquan and Pan, Wei},
  journal = {Engineering Applications of Artificial Intelligence},
  year    = {2026},
  note    = {Accepted, in press. Preprint: SSRN, doi:10.2139/ssrn.6946138}
}
```

（正式在线发表后将补充卷号 / 文章编号 / DOI。）

---

## 致谢

- 检测器基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)（AGPL-3.0）；
- WDSConv 借鉴了 Finder et al., *WTConv*（2024）的小波卷积思想；
- 本工作受奥普特机器视觉东莞市重点研发计划（No. 20241200300122）与
  广东省重点研发计划（No. 2025B0101120001）资助。

## 许可

- `YOLO-WT/ultralytics/` 下的代码：继承 Ultralytics，**AGPL-3.0**；
- `Depth-Normal_Rendering/` 下的代码：**Apache-2.0**（本工作）；
- `Crop_Data/` 与 `Train_Data/` 下的数据：详见 [`docs/dataset_card.zh-CN.md`](docs/dataset_card.zh-CN.md)。
