---
license: cc-by-nc-4.0
task_categories:
- object-detection
language:
- zh
- en
tags:
- weld-defect
- industrial-inspection
- depth-map
- structured-light
- lithium-ion-battery
- yolo
size_categories:
- n<1K
pretty_name: 锂电池顶盖焊接缺陷深度数据集（LUT-AD）
---

# 锂电池顶盖焊接缺陷深度数据集（LUT-AD）

来自真实工业产线的锂电池顶盖激光焊接数据集，采用**结构光高分辨率深度图**采集，
并对**六类缺陷**做了边界框标注。本数据集对应论文
*"Physically Inspired Geometry-to-Photometry Rendering for Real-Time Weld
Defect Detection"*（已被 *Engineering Applications of Artificial
Intelligence*，EAAI，2026 接收）。代码与文档：
<https://github.com/WillPanSUTD/MPF-AD>。

[ [English](dataset_card.md) ]

---

## 数据组织

数据分两层：原始裁剪深度图（可自行渲染）+ 论文中各模态训练直接使用的预渲染图。

### `Crop_Data/` —— 原始深度 + 标签

| 路径                | 格式 | 内容 |
|---------------------|------|------|
| `Crop_Data/Tif/`    | `.tif`，单通道 CV_32FC1 | 每个缺陷样本一张 640×640 的裁剪深度图，哨兵值 `< -1e30` 表示无效像素 |
| `Crop_Data/label/`  | `.txt`，YOLO 格式 | 每行一个缺陷：`class_id cx cy w h`，几何字段全部归一化到 `[0, 1]` |

这是 `Depth-Normal_Rendering/` 渲染器的标准输入——
对 `Crop_Data/Tif/*.tif` 跑一遍 Stage 1，即可复现下面所有模态。

### `Train_Data/` —— 预渲染图，按 YOLO 训练目录组织

6 种模态，每个模态按 8:1:1 切分为 train/val/test，YOLO 标签在各模态间完全一致
（切换模态无需重新标注）：

| 模态        | 描述 |
|-------------|------|
| `DepthMap/` | min-max 归一化的 8-bit 灰度深度图——朴素 2D 基线 |
| `LUT/`      | Stage 1a 输出：基于统计先验 LUT 的伪彩色图 |
| `Normal/`   | 表面法向图（每个分量从 `[-1, 1]` 映射到 `[0, 255]`，彩色编码） |
| `Diffuse/`  | 仅 Phong 漫反射分量调制（$I_d \cdot C$） |
| `Specular/` | 仅 Phong 镜面反射分量调制（$I_s \cdot C$） |
| `Phong/`    | 完整渲染图 $C' = (I_a + I_d + I_s) \cdot C$，**论文 93.8 % mAP@50 即在此模态上达成** |

每个模态目录形如：

```
Train_Data/<Modality>/
├── classes.txt
├── train/
│   ├── images/   # *.png，640×640
│   └── labels/   # *.txt，YOLO 格式
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

---

## 类别定义

类 id 与 `classes.txt`、代码仓 `cfg/datasets/Phong.yaml` 一致。

| id | 类别                       | 定义 |
|----|----------------------------|------|
| 0  | Pseudo soldering（虚焊）   | 相交平面近似直角，焊缝平面平整，焊缝高度 ≤ 0.1 mm |
| 1  | Pinhole（针孔）            | 焊线上的小针孔状凹陷，深度 ≥ 0.2 mm，长度 ≥ 0.2 mm |
| 2  | Pit（凹坑）                | 焊线上的凹陷区域，深度 ≥ 0.2 mm，长度 ≥ 0.2 mm |
| 3  | Burst（爆点）              | 焊缝上明显且体量较大的凸起，高度 ≥ 0.2 mm，长度 ≥ 0.2 mm |
| 4  | Fish-scale welding（鱼鳞焊） | 隆起的带状区域，纵向呈鱼鳞状凸起，高度 ≥ 0.2 mm，长度 ≥ 0.2 mm |
| 5  | Bump（凸起）               | 焊线上的凸起，高度 ≥ 0.2 mm，长度 ≥ 0.2 mm |

> **关于类 0**：早期内部版本的 `classes.txt` 与 dataset YAML 把类 0 写作
> `Inveracious solding`（拼写错误）。公开发布版本统一改为 `Pseudo soldering`；
> 如果在旧权重中看到原字符串，指代的是同一个类。

### 各类缺陷实例数（全数据集合计）

| 类 id | 类别 | 实例数 |
|------:|------|-------:|
| 0 | Pseudo soldering   | 214 |
| 1 | Pinhole            | 381 |
| 2 | Pit                | 251 |
| 3 | Burst              | 110 |
| 4 | Fish-scale welding | 188 |
| 5 | Bump               | 104 |
| **合计** |              | **1248** |

存在明显的类别不均衡（如 Pinhole vs. Bump），反映的是产线的真实分布。

### 数据集划分（图像数）

| 划分    | 图像数 |
|---------|------:|
| train   |   429 |
| val     |    52 |
| test    |    60 |
| **合计** | **541** |

---

## 采集设置

- **传感器**：OPT-LPC61-3D 结构光线扫描仪，已标定，安装在工件上方 30°–45° 倾角处采集焊缝轮廓；
- **运动平台**：精密 XY 平台带动电池顶盖在扫描仪下平移；
- **样本来源**：所有缺陷样本均来自真实产线检测数据，**无合成或人工诱导的缺陷**；
- **数据筛选**：每张扫描在入库前都人工检查过，剔除传感器噪声与丢点严重的样本。

---

## 使用方法

### 直接下载

2D 检测数据位于数据集仓库
[`vpan1226/MPW-AD`](https://huggingface.co/datasets/vpan1226/MPW-AD) 的 `Crop_Data/` 与 `Train_Data/` 下：

```bash
hf download vpan1226/MPW-AD --repo-type dataset \
    --include "Crop_Data/*" "Train_Data/*" --local-dir ./LUT_AD_DataSet
```

### 从 Crop_Data 重新渲染 Train_Data

`Train_Data/Phong/` 的渲染参数详见 [`docs/algorithm.zh-CN.md`](algorithm.zh-CN.md)。
编译 `Depth-Normal_Rendering/` 下的 C++ 工具，对 `Crop_Data/Tif/*.tif` 运行一次即可
复现所有模态。

---

## 许可

按 **Creative Commons Attribution-NonCommercial 4.0 International（CC BY-NC 4.0）**
协议发布。商业用途需与作者及资助方另行书面授权。

使用本数据集时请引用论文（见下），并在派生公开成果中保留对奥普特机器视觉的署名。

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

## 致谢

数据采集与整理工作受奥普特机器视觉东莞市重点研发计划（No. 20241200300122）
与广东省重点研发计划（No. 2025B0101120001）资助。
