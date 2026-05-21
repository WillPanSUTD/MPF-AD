# LUT-AD 3D 异常检测变体

本页描述基于 LUT-AD 原始数据（`Crop_Data/Tif/*.tif` + `Crop_Data/label/*.txt`）
派生的三个 3D 异常检测风格的姊妹数据集：

- `Dataset_3D/MVTec3D_Weld/` — MVTec 3D-AD 风格布局（RGB + 有序 XYZ 点云 + 像素 GT）。
- `Dataset_3D/Eyecandies_Weld/` — Eyecandies 风格多模态布局（5 种光度渲染 + 原始深度 + GT）。
- `Dataset_3D/Real3D_Weld/` — Real3D-AD 风格点云布局（4 个原型模板 + 每个测试源的 PCD 与逐点二值标签）。

## 生成流程

完全由 `cfg/dataset_3d.yaml` 与 `scripts/dataset_3d/` 下的 8 个脚本驱动：

```bash
python -m scripts.dataset_3d.harvest_good
python -m scripts.dataset_3d.segment_sam2           # 12GB 消费级 GPU 上约 1 小时
python -m scripts.dataset_3d.segment_depth_threshold
python -m scripts.dataset_3d.unproject_xyz
python -m scripts.dataset_3d.build_mvtec3d
python -m scripts.dataset_3d.build_eyecandies
python -m scripts.dataset_3d.build_real3d
python -m scripts.dataset_3d.verify_dataset
```

输出位于 `Dataset_3D/`（已 gitignore），中间缓存位于 `.cache/`（同样 gitignore）。

## 坐标尺度

| 轴 | 值 |
|---|---|
| `pixel_size_mm_x` | 0.016 |
| `pixel_size_mm_y` | 0.016 |
| `depth_scale_mm` | 1.0 |
| `invalid_threshold` | -1.0e30 |

## 已知限制

- **SAM2 质量：** 在 20 张随机源图上抽检；找不到内部分割时回退为 bbox 矩形（少见）。
  评估对 mask 选择的敏感度时，请同时报告 `gt`（SAM2）与 `gt_geom`（深度阈值）。
- **类别不均衡：** 继承自原始数据。每类**出现**次数（同一图含多类时分别计入每类）：
  `pseudo_soldering: 212, pinhole: 217, pit: 228, burst: 100, fish_scale_welding: 174, bump: 97`。
  541 张异常源图中 270 张为单类、271 张为多类。构建管线产出的
  `test/<class>/` 单类子目录实际样本数：
  `pseudo_soldering: 206, pinhole: 11, pit: 10, burst: 3, fish_scale_welding: 21, bump: 19`；
  另 271 张多类图统一归入 `test/combined/`。**多数含某缺陷的图都是多类的**，
  因此每个 `test/<class>/` 单类目录的数量都明显偏低，绝大多数样本落在 `combined/` 与 `pseudo_soldering/`。
- **跳过的原生 good 源：** 8 张源 `.tif` 标签为空，但因渲染流程只对 541 张异常图
  做 Phong/LUT/Diffuse/Specular/Normal 渲染，这 8 张缺少 RGB 模态，
  在 harvest 阶段会被跳过。好-patch 池仅来自 541 张异常源图。
- **Real3D 模板：** 焊缝几何形态因实例而异，模板是"原型 patch"而非整体对象实例；
  刚体配准 baseline 需要适配。

## 可复现性

每个脚本在 `cfg/dataset_3d.yaml` 给定时均为确定性流程，唯一的随机源是
`good_harvest.shuffle_seed = 42`，它同时控制 70/15/15 划分、每源随机
子采样以及 Real3D 模板选择。
