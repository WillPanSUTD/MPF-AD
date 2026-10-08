import os
import random
import shutil
from collections import defaultdict


def split_train_val_test(label_dir, img_dir, output_dir, split_ratio=(0.8, 0.1, 0.1)):
    """
    按比例划分YOLO格式数据集为train、val、test，生成完整目录结构并复制文件

    参数:
        label_dir: 标签文件夹路径（包含classes.txt和txt标签文件）
        img_dir: 图片文件夹路径（图片与标签文件同名，后缀不同）
        output_dir: 根输出目录（会在其下创建train/val/test文件夹）
        split_ratio: 划分比例 (train, val, test)，默认(0.8, 0.1, 0.1)
    """
    # 检查输入有效性
    if not os.path.exists(label_dir):
        print(f"错误：标签文件夹不存在 - {label_dir}")
        return
    if not os.path.exists(img_dir):
        print(f"错误：图片文件夹不存在 - {img_dir}")
        return

    # 1. 创建目标目录结构（train/images, train/labels, val/..., test/...）
    dataset_types = ['train', 'val', 'test']
    for dtype in dataset_types:
        # 图片目录
        img_dst_dir = os.path.join(output_dir, dtype, 'images')
        os.makedirs(img_dst_dir, exist_ok=True)
        # 标签目录
        label_dst_dir = os.path.join(output_dir, dtype, 'labels')
        os.makedirs(label_dst_dir, exist_ok=True)

    # 2. 读取类别列表（读取classes.txt）
    class_file = os.path.join(label_dir, "classes.txt")
    if not os.path.exists(class_file):
        print(f"错误：未找到classes.txt - {class_file}")
        return
    with open(class_file, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]
    # 复制classes.txt到输出目录根目录（方便训练时使用）
    shutil.copy2(class_file, os.path.join(output_dir, 'classes.txt'))

    # 3. 收集所有有效样本（图片+标签对应）
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']  # 支持的图片格式
    label_files = [f for f in os.listdir(label_dir)
                   if f.endswith('.txt') and f != 'classes.txt']  # 排除classes.txt

    samples = []  # 存储有效样本：(图片路径, 标签路径, 包含的类别列表)
    for label_file in label_files:
        # 标签文件路径
        label_path = os.path.join(label_dir, label_file)
        # 对应图片文件路径（尝试所有支持的格式）
        base_name = os.path.splitext(label_file)[0]  # 文件名（不含后缀）
        img_path = None
        for ext in img_extensions:
            candidate = os.path.join(img_dir, base_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if not img_path:
            print(f"警告：未找到标签对应的图片 - {label_file}，已跳过")
            continue

        # 读取标签文件，获取包含的类别
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        sample_classes = []
        for line in lines:
            parts = line.split()
            if len(parts) == 0:
                continue
            try:
                cls_id = int(parts[0])
                if 0 <= cls_id < len(classes):
                    sample_classes.append(classes[cls_id])
            except ValueError:
                continue  # 跳过格式错误的行
        if not sample_classes:
            print(f"警告：标签文件无有效类别 - {label_file}，已跳过")
            continue

        samples.append((img_path, label_path, sample_classes))

    if not samples:
        print("错误：未找到有效样本（图片+标签对应且包含有效类别）")
        return
    print(f"共找到 {len(samples)} 个有效样本")

    # 4. 按类别分组（一个样本可能属于多个类别，会被分到多个组）
    class_to_samples = defaultdict(list)
    for idx, (img_path, label_path, sample_classes) in enumerate(samples):
        for cls in sample_classes:
            class_to_samples[cls].append(idx)  # 存储样本索引

    # 5. 分层抽样：为每个类别划分样本，确保每个类别比例符合要求
    train_indices = set()
    val_indices = set()
    test_indices = set()
    random.seed(42)  # 固定随机种子，保证划分可复现

    for cls in classes:
        if cls not in class_to_samples:
            print(f"警告：类别 {cls} 无样本")
            continue

        # 获取该类别的所有样本索引（去重，避免同一样本被重复处理）
        cls_samples = list(set(class_to_samples[cls]))
        random.shuffle(cls_samples)  # 打乱顺序
        total = len(cls_samples)

        # 计算划分数量（确保val和test至少有1个样本，当总数较小时）
        train_num = int(total * split_ratio[0])
        val_num = int(total * split_ratio[1])
        # 剩余样本分配给test（处理四舍五入误差）
        test_num = total - train_num - val_num

        # 分配索引到对应集合（注意：已分配的样本不再重复分配）
        for i, idx in enumerate(cls_samples):
            if idx in train_indices or idx in val_indices or idx in test_indices:
                continue  # 跳过已分配的样本
            if i < train_num:
                train_indices.add(idx)
            elif i < train_num + val_num:
                val_indices.add(idx)
            else:
                test_indices.add(idx)

    # 处理可能未被分配的样本（分配到train）
    all_indices = set(range(len(samples)))
    unassigned = all_indices - train_indices - val_indices - test_indices
    train_indices.update(unassigned)
    if unassigned:
        print(f"注意：{len(unassigned)} 个样本未被分配，已加入训练集")

    # 6. 复制文件到对应目录，并生成路径索引文件
    def copy_files(indices, dtype):
        """复制样本到指定数据集目录（train/val/test）"""
        img_dst = os.path.join(output_dir, dtype, 'images')
        label_dst = os.path.join(output_dir, dtype, 'labels')
        paths = []  # 存储图片在新位置的路径（用于生成txt索引）
        for i in indices:
            img_path, label_path, _ = samples[i]
            # 复制图片
            img_name = os.path.basename(img_path)
            img_dst_path = os.path.join(img_dst, img_name)
            shutil.copy2(img_path, img_dst_path)  # 保留文件元数据
            # 复制标签
            label_name = os.path.basename(label_path)
            label_dst_path = os.path.join(label_dst, label_name)
            shutil.copy2(label_path, label_dst_path)
            # 记录图片新路径（绝对路径，方便YOLO读取）
            paths.append(os.path.abspath(img_dst_path))
        return paths

    # 复制训练集
    print("\n正在复制训练集文件...")
    train_paths = copy_files(train_indices, 'train')
    # 复制验证集
    print("正在复制验证集文件...")
    val_paths = copy_files(val_indices, 'val')
    # 复制测试集
    print("正在复制测试集文件...")
    test_paths = copy_files(test_indices, 'test')

    # 7. 保存路径索引文件（train.txt等，每行是图片在新位置的绝对路径）
    def save_list(file_path, data_list):
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data_list:
                f.write(f"{item}\n")

    save_list(os.path.join(output_dir, "train.txt"), train_paths)
    save_list(os.path.join(output_dir, "val.txt"), val_paths)
    save_list(os.path.join(output_dir, "test.txt"), test_paths)

    # 8. 统计各数据集的类别分布，验证划分效果
    def count_classes(sample_indices):
        count = defaultdict(int)
        for i in sample_indices:
            for cls in samples[i][2]:
                count[cls] += 1
        return count

    train_count = count_classes(train_indices)
    val_count = count_classes(val_indices)
    test_count = count_classes(test_indices)

    # 输出统计结果
    print("\n数据集划分完成！")
    print(f"训练集（train）：{len(train_paths)} 张图片，{len(train_paths)} 个标签")
    print(f"验证集（val）：{len(val_paths)} 张图片，{len(val_paths)} 个标签")
    print(f"测试集（test）：{len(test_paths)} 张图片，{len(test_paths)} 个标签")
    print("\n各数据集类别分布：")
    print(f"{'类别':<15} {'train':<8} {'val':<8} {'test':<8}")
    print("-" * 40)
    for cls in classes:
        print(f"{cls:<15} {train_count.get(cls, 0):<8} {val_count.get(cls, 0):<8} {test_count.get(cls, 0):<8}")
    print(f"\n所有文件已保存至：{os.path.abspath(output_dir)}")


# 使用示例
if __name__ == "__main__":
    # 请根据实际路径修改以下参数
    label_directory = "/path/to/LUT-AD/Crop_Data/label"  # 标签文件夹路径（含classes.txt）
    image_directory = "/path/to/LUT-AD/Crop_Data/image/Specular"  # 图片文件夹路径
    output_directory = "/path/to/LUT-AD/Train_Data_new/YOLO/Specular"  # 根输出目录

    # 划分比例：train=80%，val=10%，test=10%
    split_train_val_test(
        label_dir=label_directory,
        img_dir=image_directory,
        output_dir=output_directory,
        split_ratio=(0.8, 0.1, 0.1)
    )