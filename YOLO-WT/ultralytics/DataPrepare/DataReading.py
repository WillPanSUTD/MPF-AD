import os


def count_yolo_classes(label_dir):
    """
    统计YOLO格式标签文件夹中各类别的数量

    参数:
        label_dir: 标签文件夹路径（包含class.txt和所有txt标签文件）
    """
    # 1. 读取class.txt获取类别名称
    class_file = os.path.join(label_dir, "classes.txt")
    if not os.path.exists(class_file):
        print(f"错误：未在{label_dir}中找到classes.txt文件")
        return

    # 读取类别列表（处理可能的空行）
    with open(class_file, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]
    if not classes:
        print("错误：classes.txt中未找到有效类别名称")
        return

    # 初始化计数器（类别名称: 数量）
    class_count = {cls: 0 for cls in classes}
    unknown_count = 0  # 记录未知类别（索引超出范围的情况）

    # 2. 遍历所有标签文件
    label_files = [f for f in os.listdir(label_dir)
                   if f.endswith('.txt') and f != 'classes.txt']  # 排除class.txt本身

    if not label_files:
        print(f"警告：在{label_dir}中未找到任何标签文件（.txt）")
        return

    # 3. 统计每个文件中的类别
    for file in label_files:
        file_path = os.path.join(label_dir, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:  # 跳过空行
                    continue

                # YOLO格式：第一列为类别索引，后面为坐标
                parts = line.split()
                if len(parts) < 1:  # 无效行
                    continue

                try:
                    class_idx = int(parts[0])
                    # 检查索引是否在有效范围内
                    if 0 <= class_idx < len(classes):
                        class_name = classes[class_idx]
                        class_count[class_name] += 1
                    else:
                        unknown_count += 1
                except ValueError:
                    # 无法转换为整数的情况（无效标签）
                    continue

        except Exception as e:
            print(f"处理文件{file}时出错：{str(e)}")
            continue

    # 4. 输出统计结果
    print(f"类别统计结果（共{len(label_files)}个标签文件）：")
    print("-" * 40)
    # 按类别顺序输出
    for cls in classes:
        print(f"{cls}: {class_count[cls]}")

    # 输出未知类别数量（如果有）
    if unknown_count > 0:
        print("-" * 40)
        print(f"未知类别（索引超出范围）：{unknown_count}")


# 使用示例
if __name__ == "__main__":
    # 请替换为你的标签文件夹路径
    label_directory = "/path/to/LUT-AD/Crop_Data/label"
    count_yolo_classes(label_directory)