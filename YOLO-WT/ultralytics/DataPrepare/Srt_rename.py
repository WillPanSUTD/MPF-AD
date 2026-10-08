import os
import shutil


def rename_and_copy_srt_files(source_folders, target_folder):
    """
    批量重命名多个文件夹中的srt文件（从1开始连续编号），并复制到目标文件夹

    Args:
        source_folders (list): 包含srt文件的源文件夹路径列表
        target_folder (str): 目标文件夹路径（不存在则自动创建）
    """
    # 确保目标文件夹存在
    os.makedirs(target_folder, exist_ok=True)

    # 初始化文件编号
    file_number = 1

    # 遍历所有源文件夹
    for folder in source_folders:
        # 检查源文件夹是否存在
        if not os.path.isdir(folder):
            print(f"警告：文件夹 {folder} 不存在，已跳过")
            continue

        # 遍历文件夹中的所有文件
        for filename in os.listdir(folder):
            # 只处理.srt后缀的文件
            if filename.lower().endswith('.srt'):
                # 源文件完整路径
                source_path = os.path.join(folder, filename)

                # 新文件名（带.srt后缀）
                new_filename = f"{file_number}.srt"
                # 目标文件完整路径
                target_path = os.path.join(target_folder, new_filename)

                # 复制文件并改名
                try:
                    shutil.copy2(source_path, target_path)  # copy2会保留文件元数据
                    print(f"成功复制：{source_path} → {target_path}")
                    file_number += 1  # 编号自增
                except Exception as e:
                    print(f"错误：复制 {source_path} 失败 - {str(e)}")

    print(f"\n操作完成！共处理 {file_number - 1} 个srt文件，已保存到 {target_folder}")


def rename_and_copy_bmp_files(source_folders, target_folder):
    """
    批量重命名多个文件夹中的bmp文件（从1开始连续编号），并复制到目标文件夹

    Args:
        source_folders (list): 包含bmp文件的源文件夹路径列表
        target_folder (str): 目标文件夹路径（不存在则自动创建）
    """
    # 确保目标文件夹存在
    os.makedirs(target_folder, exist_ok=True)

    # 初始化文件编号
    file_number = 1

    # 遍历所有源文件夹
    for folder in source_folders:
        # 检查源文件夹是否存在
        if not os.path.isdir(folder):
            print(f"警告：文件夹 {folder} 不存在，已跳过")
            continue

        # 遍历文件夹中的所有文件
        for filename in os.listdir(folder):
            # 只处理.bmp后缀的文件（忽略大小写，兼容.BMP/.bmp）
            if filename.lower().endswith('.bmp'):
                # 源文件完整路径
                source_path = os.path.join(folder, filename)

                # 新文件名（带.bmp后缀）
                new_filename = f"{file_number}.bmp"
                # 目标文件完整路径
                target_path = os.path.join(target_folder, new_filename)

                # 复制文件并改名
                try:
                    shutil.copy2(source_path, target_path)  # copy2保留文件元数据（创建时间等）
                    print(f"成功复制：{source_path} → {target_path}")
                    file_number += 1  # 编号自增
                except Exception as e:
                    print(f"错误：复制 {source_path} 失败 - {str(e)}")

    print(f"\n操作完成！共处理 {file_number - 1} 个bmp文件，已保存到 {target_folder}")

# ====================== 请在这里修改配置 ======================
# 配置1：需要处理的文件夹路径列表（可以添加多个）
SOURCE_FOLDERS = [
    r"/path/to/LUT-AD/Dataset/1_ORG/1无分类/Depth_Img",
    r"/path/to/LUT-AD/Dataset/1_ORG/2有分类/凹坑/Depth_Img",
    r"/path/to/LUT-AD/Dataset/1_ORG/2有分类/爆点/Depth_Img",
    r"/path/to/LUT-AD/Dataset/1_ORG/2有分类/凸起/Depth_Img",
    r"/path/to/LUT-AD/Dataset/1_ORG/2有分类/虚焊/Depth_Img",
    r"/path/to/LUT-AD/Dataset/1_ORG/2有分类/鱼纹/Depth_Img",
    r"/path/to/LUT-AD/Dataset/1_ORG/2有分类/针孔/Depth_Img",
    # 可以继续添加更多文件夹，比如 r"D:\视频\字幕文件夹"
]

# 配置2：目标文件夹路径（重命名后的文件会复制到这里）
TARGET_FOLDER = r"/path/to/LUT-AD/Org_Data/image/DepthMap"
# =============================================================

# 执行主函数
if __name__ == "__main__":
    rename_and_copy_bmp_files(SOURCE_FOLDERS, TARGET_FOLDER)