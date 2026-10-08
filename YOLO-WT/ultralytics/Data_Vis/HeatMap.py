# # 忽略警告信息，避免输出干扰（新手调试时可注释掉，查看完整警告）
# import warnings
#
# warnings.filterwarnings('ignore')
# warnings.simplefilter('ignore')
#
# # 导入核心依赖库
# import torch  # PyTorch核心，用于模型加载、张量运算
# import cv2  # OpenCV，用于图像读取、处理、绘制
# import os  # 操作系统接口，用于文件/目录操作
# import shutil  # 高级文件操作，用于删除目录
# import copy  # 深拷贝，避免修改原模型
# import numpy as np  # 数值计算，处理图像数组
#
# np.random.seed(0)  # 设置numpy随机种子，保证结果可复现
# from PIL import Image  # PIL库，用于图像保存
# from ultralytics import YOLO  # 导入Ultralytics YOLO模型
# from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM, KPCA_CAM, AblationCAM  # 仅保留GradCAM（检测任务用）
# from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image  # CAM图像处理工具
#
#
# # -------------------------- 核心工具函数：图像预处理（YOLO官方） --------------------------
# def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
#     """
#     YOLO专用图像预处理：保持宽高比缩放+填充，满足模型输入要求
#     :param im: 原始图像（np.ndarray）
#     :param new_shape: 目标尺寸，默认640x640
#     :param color: 填充颜色（灰边），默认(114,114,114)
#     :param auto: 是否自动填充（保持宽高比）
#     :param scaleFill: 是否拉伸填充（不保持宽高比）
#     :param scaleup: 是否允许放大图像（仅缩小不放大可提升精度）
#     :param stride: 模型步长，需填充为stride的整数倍
#     :return: 处理后的图像、缩放比例、填充的边界（top, bottom, left, right）
#     """
#     # 获取原始图像尺寸 [height, width]
#     shape = im.shape[:2]
#     # 如果new_shape是整数，转为(宽,高)元组
#     if isinstance(new_shape, int):
#         new_shape = (new_shape, new_shape)
#
#     # 计算缩放比例（新尺寸/原始尺寸，取最小比例保证图像全部在框内）
#     r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
#     # 仅缩小，不放大（避免低分辨率图像放大失真）
#     if not scaleup:
#         r = min(r, 1.0)
#
#     # 计算缩放后的尺寸（未填充）
#     ratio = r, r  # 宽、高缩放比例
#     new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
#     # 计算需要填充的宽度和高度
#     dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
#     # 自动填充：填充后尺寸为stride的整数倍（YOLO要求）
#     if auto:
#         dw, dh = np.mod(dw, stride), np.mod(dh, stride)
#     # 拉伸填充：直接填充到目标尺寸（不保持宽高比）
#     elif scaleFill:
#         dw, dh = 0.0, 0.0
#         new_unpad = (new_shape[1], new_shape[0])
#         ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
#
#     # 将填充量平分到图像两侧（上下/左右各一半）
#     dw /= 2
#     dh /= 2
#
#     # 缩放图像（如果尺寸不一致）
#     if shape[::-1] != new_unpad:
#         im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
#     # 计算上下左右填充的像素数（四舍五入）
#     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
#     # 给图像添加边界（填充灰边）
#     im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
#     return im, ratio, (top, bottom, left, right)
#
#
# # -------------------------- CAM核心类：捕获模型激活值和梯度 --------------------------
# class ActivationsAndGradients:
#     """
#     用于捕获YOLO模型指定层的激活值和梯度（CAM算法核心）
#     """
#
#     def __init__(self, model, target_layers, reshape_transform):
#         """
#         初始化：注册钩子函数捕获激活值和梯度
#         :param model: YOLO模型
#         :param target_layers: 需要捕获的目标层列表
#         :param reshape_transform: 形状变换函数（这里用不到，设为None）
#         """
#         self.model = model  # YOLO模型
#         self.gradients = []  # 存储梯度
#         self.activations = []  # 存储激活值
#         self.reshape_transform = reshape_transform  # 形状变换（None）
#         self.handles = []  # 存储钩子句柄（用于释放）
#
#         # 为每个目标层注册钩子函数
#         for target_layer in target_layers:
#             # 注册前向钩子：捕获激活值
#             self.handles.append(target_layer.register_forward_hook(self.save_activation))
#             # 注册前向钩子：捕获梯度（规避PyTorch backward hook的bug）
#             self.handles.append(target_layer.register_forward_hook(self.save_gradient))
#
#     def save_activation(self, module, input, output):
#         """
#         前向钩子：保存目标层的激活值
#         :param module: 目标层模块
#         :param input: 模块输入
#         :param output: 模块输出（激活值）
#         """
#         activation = output
#         # 形状变换（这里用不到，直接保存）
#         if self.reshape_transform is not None:
#             activation = self.reshape_transform(activation)
#         # 将激活值转到CPU并脱离计算图，存入列表
#         self.activations.append(activation.cpu().detach())
#
#     def save_gradient(self, module, input, output):
#         """
#         前向钩子：保存目标层的梯度
#         :param module: 目标层模块
#         :param input: 模块输入
#         :param output: 模块输出
#         """
#         # 仅处理需要梯度的张量
#         if not hasattr(output, "requires_grad") or not output.requires_grad:
#             return
#
#         # 定义梯度存储函数（梯度反向传播时调用）
#         def _store_grad(grad):
#             if self.reshape_transform is not None:
#                 grad = self.reshape_transform(grad)
#             # 梯度反向传播，所以新梯度插入到列表头部
#             self.gradients = [grad.cpu().detach()] + self.gradients
#
#         # 注册梯度钩子：反向传播时捕获梯度
#         output.register_hook(_store_grad)
#
#     def post_process(self, result):
#         """
#         YOLO检测任务输出后处理：提取类别得分和框坐标，并按置信度排序
#         :param result: YOLO原始输出
#         :return: 排序后的类别得分、框坐标
#         """
#         # 检测任务：输出格式 [框数, 4(框)+类别数]
#         logits_ = result[:, 4:]  # 类别得分
#         boxes_ = result[:, :4]  # 框坐标（xywh）
#         # 按类别得分最大值降序排序，获取索引
#         sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
#         # 按排序索引重新排列类别得分和框坐标
#         return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[
#             indices[0]]
#
#     def __call__(self, x):
#         """
#         前向传播：执行模型推理，捕获激活值和梯度，返回处理后的输出
#         :param x: 输入张量 [1,3,640,640]
#         :return: 处理后的检测结果 [类别得分, 框坐标]
#         """
#         # 清空之前的激活值和梯度
#         self.gradients = []
#         self.activations = []
#         # 模型前向推理
#         model_output = self.model(x)
#         # 仅处理检测任务
#         post_result, pre_post_boxes = self.post_process(model_output[0])
#         return [[post_result, pre_post_boxes]]
#
#     def release(self):
#         """释放钩子句柄，避免内存泄漏"""
#         for handle in self.handles:
#             handle.remove()
#
#
# # -------------------------- 检测任务目标函数：计算CAM损失 --------------------------
# class yolo_detect_target(torch.nn.Module):
#     """
#     YOLO检测任务的CAM目标函数：计算损失值（用于反向传播求梯度）
#     """
#
#     def __init__(self, ouput_type, conf, ratio, end2end) -> None:
#         """
#         初始化
#         :param ouput_type: 损失计算类型（class/box/all）
#         :param conf: 置信度阈值（过滤低置信度目标）
#         :param ratio: 参与计算的目标比例（避免计算量过大）
#         :param end2end: 是否端到端模型（这里固定为False）
#         """
#         super().__init__()
#         self.ouput_type = ouput_type  # 损失类型
#         self.conf = conf  # 置信度阈值
#         self.ratio = ratio  # 目标比例
#         self.end2end = end2end  # 端到端标记
#
#     def forward(self, data):
#         """
#         前向传播：计算损失值（类别得分/框坐标求和）
#         :param data: [类别得分, 框坐标]
#         :return: 损失值（求和结果）
#         """
#         post_result, pre_post_boxes = data  # 解包数据
#         result = []  # 存储参与损失计算的数值
#
#         # 遍历前N个目标（N=总目标数*ratio）
#         for i in range(int(post_result.size(0) * self.ratio)):
#             # 过滤低置信度目标（低于阈值则停止）
#             if float(post_result[i].max()) < self.conf:
#                 break
#             # 按类型计算损失
#             if self.ouput_type == 'class' or self.ouput_type == 'all':
#                 # 类别得分：取最大值加入损失
#                 result.append(post_result[i].max())
#             elif self.ouput_type == 'box' or self.ouput_type == 'all':
#                 # 框坐标：4个值都加入损失
#                 for j in range(4):
#                     result.append(pre_post_boxes[i, j])
#         # 返回损失总和（用于反向传播）
#         return sum(result)
#
#
# # -------------------------- 核心类：YOLO检测任务热力图生成 --------------------------
# class yolo_heatmap:
#     """
#     YOLO检测任务热力图生成器：封装CAM算法全流程
#     """
#
#     def __init__(self, weight, device, method, layer, backward_type, conf_threshold, ratio, show_result, renormalize,
#                  task, img_size):
#         """
#         初始化YOLO热力图生成器
#         :param weight: YOLO权重文件路径
#         :param device: 运行设备（cuda:0/cpu）
#         :param method: CAM算法（仅GradCAM）
#         :param layer: 目标层索引列表
#         :param backward_type: 损失类型（class/box/all）
#         :param conf_threshold: 置信度阈值
#         :param ratio: 目标比例
#         :param show_result: 是否绘制检测框
#         :param renormalize: 是否将热力图限制在检测框内
#         :param task: 任务类型（固定为detect）
#         :param img_size: 输入图像尺寸
#         """
#         # 初始化设备
#         device = torch.device(device)
#         # 加载YOLO模型（含权重）
#         model_yolo = YOLO(weight)
#         # 获取模型类别名称（如['person', 'car']）
#         model_names = model_yolo.names
#         print(f'模型类别信息:{model_names}')
#
#         # 深拷贝YOLO模型（避免修改原模型）
#         model = copy.deepcopy(model_yolo.model)
#         # 将模型转到指定设备
#         model.to(device)
#         # 打印模型信息
#         model.info()
#         # 开启模型参数梯度（CAM需要反向传播）
#         for p in model.parameters():
#             p.requires_grad_(True)
#         # 模型设为评估模式（不启用BN/Dropout）
#         model.eval()
#
#         # 固定任务为detect
#         model.task = task
#         # 新增end2end属性（避免AttributeError）
#         if not hasattr(model, 'end2end'):
#             model.end2end = False
#
#         # 创建检测任务目标函数
#         target = yolo_detect_target(backward_type, conf_threshold, ratio, model.end2end)
#         # 获取目标层（按索引取模型层）
#         target_layers = [model.model[l] for l in layer]
#         # 初始化GradCAM算法
#         method = eval(method)(model, target_layers)
#         # 绑定激活值/梯度捕获类
#         method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)
#
#         # 生成随机颜色（用于绘制检测框，按类别数生成）
#         colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int32)
#         # 将所有局部变量设为实例属性（方便其他方法调用）
#         self.__dict__.update(locals())
#
#     def draw_detections(self, box, color, name, img):
#         """
#         绘制检测框和类别名称（备用函数，实际用YOLO内置plot）
#         :param box: 框坐标（xyxy）
#         :param color: 框颜色
#         :param name: 类别名称
#         :param img: 绘制图像
#         :return: 绘制后的图像
#         """
#         xmin, ymin, xmax, ymax = list(map(int, list(box)))
#         # 绘制矩形框
#         cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
#         # 绘制类别名称
#         cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2,
#                     lineType=cv2.LINE_AA)
#         return img
#
#     def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
#         """
#         将热力图归一化到检测框内（框外置0），增强目标区域可视化效果
#         :param boxes: 检测框列表（xyxy）
#         :param image_float_np: 归一化后的原始图像（0-1）
#         :param grayscale_cam: 灰度热力图
#         :return: 归一化后的热力图叠加图像
#         """
#         # 初始化归一化热力图（全0）
#         renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
#         # 遍历每个检测框
#         for x1, y1, x2, y2 in boxes:
#             # 确保框坐标在图像范围内
#             x1, y1 = max(x1, 0), max(y1, 0)
#             x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
#             # 框内热力图归一化到0-1
#             renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
#         # 整体归一化
#         renormalized_cam = scale_cam_image(renormalized_cam)
#         # 将热力图叠加到原始图像
#         eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
#         return eigencam_image_renormalized
#
#     def process(self, img_path, save_path):
#         """
#         处理单张图像：生成热力图并保存
#         :param img_path: 输入图像路径
#         :param save_path: 输出图像保存路径
#         """
#         # 1. 读取图像（兼容中文路径）
#         try:
#             # 从文件读取字节流，解码为图像（解决cv2.imread中文路径问题）
#             img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
#         except:
#             print(f"警告... {img_path} 读取失败。")
#             return
#
#         # 2. 图像预处理（letterbox缩放+填充）
#         img, _, (top, bottom, left, right) = letterbox(img, new_shape=(self.img_size, self.img_size), auto=True)
#         # 转换色彩空间：BGR（OpenCV）→ RGB（PIL/CAM）
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         # 归一化到0-1（CAM算法要求）
#         img = np.float32(img) / 255.0
#
#         # 3. 转换为模型输入张量：HWC → CHW → 加batch维度 → 转到指定设备
#         tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
#         print(f'模型输入张量尺寸:{tensor.size()}')
#
#         # 4. 生成GradCAM热力图
#         try:
#             # 执行CAM算法：生成灰度热力图 [1, H, W]
#             grayscale_cam = self.method(tensor, [self.target])
#         except AttributeError as e:
#             print(f"警告... CAM算法执行失败: {e}")
#             return
#         # 去除batch维度，得到 [H, W] 灰度热力图
#         grayscale_cam = grayscale_cam[0, :]
#         # 将热力图叠加到原始图像（生成RGB彩色热力图）
#         cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
#
#         # 5. YOLO推理：获取检测结果
#         pred = self.model_yolo.predict(tensor, conf=self.conf_threshold, iou=0.7)[0]
#         # 可选：将热力图限制在检测框内
#         if self.renormalize:
#             # 提取检测框坐标（xyxy）并转为int
#             boxes = pred.boxes.xyxy.cpu().detach().numpy().astype(np.int32)
#             # 热力图归一化到检测框内
#             cam_image = self.renormalize_cam_in_bounding_boxes(boxes, img, grayscale_cam)
#         # 可选：绘制检测框/置信度
#         if self.show_result:
#             cam_image = pred.plot(
#                 img=cam_image,
#                 conf=True,  # 显示置信度
#                 font_size=None,  # 自动计算字体大小
#                 line_width=None,  # 自动计算线条宽度
#                 labels=False,  # 不显示类别标签（仅显示置信度）
#             )
#
#         # 6. 后处理：去除填充的边界，恢复原始图像尺寸
#         cam_image = cam_image[top:cam_image.shape[0] - bottom, left:cam_image.shape[1] - right]
#         # 转换为PIL图像并保存
#         cam_image = Image.fromarray(cam_image)
#         cam_image.save(save_path)
#
#     def __call__(self, img_path, save_path):
#         """
#         对外调用接口：处理单张/批量图像
#         :param img_path: 输入图像路径（文件/文件夹）
#         :param save_path: 输出保存目录
#         """
#         # 1. 清理保存目录（如果存在则删除）
#         if os.path.exists(save_path):
#             shutil.rmtree(save_path)
#         # 2. 创建保存目录（不存在则创建）
#         os.makedirs(save_path, exist_ok=True)
#
#         # 3. 判断输入类型：文件夹（批量）/文件（单张）
#         if os.path.isdir(img_path):
#             # 批量处理：遍历文件夹内所有图像
#             for img_path_ in os.listdir(img_path):
#                 self.process(f'{img_path}/{img_path_}', f'{save_path}/{img_path_}')
#         else:
#             # 单张处理：保存为result.png
#             self.process(img_path, f'{save_path}/heatmap.png')
#
#
# # -------------------------- 配置参数：可根据需求修改 --------------------------
# def get_params():
#     """
#     配置参数：集中管理所有超参数，方便修改
#     """
#     params = {
#         'weight': 'Phong_Comparison_Experiments/train/DepthMap-250-32-640-SGD/weights/best.pt',  # YOLO权重路径
#         'device': 'cuda:0',  # 运行设备（cpu/cuda:0）
#         'method': 'GradCAMPlusPlus',  # GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM, KPCA_CAM
#         'layer': [16, 19, 22],  # YOLO目标层索引（可调整）[3, 7, 11,14,16, 19, 22]
#         'backward_type': 'all',  # 损失类型（class/box/all）
#         'conf_threshold': 0.5,  # 置信度阈值（过滤低置信度目标）
#         'ratio': 0.02,  # 参与计算的目标比例（0.02-0.1）
#         'show_result': False,  # 是否绘制检测框/置信度
#         'renormalize': False,  # 是否将热力图限制在检测框内
#         'task': 'detect',  # 固定为detect
#         'img_size': 640,  # 输入图像尺寸
#     }
#     return params
#
# # -------------------------- 主函数：执行热力图生成 --------------------------
# if __name__ == '__main__':
#     # 1. 获取配置参数
#     params = get_params()
#     # 2. 初始化YOLO热力图生成器
#     model = yolo_heatmap(**params)
#     # 3. 执行热力图生成：输入图像路径 → 输出保存目录
#     # model(r'assets/test.bmp', 'result')  # 单张图像
#     model(r'DataPrepare/EXP1_3D/DepthMap/27.bmp', r'result')  # 批量图像（注释掉上一行，启用这一行）


# # 忽略警告信息，避免输出干扰
# import warnings
#
# warnings.filterwarnings('ignore')
# warnings.simplefilter('ignore')
#
# # 导入核心依赖库
# import torch
# import cv2
# import os
# import shutil
# import copy
# import csv
# import numpy as np
#
# np.random.seed(0)
# from PIL import Image
# from ultralytics import YOLO
# from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM, \
#     KPCA_CAM, AblationCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
# # 新增matplotlib相关导入
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图库
#
# plt.switch_backend('TkAgg')
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 确保英文显示正常
# plt.rcParams['axes.unicode_minus'] = False
#
#
# # -------------------------- 预处理 --------------------------
# def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
#     shape = im.shape[:2]
#     if isinstance(new_shape, int):
#         new_shape = (new_shape, new_shape)
#     r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
#     if not scaleup:
#         r = min(r, 1.0)
#     ratio = (r, r)
#     new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
#     dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
#     if auto:
#         dw, dh = np.mod(dw, stride), np.mod(dh, stride)
#     elif scaleFill:
#         dw, dh = 0.0, 0.0
#         new_unpad = (new_shape[1], new_shape[0])
#         ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
#     dw /= 2
#     dh /= 2
#     if shape[::-1] != new_unpad:
#         im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
#     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
#     im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
#     return im, ratio, (top, bottom, left, right)
#
#
# # -------------------------- 生成方格数据（不归一化） --------------------------
# def get_attention_grid_raw(grayscale_cam, grid_size=20):
#     h, w = grayscale_cam.shape
#     grid_data = np.zeros((h, w), dtype=np.float32)
#     for y in range(0, h, grid_size):
#         for x in range(0, w, grid_size):
#             x2 = min(x + grid_size, w)
#             y2 = min(y + grid_size, h)
#             val = np.mean(grayscale_cam[y:y2, x:x2])
#             grid_data[y:y2, x:x2] = val
#     return grid_data
#
#
# # -------------------------- 2D可视化（白底→红 + colorbar，不归一化） --------------------------
# def plot_attention_raw_with_colorbar(grid_data, save_path, show_plot=True):
#     # 白 → 红 颜色映射
#     colors = [(1, 1, 1), (1, 0, 0)]
#     cmap = mcolors.LinearSegmentedColormap.from_list("white2red", colors, N=256)
#
#     fig, ax = plt.subplots(figsize=(7, 7))
#     ax.axis('off')
#
#     # 直接用原始值，不归一化
#     im = ax.imshow(grid_data, cmap=cmap)
#
#     # 右侧 colorbar
#     cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.0)
#     if show_plot:
#         plt.show()
#     plt.close()
#
#
# # -------------------------- 3D可视化（XY=像素坐标，Z=响应值） --------------------------
# def plot_attention_3d(grid_data, save_path, show_plot=True):
#     """
#     3D曲面图可视化attention响应矩阵
#     :param grid_data: 响应值矩阵（h, w）
#     :param save_path: 保存路径
#     :param show_plot: 是否弹出显示窗口
#     """
#     h, w = grid_data.shape
#
#     # 生成XY坐标轴（对应图像像素坐标）
#     x = np.arange(0, w, 1)
#     y = np.arange(0, h, 1)
#     X, Y = np.meshgrid(x, y)
#
#     # 创建3D画布
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#
#     # 白 → 红 颜色映射
#     colors = [(1, 1, 1), (1, 0, 0)]
#     cmap = mcolors.LinearSegmentedColormap.from_list("white2red", colors, N=256)
#
#     # 绘制3D曲面图
#     surf = ax.plot_surface(X, Y, grid_data,
#                            cmap=cmap,
#                            linewidth=0,
#                            antialiased=True,
#                            alpha=0.8)  # 透明度
#
#     # 设置坐标轴标签
#     ax.set_xlabel('X Pixel', fontsize=18, labelpad=10)
#     ax.set_ylabel('Y Pixel', fontsize=18, labelpad=10)
#     ax.set_zlabel('Response Value', fontsize=18, labelpad=10)
#
#     # 设置坐标轴范围
#     ax.set_xlim(0, w)
#     ax.set_ylim(0, h)
#
#     # 添加颜色条（对应Z轴响应值）
#     cbar = fig.colorbar(surf, ax=ax, shrink=0.8, aspect=20, pad=0.01)
#
#     # 设置视角（可调整azim和elev改变观察角度）
#     ax.view_init(elev=30, azim=45)
#
#     # 优化布局
#     plt.tight_layout()
#
#     # 保存高分辨率图像
#     plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
#
#     # 弹出显示窗口
#     if show_plot:
#         plt.show()
#
#     # 关闭画布释放内存
#     plt.close()
#
#
# # -------------------------- 保存metric --------------------------
# def save_metric_raw(grayscale_cam, grid_size, save_path):
#     h, w = grayscale_cam.shape
#     data = [['row', 'col', 'x1', 'y1', 'x2', 'y2', 'mean_response']]
#     row = 0
#     for y in range(0, h, grid_size):
#         col = 0
#         for x in range(0, w, grid_size):
#             x2 = min(x + grid_size, w)
#             y2 = min(y + grid_size, h)
#             v = np.mean(grayscale_cam[y:y2, x:x2])
#             data.append([row, col, x, y, x2, y2, round(float(v), 4)])
#             col += 1
#         row += 1
#     with open(save_path, 'w', newline='', encoding='utf-8') as f:
#         csv.writer(f).writerows(data)
#
#
# # -------------------------- CAM 相关 --------------------------
# class ActivationsAndGradients:
#     def __init__(self, model, layers, transform):
#         self.model = model
#         self.gradients = []
#         self.activations = []
#         self.transform = transform
#         self.handles = []
#         for l in layers:
#             self.handles.append(l.register_forward_hook(self.save_act))
#             self.handles.append(l.register_forward_hook(self.save_grad))
#
#     def save_act(self, m, i, o):
#         a = o
#         if self.transform: a = self.transform(a)
#         self.activations.append(a.detach().cpu())
#
#     def save_grad(self, m, i, o):
#         if not o.requires_grad: return
#
#         def _g(grad):
#             if self.transform: grad = self.transform(grad)
#             self.gradients = [grad.detach().cpu()] + self.gradients
#
#         o.register_hook(_g)
#
#     def __call__(self, x):
#         self.gradients = []
#         self.activations = []
#         out = self.model(x)
#         return out
#
#     def release(self):
#         for h in self.handles: h.remove()
#
#
# class YoloTarget(torch.nn.Module):
#     def __init__(self, mode, conf, ratio):
#         super().__init__()
#         self.mode = mode
#         self.conf = conf
#         self.ratio = ratio
#
#     def forward(self, x):
#         res = []
#         for i in range(int(x[0][0].size(0) * self.ratio)):
#             v = x[0][0][i].max()
#             if v < self.conf: break
#             res.append(v)
#         return sum(res)
#
#
# # -------------------------- 主类 --------------------------
# class YOLOHeatmap:
#     def __init__(self, weight, device, method, layers, conf=0.5, grid_size=20):
#         self.dev = torch.device(device)
#         self.yolo = YOLO(weight).to(self.dev)
#         self.model = self.yolo.model
#         self.model.eval()
#         for p in self.model.parameters(): p.requires_grad_(True)
#         self.target_layers = [self.model.model[i] for i in layers]
#         self.cam = eval(method)(self.model, self.target_layers)
#         self.cam.activations_and_grads = ActivationsAndGradients(self.model, self.target_layers, None)
#         self.target = YoloTarget('all', conf, 0.02)
#         self.conf = conf
#         self.grid_size = grid_size
#
#     def process(self, img_path, save_dir):
#         img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
#         name = os.path.splitext(os.path.basename(img_path))[0]
#
#         img, _, (t, b, l, r) = letterbox(img, 640)
#         rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         norm = rgb.astype(np.float32) / 255
#         tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(self.dev)
#
#         # 原始 CAM
#         cam = self.cam(tensor, [self.target])[0, :]
#
#         # 原图热力图
#         vis = show_cam_on_image(norm, cam, use_rgb=True)
#         vis = vis[t:vis.shape[0] - b, l:vis.shape[1] - r]
#         Image.fromarray(vis).save(f"{save_dir}/{name}_heatmap.png")
#
#         # 生成方格化响应数据
#         grid = get_attention_grid_raw(cam, self.grid_size)
#         grid = grid[t:grid.shape[0] - b, l:grid.shape[1] - r]
#
#         # 1. 生成2D attention map（原有）
#         plot_attention_raw_with_colorbar(grid, f"{save_dir}/{name}_attention_2d.png", show_plot=True)
#
#         # 2. 额外生成3D attention map（新增）
#         plot_attention_3d(grid, f"{save_dir}/{name}_attention_3d.png", show_plot=True)
#
#         # 保存 metric
#         save_metric_raw(cam, self.grid_size, f"{save_dir}/{name}_metric.csv")
#
#     def __call__(self, img_path, save_dir):
#         os.makedirs(save_dir, exist_ok=True)
#         self.process(img_path, save_dir)
#
#
# # ==================== 运行 ====================
# if __name__ == "__main__":
#     model = YOLOHeatmap(
#         weight="Phong_Comparison_Experiments/train/DepthMap-250-32-640-SGD/weights/best.pt",
#         device="cuda:0",
#         method="GradCAMPlusPlus",
#         layers=[16, 19, 22],
#         conf=0.5,
#         grid_size=4
#     )
#     # 运行
#     model(r'DataPrepare/EXP1_3D/DepthMap/81.bmp', r'result')


# 忽略警告信息，避免输出干扰
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# 导入核心依赖库
import torch
import cv2
import os
import shutil
import copy
import csv
import numpy as np

np.random.seed(0)
from PIL import Image
from ultralytics import YOLO
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM, \
    KPCA_CAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
# 新增matplotlib相关导入
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图库

plt.switch_backend('TkAgg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 确保英文显示正常
plt.rcParams['axes.unicode_minus'] = False


# -------------------------- 预处理 --------------------------
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = (r, r)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (top, bottom, left, right)


# -------------------------- 生成方格数据（归一化到0~1） --------------------------
def get_attention_grid_normalized(grayscale_cam, grid_size=20):
    h, w = grayscale_cam.shape
    # 第一步：全局min-max归一化到0~1
    cam_min = np.min(grayscale_cam)
    cam_max = np.max(grayscale_cam)
    if cam_max - cam_min < 1e-6:  # 避免除0
        cam_normalized = np.zeros_like(grayscale_cam)
    else:
        cam_normalized = (grayscale_cam - cam_min) / (cam_max - cam_min)

    # 第二步：生成方格化数据
    grid_data = np.zeros((h, w), dtype=np.float32)
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            x2 = min(x + grid_size, w)
            y2 = min(y + grid_size, h)
            val = np.mean(cam_normalized[y:y2, x:x2])
            grid_data[y:y2, x:x2] = val
    return grid_data


# -------------------------- 2D可视化（归一化到0~1 + 固定colorbar刻度） --------------------------
def plot_attention_normalized_with_colorbar(grid_data, save_path, show_plot=True):
    # 白 → 红 颜色映射
    colors = [(1, 1, 1), (1, 0, 0)]
    cmap = mcolors.LinearSegmentedColormap.from_list("white2red", colors, N=256)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.axis('off')

    # 绘制归一化后的响应图，强制vmin/vmax=0~1
    im = ax.imshow(grid_data, cmap=cmap, vmin=0.0, vmax=1.0)

    # 右侧 colorbar，固定刻度0~1
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # 设置colorbar刻度为0~1，间隔0.2
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    cbar.set_label('Normalized Response', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.0)
    if show_plot:
        plt.show()
    plt.close()


# -------------------------- 3D可视化（归一化到0~1 + 固定colorbar刻度） --------------------------
def plot_attention_3d_normalized(grid_data, save_path, show_plot=True):
    """
    3D曲面图可视化归一化后的attention响应矩阵（0~1）
    """
    h, w = grid_data.shape

    # 生成XY坐标轴（对应图像像素坐标）
    x = np.arange(0, w, 1)
    y = np.arange(0, h, 1)
    X, Y = np.meshgrid(x, y)

    # 创建3D画布
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 白 → 红 颜色映射
    colors = [(1, 1, 1), (1, 0, 0)]
    cmap = mcolors.LinearSegmentedColormap.from_list("white2red", colors, N=256)

    # 绘制3D曲面图，强制vmin/vmax=0~1
    surf = ax.plot_surface(X, Y, grid_data,
                           cmap=cmap,
                           linewidth=0,
                           antialiased=True,
                           alpha=0.8,
                           vmin=0.0,
                           vmax=1.0)  # 固定颜色映射范围

    # 设置坐标轴标签
    ax.set_xlabel('X Pixel', fontsize=18, labelpad=10)
    ax.set_ylabel('Y Pixel', fontsize=18, labelpad=10)
    ax.set_zlabel('Response Value', fontsize=18, labelpad=10)

    # 设置坐标轴范围
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_zlim(0.0, 1.0)  # Z轴固定0~1

    # 添加颜色条（固定刻度0~1）
    cbar = fig.colorbar(surf, ax=ax, shrink=0.8, aspect=20, pad=0.01)
    # 固定colorbar刻度和标签
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])

    # 设置视角（可调整azim和elev改变观察角度）
    ax.view_init(elev=30, azim=45)

    # 优化布局
    plt.tight_layout()

    # 保存高分辨率图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)

    # 弹出显示窗口
    if show_plot:
        plt.show()

    # 关闭画布释放内存
    plt.close()


# -------------------------- 保存metric（同时保存原始值和归一化值） --------------------------
def save_metric_normalized(grayscale_cam, grid_size, save_path):
    h, w = grayscale_cam.shape
    # 计算归一化参数
    cam_min = np.min(grayscale_cam)
    cam_max = np.max(grayscale_cam)

    data = [['row', 'col', 'x1', 'y1', 'x2', 'y2', 'raw_response', 'normalized_response']]
    row = 0
    for y in range(0, h, grid_size):
        col = 0
        for x in range(0, w, grid_size):
            x2 = min(x + grid_size, w)
            y2 = min(y + grid_size, h)
            raw_val = np.mean(grayscale_cam[y:y2, x:x2])
            # 计算归一化值
            if cam_max - cam_min < 1e-6:
                norm_val = 0.0
            else:
                norm_val = (raw_val - cam_min) / (cam_max - cam_min)
            data.append([row, col, x, y, x2, y2, round(float(raw_val), 4), round(float(norm_val), 4)])
            col += 1
        row += 1
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(data)


# -------------------------- CAM 相关 --------------------------
class ActivationsAndGradients:
    def __init__(self, model, layers, transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.transform = transform
        self.handles = []
        for l in layers:
            self.handles.append(l.register_forward_hook(self.save_act))
            self.handles.append(l.register_forward_hook(self.save_grad))

    def save_act(self, m, i, o):
        a = o
        if self.transform: a = self.transform(a)
        self.activations.append(a.detach().cpu())

    def save_grad(self, m, i, o):
        if not o.requires_grad: return

        def _g(grad):
            if self.transform: grad = self.transform(grad)
            self.gradients = [grad.detach().cpu()] + self.gradients

        o.register_hook(_g)

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        out = self.model(x)
        return out

    def release(self):
        for h in self.handles: h.remove()


class YoloTarget(torch.nn.Module):
    def __init__(self, mode, conf, ratio):
        super().__init__()
        self.mode = mode
        self.conf = conf
        self.ratio = ratio

    def forward(self, x):
        res = []
        for i in range(int(x[0][0].size(0) * self.ratio)):
            v = x[0][0][i].max()
            if v < self.conf: break
            res.append(v)
        return sum(res)


# -------------------------- 主类 --------------------------
class YOLOHeatmap:
    def __init__(self, weight, device, method, layers, conf=0.5, grid_size=20):
        self.dev = torch.device(device)
        self.yolo = YOLO(weight).to(self.dev)
        self.model = self.yolo.model
        self.model.eval()
        for p in self.model.parameters(): p.requires_grad_(True)
        self.target_layers = [self.model.model[i] for i in layers]
        self.cam = eval(method)(self.model, self.target_layers)
        self.cam.activations_and_grads = ActivationsAndGradients(self.model, self.target_layers, None)
        self.target = YoloTarget('all', conf, 0.02)
        self.conf = conf
        self.grid_size = grid_size

    def process(self, img_path, save_dir):
        img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
        name = os.path.splitext(os.path.basename(img_path))[0]

        img, _, (t, b, l, r) = letterbox(img, 640)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        norm = rgb.astype(np.float32) / 255
        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(self.dev)

        # 原始 CAM
        cam = self.cam(tensor, [self.target])[0, :]

        # 原图热力图
        vis = show_cam_on_image(norm, cam, use_rgb=True)
        vis = vis[t:vis.shape[0] - b, l:vis.shape[1] - r]
        Image.fromarray(vis).save(f"{save_dir}/{name}_heatmap.png")

        # 生成归一化后的方格响应数据（0~1）
        grid = get_attention_grid_normalized(cam, self.grid_size)
        grid = grid[t:grid.shape[0] - b, l:grid.shape[1] - r]

        # 1. 生成2D attention map（归一化到0~1）
        plot_attention_normalized_with_colorbar(grid, f"{save_dir}/{name}_attention_2d.png", show_plot=True)

        # 2. 生成3D attention map（归一化到0~1）
        plot_attention_3d_normalized(grid, f"{save_dir}/{name}_attention_3d.png", show_plot=True)

        # 保存metric（原始值+归一化值）
        # save_metric_normalized(cam, self.grid_size, f"{save_dir}/{name}_metric.csv")

    def __call__(self, img_path, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.process(img_path, save_dir)


# ==================== 运行 ====================
if __name__ == "__main__":
    model = YOLOHeatmap(
        weight="Phong_Exp/train/Phong-V11-250-16-640-SGD/weights/best.pt",
        device="cuda:0",
        method="GradCAMPlusPlus",
        layers=[16, 19, 22],
        conf=0.5,
        grid_size=4
    )
    # 运行
    model(r'/path/to/LUT-AD/EXP/EXP3/ResponseMatric/GT/Phong/172.bmp', r'/path/to/LUT-AD/EXP/EXP3/ResponseMatric/Phong')