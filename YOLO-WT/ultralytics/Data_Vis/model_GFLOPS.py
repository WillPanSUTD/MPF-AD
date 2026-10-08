from sympy import false
from ultralytics.models import YOLO  # 假设 Model 定义在 yolo.py 中
from ultralytics.models import RTDETR
# 创建模型
#model = RTDETR(model='cfg/models/rt-detr/rtdetr-resnet50.yaml')
model = YOLO(model='cfg/models/YOLO-WT/WDSConv.yaml')
print(model.info(detailed=True,verbose=True))
