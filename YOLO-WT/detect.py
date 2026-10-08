from ultralytics import YOLO
if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'Abl_Exp/train/YOLO-WT-250-16-640-SGD-seed42/weights/best.pt')
    model.predict(source=r'assets/test.bmp',
                  device='0',
                  imgsz=640,
                  save=False,
                  project='runs/detect',
                  name='LUT',
                  show=False,
                  )

# model参数:该参数可以填入模型文件路径
# source参数:该参数可以填入需要推理的图片或者视频路径，如果打开摄像头推理则填入0就行
# save参数:该参数填入True，代表把推理结果保存下来，默认是不保存的，所以一般都填入True
# show参数:该参数填入True，代表把推理结果以窗口形式显示出来，默认是显示的，这个参数根据自己需求打开就行，不显示填False