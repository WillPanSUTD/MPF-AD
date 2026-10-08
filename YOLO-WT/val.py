from ultralytics.models import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == '__main__':

    model = YOLO('Abl_Exp/train/YOLO-WT-250-16-640-SGD-seed42/weights/best.pt')
    model.val(data='cfg/datasets/Phong.yaml',
              imgsz=640,
              batch=16,
              save_json=True,
              device='0',
              workers=0,
              project='Abl_Exp/val',
              name='YOLO-WT-250-16-640-SGD',)


