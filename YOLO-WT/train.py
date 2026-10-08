import warnings

from ultralytics import YOLO, settings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    settings["tensorboard"] = True
    # Main paper run: YOLO-WT on the Phong-rendered modality, no pretrained weights.
    # Swap `data=` for another cfg/datasets/*.yaml to reproduce the modality ablation,
    # or `model=` for cfg/models/YOLO-WT/{WDSConv,IWUpSample}.yaml for the module ablation.
    model = YOLO(model="cfg/models/YOLO-WT/YOLO-WT.yaml")
    model.train(
        data="cfg/datasets/Phong.yaml",
        imgsz=640,
        epochs=250,
        batch=16,
        workers=0,
        device="0",
        optimizer="SGD",
        resume=False,
        project="Abl_Exp/train",
        name="YOLO-WT-250-16-640-SGD",
        single_cls=False,
        cache=False,
        patience=0,
    )
