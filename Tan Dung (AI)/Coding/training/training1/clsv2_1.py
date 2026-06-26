from ultralytics import YOLO
import torch
import multiprocessing
import os

# On Windows, use file-system based shared memory to avoid
# "Couldn't open shared file mapping" DataLoader errors.
if os.name == "nt":
    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
    except Exception:
        pass
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except Exception:
        pass

def main():
    model = YOLO("yolov8n-cls.pt")

    model.train(
        data=r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)\Datasets\File Traning\v2_split_flat",
        epochs=120,
        imgsz=416,
        batch=16,
        device=0,
        workers=4,
        project="runs/classify",
        name="clsv2_1"
    )

if __name__ == "__main__":
    main()