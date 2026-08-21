"""So sanh YOLOv8n-cls (dang deploy that) voi vai classifier NHE khac (KHONG
phai detector) tren cung 1 nguon crop nguoi that -- phuc vu slide "vi sao chon
YOLOv8n-cls". Tai su dung 2 video benchmark cu (Datasets/File Test 3/
gialap_vidnga_rtsp_1.mp4 va _2.mp4 -- da dung truoc day de so YOLOv8n vs
YOLOv8x DETECTOR, xem Coding/Test/Test 4/visualize_file_test3.py) de crop
nguoi that lam input.

QUAN TRONG -- gioi han ro rang:
- Cac classifier khac (MobileNetV3-Small, ShuffleNetV2 x1.0, EfficientNet-B0,
  SqueezeNet1.1) dung trong so ImageNet CO SAN (torchvision), CHUA fine-tune
  tren posture data -- vi vay CHI so sanh chi phi tinh toan (FPS/params/kich
  thuoc model), KHONG so sanh accuracy (khong cong bang, YOLOv8n-cls da duoc
  fine-tune rieng con cac model kia thi chua).
- Chay tren MAY DEV (khong phai Jetson Nano that -- SSH toi Jetson bi
  timeout luc chay, xem log). Danh dau ro la so lieu TUONG DOI (dung de so
  sanh CAC MODEL VOI NHAU tren CUNG 1 may), KHONG phai so lieu FPS/RAM tren
  Jetson nhu cac bang khac trong report.
"""
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO

AI_ROOT = Path(__file__).resolve().parents[3]
VIDEOS = [
    AI_ROOT / "Datasets" / "File Test 3" / "gialap_vidnga_rtsp_1.mp4",
    AI_ROOT / "Datasets" / "File Test 3" / "gialap_vidnga_rtsp_2.mp4",
]
YOLO_CLS_CKPT = (AI_ROOT / "Coding" / "training" / "training 3" / "runs" / "classify"
                 / "Fall_Detection_Advanced_Loss" / "YOLOv8n_AFCL_Balanced-8" / "weights" / "best.pt")
OUT_CSV = Path(__file__).resolve().parent / "benchmark_lightweight_classifiers.csv"

DEVICE = "cpu"  # nhat quan CPU cho CA cac model -- so sanh TUONG DOI giua cac
# model voi nhau tren CUNG 1 thiet bi, khong phai so lieu Jetson that.
N_WARMUP = 5
N_REPEAT = 3  # lap lai toan bo tap crop 3 lan de FPS on dinh hon
IMGSZ = 224
MAX_CROPS = 150  # gioi han so crop thu thap (tranh qua lau), du de do FPS on dinh


def collect_person_crops(videos, max_crops=MAX_CROPS):
    detector = YOLO(str(AI_ROOT / "yolov8n.pt"))
    crops = []
    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        while cap.isOpened() and len(crops) < max_crops:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % 5 != 0:
                continue
            results = detector.predict(frame, classes=[0], conf=0.4, verbose=False)[0]
            if results.boxes is None:
                continue
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size == 0:
                    continue
                crops.append(crop)
                if len(crops) >= max_crops:
                    break
        cap.release()
        print(f"  {video_path.name}: da thu thap den {len(crops)} crop")
    return crops


def build_models():
    """Tra ve dict {ten: (model_eval, preprocess_fn, n_params, so luot pass-forward
    can 1 anh)}. Tat ca deu .eval(), khong train."""
    out = {}

    # ---- YOLOv8n-cls dang deploy (fine-tuned that tren v4_split_flat) ----
    yolo_cls = YOLO(str(YOLO_CLS_CKPT))

    def yolo_predict(crop_bgr):
        yolo_cls.predict(crop_bgr, verbose=False, imgsz=IMGSZ)

    n_params_yolo = sum(p.numel() for p in yolo_cls.model.parameters())
    out["YOLOv8n-cls (deployed, fine-tuned)"] = (yolo_predict, n_params_yolo, str(YOLO_CLS_CKPT))

    # ---- torchvision lightweight classifiers (ImageNet pretrained, CHUA fine-tune) ----
    tv_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMGSZ, IMGSZ)),
        transforms.ToTensor(),
    ])

    tv_models = {
        "MobileNetV3-Small (ImageNet, chua fine-tune)": models.mobilenet_v3_small(weights="DEFAULT"),
        "ShuffleNetV2 x1.0 (ImageNet, chua fine-tune)": models.shufflenet_v2_x1_0(weights="DEFAULT"),
        "EfficientNet-B0 (ImageNet, chua fine-tune)": models.efficientnet_b0(weights="DEFAULT"),
        "SqueezeNet1.1 (ImageNet, chua fine-tune)": models.squeezenet1_1(weights="DEFAULT"),
    }
    for name, m in tv_models.items():
        m.eval().to(DEVICE)
        n_params = sum(p.numel() for p in m.parameters())

        def make_predict(model_ref):
            def predict(crop_bgr):
                x = tv_transform(crop_bgr).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    model_ref(x)
            return predict

        out[name] = (make_predict(m), n_params, None)

    return out


def measure_fps(predict_fn, crops):
    for c in crops[:N_WARMUP]:
        predict_fn(c)
    t0 = time.perf_counter()
    n_runs = 0
    for _ in range(N_REPEAT):
        for c in crops:
            predict_fn(c)
            n_runs += 1
    elapsed = time.perf_counter() - t0
    return n_runs / elapsed, elapsed / n_runs * 1000


def main():
    print("Thu thap crop nguoi that tu 2 video benchmark cu...")
    crops = collect_person_crops(VIDEOS)
    print(f"Tong so crop dung de benchmark: {len(crops)}\n")

    print("Nap cac model (YOLOv8n-cls fine-tuned + 4 classifier nhe ImageNet pretrained)...")
    model_map = build_models()

    rows = []
    for name, (predict_fn, n_params, ckpt_path) in model_map.items():
        print(f"Do FPS: {name} ...")
        fps, ms_per_img = measure_fps(predict_fn, crops)
        size_mb = None
        if ckpt_path:
            size_mb = Path(ckpt_path).stat().st_size / (1024 * 1024)
        rows.append({
            "model": name,
            "params_M": round(n_params / 1e6, 2),
            "size_MB": round(size_mb, 1) if size_mb else None,
            "fps_cpu_devmachine": round(fps, 2),
            "ms_per_image": round(ms_per_img, 2),
        })
        print(f"  -> {fps:.2f} FPS ({ms_per_img:.2f} ms/anh), {n_params/1e6:.2f}M params")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nDa luu: {OUT_CSV}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
