"""
Pipeline MVP: detect nguoi (YOLOv8n) -> crop bbox (+padding) -> classify tu the (5 lop)
Thay the cach lam cu (classify ca khung hinh) bang dung co che PGIE->SGIE ma
DeepStream cua Dung se dung tren Jetson: detect truoc, chi classify vung co nguoi.
"""
import argparse
import os
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

# 1_Single_Camera_Pipeline/pipeline_code/detect_classify_pipeline.py ->
# 1_Single_Camera_Pipeline/models/ (model di kem trong CHINH goi ban giao nay,
# khac duong dan du an dev goc).
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

DEFAULT_DETECTOR = MODELS_DIR / "person_detector.pt"
DEFAULT_CLASSIFIER = MODELS_DIR / "pose_classifier.pt"

CLASS_NAMES = ["bend", "exercise", "lie", "sit", "stand"]
PERSON_CLASS_ID = 0  # COCO: person
PAD_RATIO = 0.20  # padding quanh bbox, khop quy uoc trong mqtt_schema_v2.json
DET_CONF_THRESHOLD = 0.4
FRAME_SKIP = 5  # ~6 khung hinh/giay o video 30fps, du de khong bo sot chuyen dong nga

# Da phat hien qua debug: YOLOv8n (ca ban lon hon yolov8s) hoan toan KHONG detect
# duoc nguoi khi nam det tren san nhin tu goc camera cao/nghieng (thu conf tu 0.4
# xuong 0.02, tang imgsz len 960 -- van 0 box). Day la han che that cua detector
# COCO-pretrained voi tu the nam det (khac han bbox dung/ngoi quen thuoc). Khong
# sua duoc bang threshold -- xu ly bang cach GIU TAM vi tri bbox cuoi cung khi mat
# dau dot ngot trong vai frame lien tiep (hop ly vi nguoi vua nga thi gan nhu
# khong di chuyen), thay vi mat toan bo tin hieu ngay khi detector "mu" 1 frame.
GRACE_FRAMES = 6  # so lan lien tiep duoc giu bbox cu khi mat dau (o frame_skip=5, 25fps ~ 1.2s)
# Da thu 3/6/10 tren 64 video: 3 kem hon (F1 0.755), 6 va 10 cho ket qua GIONG HET
# nhau (da bao hoa) -> chon 6, khong can giu lau hon khong can thiet.


def crop_with_padding(frame, box_xyxy, pad_ratio=PAD_RATIO):
    x1, y1, x2, y2 = box_xyxy
    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = w * pad_ratio, h * pad_ratio
    H, W = frame.shape[:2]
    nx1 = max(0, int(x1 - pad_x))
    ny1 = max(0, int(y1 - pad_y))
    nx2 = min(W, int(x2 + pad_x))
    ny2 = min(H, int(y2 + pad_y))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return frame[ny1:ny2, nx1:nx2]


class DetectClassifyPipeline:
    """Tai model 1 lan, dung lai cho nhieu video (tranh load lai moi lan goi)."""

    def __init__(self, detector_path=DEFAULT_DETECTOR, classifier_path=DEFAULT_CLASSIFIER):
        self.detector = YOLO(str(detector_path))
        self.classifier = YOLO(str(classifier_path))

    def _classify_and_record(self, frame, frame_idx, person_id, xyxy, det_confidence, held):
        crop = crop_with_padding(frame, xyxy)
        if crop is None or crop.size == 0:
            return None
        cls_result = self.classifier.predict(crop, verbose=False)[0]
        cls_id = int(cls_result.probs.top1)
        pose_confidence = float(cls_result.probs.top1conf)
        return {
            "frame_id": frame_idx,
            "person_id": person_id,
            "bbox_x1": xyxy[0], "bbox_y1": xyxy[1],
            "bbox_x2": xyxy[2], "bbox_y2": xyxy[3],
            "det_confidence": det_confidence,
            "pose": CLASS_NAMES[cls_id],
            "pose_confidence": pose_confidence,
            "held": held,  # True = bbox giu tam vi mat dau, khong phai detector that su thay
        }

    def process_video(self, video_path, frame_skip=FRAME_SKIP, det_conf=DET_CONF_THRESHOLD,
                       grace_frames=GRACE_FRAMES):
        """Tra ve list record: {frame_id, person_id, bbox_xyxy, det_confidence, pose,
        pose_confidence, held}. held=True nghia la frame nay detector khong thay nguoi,
        dang dung tam bbox lan cuoi cung con thay (xem GRACE_FRAMES o tren)."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []

        records = []
        frame_idx = 0
        last_box = {}  # person_id -> {"bbox": [...], "missed": int}

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            # .track() thay vi .predict() de co san person_id (ByteTrack mac dinh cua Ultralytics)
            det_results = self.detector.track(
                frame, persist=True, classes=[PERSON_CLASS_ID],
                conf=det_conf, verbose=False,
            )[0]

            current_ids = set()
            if det_results.boxes is not None:
                for box in det_results.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    det_confidence = float(box.conf[0])
                    person_id = int(box.id[0]) if box.id is not None else -1
                    current_ids.add(person_id)
                    last_box[person_id] = {"bbox": xyxy, "missed": 0}

                    rec = self._classify_and_record(frame, frame_idx, person_id, xyxy, det_confidence, held=False)
                    if rec:
                        records.append(rec)

            # Nguoi vua mat dau dot ngot (co trong last_box nhung khong o frame nay):
            # giu tam bbox cu trong grace_frames lan tiep theo thay vi mat het tin hieu.
            for person_id, info in list(last_box.items()):
                if person_id in current_ids:
                    continue
                info["missed"] += 1
                if info["missed"] > grace_frames:
                    del last_box[person_id]
                    continue
                rec = self._classify_and_record(
                    frame, frame_idx, person_id, info["bbox"], det_confidence=0.0, held=True,
                )
                if rec:
                    records.append(rec)

        cap.release()
        return records


def main():
    parser = argparse.ArgumentParser(description="Chay pipeline detect+crop+classify tren 1 video, xuat CSV")
    parser.add_argument("video", help="Duong dan video input")
    parser.add_argument("--out", default="pipeline_output.csv", help="File CSV output")
    parser.add_argument("--detector", default=str(DEFAULT_DETECTOR))
    parser.add_argument("--classifier", default=str(DEFAULT_CLASSIFIER))
    parser.add_argument("--frame-skip", type=int, default=FRAME_SKIP)
    args = parser.parse_args()

    pipeline = DetectClassifyPipeline(args.detector, args.classifier)
    records = pipeline.process_video(args.video, frame_skip=args.frame_skip)

    df = pd.DataFrame(records)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"Da xu ly {len(records)} record tu {args.video}")
    print(f"Da luu: {os.path.abspath(args.out)}")
    if len(records):
        print(df["pose"].value_counts())


if __name__ == "__main__":
    main()
