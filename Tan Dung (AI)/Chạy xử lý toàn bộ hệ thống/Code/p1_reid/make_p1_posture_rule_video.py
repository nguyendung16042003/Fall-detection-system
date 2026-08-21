"""
P1 -- bai test "khong bao gia": chay phan loai tu the DON CAMERA (SharedBackbone
.classify_from_grid(), tai dung nguyen tu P3 -- KHONG can P2/P3 fusion vi P1
theo thiet ke luon dam bao chi 1 nguoi/1 camera tai 1 thoi diem, "vung chet"
giua 2 cam) + RULE THAT (Coding/Pipeline/fall_rule.py) tren Scene 5 (P1 Data 2,
nguoi di qua vung chet 4 lan, KHONG PHAI canh nga).

KY VONG RO: 0 fall event -- day la bai kiem tra KHONG BAO GIA (false-positive
check), khong phai bai kiem tra Sensitivity (P1 khong co canh nga that nao,
xem audit truoc do). Neu ra event thi la false positive can xem lai, khong
phai ket qua "tot".
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p3_cross_camera"))
PIPELINE_DIR = Path(__file__).resolve().parents[3] / "Coding" / "Pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

from shared_backbone import SharedBackbone, preprocess_crop  # noqa: E402
from fall_rule import detect_fall_events  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
P1_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "File Run Problem 1" / "P1 test"
CKPT = (AI_ROOT / "Coding" / "training" / "training 3" / "runs" / "classify"
        / "Fall_Detection_Advanced_Loss" / "YOLOv8n_AFCL_Balanced-8" / "weights" / "best.pt")
CLASS_NAMES = ["bend", "exercise", "lie", "sit", "stand"]
DET_CONF = 0.4  # khop dung cac script P1 khac (khac P3, giu tham so rieng cua P1)
TILE_H = 540
FALL_BANNER_MS = 1500

CAM1_VIDEO = P1_DIR / "Data 2" / "Scene 5-CAM 1.mp4"
CAM2_VIDEO = P1_DIR / "Data 2" / "Scene 5-CAM 2.mp4"
OUT_DIR = (AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Số liệu thống kê"
           / "P1" / "4. Posture + Rule check (khong bao gia)")


def detect_track(detector, frame, conf=DET_CONF):
    """P1 dam bao <=1 nguoi/frame -- lay box dau tien neu co, kem track_id
    (dung .track() de co ID on dinh xuyen video, giong cac script P1 khac)."""
    results = detector.track(frame, classes=[0], conf=conf, persist=True, verbose=False)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return None
    box = results.boxes[0]
    if box.id is None:
        return None
    track_id = int(box.id[0])
    xyxy = tuple(box.xyxy[0].cpu().numpy().tolist())
    det_conf = float(box.conf[0])
    return track_id, xyxy, det_conf


def classify_single(backbone, frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return None, None
    x = preprocess_crop(crop)
    with torch.no_grad():
        grid = backbone(x)
        logits = backbone.classify_from_grid(grid)
        probs = torch.softmax(logits, dim=1)[0]
    return CLASS_NAMES[int(probs.argmax())], float(probs.max())


def analyze(backbone, detector, video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    per_frame = []
    records = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        det = detect_track(detector, frame, DET_CONF)
        rec = {"frame_idx": frame_idx, "bbox": None, "track_id": None, "label": None, "conf": None}
        if det is not None:
            track_id, bbox, det_conf = det
            label, conf = classify_single(backbone, frame, bbox)
            if label is not None:
                rec.update({"bbox": bbox, "track_id": track_id, "label": label, "conf": conf})
                x1, y1, x2, y2 = bbox
                records.append({
                    "frame_id": frame_idx, "person_id": track_id,
                    "pose": label, "pose_confidence": conf,
                    "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                    "det_confidence": det_conf, "held": False,
                })
        per_frame.append(rec)
    cap.release()
    return per_frame, records, fps


def render_tile(frame, rec, cam_label, fall_active):
    frame = frame.copy()
    if rec["bbox"] is not None:
        x1, y1, x2, y2 = map(int, rec["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (46, 160, 46), 3)
        text = f"ID{rec['track_id']} {rec['label']} ({rec['conf']:.2f})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), (46, 160, 46), -1)
        cv2.putText(frame, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    h, w = frame.shape[:2]
    scale = TILE_H / h
    frame = cv2.resize(frame, (int(w * scale), TILE_H))
    cv2.putText(frame, cam_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    if fall_active:
        cv2.putText(frame, "!!! FALL DETECTED !!!", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 4)
    return frame


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Nap SharedBackbone (don-camera) + 2 detector...")
    backbone = SharedBackbone(CKPT).eval()
    detector1 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    detector2 = YOLO(str(AI_ROOT / "yolov8n.pt"))

    print("Phan tich Scene 5-CAM 1...")
    per_frame1, records1, fps1 = analyze(backbone, detector1, CAM1_VIDEO)
    print("Phan tich Scene 5-CAM 2...")
    per_frame2, records2, fps2 = analyze(backbone, detector2, CAM2_VIDEO)

    n_cls1 = sum(1 for r in per_frame1 if r["label"] is not None)
    n_cls2 = sum(1 for r in per_frame2 if r["label"] is not None)
    print(f"  CAM1: {len(per_frame1)} frame, {n_cls1} frame co phan loai")
    print(f"  CAM2: {len(per_frame2)} frame, {n_cls2} frame co phan loai")

    # 2 camera KHONG bao gio thay cung 1 nguoi cung luc (vung chet, da xac
    # nhan qua audit) -- nen goi rule RIENG cho tung camera (khong gop track_id
    # qua 2 cam, vi la 2 detector/track doc lap).
    events1 = detect_fall_events(records1, fps=fps1)
    events2 = detect_fall_events(records2, fps=fps2)
    print(f"\nRULE THAT -- CAM1: {len(events1)} su kien | CAM2: {len(events2)} su kien")
    for e in events1:
        print(f"  CAM1 -> t={e['timestamp_ms']/1000:.2f}s person_id={e['person_id']} trigger={e['rule']['trigger']}")
    for e in events2:
        print(f"  CAM2 -> t={e['timestamp_ms']/1000:.2f}s person_id={e['person_id']} trigger={e['rule']['trigger']}")
    total_events = len(events1) + len(events2)
    if total_events == 0:
        print("  -> 0 su kien nga -- DUNG NHU KY VONG (Scene 5 khong phai canh nga).")
    else:
        print(f"  -> CANH BAO: {total_events} su kien nga -- FALSE POSITIVE can xem lai "
              f"(Scene 5 khong co canh nga that nao).")
    windows1 = [(e["timestamp_ms"], e["timestamp_ms"] + FALL_BANNER_MS) for e in events1]
    windows2 = [(e["timestamp_ms"], e["timestamp_ms"] + FALL_BANNER_MS) for e in events2]

    cap1 = cv2.VideoCapture(str(CAM1_VIDEO))
    ret, f0 = cap1.read()
    cap1.release()
    tile_w_sample = int(TILE_H * f0.shape[1] / f0.shape[0])
    out_w = tile_w_sample * 2 + 10
    out_h = TILE_H

    out_path = OUT_DIR / "P1_Scene5_posture_rule_ghep.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps1, (out_w, out_h))
    assert writer.isOpened(), f"VideoWriter khong mo duoc: {out_path}"

    cap1 = cv2.VideoCapture(str(CAM1_VIDEO))
    cap2 = cv2.VideoCapture(str(CAM2_VIDEO))
    n = max(len(per_frame1), len(per_frame2))
    black1 = black2 = None
    for i in range(n):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        t_ms = (i + 1) / fps1 * 1000.0
        fall1 = any(lo <= t_ms <= hi for lo, hi in windows1)
        fall2 = any(lo <= t_ms <= hi for lo, hi in windows2)
        if ret1:
            tile1 = render_tile(frame1, per_frame1[i], "Cam 1 (P1)", fall1)
            black1 = np.zeros_like(tile1)
        else:
            tile1 = black1 if black1 is not None else np.zeros((TILE_H, tile_w_sample, 3), dtype=np.uint8)
        if ret2:
            tile2 = render_tile(frame2, per_frame2[i], "Cam 2 (P1)", fall2)
            black2 = np.zeros_like(tile2)
        else:
            tile2 = black2 if black2 is not None else np.zeros((TILE_H, tile_w_sample, 3), dtype=np.uint8)
        if tile1.shape[1] != tile_w_sample:
            tile1 = cv2.resize(tile1, (tile_w_sample, TILE_H))
        if tile2.shape[1] != tile_w_sample:
            tile2 = cv2.resize(tile2, (tile_w_sample, TILE_H))
        sep = np.full((TILE_H, 10, 3), 255, dtype=np.uint8)
        combined = np.hstack([tile1, sep, tile2])
        writer.write(combined)
    cap1.release()
    cap2.release()
    writer.release()
    assert out_path.exists() and out_path.stat().st_size > 0, f"Video khong duoc ghi ra: {out_path}"
    print(f"\nDa luu: {out_path}")


if __name__ == "__main__":
    main()
