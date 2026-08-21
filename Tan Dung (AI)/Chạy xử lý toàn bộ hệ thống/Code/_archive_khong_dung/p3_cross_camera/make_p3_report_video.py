"""
Tao VIDEO minh hoa report P3 (bai cua nhom -- Cross-Camera Attention Fusion):
ghep Scene 1-Cam 1 + Scene 1-Cam 2 (P3 Data 1, canh nga) canh nhau (trai/phai),
ve bbox + nhan tu the (bend/exercise/lie/sit/stand) theo dung pipeline THAT da
dung o pose_head_risk_p3_real.py (SharedBackbone + cross_camera_fuse + homography
that, KHONG train lai). Nhan hien thi la ket qua PHAN LOAI TU THE fused (khi ca 2
cam cung thay + khop khoang cach san), khong phai DUNG/SAI dinh danh nguoi (P3
khong lam Re-ID) -- giai thich chi tiet de trong Word, video chi ve nhan lop.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p2_homography"))

from shared_backbone import SharedBackbone, preprocess_crop  # noqa: E402
from cross_camera_fuse import build_distance_matrices, sgie_forward  # noqa: E402
from homography import apply_homography, foot_point_from_bbox  # noqa: E402
from p3_fall_rule_adapter import detect_fall_events_fused  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
P3_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "File Run Problem 3" / "P3"
RESULTS_P3 = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p3"
CKPT = (AI_ROOT / "Coding" / "training" / "training 3" / "runs" / "classify"
        / "Fall_Detection_Advanced_Loss" / "YOLOv8n_AFCL_Balanced-8" / "weights" / "best.pt")
CLASS_NAMES = ["bend", "exercise", "lie", "sit", "stand"]
DET_CONF = 0.2
DIST_THRESHOLD = 1.5
SIGMA = 2.0
TILE_H = 540
FALL_BANNER_MS = 1500  # khop dung Coding/Test/Test 4/visualize_any_video.py

# Debug thuc te: YOLOv8n bo lo nguoi rat thuong xuyen tren Cam 2 cua Scene 1
# (nguoi nam det, goc camera cao/nghieng, nen lon xon nhieu do vat -- da do thu
# 18/150 frame detect duoc). Day la han che THAT cua detector COCO-pretrained voi
# tu the nam det, DA duoc phat hien va xu ly dung cach trong pipeline chuan dong
# goi (Coding/Pipeline/detect_classify_pipeline.py, xem comment GRACE_FRAMES o do)
# -- ap dung LAI dung co che do: .track() + giu tam bbox cu (tren FRAME HIEN TAI,
# khong dung anh cu) trong ~1.2s khi mat dau dot ngot, thay vi mat tin hieu ngay.
GRACE_MS = 1200  # khop dung thoi luong da validate (GRACE_FRAMES=6 o frame_skip=5, ~25-30fps)

DATA_NAME = "P3 Data 1"
SCENE_ID = 1
CAM1_VIDEO = P3_DIR / DATA_NAME / f"Scene {SCENE_ID}-Cam 1.mp4"
CAM2_VIDEO = P3_DIR / DATA_NAME / f"Scene {SCENE_ID}-Cam 2.mp4"
OUT_DIR = (AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Số liệu thống kê"
           / "P3" / "1. Bai cua nhom (P3 Fusion)")


def load_h_by_height():
    with open(RESULTS_P3 / "h_by_height_p3.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {cam: {float(h): np.array(H) for h, H in d.items()} for cam, d in raw.items()}


def load_h_floor():
    with open(RESULTS_P3 / "homography_matrices_p3.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {cam: np.array(H) for cam, H in raw.items()}


def detect_one_tracked(detector, frame, held_state, grace_frames, conf=DET_CONF):
    """Nhu detect_classify_pipeline.py: .track() de co ID on dinh; neu mat dau
    dot ngot thi giu tam bbox cu (ap len FRAME HIEN TAI, khong dung anh cu) toi
    da grace_frames lien tiep truoc khi coi la that su mat nguoi. held_state la
    dict {"bbox":.., "missed": int} duoc cap nhat tai cho qua tung lan goi."""
    results = detector.track(frame, classes=[0], conf=conf, persist=True, verbose=False)[0]
    if results.boxes is not None and len(results.boxes) > 0:
        box = results.boxes[0]
        bbox = tuple(box.xyxy[0].cpu().numpy().tolist())
        held_state["bbox"] = bbox
        held_state["missed"] = 0
        return bbox, False
    if held_state["bbox"] is not None and held_state["missed"] < grace_frames:
        held_state["missed"] += 1
        return held_state["bbox"], True
    held_state["bbox"] = None
    return None, False


def analyze(backbone, detector1, detector2, H_floor, H_by_height):
    cap1 = cv2.VideoCapture(str(CAM1_VIDEO))
    cap2 = cv2.VideoCapture(str(CAM2_VIDEO))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    grace_frames = max(1, round(GRACE_MS / 1000.0 * fps))
    per_frame = []  # (frame_idx, bbox1, bbox2, label, conf, fused)
    frame_idx = 0
    held1 = {"bbox": None, "missed": 0}
    held2 = {"bbox": None, "missed": 0}
    n_held1 = n_held2 = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        frame_idx += 1

        bbox1, is_held1 = detect_one_tracked(detector1, frame1, held1, grace_frames)
        bbox2, is_held2 = detect_one_tracked(detector2, frame2, held2, grace_frames)
        n_held1 += is_held1
        n_held2 += is_held2
        rec = {"frame_idx": frame_idx, "bbox1": bbox1, "bbox2": bbox2,
               "held1": is_held1, "held2": is_held2,
               "label": None, "conf": None, "fused": False}

        if bbox1 is not None and bbox2 is not None:
            foot1 = apply_homography(H_floor["P3 Cam1"], foot_point_from_bbox(bbox1))
            foot2 = apply_homography(H_floor["P3 Cam2"], foot_point_from_bbox(bbox2))
            dist = np.linalg.norm(np.array(foot1) - np.array(foot2))
            crop1 = frame1[int(bbox1[1]):int(bbox1[3]), int(bbox1[0]):int(bbox1[2])]
            crop2 = frame2[int(bbox2[1]):int(bbox2[3]), int(bbox2[0]):int(bbox2[2])]
            if dist <= DIST_THRESHOLD and crop1.size > 0 and crop2.size > 0:
                x1 = preprocess_crop(crop1)
                x2 = preprocess_crop(crop2)
                with torch.no_grad():
                    grid1 = backbone(x1)
                    h, w = grid1.shape[2], grid1.shape[3]
                    dist_1to2, dist_2to1 = build_distance_matrices(
                        (h, w), foot1, foot2, H_by_height["P3 Cam1"], H_by_height["P3 Cam2"],
                        bbox1, bbox2)
                    v_fused = sgie_forward(backbone, x1, x2, has_both_cams=True,
                                            dist_1to2=dist_1to2, dist_2to1=dist_2to1, sigma=SIGMA)
                    logits_fused = backbone.classify_from_vector(v_fused)
                    probs_fused = torch.softmax(logits_fused, dim=1)[0]
                rec["label"] = CLASS_NAMES[int(probs_fused.argmax())]
                rec["conf"] = float(probs_fused.max())
                rec["fused"] = True
        per_frame.append(rec)
    cap1.release()
    cap2.release()
    print(f"  grace_frames={grace_frames} (~{GRACE_MS}ms @ {fps:.1f}fps) "
          f"-- Cam1 held {n_held1} frame, Cam2 held {n_held2} frame")
    return per_frame, fps


def render_tile(frame, bbox, label, conf, fused, cam_label, held=False, fall_active=False):
    frame = frame.copy()
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        color = (46, 160, 46) if fused else (180, 180, 60)
        thickness = 4 if held else 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        if label is not None:
            text = f"{label} ({conf:.2f})" + (" [fused]" if fused else "") + (" (held)" if held else "")
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 12)), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
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
    print("Nap SharedBackbone + 2 detector + calib that P3...")
    backbone = SharedBackbone(CKPT).eval()
    detector1 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    detector2 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    H_by_height = load_h_by_height()
    H_floor = load_h_floor()

    print(f"Phan tich {DATA_NAME} / Scene {SCENE_ID} (Cam 1 + Cam 2)...")
    per_frame, fps = analyze(backbone, detector1, detector2, H_floor, H_by_height)

    n_fused = sum(1 for r in per_frame if r["fused"])
    n_lie = sum(1 for r in per_frame if r["fused"] and r["label"] == "lie")
    print(f"  {len(per_frame)} frame, {n_fused} frame gop duoc ca 2 cam, "
          f"{n_lie}/{n_fused} frame gop nhan dung 'lie' (nga)")

    events = detect_fall_events_fused(per_frame, fps)
    print(f"\nRULE THAT (fall_rule.detect_fall_events) -- {len(events)} su kien nga:")
    for e in events:
        print(f"  -> t={e['timestamp_ms']/1000:.2f}s person_id={e['person_id']} "
              f"trigger={e['rule']['trigger']} class_before={e['detection']['class_before']}")
    if not events:
        print("  -> Khong phat hien su kien nga nao.")
    event_windows = [(e["timestamp_ms"], e["timestamp_ms"] + FALL_BANNER_MS) for e in events]

    cap1 = cv2.VideoCapture(str(CAM1_VIDEO))
    ret, f0 = cap1.read()
    cap1.release()
    tile_w_sample = int(TILE_H * f0.shape[1] / f0.shape[0])
    out_w = tile_w_sample * 2 + 10
    out_h = TILE_H

    out_path = OUT_DIR / f"P3_video_Scene{SCENE_ID}_Cam1_Cam2_ghep.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
    assert writer.isOpened(), f"VideoWriter khong mo duoc: {out_path}"

    cap1 = cv2.VideoCapture(str(CAM1_VIDEO))
    cap2 = cv2.VideoCapture(str(CAM2_VIDEO))
    for rec in per_frame:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        t_ms = rec["frame_idx"] / fps * 1000.0
        fall_active = any(lo <= t_ms <= hi for lo, hi in event_windows)
        tile1 = render_tile(frame1, rec["bbox1"], rec["label"], rec["conf"], rec["fused"], "Cam 1",
                            held=rec["held1"], fall_active=fall_active)
        tile2 = render_tile(frame2, rec["bbox2"], rec["label"], rec["conf"], rec["fused"], "Cam 2",
                            held=rec["held2"], fall_active=fall_active)
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
