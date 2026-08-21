"""
P1 -- BAN THU NGHIEM voi OSNet_x0.25 (sau Knowledge Distillation tu
OSNet_x1.0, xem train_reid_kd_osnet025.py) thay cho OSNet_x0.5 goc. Giong het
make_p1_reid_rule_video.py (ReID + RULE THAT chay ngam) -- CHI khac o
extractor va ten file output, de khong ghi de video x0.5 da co san (dung lam
doi chieu truc tiep).

Luu y da bao truoc: eval Rank-1/mAP CHINH THUC tren bo query/gallery chuan
cho thay x0.25 KD (Rank-1 52.00%, mAP 27.46%) THAP HON x0.5 (71.14%/41.99%) --
danh doi lay toc do nhanh hon ~4 lan. Video nay de xac nhan truc quan tren
Scene 5 thuc te, khong phai de "chung minh x0.25 tot hon".
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

from msmt17_dataset import get_eval_transform  # noqa: E402  (giu import goc, torchreid can)
from torchreid.reid.utils import FeatureExtractor  # noqa: E402
from make_p1_scene5_both_cams_video import (  # noqa: E402
    ENROLLMENT, DET_CONF, crop_person, build_gallery, match_gallery,
)
from shared_backbone import SharedBackbone, preprocess_crop  # noqa: E402
from fall_rule import detect_fall_events  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
P1_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "File Run Problem 1" / "P1 test"
CKPT = (AI_ROOT / "Coding" / "training" / "training 3" / "runs" / "classify"
        / "Fall_Detection_Advanced_Loss" / "YOLOv8n_AFCL_Balanced-8" / "weights" / "best.pt")
OSNET_KD_CKPT = Path(__file__).resolve().parent / "osnet_x0_25_kd_msmt17_best.pt"
CLASS_NAMES = ["bend", "exercise", "lie", "sit", "stand"]
TILE_H = 540
FALL_BANNER_MS = 1500

CAM1_VIDEO = P1_DIR / "Data 2" / "Scene 5-CAM 1.mp4"
CAM2_VIDEO = P1_DIR / "Data 2" / "Scene 5-CAM 2.mp4"
OUT_DIR = (AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Số liệu thống kê"
           / "P1" / "4. Posture + Rule check (khong bao gia)")


def classify_pose(backbone, crop):
    x = preprocess_crop(crop)
    with torch.no_grad():
        grid = backbone(x)
        logits = backbone.classify_from_grid(grid)
        probs = torch.softmax(logits, dim=1)[0]
    return CLASS_NAMES[int(probs.argmax())], float(probs.max())


def analyze_video(video_path, embed_fn, backbone, detector, gallery, next_wrong_label_start=2):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    per_frame = []
    track_embs = {}
    pose_records = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = crop_person(detector, frame)
        for track_id, bbox, crop in dets:
            per_frame.append((frame_idx, track_id, bbox))
            buf = track_embs.setdefault(track_id, [])
            if len(buf) < 10:
                buf.append(embed_fn(crop))
            pose, pose_conf = classify_pose(backbone, crop)
            x1, y1, x2, y2 = bbox
            pose_records.append({
                "frame_id": frame_idx, "person_id": track_id,
                "pose": pose, "pose_confidence": pose_conf,
                "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                "det_confidence": 1.0, "held": False,
            })
        frame_idx += 1
    cap.release()

    track_result = {}
    next_wrong_label = next_wrong_label_start
    for track_id, embs in track_embs.items():
        if len(embs) < 5:
            continue
        avg = np.mean(embs, axis=0)
        avg = avg / np.linalg.norm(avg)
        pred, score = match_gallery(avg, gallery)
        correct = pred == "A"
        if correct:
            label = "Person 1"
        else:
            label = f"Person {next_wrong_label}"
            next_wrong_label += 1
        track_result[track_id] = {"pred": pred, "score": score, "correct": correct, "label": label}

    frame_tracks = {}
    for frame_idx, track_id, bbox in per_frame:
        frame_tracks.setdefault(frame_idx, []).append((track_id, bbox))

    events = detect_fall_events(pose_records, fps=fps)
    return frame_tracks, track_result, fps, next_wrong_label, events


def render_tile(frame, frame_tracks_at_idx, track_result, cam_label, fall_active=False):
    frame = frame.copy()
    for track_id, bbox in frame_tracks_at_idx:
        res = track_result.get(track_id)
        if res is None:
            continue
        color = (46, 160, 46) if res["correct"] else (40, 40, 220)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        text = f"{res['label']} ({res['score']:.2f})"
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
    print("Nap OSNet x0.25 (KD) + SharedBackbone (posture, chay ngam) + detector...")
    backbone = SharedBackbone(CKPT).eval()
    detector1 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    detector2 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    detector_gallery = YOLO(str(AI_ROOT / "yolov8n.pt"))
    extractor = FeatureExtractor(model_name="osnet_x0_25", model_path=str(OSNET_KD_CKPT), device="cpu")

    def embed(crop):
        emb = extractor([crop])
        emb = torch.nn.functional.normalize(emb, dim=1)
        return emb.cpu().numpy().flatten()

    print("Xay Gallery (A, B tu Data 1)...")
    gallery = build_gallery(embed, detector_gallery)

    print("Phan tich Scene 5-CAM 1 (ReID + posture ngam)...")
    ft1, res1, fps1, next_label, events1 = analyze_video(
        CAM1_VIDEO, embed, backbone, detector1, gallery, next_wrong_label_start=2)
    print("  ReID CAM 1:")
    for tid, r in res1.items():
        print(f"    track_id={tid} -> {r['label']} (score={r['score']:.4f}, {'dung' if r['correct'] else 'sai'})")
    print(f"  RULE THAT CAM 1: {len(events1)} su kien nga")
    for e in events1:
        print(f"    -> t={e['timestamp_ms']/1000:.2f}s person_id={e['person_id']} trigger={e['rule']['trigger']}")

    print("Phan tich Scene 5-CAM 2 (ReID + posture ngam)...")
    ft2, res2, fps2, _, events2 = analyze_video(
        CAM2_VIDEO, embed, backbone, detector2, gallery, next_wrong_label_start=next_label)
    print("  ReID CAM 2:")
    for tid, r in res2.items():
        print(f"    track_id={tid} -> {r['label']} (score={r['score']:.4f}, {'dung' if r['correct'] else 'sai'})")
    print(f"  RULE THAT CAM 2: {len(events2)} su kien nga")
    for e in events2:
        print(f"    -> t={e['timestamp_ms']/1000:.2f}s person_id={e['person_id']} trigger={e['rule']['trigger']}")

    n_correct = sum(1 for r in list(res1.values()) + list(res2.values()) if r["correct"])
    n_total = len(res1) + len(res2)
    print(f"\nTONG ReID (x0.25 KD): {n_correct}/{n_total} dung")
    total_events = len(events1) + len(events2)
    if total_events == 0:
        print("TONG RULE: 0 su kien nga -- DUNG NHU KY VONG (Scene 5 khong phai canh nga).")
    else:
        print(f"TONG RULE: CANH BAO {total_events} su kien nga -- can xem lai (khong phai canh nga that).")

    windows1 = [(e["timestamp_ms"], e["timestamp_ms"] + FALL_BANNER_MS) for e in events1]
    windows2 = [(e["timestamp_ms"], e["timestamp_ms"] + FALL_BANNER_MS) for e in events2]

    cap1 = cv2.VideoCapture(str(CAM1_VIDEO))
    cap2 = cv2.VideoCapture(str(CAM2_VIDEO))
    n1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    n2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    n_max = max(n1, n2)
    fps_out = fps1

    ret, f0 = cap1.read()
    cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
    tile_w_sample = int(TILE_H * f0.shape[1] / f0.shape[0])
    out_w = tile_w_sample * 2 + 10
    out_h = TILE_H

    out_path = OUT_DIR / "P1_Scene5_ReID_rule_ghep_x0_25_KD.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_out, (out_w, out_h))
    assert writer.isOpened(), f"VideoWriter khong mo duoc: {out_path}"

    idx = 0
    black1 = black2 = None
    while idx < n_max:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        t_ms = idx / fps_out * 1000.0
        fall1 = any(lo <= t_ms <= hi for lo, hi in windows1)
        fall2 = any(lo <= t_ms <= hi for lo, hi in windows2)
        if ret1:
            tile1 = render_tile(frame1, ft1.get(idx, []), res1, "CAM 1", fall_active=fall1)
            black1 = np.zeros_like(tile1)
        else:
            tile1 = black1 if black1 is not None else np.zeros((TILE_H, tile_w_sample, 3), dtype=np.uint8)
        if ret2:
            tile2 = render_tile(frame2, ft2.get(idx, []), res2, "CAM 2", fall_active=fall2)
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
        idx += 1
    cap1.release()
    cap2.release()
    writer.release()
    assert out_path.exists() and out_path.stat().st_size > 0, f"Video khong duoc ghi ra: {out_path}"
    print(f"\nDa luu: {out_path}")


if __name__ == "__main__":
    main()
