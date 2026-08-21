"""
Tao 2 video report cho Re-Identification (bai cua nhom, OSNet_x0.25 sau
Knowledge Distillation): video CHINH ghep 2 camera Scene 5 (giong het
make_p1_scene5_both_cams_video.py, chi doi model) + video phu Dang ky+Test
(giong het make_p1_report_video.py, chi doi model). Xuat vao folder
"Re-Identification/1. Bai cua nhom (OSNet x0.25 - Knowledge Distillation)".
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchreid.reid.utils import FeatureExtractor

sys.path.insert(0, str(Path(__file__).resolve().parent))

AI_ROOT = Path(__file__).resolve().parents[3]
P1_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "File Run Problem 1" / "P1 test"
OSNET_KD_CKPT = Path(__file__).resolve().parent / "osnet_x0_25_kd_msmt17_best.pt"
CAM1_VIDEO = P1_DIR / "Data 2" / "Scene 5-CAM 1.mp4"
CAM2_VIDEO = P1_DIR / "Data 2" / "Scene 5-CAM 2.mp4"
ENROLL_CLIP = P1_DIR / "Data 1" / "Scene 2-CAM 2.mp4"
ENROLLMENT = {
    "A": [P1_DIR / "Data 1" / "Scene 1-CAM 1.mp4", P1_DIR / "Data 1" / "Scene 2-CAM 2.mp4"],
    "B": [P1_DIR / "Data 1" / "Scene 3-CAM 1.mp4", P1_DIR / "Data 1" / "Scene 4-CAM 2.mp4"],
}
OUT_DIR = (AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Số liệu thống kê"
           / "Re-Identification" / "1. Bai cua nhom (OSNet x0.25 KD)")
DET_CONF = 0.4
TILE_H = 540


def crop_person(detector, frame, conf=DET_CONF):
    results = detector.track(frame, classes=[0], conf=conf, persist=True, verbose=False)[0]
    out = []
    if results.boxes is None:
        return out
    for box in results.boxes:
        if box.id is None:
            continue
        track_id = int(box.id[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size > 0:
            out.append((track_id, (x1, y1, x2, y2), crop))
    return out


def build_gallery(embed_fn, detector):
    gallery = {}
    for person, videos in ENROLLMENT.items():
        embs_all = []
        for v in videos:
            cap = cv2.VideoCapture(str(v))
            embs = []
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % 5 == 0:
                    people = crop_person(detector, frame)
                    if people:
                        _, _, crop = people[0]
                        embs.append(embed_fn(crop))
                idx += 1
            cap.release()
            if embs:
                avg = np.mean(embs, axis=0)
                embs_all.append(avg / np.linalg.norm(avg))
        if embs_all:
            avg = np.mean(embs_all, axis=0)
            gallery[person] = avg / np.linalg.norm(avg)
    return gallery


def match_gallery(query_emb, gallery, threshold=0.6):
    best_id, best_score = None, -1.0
    for person, gal_emb in gallery.items():
        score = float(np.dot(query_emb, gal_emb))
        if score > best_score:
            best_id, best_score = person, score
    if best_score > threshold:
        return best_id, best_score
    return "NEW", best_score


def analyze_video(video_path, embed_fn, detector, gallery, next_wrong_label_start=2):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    per_frame = []
    track_embs = {}
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
    return frame_tracks, track_result, fps, next_wrong_label


def render_tile(frame, frame_tracks_at_idx, track_result, cam_label):
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
    return frame


def make_title_card(text, size, seconds, fps):
    w, h = size
    frame = np.full((h, w, 3), 30, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 1.4, 3)
    cv2.putText(frame, text, ((w - tw) // 2, (h + th) // 2), font, 1.4, (255, 255, 255), 3)
    return [frame] * int(seconds * fps)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Nap OSNet x0.25 (Knowledge Distillation)...")
    detector1 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    detector2 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    detector_gallery = YOLO(str(AI_ROOT / "yolov8n.pt"))
    detector_secondary = YOLO(str(AI_ROOT / "yolov8n.pt"))
    extractor = FeatureExtractor(model_name="osnet_x0_25", model_path=str(OSNET_KD_CKPT), device="cpu")

    def embed(crop):
        emb = extractor([crop])
        emb = torch.nn.functional.normalize(emb, dim=1)
        return emb.cpu().numpy().flatten()

    print("Xay Gallery (A, B tu Data 1)...")
    gallery = build_gallery(embed, detector_gallery)

    # ---------- Video 1: CHINH -- ghep 2 camera Scene 5 ----------
    print("Phan tich Scene 5-CAM 1...")
    ft1, res1, fps1, next_label = analyze_video(CAM1_VIDEO, embed, detector1, gallery, next_wrong_label_start=2)
    print("  Ket qua CAM 1:")
    for tid, r in res1.items():
        print(f"    track_id={tid} -> {r['label']} (score={r['score']:.4f}, {'dung' if r['correct'] else 'sai'})")

    print("Phan tich Scene 5-CAM 2...")
    ft2, res2, fps2, _ = analyze_video(CAM2_VIDEO, embed, detector2, gallery, next_wrong_label_start=next_label)
    print("  Ket qua CAM 2:")
    for tid, r in res2.items():
        print(f"    track_id={tid} -> {r['label']} (score={r['score']:.4f}, {'dung' if r['correct'] else 'sai'})")

    n_correct = sum(1 for r in list(res1.values()) + list(res2.values()) if r["correct"])
    n_total = len(res1) + len(res2)
    print(f"\nTONG (CAM1+CAM2, 4 case goc): {n_correct}/{n_total} dung")

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

    out_path1 = OUT_DIR / "OSNet_x025KD_Scene5_ghep.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path1), fourcc, fps_out, (out_w, out_h))
    assert writer.isOpened(), f"VideoWriter khong mo duoc: {out_path1}"

    idx = 0
    black1 = black2 = None
    while idx < n_max:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if ret1:
            tile1 = render_tile(frame1, ft1.get(idx, []), res1, "CAM 1")
            black1 = np.zeros_like(tile1)
        else:
            tile1 = black1 if black1 is not None else np.zeros((TILE_H, tile_w_sample, 3), dtype=np.uint8)
        if ret2:
            tile2 = render_tile(frame2, ft2.get(idx, []), res2, "CAM 2")
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
    assert out_path1.exists() and out_path1.stat().st_size > 0, f"Video khong duoc ghi ra: {out_path1}"
    print(f"\nDa luu: {out_path1}")

    # ---------- Video 2: PHU -- Dang ky + Test (CAM2 only) ----------
    print("\nTao video phu (Dang ky + Test, CAM2 only)...")
    ft_sec, res_sec, fps_sec, _ = analyze_video(CAM2_VIDEO, embed, detector_secondary, gallery, next_wrong_label_start=2)
    cap_ref = cv2.VideoCapture(str(CAM2_VIDEO))
    w = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_ref.release()

    out_path2 = OUT_DIR / "OSNet_x025KD_dangky_test.mp4"
    writer2 = cv2.VideoWriter(str(out_path2), fourcc, fps_sec, (w, h))
    assert writer2.isOpened(), f"VideoWriter khong mo duoc: {out_path2}"

    for f in make_title_card("OSNet_x0.25_KD -- Doan 1: Dang ky Person 1 (Enrollment)", (w, h), 2.5, fps_sec):
        writer2.write(f)
    cap_enroll = cv2.VideoCapture(str(ENROLL_CLIP))
    while True:
        ret, frame = cap_enroll.read()
        if not ret:
            break
        cv2.putText(frame, "DANG KY: Person 1", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (46, 200, 80), 3)
        writer2.write(frame)
    cap_enroll.release()

    for f in make_title_card("OSNet_x0.25_KD -- Doan 2: Test (nguoi di qua vung chet nhieu lan)", (w, h), 2.5, fps_sec):
        writer2.write(f)
    cap_test = cv2.VideoCapture(str(CAM2_VIDEO))
    frame_idx = 0
    while True:
        ret, frame = cap_test.read()
        if not ret:
            break
        for track_id, bbox in ft_sec.get(frame_idx, []):
            res = res_sec.get(track_id)
            if res is None:
                continue
            color = (46, 160, 46) if res["correct"] else (40, 40, 220)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            text = f"{res['label']} ({res['score']:.2f})"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 12)), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        writer2.write(frame)
        frame_idx += 1
    cap_test.release()
    writer2.release()
    assert out_path2.exists() and out_path2.stat().st_size > 0, f"Video khong duoc ghi ra: {out_path2}"
    print(f"Da luu: {out_path2}")


if __name__ == "__main__":
    main()
