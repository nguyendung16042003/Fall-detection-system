"""Video demo cho Phuong an 4 (Teacher Assistant, model tot nhat cua nhom
trong cac ban KD da thu -- "Bai cua nhom" trong so sanh voi Shared Backbone
va MSINet), dung PATTERN y het cac model khac: xuat 2 video --
  (1) Scene5_CAM1_CAM2_ghep: ca 2 camera Scene 5 (nguoi di qua vung chet)
      hien thi song song, nhan Person N theo ket qua gallery-matching.
  (2) dang_ky_va_test: doan Dang ky (clip Scene 2-CAM 2) + doan Test (toan bo
      Scene 5-CAM 2), bbox mau + nhan Person N.
Copy tu make_p1_msinet_video.py, chi doi embed_fn sang checkpoint PA4
(osnet_x0_25_kd_pa4_via_ta_msmt17_best.pt, dung build_model+load_pretrained_weights
giong compare_tang2_pa4.py).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchreid.reid.models import build_model
from torchreid.reid.utils import load_pretrained_weights
from torchvision import transforms as T
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))

PA4_CKPT = Path(__file__).resolve().parent / "osnet_x0_25_kd_pa4_via_ta_msmt17_best.pt"
PA4_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PA4_TRANSFORM = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def make_pa4_embed_fn(device=PA4_DEVICE):
    model = build_model("osnet_x0_25", num_classes=1041, loss="softmax", pretrained=False)
    load_pretrained_weights(model, str(PA4_CKPT))
    model.eval().to(device)

    def embed(crop_bgr):
        from PIL import Image
        pil_img = Image.fromarray(crop_bgr[:, :, ::-1])
        tensor = PA4_TRANSFORM(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(tensor)
        feat = torch.nn.functional.normalize(feat, dim=1)
        return feat.cpu().numpy().flatten()

    return embed

AI_ROOT = Path(__file__).resolve().parents[3]
P1_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "File Run Problem 1" / "P1 test"
AUG_DIR = P1_DIR / "Data 2 (augmented)"
CAM1_VIDEO = P1_DIR / "Data 2" / "Scene 5-CAM 1.mp4"
CAM2_VIDEO = P1_DIR / "Data 2" / "Scene 5-CAM 2.mp4"
ENROLL_CLIP = P1_DIR / "Data 1" / "Scene 2-CAM 2.mp4"
TEST_VIDEO = CAM2_VIDEO


def get_variant_video(cam, variant):
    """variant='original' -> Data 2 goc; nguoc lai -> Data 2 (augmented)."""
    if variant == "original":
        return P1_DIR / "Data 2" / f"Scene 5-{cam}.mp4"
    return AUG_DIR / f"Scene 5-{cam}_{variant}.mp4"
ENROLLMENT = {
    "A": [P1_DIR / "Data 1" / "Scene 1-CAM 1.mp4", P1_DIR / "Data 1" / "Scene 2-CAM 2.mp4"],
    "B": [P1_DIR / "Data 1" / "Scene 3-CAM 1.mp4", P1_DIR / "Data 1" / "Scene 4-CAM 2.mp4"],
}
OUT_DIR = (AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Số liệu thống kê" / "Re-Identification"
           / "Phương pháp 4 và backbone và MSI" / "0. Bai cua nhom (PA4)")
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
        for track_id, bbox, crop in crop_person(detector, frame):
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
        label = "Person 1" if correct else f"Person {next_wrong_label}"
        if not correct:
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


def make_ghep_video(embed_fn, detector1, detector2, gallery, cam1_video, cam2_video, out_name, tag):
    print(f"Phan tich {tag}-CAM 1...")
    ft1, res1, fps1, next_label = analyze_video(cam1_video, embed_fn, detector1, gallery, next_wrong_label_start=2)
    for tid, r in res1.items():
        print(f"    CAM1 track_id={tid} -> {r['label']} (score={r['score']:.4f}, {'dung' if r['correct'] else 'sai'})")

    print(f"Phan tich {tag}-CAM 2...")
    ft2, res2, fps2, _ = analyze_video(cam2_video, embed_fn, detector2, gallery, next_wrong_label_start=next_label)
    for tid, r in res2.items():
        print(f"    CAM2 track_id={tid} -> {r['label']} (score={r['score']:.4f}, {'dung' if r['correct'] else 'sai'})")

    n_correct = sum(1 for r in list(res1.values()) + list(res2.values()) if r["correct"])
    n_total = len(res1) + len(res2)
    print(f"\nTONG [{tag}]: {n_correct}/{n_total} dung khi chay CA 2 camera")

    cap1 = cv2.VideoCapture(str(cam1_video))
    cap2 = cv2.VideoCapture(str(cam2_video))
    n_max = max(int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps_out = fps1

    ret, f0 = cap1.read()
    cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
    tile_w = int(TILE_H * f0.shape[1] / f0.shape[0])
    out_w, out_h = tile_w * 2 + 10, TILE_H

    out_path = OUT_DIR / out_name
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (out_w, out_h))

    idx = 0
    black1 = black2 = None
    while idx < n_max:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if ret1:
            tile1 = render_tile(frame1, ft1.get(idx, []), res1, "CAM 1")
            black1 = np.zeros_like(tile1)
        else:
            tile1 = black1 if black1 is not None else np.zeros((TILE_H, tile_w, 3), dtype=np.uint8)
        if ret2:
            tile2 = render_tile(frame2, ft2.get(idx, []), res2, "CAM 2")
            black2 = np.zeros_like(tile2)
        else:
            tile2 = black2 if black2 is not None else np.zeros((TILE_H, tile_w, 3), dtype=np.uint8)
        if tile1.shape[1] != tile_w:
            tile1 = cv2.resize(tile1, (tile_w, TILE_H))
        if tile2.shape[1] != tile_w:
            tile2 = cv2.resize(tile2, (tile_w, TILE_H))
        sep = np.full((TILE_H, 10, 3), 255, dtype=np.uint8)
        writer.write(np.hstack([tile1, sep, tile2]))
        idx += 1
    cap1.release()
    cap2.release()
    writer.release()
    print(f"Da luu: {out_path}")


def make_title_card(text, size, seconds, fps):
    w, h = size
    frame = np.full((h, w, 3), 30, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 1.4, 3)
    cv2.putText(frame, text, ((w - tw) // 2, (h + th) // 2), font, 1.4, (255, 255, 255), 3)
    return [frame] * int(seconds * fps)


def make_enroll_test_video(embed_fn, detector, gallery):
    cap = cv2.VideoCapture(str(TEST_VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print("Phan tich video test...")
    per_frame = []
    track_embs = {}
    cap = cv2.VideoCapture(str(TEST_VIDEO))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for track_id, bbox, crop in crop_person(detector, frame):
            per_frame.append((frame_idx, track_id, bbox))
            buf = track_embs.setdefault(track_id, [])
            if len(buf) < 10:
                buf.append(embed_fn(crop))
        frame_idx += 1
    cap.release()

    track_result = {}
    next_wrong_label = 2
    for track_id, embs in track_embs.items():
        if len(embs) < 5:
            continue
        avg = np.mean(embs, axis=0)
        avg = avg / np.linalg.norm(avg)
        pred, score = match_gallery(avg, gallery)
        correct = pred == "A"
        label = "Person 1" if correct else f"Person {next_wrong_label}"
        if not correct:
            next_wrong_label += 1
        track_result[track_id] = {"pred": pred, "score": score, "correct": correct, "label": label}
    frame_tracks = {}
    for frame_idx, track_id, bbox in per_frame:
        frame_tracks.setdefault(frame_idx, []).append((track_id, bbox))

    out_path = OUT_DIR / "dangky_test.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for f in make_title_card("Phuong an 4 -- Doan 1: Dang ky Person 1 (Enrollment)", (w, h), 2.5, fps):
        writer.write(f)
    cap_enroll = cv2.VideoCapture(str(ENROLL_CLIP))
    while True:
        ret, frame = cap_enroll.read()
        if not ret:
            break
        cv2.putText(frame, "DANG KY: Person 1", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (46, 200, 80), 3)
        writer.write(frame)
    cap_enroll.release()

    for f in make_title_card("Phuong an 4 -- Doan 2: Test (nguoi di qua vung chet nhieu lan)", (w, h), 2.5, fps):
        writer.write(f)
    cap_test = cv2.VideoCapture(str(TEST_VIDEO))
    frame_idx = 0
    while True:
        ret, frame = cap_test.read()
        if not ret:
            break
        for track_id, bbox in frame_tracks.get(frame_idx, []):
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
        writer.write(frame)
        frame_idx += 1
    cap_test.release()
    writer.release()
    print(f"Da luu: {out_path}")
    for track_id, res in track_result.items():
        print(f"    track_id={track_id} -> {res['label']} (score={res['score']:.4f}, "
              f"{'dung' if res['correct'] else 'sai'})")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Nap Phuong an 4 (Teacher Assistant checkpoint)...")
    embed_fn = make_pa4_embed_fn()
    detector1 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    detector2 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    detector_gallery = YOLO(str(AI_ROOT / "yolov8n.pt"))
    detector_test = YOLO(str(AI_ROOT / "yolov8n.pt"))

    print("Xay Gallery...")
    gallery = build_gallery(embed_fn, detector_gallery)

    # Nhieu "scene" demo -- P1 test chi co dung 1 vi tri quay that (Scene 5,
    # nguoi di qua vung chet) nhung co 5 bien the dieu kien anh sang/nhieu/
    # nen/lat da quay/tao san (Data 2 (augmented)) -- dung lam nhieu "scene"
    # demo khac nhau thay vi chi 1 video goc, theo yeu cau nguoi dung.
    variants = [
        ("original", "Scene5_ghep.mp4", "Scene 5 (goc)"),
        ("aug-flip", "Scene5_ghep_flip.mp4", "Scene 5 (lat anh)"),
        ("aug-noise", "Scene5_ghep_noise.mp4", "Scene 5 (nhieu)"),
        ("aug-bright_down", "Scene5_ghep_toi.mp4", "Scene 5 (toi hon)"),
    ]
    for variant, out_name, tag in variants:
        cam1 = get_variant_video("CAM 1", variant)
        cam2 = get_variant_video("CAM 2", variant)
        if not cam1.exists() or not cam2.exists():
            print(f"\n[{tag}] thieu file, bo qua")
            continue
        print(f"\n=== Video: {tag} ===")
        make_ghep_video(embed_fn, detector1, detector2, gallery, cam1, cam2, out_name, tag)

    print("\n=== Video: Dang ky + Test ===")
    make_enroll_test_video(embed_fn, detector_test, gallery)


if __name__ == "__main__":
    main()
