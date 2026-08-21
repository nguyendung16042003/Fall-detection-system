"""
So sanh Test Tang 2 (True Positive Rate) giua OSNet x0.5 (da chon) va
backbone dung chung (ban cu, gio la "doi thu" tu-so-sanh -- xem
"Xu ly van de 1 moi.md") tren 6 bien the cua video test that (goc + 5
augment an toan: sang/toi/nhieu/nen/lat -- xem make_p1_test_augmentations.py).

QUAN TRONG: 6 bien the la CUNG 1 video test that (2 lan A xuat hien lai) --
6 lan chay = 12 "case nhan dang", KHONG PHAI 12 su kien doc lap. Bao cao
ro ca muc goc (2 case, dang tin nhat) va muc mo rong (12 case, robustness).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reid_model import ReIDModel  # noqa: E402
from msmt17_dataset import get_eval_transform  # noqa: E402
from torchreid.reid.utils import FeatureExtractor  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
CKPT_BACKBONE = (AI_ROOT / "Coding" / "training" / "training 3" / "runs" / "classify"
                  / "Fall_Detection_Advanced_Loss" / "YOLOv8n_AFCL_Balanced-8" / "weights" / "best.pt")
CKPT_SHARED_HEAD = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p1" / "reid_head_msmt17_frozen_backbone.pt"
OSNET_CKPT = Path(__file__).resolve().parent / "osnet_x0_5_msmt17.pth"
P1_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "File Run Problem 1" / "P1 test"
RESULTS_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENROLLMENT = {
    "A": [P1_DIR / "Data 1" / "Scene 1-CAM 1.mp4", P1_DIR / "Data 1" / "Scene 2-CAM 2.mp4"],
    "B": [P1_DIR / "Data 1" / "Scene 3-CAM 1.mp4", P1_DIR / "Data 1" / "Scene 4-CAM 2.mp4"],
}
ORIGINAL_TEST = P1_DIR / "Data 2" / "Scene 5-CAM 2.mp4"
AUG_DIR = P1_DIR / "Data 2 (augmented)"
VARIANTS = ["original", "aug-bright_up", "aug-bright_down", "aug-noise", "aug-compress", "aug-flip"]
EXPECTED_ID = "A"
DET_CONF = 0.4
MIN_BUFFER_FRAMES = 5
MAX_BUFFER_FRAMES = 10
MATCH_THRESHOLD = 0.6


def get_test_video(variant):
    if variant == "original":
        return ORIGINAL_TEST
    return AUG_DIR / f"Scene 5-CAM 2_{variant}.mp4"


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
            out.append((track_id, crop))
    return out


# ---------- OSNet x0.5 ----------
def osnet_embed(extractor, crop_bgr):
    emb = extractor([crop_bgr])
    emb = torch.nn.functional.normalize(emb, dim=1)
    return emb.cpu().numpy().flatten()


def build_gallery_osnet(extractor, detector):
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
                        _, crop = people[0]
                        embs.append(osnet_embed(extractor, crop))
                idx += 1
            cap.release()
            if embs:
                avg = np.mean(embs, axis=0)
                embs_all.append(avg / np.linalg.norm(avg))
        if embs_all:
            avg = np.mean(embs_all, axis=0)
            gallery[person] = avg / np.linalg.norm(avg)
    return gallery


def run_test_osnet(extractor, detector, video_path):
    cap = cv2.VideoCapture(str(video_path))
    buffers = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for track_id, crop in crop_person(detector, frame):
            buf = buffers.setdefault(track_id, [])
            if len(buf) < MAX_BUFFER_FRAMES:
                buf.append(osnet_embed(extractor, crop))
    cap.release()
    queries = []
    for track_id, embs in buffers.items():
        if len(embs) < MIN_BUFFER_FRAMES:
            continue
        avg = np.mean(embs, axis=0)
        queries.append((track_id, avg / np.linalg.norm(avg)))
    return queries


# ---------- Backbone dung chung (cu) ----------
def shared_embed(model, transform, crop_bgr):
    from PIL import Image
    pil_img = Image.fromarray(crop_bgr[:, :, ::-1])
    tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    emb = model.inference_embedding(tensor)
    return emb.cpu().numpy().flatten()


def build_gallery_shared(model, transform, detector):
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
                        _, crop = people[0]
                        embs.append(shared_embed(model, transform, crop))
                idx += 1
            cap.release()
            if embs:
                avg = np.mean(embs, axis=0)
                embs_all.append(avg / np.linalg.norm(avg))
        if embs_all:
            avg = np.mean(embs_all, axis=0)
            gallery[person] = avg / np.linalg.norm(avg)
    return gallery


def run_test_shared(model, transform, detector, video_path):
    cap = cv2.VideoCapture(str(video_path))
    buffers = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for track_id, crop in crop_person(detector, frame):
            buf = buffers.setdefault(track_id, [])
            if len(buf) < MAX_BUFFER_FRAMES:
                buf.append(shared_embed(model, transform, crop))
    cap.release()
    queries = []
    for track_id, embs in buffers.items():
        if len(embs) < MIN_BUFFER_FRAMES:
            continue
        avg = np.mean(embs, axis=0)
        queries.append((track_id, avg / np.linalg.norm(avg)))
    return queries


def match_gallery(query_emb, gallery, threshold=MATCH_THRESHOLD):
    best_id, best_score = None, -1.0
    for person, gal_emb in gallery.items():
        score = float(np.dot(query_emb, gal_emb))
        if score > best_score:
            best_id, best_score = person, score
    if best_score > threshold:
        return best_id, best_score
    return "NEW", best_score


def main():
    print("Nap detector...")
    detector_osnet = YOLO(str(AI_ROOT / "yolov8n.pt"))
    detector_shared = YOLO(str(AI_ROOT / "yolov8n.pt"))

    print("Nap OSNet x0.5...")
    osnet_extractor = FeatureExtractor(model_name="osnet_x0_5", model_path=str(OSNET_CKPT), device="cpu")
    print("Nap backbone dung chung (cu)...")
    shared_model = ReIDModel(CKPT_BACKBONE, num_classes=1041, freeze_backbone=True).to(DEVICE)
    shared_model.load_state_dict(torch.load(CKPT_SHARED_HEAD, map_location=DEVICE))
    shared_model.eval()
    shared_transform = get_eval_transform(224)

    print("Xay Gallery (2 model)...")
    gallery_osnet = build_gallery_osnet(osnet_extractor, detector_osnet)
    gallery_shared = build_gallery_shared(shared_model, shared_transform, detector_shared)

    rows = []
    for variant in VARIANTS:
        video_path = get_test_video(variant)
        if not video_path.exists():
            print(f"[{variant}] thieu file, bo qua")
            continue

        queries_osnet = run_test_osnet(osnet_extractor, detector_osnet, video_path)
        queries_shared = run_test_shared(shared_model, shared_transform, detector_shared, video_path)

        for track_id, q_emb in queries_osnet:
            pred, score = match_gallery(q_emb, gallery_osnet)
            correct = pred == EXPECTED_ID
            print(f"[{variant}/OSNet] track_id={track_id} pred={pred} score={score:.4f} -- {'DUNG' if correct else 'SAI'}")
            rows.append({"variant": variant, "model": "osnet_x0_5", "track_id": track_id,
                         "pred": pred, "score": round(score, 4), "correct": correct})

        for track_id, q_emb in queries_shared:
            pred, score = match_gallery(q_emb, gallery_shared)
            correct = pred == EXPECTED_ID
            print(f"[{variant}/Shared] track_id={track_id} pred={pred} score={score:.4f} -- {'DUNG' if correct else 'SAI'}")
            rows.append({"variant": variant, "model": "shared_backbone", "track_id": track_id,
                         "pred": pred, "score": round(score, 4), "correct": correct})

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "p1_tang2_augmented_comparison.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    for model_name in ["osnet_x0_5", "shared_backbone"]:
        sub = df[df.model == model_name]
        orig = sub[sub.variant == "original"]
        print(f"\n--- {model_name} ---")
        print(f"  2 case GOC: {orig.correct.sum()}/{len(orig)} dung ({orig.correct.mean():.1%})")
        print(f"  {len(sub)} case MO RONG (6 bien the): {sub.correct.sum()}/{len(sub)} dung ({sub.correct.mean():.1%})")
    print("=" * 70)
    print(f"\nDa luu: {out_csv}")


if __name__ == "__main__":
    main()
