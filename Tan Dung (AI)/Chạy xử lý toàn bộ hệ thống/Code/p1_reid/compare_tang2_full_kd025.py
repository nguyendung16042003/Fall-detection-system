"""
Ban thay the compare_tang2_full.py: dung OSNet_x0.25 (sau Knowledge
Distillation tu x1.0, xem train_reid_kd_osnet025.py) thay cho OSNet_x0.5 lam
"bai cua nhom", giu nguyen backbone dung chung lam doi thu tu-so-sanh de so
sanh cong bang (cung data, cung quy trinh). Ca 2 camera Scene 5 x 6 bien the
(goc + 5 augment) = 24 case moi model.
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
OSNET_KD_CKPT = Path(__file__).resolve().parent / "osnet_x0_25_kd_msmt17_best.pt"
P1_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "File Run Problem 1" / "P1 test"
AUG_DIR = P1_DIR / "Data 2 (augmented)"
ENROLLMENT = {
    "A": [P1_DIR / "Data 1" / "Scene 1-CAM 1.mp4", P1_DIR / "Data 1" / "Scene 2-CAM 2.mp4"],
    "B": [P1_DIR / "Data 1" / "Scene 3-CAM 1.mp4", P1_DIR / "Data 1" / "Scene 4-CAM 2.mp4"],
}
RESULTS_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DET_CONF = 0.4
VARIANTS = ["original", "aug-bright_up", "aug-bright_down", "aug-noise", "aug-compress", "aug-flip"]
CAMS = ["CAM 1", "CAM 2"]


def get_video(cam, variant):
    if variant == "original":
        return P1_DIR / "Data 2" / f"Scene 5-{cam}.mp4"
    return AUG_DIR / f"Scene 5-{cam}_{variant}.mp4"


def crop_person(detector, frame, conf=DET_CONF):
    results = detector.track(frame, classes=[0], conf=conf, persist=True, verbose=False)[0]
    out = []
    if results.boxes is None:
        return out
    for box in results.boxes:
        if box.id is None:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size > 0:
            out.append((int(box.id[0]), crop))
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
                        _, crop = people[0]
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


def run_test(video_path, embed_fn, detector, gallery):
    cap = cv2.VideoCapture(str(video_path))
    buffers = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for track_id, crop in crop_person(detector, frame):
            buf = buffers.setdefault(track_id, [])
            if len(buf) < 10:
                buf.append(embed_fn(crop))
    cap.release()
    results = []
    for track_id, embs in buffers.items():
        if len(embs) < 5:
            continue
        avg = np.mean(embs, axis=0)
        avg = avg / np.linalg.norm(avg)
        pred, score = match_gallery(avg, gallery)
        results.append((track_id, pred, score, pred == "A"))
    return results


def evaluate_model(tag, embed_fn, detector, gallery):
    rows = []
    for cam in CAMS:
        for variant in VARIANTS:
            v = get_video(cam, variant)
            if not v.exists():
                print(f"  [{tag}/{cam}/{variant}] thieu file, bo qua")
                continue
            results = run_test(v, embed_fn, detector, gallery)
            for track_id, pred, score, correct in results:
                rows.append({"model": tag, "cam": cam, "variant": variant,
                             "track_id": track_id, "pred": pred, "score": round(score, 4),
                             "correct": correct})
            print(f"  [{tag}/{cam}/{variant}] {len(results)} track -> "
                  f"{sum(r[3] for r in results)}/{len(results)} dung")
    return rows


def main():
    print("Nap OSNet x0.25 (KD)...")
    detector_osnet = YOLO(str(AI_ROOT / "yolov8n.pt"))
    osnet_extractor = FeatureExtractor(model_name="osnet_x0_25", model_path=str(OSNET_KD_CKPT), device="cpu")

    def osnet_embed(crop):
        emb = osnet_extractor([crop])
        emb = torch.nn.functional.normalize(emb, dim=1)
        return emb.cpu().numpy().flatten()

    print("Nap backbone dung chung...")
    detector_shared = YOLO(str(AI_ROOT / "yolov8n.pt"))
    shared_model = ReIDModel(CKPT_BACKBONE, num_classes=1041, freeze_backbone=True).to(DEVICE)
    shared_model.load_state_dict(torch.load(CKPT_SHARED_HEAD, map_location=DEVICE))
    shared_model.eval()
    shared_transform = get_eval_transform(224)

    def shared_embed(crop):
        from PIL import Image
        pil_img = Image.fromarray(crop[:, :, ::-1])
        tensor = shared_transform(pil_img).unsqueeze(0).to(DEVICE)
        emb = shared_model.inference_embedding(tensor)
        return emb.cpu().numpy().flatten()

    print("Xay Gallery (2 model)...")
    gallery_osnet = build_gallery(osnet_embed, detector_osnet)
    gallery_shared = build_gallery(shared_embed, detector_shared)

    print("\nDanh gia OSNet x0.25 KD (ca 2 camera x 6 bien the)...")
    rows_osnet = evaluate_model("osnet_x0_25_kd", osnet_embed, detector_osnet, gallery_osnet)

    print("\nDanh gia Backbone dung chung (ca 2 camera x 6 bien the)...")
    rows_shared = evaluate_model("shared_backbone", shared_embed, detector_shared, gallery_shared)

    df = pd.DataFrame(rows_osnet + rows_shared)
    out_csv = RESULTS_DIR / "p1_tang2_full_comparison_kd025.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    for model_name in ["osnet_x0_25_kd", "shared_backbone"]:
        sub = df[df.model == model_name]
        orig = sub[sub.variant == "original"]
        print(f"\n--- {model_name} ---")
        print(f"  4 case GOC (ca 2 camera, chua augment): {orig.correct.sum()}/{len(orig)} dung ({orig.correct.mean():.1%})")
        print(f"  {len(sub)} case DAY DU (2 camera x 6 bien the): {sub.correct.sum()}/{len(sub)} dung ({sub.correct.mean():.1%})")
    print("=" * 70)
    print(f"\nDa luu: {out_csv}")


if __name__ == "__main__":
    main()
