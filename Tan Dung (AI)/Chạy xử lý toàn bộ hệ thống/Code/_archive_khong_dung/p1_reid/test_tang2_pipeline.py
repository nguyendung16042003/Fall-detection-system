"""
P1 Test Tang 2 -- chay dung pipeline 4 lop (rut gon, 1 tien trinh -- khong
MQTT/cloud that vi chi test logic) tren data tu quay, dung Re-ID Head DA
TRAIN (khong phai OSNet). Doi chieu voi osnet_baseline.py (Phase 0.1) de so
sanh 2 dong "True Positive Rate".

LOP 1: extract_embedding = model.inference_embedding (backbone dong bang +
       Re-ID head train o Giai doan A).
LOP 2: buffer toi da 10 frame/track_id (dung MAX_QUERY_FRAMES=10 nhu huong
       dan "Buffer 10 frame").
LOP 3+4: rut gon 1 tien trinh (khong MQTT that) -- trung binh embedding trong
       buffer, chuan hoa L2, so cosine similarity voi Gallery, nguong 0.6
       (dung nguong nhu osnet_baseline.py de so sanh cong bang).
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

AI_ROOT = Path(__file__).resolve().parents[3]
CKPT = (AI_ROOT / "Coding" / "training" / "training 3" / "runs" / "classify"
        / "Fall_Detection_Advanced_Loss" / "YOLOv8n_AFCL_Balanced-8" / "weights" / "best.pt")
P1_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "File Run Problem 1" / "P1 test"
RESULTS_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENROLLMENT = {
    "A": [P1_DIR / "Data 1" / "Scene 1-CAM 1.mp4", P1_DIR / "Data 1" / "Scene 2-CAM 2.mp4"],
    "B": [P1_DIR / "Data 1" / "Scene 3-CAM 1.mp4", P1_DIR / "Data 1" / "Scene 4-CAM 2.mp4"],
}
TEST_VIDEO = P1_DIR / "Data 2" / "Scene 5-CAM 2.mp4"
TEST_EXPECTED_ID = "A"

DET_CONF = 0.4
MIN_BUFFER_FRAMES = 5
MAX_BUFFER_FRAMES = 10  # "Buffer 10 frame" theo dung dac ta Lop 2
MATCH_THRESHOLD = 0.6


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


@torch.no_grad()
def extract_embedding(model, transform, crop_bgr):
    """LOP 1."""
    from PIL import Image
    pil_img = Image.fromarray(crop_bgr[:, :, ::-1])
    tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    emb = model.inference_embedding(tensor)
    return emb.cpu().numpy().flatten()


def build_gallery_entry(video_path, detector, model, transform, sample_every_n_frames=5):
    cap = cv2.VideoCapture(str(video_path))
    embeddings = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every_n_frames == 0:
            people = crop_person(detector, frame)
            if people:
                _, crop = people[0]
                embeddings.append(extract_embedding(model, transform, crop))
        frame_idx += 1
    cap.release()
    if not embeddings:
        return None
    avg = np.mean(embeddings, axis=0)
    return avg / np.linalg.norm(avg)


def build_gallery(detector, model, transform):
    gallery = {}
    for person, videos in ENROLLMENT.items():
        embs = [build_gallery_entry(v, detector, model, transform) for v in videos]
        embs = [e for e in embs if e is not None]
        if not embs:
            print(f"  CANH BAO: khong trich duoc embedding nao cho {person}")
            continue
        avg = np.mean(embs, axis=0)
        gallery[person] = avg / np.linalg.norm(avg)
        print(f"  Gallery[{person}] xay tu {len(embs)}/{len(videos)} video enrollment")
    return gallery


def extract_test_queries(video_path, detector, model, transform):
    """LOP 2 -- buffer toi da MAX_BUFFER_FRAMES/track_id, moi track_id moi la
    1 query doc lap (1 lan xuat hien lai)."""
    cap = cv2.VideoCapture(str(video_path))
    buffers = {}
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for track_id, crop in crop_person(detector, frame):
            buf = buffers.setdefault(track_id, [])
            if len(buf) < MAX_BUFFER_FRAMES:
                buf.append(extract_embedding(model, transform, crop))
        frame_idx += 1
    cap.release()

    queries = []
    for track_id, embs in buffers.items():
        if len(embs) < MIN_BUFFER_FRAMES:
            continue
        avg = np.mean(embs, axis=0)
        queries.append((track_id, avg / np.linalg.norm(avg)))
    return queries


def match_gallery(query_emb, gallery, threshold=MATCH_THRESHOLD):
    """LOP 3+4 rut gon (1 tien trinh, khong MQTT that)."""
    best_id, best_score = None, -1.0
    for person, gal_emb in gallery.items():
        score = float(np.dot(query_emb, gal_emb))
        if score > best_score:
            best_id, best_score = person, score
    if best_score > threshold:
        return best_id, best_score
    return "NEW", best_score


def main():
    print("Nap detector (YOLOv8n) + Re-ID model (backbone dong bang + head da train)...")
    detector = YOLO(str(AI_ROOT / "yolov8n.pt"))
    model = ReIDModel(CKPT, num_classes=1041, freeze_backbone=True).to(DEVICE)
    state = torch.load(RESULTS_DIR / "reid_head_msmt17_frozen_backbone.pt", map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    transform = get_eval_transform(224)

    print("\n[1/3] Xay Gallery tu video Enrollment...")
    gallery = build_gallery(detector, model, transform)

    print(f"\n[2/3] Chay test tren {TEST_VIDEO.name} (tach tu dong theo track_id, LOP 1+2)...")
    queries = extract_test_queries(TEST_VIDEO, detector, model, transform)
    print(f"  Phat hien {len(queries)} lan xuat hien doc lap (track_id) trong video test")

    print("\n[3/3] So khop Gallery (LOP 3+4 rut gon), tinh True Positive Rate...")
    rows = []
    for track_id, q_emb in queries:
        predicted, score = match_gallery(q_emb, gallery)
        rows.append({"track_id": track_id, "expected": TEST_EXPECTED_ID,
                      "predicted_own_model": predicted, "score_own_model": round(score, 4)})
        print(f"  track_id={track_id} | expected={TEST_EXPECTED_ID} | "
              f"predicted={predicted} | score={score:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "reid_own_model_tang2_results.csv", index=False, encoding="utf-8-sig")
    tpr = (df["predicted_own_model"] == df["expected"]).sum() / len(df) if len(df) else 0.0

    print("\n" + "=" * 60)
    print(f"Re-ID Head TU TRAIN -- True Positive Rate: {tpr:.2%}")
    print("=" * 60)
    print(f"\nDa luu: {RESULTS_DIR / 'reid_own_model_tang2_results.csv'}")


if __name__ == "__main__":
    main()
