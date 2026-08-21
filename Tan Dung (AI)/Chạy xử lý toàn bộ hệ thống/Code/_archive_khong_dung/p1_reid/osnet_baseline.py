"""
Baseline doi thu P1 -- OSNet pretrained (Torchreid), theo dung
"File Run Problem 1/Context mo phong doi thu 1.pdf". Muc tieu: so True Positive
Rate cua OSNet (khong train, tong quat) voi Re-ID head tich hop trong SGIE
(se lam o Phase 3, sau khi co backbone/Re-ID head that).

Khac voi ban mau trong PDF (gia dinh san p1_test_labels.csv voi clip_id tro
toi cac clip da cat san): du lieu thuc te quay theo "Kich ban P1.docx" chi co
2 video LIEN TUC cho Canh 5 (Scene 5-CAM 1/2.mp4, nguoi A di qua vung chet
4 lan lien tiep trong cung 1 video), khong phai cac clip rieng le. Vi vay
dung YOLOv8n .track() de TU DONG tach moi lan nguoi A xuat hien lai o Cam 2
thanh 1 "query" rieng (moi track_id moi = 1 lan quay lai), thay vi cat tay
tung clip.

Nhan ky vong: theo dung kich ban, Canh 5 chi co nguoi A -> moi query deu ky
vong khop "A".
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from torchreid.reid.utils import FeatureExtractor
from ultralytics import YOLO

AI_ROOT = Path(__file__).resolve().parents[3]
P1_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "File Run Problem 1" / "P1 test"
OUT_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# QUAN TRONG: FeatureExtractor KHONG dua model_path se CHI nap ImageNet weights
# (khong phai model Re-ID that) -- da phat hien loi nay khi debug Rank-1/mAP.
# CHOT DUNG osnet_x0_5 (nhe, ~0.6M tham so) -- da do FPS/RAM tren Jetson Nano
# 4GB that dat ca 3 tieu chi (18.15 FPS/2.31GB RAM/41.8C, worst-case chung
# voi YOLOv8n 320x320). Checkpoint DA TRAIN+TEST TREN MSMT17 that (Rank-1
# 69.7%/mAP 37.5% theo MODEL_ZOO chinh thuc cua deep-person-reid).
OSNET_NAME = "osnet_x0_5"
OSNET_CKPT = Path(__file__).resolve().parent / "osnet_x0_5_msmt17.pth"

# Enrollment: Canh 1 (A-Cam1), Canh 2 (A-Cam2), Canh 3 (B-Cam1), Canh 4 (B-Cam2)
ENROLLMENT = {
    "A": [P1_DIR / "Data 1" / "Scene 1-CAM 1.mp4", P1_DIR / "Data 1" / "Scene 2-CAM 2.mp4"],
    "B": [P1_DIR / "Data 1" / "Scene 3-CAM 1.mp4", P1_DIR / "Data 1" / "Scene 4-CAM 2.mp4"],
}
# Canh 5: A di qua vung chet 4 lan, chi quay Cam2 (noi A xuat hien lai) can test
TEST_VIDEO = P1_DIR / "Data 2" / "Scene 5-CAM 2.mp4"
TEST_EXPECTED_ID = "A"  # theo kich ban, Canh 5 chi co nhan vat A

DET_CONF = 0.4
MIN_QUERY_FRAMES = 5
MAX_QUERY_FRAMES = 10
MATCH_THRESHOLD = 0.6


def crop_person(detector, frame, conf=DET_CONF):
    """Tra ve list (track_id, crop) cho tat ca nguoi detect duoc trong frame."""
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


def build_gallery_entry(video_path, detector, extractor, sample_every_n_frames=5):
    """1 vector embedding trung binh dai dien cho 1 video enrollment (1 nguoi
    xuat hien duy nhat trong tung canh quay theo kich ban)."""
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
                # enrollment moi canh chi co 1 nguoi -> lay bbox dau tien
                _, crop = people[0]
                emb = extractor(crop)
                embeddings.append(emb.cpu().numpy().flatten())
        frame_idx += 1
    cap.release()
    if not embeddings:
        return None
    avg = np.mean(embeddings, axis=0)
    return avg / np.linalg.norm(avg)


def build_gallery(detector, extractor):
    gallery = {}
    for person, videos in ENROLLMENT.items():
        embs = [build_gallery_entry(v, detector, extractor) for v in videos]
        embs = [e for e in embs if e is not None]
        if not embs:
            print(f"  CANH BAO: khong trich duoc embedding nao cho {person}")
            continue
        avg = np.mean(embs, axis=0)
        gallery[person] = avg / np.linalg.norm(avg)
        print(f"  Gallery[{person}] xay tu {len(embs)}/{len(videos)} video enrollment")
    return gallery


def extract_test_queries(video_path, detector, extractor):
    """Chay .track() xuyen suot video test, gom frame theo track_id, moi
    track_id moi = 1 lan nguoi xuat hien lai (1 query doc lap). Tra ve list
    (track_id, query_embedding)."""
    cap = cv2.VideoCapture(str(video_path))
    buffers = {}  # track_id -> list[embedding]
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for track_id, crop in crop_person(detector, frame):
            buf = buffers.setdefault(track_id, [])
            if len(buf) < MAX_QUERY_FRAMES:
                emb = extractor(crop)
                buf.append(emb.cpu().numpy().flatten())
        frame_idx += 1
    cap.release()

    queries = []
    for track_id, embs in buffers.items():
        if len(embs) < MIN_QUERY_FRAMES:
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
    print(f"Dang khoi tao detector (YOLOv8n) + extractor ({OSNET_NAME}, MSMT17 that)...")
    detector = YOLO(str(AI_ROOT / "yolov8n.pt"))
    extractor = FeatureExtractor(model_name=OSNET_NAME, model_path=str(OSNET_CKPT), device="cpu")

    print("\n[1/3] Xay Gallery tu video Enrollment...")
    gallery = build_gallery(detector, extractor)
    np.save(OUT_DIR / "osnet_gallery.npy", gallery)

    print(f"\n[2/3] Chay test tren {TEST_VIDEO.name} (tach tu dong theo track_id)...")
    queries = extract_test_queries(TEST_VIDEO, detector, extractor)
    print(f"  Phat hien {len(queries)} lan xuat hien doc lap (track_id) trong video test")

    print("\n[3/3] So khop Gallery, tinh True Positive Rate...")
    rows = []
    for track_id, q_emb in queries:
        predicted, score = match_gallery(q_emb, gallery)
        rows.append({
            "track_id": track_id,
            "expected": TEST_EXPECTED_ID,
            "predicted_osnet": predicted,
            "score_osnet": round(score, 4),
        })
        print(f"  track_id={track_id} | expected={TEST_EXPECTED_ID} | "
              f"predicted={predicted} | score={score:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "osnet_baseline_results.csv", index=False, encoding="utf-8-sig")

    if len(df):
        tpr = (df["predicted_osnet"] == df["expected"]).sum() / len(df)
    else:
        tpr = 0.0

    print("\n" + "=" * 60)
    print(f"OSNet pretrained -- True Positive Rate (nhan dung A khi quay lai): {tpr:.2%}")
    print(f"(Luu y: Gallery chi co A/B, video test chi co A -> day la True Positive")
    print(f" Rate, KHONG phai Match Accuracy day du -- theo dung luu y trong tai lieu)")
    print("=" * 60)
    print(f"\nDa luu: {OUT_DIR / 'osnet_baseline_results.csv'}")
    print(f"Da luu: {OUT_DIR / 'osnet_gallery.npy'}")


if __name__ == "__main__":
    main()
