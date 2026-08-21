"""Video demo 1/3 -- REID da nguoi, dung checkpoint Phuong an 4
(osnet_x0_25_kd_pa4_via_ta_msmt17_best.pt). Du lieu quay moi trong
"Chay xu ly toan bo he thong/Video demo/REID/" (xem Kich ban.docx):
  - Data dang ky: 4 clip enrollment (CAM1-A, CAM2-A, CAM1-B, CAM2-B), moi
    clip 1 nguoi quay du 4 goc (thang/trai/sau/phai) lien tuc.
  - Data REID: 1 scene x 2 cam -- B dung yen o Cam 2, A di tu Cam 1 sang
    Cam 2 (ca 2 cung o Cam 2), dung yen 5s, roi A quay lai Cam 1 truoc, B
    di theo sau.

Khac ban goc make_p1_pa4_video.py: nhan hien thi TRUC TIEP ket qua
match_gallery (Person A / Person B / New) thay vi nhi phan dung/sai, vi
CA 2 nguoi deu la "dung" neu nhan dung ten -- muc dich la kiem tra he
thong co tach DUNG 2 ID rieng biet khi co >=2 nguoi hay khong.
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

AI_ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Video demo"
REG_DIR = DEMO_DIR / "REID" / "Data đăng ký"
TEST_DIR = DEMO_DIR / "REID" / "Data REID"
OUT_DIR = DEMO_DIR / "Video cuối cùng"

CAM1_VIDEO = TEST_DIR / "CAM 1- Scene 1.avi"
CAM2_VIDEO = TEST_DIR / "CAM 2- Scene 1.avi"
ENROLLMENT = {
    "A": [REG_DIR / "CAM 1- Người A.avi", REG_DIR / "CAM 2- Người A.avi"],
    "B": [REG_DIR / "CAM 1- Người B.avi", REG_DIR / "CAM 2- Người B.mp4"],
}

PA4_CKPT = Path(__file__).resolve().parent / "osnet_x0_25_kd_pa4_via_ta_msmt17_best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DET_CONF = 0.4
TILE_H = 540
MATCH_THRESHOLD = 0.6
TRANSFORM = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def make_pa4_embed_fn(device=DEVICE):
    model = build_model("osnet_x0_25", num_classes=1041, loss="softmax", pretrained=False)
    load_pretrained_weights(model, str(PA4_CKPT))
    model.eval().to(device)

    def embed(crop_bgr):
        from PIL import Image
        pil_img = Image.fromarray(crop_bgr[:, :, ::-1])
        tensor = TRANSFORM(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(tensor)
        feat = torch.nn.functional.normalize(feat, dim=1)
        return feat.cpu().numpy().flatten()

    return embed


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
            print(f"  Gallery[{person}]: {len(embs_all)} clip trung binh")
    return gallery


def match_gallery(query_emb, gallery, threshold=MATCH_THRESHOLD):
    best_id, best_score = None, -1.0
    for person, gal_emb in gallery.items():
        score = float(np.dot(query_emb, gal_emb))
        if score > best_score:
            best_id, best_score = person, score
    if best_score > threshold:
        return best_id, best_score
    return "New", best_score


def analyze_video(video_path, embed_fn, detector, gallery):
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
    for track_id, embs in track_embs.items():
        if len(embs) < 5:
            continue
        avg = np.mean(embs, axis=0)
        avg = avg / np.linalg.norm(avg)
        pred, score = match_gallery(avg, gallery)
        track_result[track_id] = {"pred": pred, "score": score}
    frame_tracks = {}
    for frame_idx, track_id, bbox in per_frame:
        frame_tracks.setdefault(frame_idx, []).append((track_id, bbox))
    return frame_tracks, track_result, fps


def render_tile(frame, frame_tracks_at_idx, track_result, cam_label):
    frame = frame.copy()
    for track_id, bbox in frame_tracks_at_idx:
        res = track_result.get(track_id)
        if res is None:
            continue
        color = (46, 160, 46) if res["pred"] in ("A", "B") else (40, 40, 220)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        text = f"Person {res['pred']} ({res['score']:.2f})"
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
    scale, thick = 1.1, 3
    lines = text.split("\n")
    (tw, th), _ = cv2.getTextSize(lines[0], font, scale, thick)
    y0 = h // 2 - (len(lines) - 1) * (th + 15) // 2
    for i, line in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(line, font, scale, thick)
        cv2.putText(frame, line, ((w - tw) // 2, y0 + i * (th + 25)), font, scale, (255, 255, 255), thick)
    return [frame] * int(seconds * fps)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Nap Phuong an 4 (Teacher Assistant checkpoint)...")
    embed_fn = make_pa4_embed_fn()
    detector_gallery = YOLO(str(AI_ROOT / "yolov8n.pt"))
    detector1 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    detector2 = YOLO(str(AI_ROOT / "yolov8n.pt"))

    print("Xay Gallery tu Data dang ky...")
    gallery = build_gallery(embed_fn, detector_gallery)

    print("Phan tich CAM 1 (Data REID)...")
    ft1, res1, fps1 = analyze_video(CAM1_VIDEO, embed_fn, detector1, gallery)
    for tid, r in res1.items():
        print(f"    CAM1 track_id={tid} -> Person {r['pred']} (score={r['score']:.4f})")

    print("Phan tich CAM 2 (Data REID)...")
    ft2, res2, fps2 = analyze_video(CAM2_VIDEO, embed_fn, detector2, gallery)
    for tid, r in res2.items():
        print(f"    CAM2 track_id={tid} -> Person {r['pred']} (score={r['score']:.4f})")

    n_A = sum(1 for r in list(res1.values()) + list(res2.values()) if r["pred"] == "A")
    n_B = sum(1 for r in list(res1.values()) + list(res2.values()) if r["pred"] == "B")
    n_new = sum(1 for r in list(res1.values()) + list(res2.values()) if r["pred"] == "New")
    print(f"\nTONG: {n_A} track -> A, {n_B} track -> B, {n_new} track -> New (khong khop)")

    cap1 = cv2.VideoCapture(str(CAM1_VIDEO))
    cap2 = cv2.VideoCapture(str(CAM2_VIDEO))
    n_max = max(int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps_out = fps1
    ret, f0 = cap1.read()
    cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
    tile_w = int(TILE_H * f0.shape[1] / f0.shape[0])
    out_w, out_h = tile_w * 2 + 10, TILE_H

    out_path = OUT_DIR / "1_REID_da_nguoi.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (out_w, out_h))

    title = "Kiem tra Re-Identification da nguoi\n(Phuong an 4 - Teacher Assistant)"
    for f in make_title_card(title, (out_w, out_h), 3.0, fps_out):
        writer.write(f)

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
    print(f"\nDa luu: {out_path}")


if __name__ == "__main__":
    main()
