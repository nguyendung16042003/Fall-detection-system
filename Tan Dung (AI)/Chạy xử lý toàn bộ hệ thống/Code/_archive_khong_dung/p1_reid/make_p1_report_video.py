"""
Tao VIDEO minh hoa report P1 (KHONG dung anh tinh) -- ghep 2 doan:
  (1) Doan Dang ky: clip Enrollment Person 1 (Data 1/Scene 2-CAM 2 -- dung
      CAM 2 de dong nhat camera voi doan test, tranh gay hieu nham do goc
      quay khac nhau).
  (2) Doan Test: toan bo Scene 5-CAM 2.mp4 (nguoi di qua vung chet 4 lan),
      ve bbox+nhan "Person N" theo KET QUA CUOI CUNG cua tung track (Person 1
      = khop dung nguoi da dang ky; Person 2/3/... = khong khop, danh so
      rieng cho tung truong hop khac nhau de phan biet -- KHONG in chu
      DUNG/SAI len video, giai thich rieng trong Word).

Xuat 1 video/model (OSNet x0.5, Backbone dung chung) vao dung folder report.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
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
ENROLL_CLIP = P1_DIR / "Data 1" / "Scene 2-CAM 2.mp4"
TEST_VIDEO = P1_DIR / "Data 2" / "Scene 5-CAM 2.mp4"
ENROLLMENT = {
    "A": [P1_DIR / "Data 1" / "Scene 1-CAM 1.mp4", P1_DIR / "Data 1" / "Scene 2-CAM 2.mp4"],
    "B": [P1_DIR / "Data 1" / "Scene 3-CAM 1.mp4", P1_DIR / "Data 1" / "Scene 4-CAM 2.mp4"],
}
OUT_DIR_OWN = (AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Số liệu thống kê"
               / "P1" / "1. Bai cua nhom (OSNet x0.5)")
OUT_DIR_COMPETITOR = (AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Số liệu thống kê"
                       / "P1" / "2. Doi thu (Backbone dung chung)")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DET_CONF = 0.4


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


def analyze_test_video(embed_fn, detector, gallery):
    """Chay 1 lan qua video test, tra ve: (a) danh sach (frame_idx, track_id,
    bbox) moi frame, (b) ket qua cuoi cung moi track_id (pred, score)."""
    cap = cv2.VideoCapture(str(TEST_VIDEO))
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
    next_wrong_label = 2
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
    return per_frame, track_result


def make_title_card(text, size, seconds, fps):
    w, h = size
    frame = np.full((h, w, 3), 30, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 1.4, 3)
    cv2.putText(frame, text, ((w - tw) // 2, (h + th) // 2), font, 1.4, (255, 255, 255), 3)
    return [frame] * int(seconds * fps)


def write_video(model_tag, embed_fn, detector, gallery, out_dir):
    cap = cv2.VideoCapture(str(TEST_VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"  [{model_tag}] Phan tich video test...")
    per_frame, track_result = analyze_test_video(embed_fn, detector, gallery)
    frame_tracks = {}
    for frame_idx, track_id, bbox in per_frame:
        frame_tracks.setdefault(frame_idx, []).append((track_id, bbox))

    out_path = out_dir / f"{model_tag}_video_dang_ky_va_test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    print(f"  [{model_tag}] Ghi doan tieu de + Dang ky...")
    for f in make_title_card(f"{model_tag} -- Doan 1: Dang ky Person 1 (Enrollment)", (w, h), 2.5, fps):
        writer.write(f)
    cap_enroll = cv2.VideoCapture(str(ENROLL_CLIP))
    while True:
        ret, frame = cap_enroll.read()
        if not ret:
            break
        cv2.putText(frame, "DANG KY: Person 1", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (46, 200, 80), 3)
        writer.write(frame)
    cap_enroll.release()

    print(f"  [{model_tag}] Ghi doan Test...")
    for f in make_title_card(f"{model_tag} -- Doan 2: Test (nguoi di qua vung chet nhieu lan)", (w, h), 2.5, fps):
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

    print(f"  [{model_tag}] Da luu: {out_path.name}")
    for track_id, res in track_result.items():
        print(f"    track_id={track_id} -> {res['label']} (score={res['score']:.4f}, "
              f"{'dung' if res['correct'] else 'sai'})")
    return out_path, track_result


def main():
    OUT_DIR_OWN.mkdir(parents=True, exist_ok=True)
    OUT_DIR_COMPETITOR.mkdir(parents=True, exist_ok=True)

    print("Nap OSNet x0.5...")
    detector_osnet = YOLO(str(AI_ROOT / "yolov8n.pt"))
    osnet_extractor = FeatureExtractor(model_name="osnet_x0_5", model_path=str(OSNET_CKPT), device="cpu")

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

    print("Xay Gallery...")
    gallery_osnet = build_gallery(osnet_embed, detector_osnet)
    gallery_shared = build_gallery(shared_embed, detector_shared)

    print("\nTao video OSNet x0.5 (bai cua nhom)...")
    write_video("OSNet_x0.5", osnet_embed, detector_osnet, gallery_osnet, OUT_DIR_OWN)

    print("\nTao video Backbone dung chung (doi thu)...")
    write_video("Shared_backbone", shared_embed, detector_shared, gallery_shared, OUT_DIR_COMPETITOR)


if __name__ == "__main__":
    main()
