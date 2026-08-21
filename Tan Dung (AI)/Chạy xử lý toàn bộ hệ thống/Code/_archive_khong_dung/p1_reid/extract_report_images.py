"""
Tao VIDEO minh hoa cho report P1 (ghep clip Enrollment + clip Test, co
bbox+nhan Person N, KHONG in chu DUNG/SAI len video -- phan dung/sai giai
thich rieng trong file Word cua tung folder, tranh nguoi xem hieu nham khi
xem video/anh don le).

Nhan: dung (khop dung nguoi da dang ky lam "Person 1" trong video nay) ->
luon ghi "Person 1". Sai/khong nhan dien duoc -> danh so rieng tang dan
(Person 2, Person 3, ...) cho tung truong hop khac nhau de phan biet, KHONG
mang y nghia "day la nguoi thu N that", chi de danh dau cac ca khac nhau.
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
TEST_VIDEO = P1_DIR / "Data 2" / "Scene 5-CAM 2.mp4"
ENROLLMENT = {
    "A": [P1_DIR / "Data 1" / "Scene 1-CAM 1.mp4", P1_DIR / "Data 1" / "Scene 2-CAM 2.mp4"],
    "B": [P1_DIR / "Data 1" / "Scene 3-CAM 1.mp4", P1_DIR / "Data 1" / "Scene 4-CAM 2.mp4"],
}
OUT_DIR_OWN = (AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Số liệu thống kê"
               / "P1" / "1. Bai cua nhom (OSNet x0.5)" / "images")
OUT_DIR_COMPETITOR = (AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Số liệu thống kê"
                       / "P1" / "2. Doi thu (Backbone dung chung)" / "images")
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


def draw_label(frame, bbox, text, color):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 12)), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, text, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return frame


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


PERSON_LABEL = {"A": "Person 1", "B": "Person 2", "NEW": "NEW"}


def match_gallery(query_emb, gallery, threshold=0.6):
    best_id, best_score = None, -1.0
    for person, gal_emb in gallery.items():
        score = float(np.dot(query_emb, gal_emb))
        if score > best_score:
            best_id, best_score = person, score
    if best_score > threshold:
        return best_id, best_score
    return "NEW", best_score


def process_and_save(embed_fn, detector, gallery, out_dir, tag, color_correct=(46, 125, 50), color_wrong=(50, 50, 220)):
    cap = cv2.VideoCapture(str(TEST_VIDEO))
    track_frames = {}  # track_id -> (frame_copy, bbox, embs[])
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        for track_id, bbox, crop in crop_person(detector, frame):
            entry = track_frames.setdefault(track_id, {"frame": None, "bbox": None, "embs": []})
            if len(entry["embs"]) < 10:
                entry["embs"].append(embed_fn(crop))
            # luu 1 frame dai dien (khi bbox du lon / ro net -- lay frame giua)
            if entry["frame"] is None or frame_idx % 15 == 0:
                entry["frame"] = frame.copy()
                entry["bbox"] = bbox
    cap.release()

    saved = []
    for track_id, entry in track_frames.items():
        if len(entry["embs"]) < 5:
            continue
        avg = np.mean(entry["embs"], axis=0)
        avg = avg / np.linalg.norm(avg)
        pred, score = match_gallery(avg, gallery)
        correct = pred == "A"
        color = color_correct if correct else color_wrong
        label = f"{tag}: {PERSON_LABEL.get(pred, pred)} (score={score:.2f}) {'DUNG' if correct else 'SAI'}"
        img = draw_label(entry["frame"].copy(), entry["bbox"], label, color)
        status = "dung" if correct else "sai"
        out_path = out_dir / f"{tag}_track{track_id}_{status}.jpg"
        cv2.imwrite(str(out_path), img)
        saved.append((track_id, pred, score, correct, out_path))
        print(f"  [{tag}] track={track_id} pred={pred} score={score:.4f} {'DUNG' if correct else 'SAI'} -> {out_path.name}")
    return saved


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

    print("\nTrich anh OSNet x0.5 (bai cua nhom)...")
    process_and_save(osnet_embed, detector_osnet, gallery_osnet, OUT_DIR_OWN, "OSNet")

    print("\nTrich anh Backbone dung chung (doi thu)...")
    process_and_save(shared_embed, detector_shared, gallery_shared, OUT_DIR_COMPETITOR, "Shared")

    print("\nDa luu anh vao:")
    print(f"  {OUT_DIR_OWN}")
    print(f"  {OUT_DIR_COMPETITOR}")


if __name__ == "__main__":
    main()
