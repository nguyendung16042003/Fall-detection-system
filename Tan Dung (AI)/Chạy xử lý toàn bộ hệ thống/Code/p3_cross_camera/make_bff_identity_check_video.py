"""
Ap dung Identity Association THAT (Hungarian tren khoang cach san qua
homography, homography_matching.match_two_cameras -- KHONG sua) vao Boundary
Feature Fusion, thay cho gia dinh "chi 1 nguoi/canh nen cu ghep box duy nhat"
truoc gio. KHONG hien thi nhan tu the nua (theo yeu cau) -- thay bang xac
nhan danh tinh: khoang cach san (tu Identity Association) + do tuong dong
Re-Identification (OSNet_x0.25 KD, doc lap voi homography, dung de kiem tra
cheo xem 2 camera co dung dang thay CUNG 1 nguoi khong).

Test tren Scene 2 (KHAC Scene 1 da dung moi lan truoc do).
"""
import sys
from pathlib import Path

import cv2
import json
import numpy as np
import torch
from ultralytics import YOLO
from torchreid.reid.utils import FeatureExtractor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p2_homography"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p1_reid"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "Coding" / "Pipeline"))

from homography import apply_homography, foot_point_from_bbox  # noqa: E402
from homography_matching import Track, match_two_cameras  # noqa: E402
from shared_backbone import SharedBackbone, preprocess_crop  # noqa: E402
from cross_camera_fuse import build_distance_matrices, sgie_forward  # noqa: E402
from fall_rule import detect_fall_events  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
P3_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "File Run Problem 3" / "P3"
RESULTS_P3 = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p3"
OSNET_KD_CKPT = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Code" / "p1_reid" / "osnet_x0_25_kd_msmt17_best.pt"
POSE_CKPT = (AI_ROOT / "Coding" / "training" / "training 3" / "runs" / "classify"
             / "Fall_Detection_Advanced_Loss" / "YOLOv8n_AFCL_Balanced-8" / "weights" / "best.pt")
CLASS_NAMES = ["bend", "exercise", "lie", "sit", "stand"]
SIGMA = 2.0
DET_CONF = 0.2
DIST_THRESHOLD = 1.5  # khop dung Boundary Feature Fusion (khac 0.5 cua Identity Association 1 phong)
GRACE_MS = 1200
FALL_BANNER_MS = 1500
TILE_H = 540

DATA_NAME = "P3 Data 1"
SCENE_ID = 3  # test them, khac Scene 1/2 da thu
CAM1_VIDEO = P3_DIR / DATA_NAME / f"Scene {SCENE_ID}-Cam 1.mp4"
CAM2_VIDEO = P3_DIR / DATA_NAME / f"Scene {SCENE_ID}-Cam 2.mp4"
OUT_DIR = (AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Số liệu thống kê"
           / "Boundary Feature Fusion" / "5. Identity check (Identity Association ap dung)")


def load_h_floor():
    with open(RESULTS_P3 / "homography_matrices_p3.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {cam: np.array(H) for cam, H in raw.items()}


def load_h_by_height():
    with open(RESULTS_P3 / "h_by_height_p3.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {cam: {float(h): np.array(H) for h, H in d.items()} for cam, d in raw.items()}


def classify_solo(backbone, frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return None, None
    x = preprocess_crop(crop)
    with torch.no_grad():
        grid = backbone(x)
        logits = backbone.classify_from_grid(grid)
        probs = torch.softmax(logits, dim=1)[0]
    return CLASS_NAMES[int(probs.argmax())], float(probs.max())


def detect_one_tracked(detector, frame, held_state, grace_frames, conf=DET_CONF):
    results = detector.track(frame, classes=[0], conf=conf, persist=True, verbose=False)[0]
    if results.boxes is not None and len(results.boxes) > 0:
        box = results.boxes[0]
        bbox = tuple(box.xyxy[0].cpu().numpy().tolist())
        track_id = int(box.id[0]) if box.id is not None else 0
        held_state["bbox"] = bbox
        held_state["track_id"] = track_id
        held_state["missed"] = 0
        return bbox, track_id, False
    if held_state["bbox"] is not None and held_state["missed"] < grace_frames:
        held_state["missed"] += 1
        return held_state["bbox"], held_state["track_id"], True
    held_state["bbox"] = None
    return None, None, False


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Xu ly {DATA_NAME} / Scene {SCENE_ID} (canh MOI, khac Scene 1)...")
    print("Nap 2 detector + OSNet_x0.25 KD (Re-Identification, kiem tra cheo) + homography...")
    detector1 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    detector2 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    extractor = FeatureExtractor(model_name="osnet_x0_25", model_path=str(OSNET_KD_CKPT), device="cpu")
    backbone = SharedBackbone(POSE_CKPT).eval()
    H_floor = load_h_floor()
    H_by_height = load_h_by_height()

    def embed(crop):
        emb = extractor([crop])
        emb = torch.nn.functional.normalize(emb, dim=1)
        return emb.cpu().numpy().flatten()

    cap1 = cv2.VideoCapture(str(CAM1_VIDEO))
    cap2 = cv2.VideoCapture(str(CAM2_VIDEO))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    grace_frames = max(1, round(GRACE_MS / 1000.0 * fps))

    held1 = {"bbox": None, "track_id": None, "missed": 0}
    held2 = {"bbox": None, "track_id": None, "missed": 0}
    per_frame = []
    pose_records = []  # cho fall_rule, PERSON_ID co dinh =1 (1 nguoi/canh)
    frame_idx = 0
    n_matched = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        frame_idx += 1

        bbox1, tid1, held_flag1 = detect_one_tracked(detector1, frame1, held1, grace_frames)
        bbox2, tid2, held_flag2 = detect_one_tracked(detector2, frame2, held2, grace_frames)

        rec = {"frame_idx": frame_idx, "bbox1": bbox1, "bbox2": bbox2, "matched": False,
               "floor_dist": None, "reid_sim": None}

        if bbox1 is not None and bbox2 is not None:
            t1 = Track("Cam1", tid1, bbox1,
                        ground_xy=apply_homography(H_floor["P3 Cam1"], foot_point_from_bbox(bbox1)))
            t2 = Track("Cam2", tid2, bbox2,
                        ground_xy=apply_homography(H_floor["P3 Cam2"], foot_point_from_bbox(bbox2)))
            pairs = match_two_cameras([t1], [t2], distance_threshold=DIST_THRESHOLD)
            if pairs:
                floor_dist = float(np.linalg.norm(np.array(t1.ground_xy) - np.array(t2.ground_xy)))
                crop1 = frame1[int(bbox1[1]):int(bbox1[3]), int(bbox1[0]):int(bbox1[2])]
                crop2 = frame2[int(bbox2[1]):int(bbox2[3]), int(bbox2[0]):int(bbox2[2])]
                reid_sim = None
                if crop1.size > 0 and crop2.size > 0:
                    e1, e2 = embed(crop1), embed(crop2)
                    reid_sim = float(np.dot(e1, e2))
                rec["matched"] = True
                rec["floor_dist"] = floor_dist
                rec["reid_sim"] = reid_sim
                n_matched += 1

                # -- Tu the chay NGAM (khong ve len video) qua Boundary Feature
                # Fusion that (gop 2 cam), dua vao rule that --
                if crop1.size > 0 and crop2.size > 0:
                    x1 = preprocess_crop(crop1)
                    x2 = preprocess_crop(crop2)
                    with torch.no_grad():
                        grid1 = backbone(x1)
                        h, w = grid1.shape[2], grid1.shape[3]
                        dist_1to2, dist_2to1 = build_distance_matrices(
                            (h, w), t1.ground_xy, t2.ground_xy,
                            H_by_height["P3 Cam1"], H_by_height["P3 Cam2"], bbox1, bbox2)
                        v_fused = sgie_forward(backbone, x1, x2, has_both_cams=True,
                                                dist_1to2=dist_1to2, dist_2to1=dist_2to1, sigma=SIGMA)
                        logits = backbone.classify_from_vector(v_fused)
                        probs = torch.softmax(logits, dim=1)[0]
                    pose_records.append({
                        "frame_id": frame_idx, "person_id": 1,
                        "pose": CLASS_NAMES[int(probs.argmax())], "pose_confidence": float(probs.max()),
                        "bbox_x1": bbox1[0], "bbox_y1": bbox1[1], "bbox_x2": bbox1[2], "bbox_y2": bbox1[3],
                        "det_confidence": 1.0, "held": False,
                    })
        elif bbox1 is not None:
            pose, conf = classify_solo(backbone, frame1, bbox1)
            if pose is not None:
                pose_records.append({
                    "frame_id": frame_idx, "person_id": 1, "pose": pose, "pose_confidence": conf,
                    "bbox_x1": bbox1[0], "bbox_y1": bbox1[1], "bbox_x2": bbox1[2], "bbox_y2": bbox1[3],
                    "det_confidence": 1.0, "held": False,
                })
        elif bbox2 is not None:
            pose, conf = classify_solo(backbone, frame2, bbox2)
            if pose is not None:
                pose_records.append({
                    "frame_id": frame_idx, "person_id": 1, "pose": pose, "pose_confidence": conf,
                    "bbox_x1": bbox2[0], "bbox_y1": bbox2[1], "bbox_x2": bbox2[2], "bbox_y2": bbox2[3],
                    "det_confidence": 1.0, "held": False,
                })
        per_frame.append(rec)
    cap1.release()
    cap2.release()

    events = detect_fall_events(pose_records, fps=fps)
    print(f"  RULE THAT (chay ngam, khong hien tu the): {len(events)} su kien nga")
    for e in events:
        print(f"    -> t={e['timestamp_ms']/1000:.2f}s trigger={e['rule']['trigger']}")
    event_windows = [(e["timestamp_ms"], e["timestamp_ms"] + FALL_BANNER_MS) for e in events]

    print(f"  {frame_idx} frame, {n_matched} frame Identity Association ghep dung "
          f"(khoang cach san <= {DIST_THRESHOLD}m)")
    matched_sims = [r["reid_sim"] for r in per_frame if r["reid_sim"] is not None]
    if matched_sims:
        print(f"  ReID similarity (OSNet x0.25 KD) tren cac frame ghep duoc: "
              f"avg={np.mean(matched_sims):.3f} min={np.min(matched_sims):.3f} max={np.max(matched_sims):.3f}")
        print("  (Luu y: chua co nguong da hieu chinh cho phep so cung-frame-khac-cam nay -- "
              "0.6 dung o Re-Identification la nguong cho GALLERY matching, khac bai toan. "
              "Diem so tren chi de tham khao/doi chieu voi Identity Association hinh hoc, "
              "khong tu ket luan dung/sai chi tu 1 con so chua hieu chinh.)")

    # ---------- render video ----------
    cap1 = cv2.VideoCapture(str(CAM1_VIDEO))
    ret, f0 = cap1.read()
    cap1.release()
    tile_w_sample = int(TILE_H * f0.shape[1] / f0.shape[0])
    out_w = tile_w_sample * 2 + 10
    out_h = TILE_H
    out_path = OUT_DIR / f"BFF_IdentityCheck_Scene{SCENE_ID}_ghep.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
    assert writer.isOpened(), f"VideoWriter khong mo duoc: {out_path}"

    def render_tile(frame, bbox, cam_label, matched, floor_dist, reid_sim, fall_active):
        frame = frame.copy()
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 0, 255) if fall_active else ((46, 160, 46) if matched else (180, 180, 60))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            if matched:
                sim_txt = f"{reid_sim:.2f}" if reid_sim is not None else "n/a"
                text = f"Person 1 (dist={floor_dist:.2f}m, ReID sim={sim_txt})"
            else:
                text = "not matched"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        h, w = frame.shape[:2]
        scale = TILE_H / h
        frame = cv2.resize(frame, (int(w * scale), TILE_H))
        cv2.putText(frame, cam_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        if fall_active:
            cv2.putText(frame, "!!! FALL DETECTED !!!", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 4)
        return frame

    cap1 = cv2.VideoCapture(str(CAM1_VIDEO))
    cap2 = cv2.VideoCapture(str(CAM2_VIDEO))
    for rec in per_frame:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        t_ms = rec["frame_idx"] / fps * 1000.0
        fall_active = any(lo <= t_ms <= hi for lo, hi in event_windows)
        tile1 = render_tile(frame1, rec["bbox1"], "Cam 1", rec["matched"], rec["floor_dist"], rec["reid_sim"], fall_active)
        tile2 = render_tile(frame2, rec["bbox2"], "Cam 2", rec["matched"], rec["floor_dist"], rec["reid_sim"], fall_active)
        if tile1.shape[1] != tile_w_sample:
            tile1 = cv2.resize(tile1, (tile_w_sample, TILE_H))
        if tile2.shape[1] != tile_w_sample:
            tile2 = cv2.resize(tile2, (tile_w_sample, TILE_H))
        sep = np.full((TILE_H, 10, 3), 255, dtype=np.uint8)
        combined = np.hstack([tile1, sep, tile2])
        writer.write(combined)
    cap1.release()
    cap2.release()
    writer.release()
    assert out_path.exists() and out_path.stat().st_size > 0, f"Video khong duoc ghi ra: {out_path}"
    print(f"\nDa luu: {out_path}")
    return out_path


if __name__ == "__main__":
    main()
