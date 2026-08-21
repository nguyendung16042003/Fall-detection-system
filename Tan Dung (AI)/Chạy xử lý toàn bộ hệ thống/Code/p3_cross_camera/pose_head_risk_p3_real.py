"""
P3 -- Ban CHINH THUC, dung DUNG data P3 that (P3 Data 1: nga giua 2 cam,
nua nguoi moi cam, 4 tu the; P3 Data 2: cui/ngoi, khong nga) VOI CALIB THAT
(homography_matrices_p3.json/h_by_height_p3.json, xem calibrate_p3_corridor.py)
-- thay the ban proxy truoc day dung P2 Data 2 (khac dung kich ban P3).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p2_homography"))

from shared_backbone import SharedBackbone, preprocess_crop  # noqa: E402
from cross_camera_fuse import build_distance_matrices, sgie_forward  # noqa: E402
from homography import apply_homography, foot_point_from_bbox  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
P3_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "File Run Problem 3" / "P3"
RESULTS_P3 = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p3"
CKPT = (AI_ROOT / "Coding" / "training" / "training 3" / "runs" / "classify"
        / "Fall_Detection_Advanced_Loss" / "YOLOv8n_AFCL_Balanced-8" / "weights" / "best.pt")
CLASS_NAMES = ["bend", "exercise", "lie", "sit", "stand"]

DET_CONF = 0.2
DIST_THRESHOLD = 1.5  # P3 Cam1/Cam2 nhin 2 khu vuc san khac nhau (hanh lang/bep)
# giap nhau qua nguong cua -- can nguong khoang cach LONG HON P2 (2 nguoi
# cung 1 phong, ~0.5m) vi sai so tu nhien lon hon khi ghep qua 2 mat phang
# san rieng biet, dung ĐE xac nhan "co ca 2 cam cung thay 1 su kien" khong
# phai de phan biet nhieu nguoi (P3 Data 1/2 chi co 1 nguoi).
FRAME_SKIP = 2
SIGMA = 2.0  # giu dung gia tri da chon o sigma_sweep.py truoc do

SCENES = {
    "P3 Data 1": {"scenes": [1, 2, 3, 4], "label": "Fall", "cam_prefix": "Cam"},
    "P3 Data 2": {"scenes": [1, 2], "label": "ADL", "cam_prefix": "Cam"},
}


def load_h_by_height():
    with open(RESULTS_P3 / "h_by_height_p3.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {cam: {float(h): np.array(H) for h, H in d.items()} for cam, d in raw.items()}


def load_h_floor():
    with open(RESULTS_P3 / "homography_matrices_p3.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {cam: np.array(H) for cam, H in raw.items()}


def detect_one(detector, frame, conf=DET_CONF):
    results = detector(frame, classes=[0], conf=conf, verbose=False)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return None
    box = results.boxes[0].xyxy[0].cpu().numpy()
    return tuple(box.tolist())


def process_scene(data_name, scene_id, detector1, detector2, backbone, H_floor, H_by_height):
    folder = P3_DIR / data_name
    v1 = folder / f"Scene {scene_id}-Cam 1.mp4"
    v2 = folder / f"Scene {scene_id}-Cam 2.mp4"
    cap1 = cv2.VideoCapture(str(v1))
    cap2 = cv2.VideoCapture(str(v2))
    rows = []
    frame_idx = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue

        bbox1 = detect_one(detector1, frame1)
        bbox2 = detect_one(detector2, frame2)
        if bbox1 is None or bbox2 is None:
            continue

        foot1 = apply_homography(H_floor["P3 Cam1"], foot_point_from_bbox(bbox1))
        foot2 = apply_homography(H_floor["P3 Cam2"], foot_point_from_bbox(bbox2))
        dist = np.linalg.norm(np.array(foot1) - np.array(foot2))
        if dist > DIST_THRESHOLD:
            continue

        crop1 = frame1[int(bbox1[1]):int(bbox1[3]), int(bbox1[0]):int(bbox1[2])]
        crop2 = frame2[int(bbox2[1]):int(bbox2[3]), int(bbox2[0]):int(bbox2[2])]
        if crop1.size == 0 or crop2.size == 0:
            continue
        x1 = preprocess_crop(crop1)
        x2 = preprocess_crop(crop2)

        with torch.no_grad():
            grid1 = backbone(x1)
            grid2 = backbone(x2)
            logits1 = backbone.classify_from_grid(grid1)
            logits2 = backbone.classify_from_grid(grid2)
            probs1 = torch.softmax(logits1, dim=1)[0]
            probs2 = torch.softmax(logits2, dim=1)[0]

            h, w = grid1.shape[2], grid1.shape[3]
            dist_1to2, dist_2to1 = build_distance_matrices(
                (h, w), foot1, foot2, H_by_height["P3 Cam1"], H_by_height["P3 Cam2"],
                bbox1, bbox2)
            v_fused = sgie_forward(backbone, x1, x2, has_both_cams=True,
                                    dist_1to2=dist_1to2, dist_2to1=dist_2to1, sigma=SIGMA)
            logits_fused = backbone.classify_from_vector(v_fused)
            probs_fused = torch.softmax(logits_fused, dim=1)[0]

        top1_cam1 = CLASS_NAMES[int(probs1.argmax())]
        top1_cam2 = CLASS_NAMES[int(probs2.argmax())]
        top1_fused = CLASS_NAMES[int(probs_fused.argmax())]
        rows.append({
            "data": data_name, "scene": scene_id, "frame_idx": frame_idx,
            "ground_dist_m": round(dist, 3),
            "cam1_top1": top1_cam1, "cam1_conf": round(float(probs1.max()), 4),
            "cam2_top1": top1_cam2, "cam2_conf": round(float(probs2.max()), 4),
            "fused_top1": top1_fused, "fused_conf": round(float(probs_fused.max()), 4),
            "min_single_cam_conf": round(min(float(probs1.max()), float(probs2.max())), 4),
        })

    cap1.release()
    cap2.release()
    return rows


def main():
    print("Nap SharedBackbone + 2 detector rieng + calib THAT P3...")
    backbone = SharedBackbone(CKPT).eval()
    detector1 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    detector2 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    H_by_height = load_h_by_height()
    H_floor = load_h_floor()

    all_rows = []
    for data_name, info in SCENES.items():
        for scene_id in info["scenes"]:
            print(f"=== {data_name} / Scene {scene_id} ({info['label']}) ===")
            rows = process_scene(data_name, scene_id, detector1, detector2, backbone, H_floor, H_by_height)
            print(f"  {len(rows)} frame co ca 2 cam cung thay + khop (dist<={DIST_THRESHOLD}m)")
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_csv = RESULTS_P3 / "pose_head_risk_p3_real.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    if len(df):
        avg_min_single = df["min_single_cam_conf"].mean()
        avg_fused = df["fused_conf"].mean()
        agree_rate = (df["cam1_top1"] == df["fused_top1"]).mean()
        print("\n" + "=" * 60)
        print(f"So mau: {len(df)} (tren DUNG data P3 that, khong con la proxy P2 Data 2)")
        print(f"Confidence trung binh — cam YEU HON (chua gop): {avg_min_single:.4f}")
        print(f"Confidence trung binh — SAU KHI GOP (fused): {avg_fused:.4f}")
        print(f"Chenh lech: {avg_fused - avg_min_single:+.4f} "
              f"({'TANG' if avg_fused > avg_min_single else 'GIAM'} sau khi gop)")
        print(f"Ty le fused_top1 == cam1_top1: {agree_rate:.2%}")
        print("=" * 60)
    print(f"\nDa luu: {out_csv}")


if __name__ == "__main__":
    main()
