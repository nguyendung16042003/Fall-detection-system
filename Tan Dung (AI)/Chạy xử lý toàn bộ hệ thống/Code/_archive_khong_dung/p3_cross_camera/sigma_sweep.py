"""
P3 -- Chon sigma (dac ta "no_training" muc 6): thu nhieu gia tri, so sanh
dinh luong (confidence trung binh, ty le tut manh) tren P2 Data 2 (calib
dung). Tai dung cache detect+homography (chi tinh 1 lan), doi sigma o buoc
gop de khong lap lai cong viec nang (detect nguoi + backbone forward).
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
from cross_camera_fuse import (  # noqa: E402
    build_distance_matrices, grid_to_flat, cross_camera_fuse, fuse_to_single_vector,
)
from homography import apply_homography, foot_point_from_bbox  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
P2_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "File Run Problem 2" / "P2 data test" / "P2 Data 2"
RESULTS_P2 = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p2"
RESULTS_P3 = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p3"
CKPT = (AI_ROOT / "Coding" / "training" / "training 3" / "runs" / "classify"
        / "Fall_Detection_Advanced_Loss" / "YOLOv8n_AFCL_Balanced-8" / "weights" / "best.pt")
CLASS_NAMES = ["bend", "exercise", "lie", "sit", "stand"]

DET_CONF = 0.2
DIST_THRESHOLD = 0.5
FRAME_SKIP = 5
SIGMAS = [0.5, 1.0, 2.0, 5.0]


def load_h_by_height():
    with open(RESULTS_P2 / "h_by_height.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {cam: {float(h): np.array(H) for h, H in d.items()} for cam, d in raw.items()}


def load_h_floor():
    with open(RESULTS_P2 / "homography_matrices.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {cam: np.array(H) for cam, H in raw.items()}


def detect_one(detector, frame, conf=DET_CONF):
    results = detector(frame, classes=[0], conf=conf, verbose=False)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return None
    box = results.boxes[0].xyxy[0].cpu().numpy()
    return tuple(box.tolist())


def collect_samples(detector1, detector2, backbone, H_floor, H_by_height):
    """Chi chay detect + backbone MOT LAN, luu lai grid1/grid2/dist matrices/
    single-cam probs cho tung mau -- de sigma_sweep tai dung, khong detect lai."""
    samples = []
    for scene_id in [1, 2, 3, 4, 5]:
        scene_prefix = f"Scene {scene_id}"
        cap1 = cv2.VideoCapture(str(P2_DIR / f"{scene_prefix}-CAM 1.mp4"))
        cap2 = cv2.VideoCapture(str(P2_DIR / f"{scene_prefix}-CAM 2.mp4"))
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
            foot1 = apply_homography(H_floor["CAM 1"], foot_point_from_bbox(bbox1))
            foot2 = apply_homography(H_floor["CAM 2"], foot_point_from_bbox(bbox2))
            if np.linalg.norm(np.array(foot1) - np.array(foot2)) > DIST_THRESHOLD:
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
                probs1 = torch.softmax(backbone.classify_from_grid(grid1), dim=1)[0]
                probs2 = torch.softmax(backbone.classify_from_grid(grid2), dim=1)[0]
                h, w = grid1.shape[2], grid1.shape[3]
                dist_1to2, dist_2to1 = build_distance_matrices(
                    (h, w), foot1, foot2, H_by_height["CAM 1"], H_by_height["CAM 2"],
                    bbox1, bbox2)
                F1_flat, _ = grid_to_flat(grid1)
                F2_flat, _ = grid_to_flat(grid2)
            samples.append({
                "scene": scene_prefix, "frame_idx": frame_idx,
                "F1_flat": F1_flat, "F2_flat": F2_flat,
                "dist_1to2": dist_1to2, "dist_2to1": dist_2to1,
                "min_single_conf": min(float(probs1.max()), float(probs2.max())),
                "cam1_top1": CLASS_NAMES[int(probs1.argmax())],
            })
        cap1.release()
        cap2.release()
        print(f"  {scene_prefix}: {sum(1 for s in samples if s['scene']==scene_prefix)} mau")
    return samples


def main():
    print("Nap SharedBackbone + detector + homography...")
    backbone = SharedBackbone(CKPT).eval()
    detector1 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    detector2 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    H_by_height = load_h_by_height()
    H_floor = load_h_floor()

    print("Thu thap mau (detect + backbone, chay 1 lan)...")
    samples = collect_samples(detector1, detector2, backbone, H_floor, H_by_height)
    print(f"Tong so mau: {len(samples)}")

    rows = []
    for sigma in SIGMAS:
        fused_confs = []
        agree_count = 0
        drop_strong_count = 0
        with torch.no_grad():
            for s in samples:
                F1_out, F2_out = cross_camera_fuse(
                    s["F1_flat"], s["F2_flat"], s["dist_1to2"], s["dist_2to1"], sigma=sigma)
                v = fuse_to_single_vector(F1_out, F2_out)
                probs_fused = torch.softmax(backbone.classify_from_vector(v), dim=1)[0]
                fused_conf = float(probs_fused.max())
                fused_top1 = CLASS_NAMES[int(probs_fused.argmax())]
                fused_confs.append(fused_conf)
                if fused_top1 == s["cam1_top1"]:
                    agree_count += 1
                if fused_conf - s["min_single_conf"] < -0.1:
                    drop_strong_count += 1
        rows.append({
            "sigma": sigma,
            "avg_fused_conf": round(float(np.mean(fused_confs)), 4),
            "avg_delta_vs_weaker_cam": round(float(np.mean(fused_confs) - np.mean([s["min_single_conf"] for s in samples])), 4),
            "agree_rate_with_cam1": round(agree_count / len(samples), 4),
            "pct_drop_strong": round(drop_strong_count / len(samples), 4),
        })
        print(f"sigma={sigma}: avg_fused_conf={rows[-1]['avg_fused_conf']} "
              f"delta={rows[-1]['avg_delta_vs_weaker_cam']:+.4f} "
              f"agree={rows[-1]['agree_rate_with_cam1']:.2%} "
              f"drop_manh={rows[-1]['pct_drop_strong']:.2%}")

    df = pd.DataFrame(rows)
    out_csv = RESULTS_P3 / "sigma_sweep.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nDa luu: {out_csv}")


if __name__ == "__main__":
    main()
