"""
Ban day du: chay pipeline P3 (fused, co gop) qua CA 15 canh that da quay
(P2 Data2x5 + P2 Data3x4 + P3 Data1x4 + P3 Data2x2) -- DUNG BO DATA Y HET
Espinosa da duoc test qua (xem evaluate_p2p3.py, Code/p0_espinosa_baseline)
-- de so sanh cong bang so case bat duoc nga tren TONG so case, khong chi
6 canh rieng P3. Nhan Fall/ADL lay tu kich ban quay (Kich ban P2.docx/
Kich Ban P3.docx), khong doan.

Method: quyet dinh Fall/ADL cap SCENE (video-level) bang OR-logic tren cac
frame co ca 2 cam cung thay + khop khoang cach san (giong chinh xac cach
evaluate_p2p3.py dang dung cho Espinosa, de so sanh cung 1 muc do "coarse").
Nhan Fall neu fused_top1 == "lie" o BAT KY frame nao trong canh.
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
P2_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "File Run Problem 2" / "P2 data test"
P3_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "File Run Problem 3" / "P3"
RESULTS_P2 = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p2"
RESULTS_P3 = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p3"
CKPT = (AI_ROOT / "Coding" / "training" / "training 3" / "runs" / "classify"
        / "Fall_Detection_Advanced_Loss" / "YOLOv8n_AFCL_Balanced-8" / "weights" / "best.pt")
CLASS_NAMES = ["bend", "exercise", "lie", "sit", "stand"]
DET_CONF = 0.2
FRAME_SKIP = 2
SIGMA = 2.0

# (folder, video1, video2, cam1_key, cam2_key, dist_threshold, label, scene_name)
SCENES = [
    (P2_DIR / "P2 Data 2", "Scene {}-CAM 1.mp4", "Scene {}-CAM 2.mp4", "CAM 1", "CAM 2", 0.5, 1, 0, "P2D2_S1"),
    (P2_DIR / "P2 Data 2", "Scene {}-CAM 1.mp4", "Scene {}-CAM 2.mp4", "CAM 1", "CAM 2", 0.5, 2, 0, "P2D2_S2"),
    (P2_DIR / "P2 Data 2", "Scene {}-CAM 1.mp4", "Scene {}-CAM 2.mp4", "CAM 1", "CAM 2", 0.5, 3, 0, "P2D2_S3"),
    (P2_DIR / "P2 Data 2", "Scene {}-CAM 1.mp4", "Scene {}-CAM 2.mp4", "CAM 1", "CAM 2", 0.5, 4, 0, "P2D2_S4"),
    (P2_DIR / "P2 Data 2", "Scene {}-CAM 1.mp4", "Scene {}-CAM 2.mp4", "CAM 1", "CAM 2", 0.5, 5, 1, "P2D2_S5_FALL"),
    (P2_DIR / "P2 Data 3", "Scene {}-CAM 1.mp4", "Scene {}-CAM 2.mp4", "CAM 1", "CAM 2", 0.5, 1, 0, "P2D3_S1"),
    (P2_DIR / "P2 Data 3", "Scene {}-CAM 1.mp4", "Scene {}-CAM 2.mp4", "CAM 1", "CAM 2", 0.5, 2, 0, "P2D3_S2"),
    (P2_DIR / "P2 Data 3", "Scene {}-CAM 1.mp4", "Scene {}-CAM 2.mp4", "CAM 1", "CAM 2", 0.5, 3, 0, "P2D3_S3"),
    (P2_DIR / "P2 Data 3", "Scene {}-CAM 1.mp4", "Scene {}-CAM 2.mp4", "CAM 1", "CAM 2", 0.5, 4, 1, "P2D3_S4_FALL"),
    (P3_DIR / "P3 Data 1", "Scene {}-Cam 1.mp4", "Scene {}-Cam 2.mp4", "P3 Cam1", "P3 Cam2", 1.5, 1, 1, "P3D1_S1_FALL"),
    (P3_DIR / "P3 Data 1", "Scene {}-Cam 1.mp4", "Scene {}-Cam 2.mp4", "P3 Cam1", "P3 Cam2", 1.5, 2, 1, "P3D1_S2_FALL"),
    (P3_DIR / "P3 Data 1", "Scene {}-Cam 1.mp4", "Scene {}-Cam 2.mp4", "P3 Cam1", "P3 Cam2", 1.5, 3, 1, "P3D1_S3_FALL"),
    (P3_DIR / "P3 Data 1", "Scene {}-Cam 1.mp4", "Scene {}-Cam 2.mp4", "P3 Cam1", "P3 Cam2", 1.5, 4, 1, "P3D1_S4_FALL"),
    (P3_DIR / "P3 Data 2", "Scene {}-Cam 1.mp4", "Scene {}-Cam 2.mp4", "P3 Cam1", "P3 Cam2", 1.5, 1, 0, "P3D2_S1"),
    (P3_DIR / "P3 Data 2", "Scene {}-Cam 1.mp4", "Scene {}-Cam 2.mp4", "P3 Cam1", "P3 Cam2", 1.5, 2, 0, "P3D2_S2"),
]


def load_calib(prefix):
    """prefix: 'p2' hoac 'p3' -- doc dung file calib tuong ung."""
    results_dir = RESULTS_P2 if prefix == "p2" else RESULTS_P3
    floor_name = "homography_matrices.json" if prefix == "p2" else "homography_matrices_p3.json"
    head_name = "homography_head_matrices.json" if prefix == "p2" else "homography_head_matrices_p3.json"
    hbh_name = "h_by_height.json" if prefix == "p2" else "h_by_height_p3.json"

    with open(results_dir / floor_name, encoding="utf-8") as f:
        H_floor = {k: np.array(v) for k, v in json.load(f).items()}
    with open(results_dir / hbh_name, encoding="utf-8") as f:
        raw = json.load(f)
    H_by_height = {cam: {float(h): np.array(H) for h, H in d.items()} for cam, d in raw.items()}
    return H_floor, H_by_height


def detect_one(detector, frame, conf=DET_CONF):
    results = detector(frame, classes=[0], conf=conf, verbose=False)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return None
    return tuple(results.boxes[0].xyxy[0].cpu().numpy().tolist())


def process_scene(folder, v1_tpl, v2_tpl, cam1_key, cam2_key, dist_th, scene_id,
                   detector1, detector2, backbone, H_floor, H_by_height):
    cap1 = cv2.VideoCapture(str(folder / v1_tpl.format(scene_id)))
    cap2 = cv2.VideoCapture(str(folder / v2_tpl.format(scene_id)))
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
        foot1 = apply_homography(H_floor[cam1_key], foot_point_from_bbox(bbox1))
        foot2 = apply_homography(H_floor[cam2_key], foot_point_from_bbox(bbox2))
        dist = np.linalg.norm(np.array(foot1) - np.array(foot2))
        if dist > dist_th:
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
            h, w = grid1.shape[2], grid1.shape[3]
            dist_1to2, dist_2to1 = build_distance_matrices(
                (h, w), foot1, foot2, H_by_height[cam1_key], H_by_height[cam2_key], bbox1, bbox2)
            v_fused = sgie_forward(backbone, x1, x2, has_both_cams=True,
                                    dist_1to2=dist_1to2, dist_2to1=dist_2to1, sigma=SIGMA)
            logits_fused = backbone.classify_from_vector(v_fused)
            probs_fused = torch.softmax(logits_fused, dim=1)[0]
        rows.append(CLASS_NAMES[int(probs_fused.argmax())])
    cap1.release()
    cap2.release()
    return rows


def main():
    print("Nap SharedBackbone + 2 detector + calib P2/P3...")
    backbone = SharedBackbone(CKPT).eval()
    detector1 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    detector2 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    H_floor_p2, H_by_height_p2 = load_calib("p2")
    H_floor_p3, H_by_height_p3 = load_calib("p3")

    results = []
    for folder, v1_tpl, v2_tpl, cam1_key, cam2_key, dist_th, scene_id, label, name in SCENES:
        is_p3 = cam1_key.startswith("P3")
        H_floor = H_floor_p3 if is_p3 else H_floor_p2
        H_by_height = H_by_height_p3 if is_p3 else H_by_height_p2
        preds = process_scene(folder, v1_tpl, v2_tpl, cam1_key, cam2_key, dist_th, scene_id,
                               detector1, detector2, backbone, H_floor, H_by_height)
        pred_fall = any(p == "lie" for p in preds)
        correct = "DUNG" if int(pred_fall) == label else "SAI"
        print(f"[{name}] label={label} pred={int(pred_fall)} ({len(preds)} frame hop le) -- {correct}")
        results.append({"scene": name, "label": label, "pred": int(pred_fall), "n_frames": len(preds)})

    df = pd.DataFrame(results)
    out_csv = RESULTS_P3 / "p3_fused_all15_results.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    tp = ((df.pred == 1) & (df.label == 1)).sum()
    fn = ((df.pred == 0) & (df.label == 1)).sum()
    tn = ((df.pred == 0) & (df.label == 0)).sum()
    fp = ((df.pred == 1) & (df.label == 0)).sum()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    acc = (tp + tn) / len(df)
    print("\n" + "=" * 60)
    print(f"So canh: {len(df)} (Fall={df.label.sum()}, ADL={len(df)-df.label.sum()})")
    print(f"So case BAT DUOC NGA: {tp}/{df.label.sum()} (TP/tong so canh Fall that)")
    print(f"TP={tp} FN={fn} TN={tn} FP={fp}")
    print(f"Sensitivity(bat nga)={sens:.2%} Specificity={spec:.2%} Accuracy={acc:.2%}")
    print("=" * 60)
    print(f"Da luu: {out_csv}")


if __name__ == "__main__":
    main()
