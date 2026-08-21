"""
Smoke test P3: chay 1 cap crop that tu "P3 Data 1" (nga giua 2 cam) qua toan
bo luong SharedBackbone -> cross_camera_fuse -> classify, kiem tra khong loi
va so sanh voi ket qua khong gop (1 cam rieng le).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p2_homography"))

from shared_backbone import SharedBackbone, preprocess_crop  # noqa: E402
from cross_camera_fuse import (  # noqa: E402
    build_distance_matrices, sgie_forward, grid_to_flat,
)
from homography import apply_homography, foot_point_from_bbox  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
P3_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "File Run Problem 3" / "P3"
RESULTS_P2 = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p2"
CKPT = (AI_ROOT / "Coding" / "training" / "training 3" / "runs" / "classify"
        / "Fall_Detection_Advanced_Loss" / "YOLOv8n_AFCL_Balanced-8" / "weights" / "best.pt")
CLASS_NAMES = ["bend", "exercise", "lie", "sit", "stand"]


def load_h_by_height():
    with open(RESULTS_P2 / "h_by_height.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {cam: {float(h): np.array(H) for h, H in d.items()} for cam, d in raw.items()}


def load_h_floor():
    with open(RESULTS_P2 / "homography_matrices.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {cam: np.array(H) for cam, H in raw.items()}


def get_frame_and_bbox(video_path, detector, frame_idx=10, conf=0.2):
    cap = cv2.VideoCapture(str(video_path))
    frame = None
    for _ in range(frame_idx):
        ret, frame = cap.read()
        if not ret:
            break
    cap.release()
    results = detector(frame, classes=[0], conf=conf, verbose=False)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return frame, None
    box = results.boxes[0].xyxy[0].cpu().numpy()
    return frame, tuple(box.tolist())


def find_synced_frame_with_both_boxes(video1, video2, detector, conf=0.15, max_frame=120, step=3):
    """Quet frame_idx tang dan, tim khung hinh dong bo ma CA 2 cam deu detect
    duoc nguoi (giai bai toan cam2 co the khong thay nguoi o frame ban dau)."""
    for idx in range(3, max_frame, step):
        f1, b1 = get_frame_and_bbox(video1, detector, frame_idx=idx, conf=conf)
        f2, b2 = get_frame_and_bbox(video2, detector, frame_idx=idx, conf=conf)
        if b1 is not None and b2 is not None:
            print(f"  Tim thay frame_idx={idx} ca 2 cam deu co bbox")
            return f1, b1, f2, b2
    return None, None, None, None


def main():
    print("Nap SharedBackbone + detector...")
    backbone = SharedBackbone(CKPT).eval()
    detector = YOLO(str(AI_ROOT / "yolov8n.pt"))
    H_by_height = load_h_by_height()
    H_floor = load_h_floor()

    video1 = P3_DIR / "P3 Data 1" / "Scene 1-Cam 1.mp4"
    video2 = P3_DIR / "P3 Data 1" / "Scene 1-Cam 2.mp4"

    print("Dang quet frame de tim khung ca 2 cam deu detect duoc nguoi...")
    frame1, bbox1, frame2, bbox2 = find_synced_frame_with_both_boxes(video1, video2, detector)
    print("bbox1:", bbox1)
    print("bbox2:", bbox2)
    if bbox1 is None or bbox2 is None:
        print("KHONG tim thay frame nao ca 2 cam deu detect duoc trong khoang quet.")
        return

    crop1 = frame1[int(bbox1[1]):int(bbox1[3]), int(bbox1[0]):int(bbox1[2])]
    crop2 = frame2[int(bbox2[1]):int(bbox2[3]), int(bbox2[0]):int(bbox2[2])]
    x1 = preprocess_crop(crop1)
    x2 = preprocess_crop(crop2)

    foot1 = apply_homography(H_floor["CAM 1"], foot_point_from_bbox(bbox1))
    foot2 = apply_homography(H_floor["CAM 2"], foot_point_from_bbox(bbox2))
    print("foot1 (san, m):", foot1)
    print("foot2 (san, m):", foot2)
    dist_ground = np.linalg.norm(np.array(foot1) - np.array(foot2))
    print("Khoang cach san giua 2 diem chieu (m):", round(dist_ground, 3),
          "(cang nho cang giong 1 nguoi)")

    with torch.no_grad():
        # Duong di KHONG gop (tung cam rieng)
        grid1 = backbone(x1)
        grid2 = backbone(x2)
        logits1 = backbone.classify_from_grid(grid1)
        logits2 = backbone.classify_from_grid(grid2)
        probs1 = torch.softmax(logits1, dim=1)[0]
        probs2 = torch.softmax(logits2, dim=1)[0]
        print("\n--- KHONG gop (1 cam rieng) ---")
        print("Cam1:", {CLASS_NAMES[i]: round(float(probs1[i]), 3) for i in range(5)})
        print("Cam2:", {CLASS_NAMES[i]: round(float(probs2[i]), 3) for i in range(5)})

        # Duong di CO gop (cross-camera fuse)
        h, w = grid1.shape[2], grid1.shape[3]
        dist_1to2, dist_2to1 = build_distance_matrices(
            (h, w), foot1, foot2, H_by_height["CAM 1"], H_by_height["CAM 2"],
            bbox1, bbox2)

        v_fused = sgie_forward(backbone, x1, x2, has_both_cams=True,
                                dist_1to2=dist_1to2, dist_2to1=dist_2to1, sigma=1.0)
        logits_fused = backbone.classify_from_vector(v_fused)
        probs_fused = torch.softmax(logits_fused, dim=1)[0]
        print("\n--- CO gop (cross_camera_fuse) ---")
        print("Fused:", {CLASS_NAMES[i]: round(float(probs_fused[i]), 3) for i in range(5)})

    print("\nSMOKE TEST CHAY XONG KHONG LOI.")


if __name__ == "__main__":
    main()
