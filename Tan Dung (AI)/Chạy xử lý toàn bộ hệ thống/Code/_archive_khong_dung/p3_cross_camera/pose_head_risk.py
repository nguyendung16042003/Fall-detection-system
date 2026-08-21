"""
P3 -- Do rui ro ky thuat that (dac ta "no_training" muc 5): Pose Head chi
duoc train tren dac trung DON-CAMERA, chua tung thay input da "tron" qua
cong thuc gop. So sanh confidence Pose Head khi input la anh goc thang vs
khi input da qua cross_camera_fuse().

Dung "P2 Data 2" (co calib DUNG PHONG, co san canh nga That -- Scene 5) lam
nguon danh gia dinh luong dang tin, thay vi P3 Data 1/2 (khac phong, chua
calib -- xem pose_head_risk_p3_room.py cho ban uoc luong tho rieng).
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
from homography_matching import Track, match_two_cameras  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
P2_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "File Run Problem 2" / "P2 data test" / "P2 Data 2"
RESULTS_P2 = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p2"
RESULTS_P3 = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p3"
RESULTS_P3.mkdir(parents=True, exist_ok=True)
CKPT = (AI_ROOT / "Coding" / "training" / "training 3" / "runs" / "classify"
        / "Fall_Detection_Advanced_Loss" / "YOLOv8n_AFCL_Balanced-8" / "weights" / "best.pt")
CLASS_NAMES = ["bend", "exercise", "lie", "sit", "stand"]

DET_CONF = 0.2
DIST_THRESHOLD = 0.5
FRAME_SKIP = 5
SIGMA = 2.0  # chon qua sigma_sweep.py sau khi sua 2 bug he quy chieu -- xem
# Results/p3/sigma_sweep.csv (can bang tot nhat: delta +20.8%, drop_manh 4.5%)


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


def process_scene(scene_prefix, detector1, detector2, backbone, H_floor, H_by_height):
    cap1 = cv2.VideoCapture(str(P2_DIR / f"{scene_prefix}-CAM 1.mp4"))
    cap2 = cv2.VideoCapture(str(P2_DIR / f"{scene_prefix}-CAM 2.mp4"))
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

        foot1 = apply_homography(H_floor["CAM 1"], foot_point_from_bbox(bbox1))
        foot2 = apply_homography(H_floor["CAM 2"], foot_point_from_bbox(bbox2))
        dist = np.linalg.norm(np.array(foot1) - np.array(foot2))
        if dist > DIST_THRESHOLD:
            continue  # khong phai cung 1 nguoi -- bo qua frame nay

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
                (h, w), foot1, foot2, H_by_height["CAM 1"], H_by_height["CAM 2"],
                bbox1, bbox2)
            v_fused = sgie_forward(backbone, x1, x2, has_both_cams=True,
                                    dist_1to2=dist_1to2, dist_2to1=dist_2to1, sigma=SIGMA)
            logits_fused = backbone.classify_from_vector(v_fused)
            probs_fused = torch.softmax(logits_fused, dim=1)[0]

        top1_cam1 = CLASS_NAMES[int(probs1.argmax())]
        top1_cam2 = CLASS_NAMES[int(probs2.argmax())]
        top1_fused = CLASS_NAMES[int(probs_fused.argmax())]
        rows.append({
            "scene": scene_prefix,
            "frame_idx": frame_idx,
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
    print("Nap SharedBackbone + 2 detector rieng...")
    backbone = SharedBackbone(CKPT).eval()
    detector1 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    detector2 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    H_by_height = load_h_by_height()
    H_floor = load_h_floor()

    all_rows = []
    for scene_id in [1, 2, 3, 4, 5]:
        scene_prefix = f"Scene {scene_id}"
        print(f"=== P2 Data 2 / {scene_prefix} ===")
        rows = process_scene(scene_prefix, detector1, detector2, backbone, H_floor, H_by_height)
        print(f"  {len(rows)} frame co ca 2 cam cung thay + khop cap (dist<={DIST_THRESHOLD}m)")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_csv = RESULTS_P3 / "pose_head_risk_p2data2.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    if len(df):
        avg_min_single = df["min_single_cam_conf"].mean()
        avg_fused = df["fused_conf"].mean()
        agree_rate = (df["cam1_top1"] == df["fused_top1"]).mean()
        print("\n" + "=" * 60)
        print(f"So mau: {len(df)}")
        print(f"Confidence trung binh — cam YEU HON trong 2 cam (single, chua gop): {avg_min_single:.4f}")
        print(f"Confidence trung binh — SAU KHI GOP (fused): {avg_fused:.4f}")
        print(f"Chenh lech: {avg_fused - avg_min_single:+.4f} "
              f"({'TANG' if avg_fused > avg_min_single else 'GIAM'} sau khi gop)")
        print(f"Ty le fused_top1 == cam1_top1 (giu nguyen huong doan cam manh hon): {agree_rate:.2%}")
        print("=" * 60)
    print(f"\nDa luu: {out_csv}")


if __name__ == "__main__":
    main()
