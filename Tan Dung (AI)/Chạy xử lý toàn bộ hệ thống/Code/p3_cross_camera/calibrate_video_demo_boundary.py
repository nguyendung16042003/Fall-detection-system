"""
Calib homography MOI cho video demo Boundary (phong/set-up hoan toan moi,
KHONG lien quan calib P2/P3 cu). Video LIEN TUC 1 file/cam (tu dong phat
hien 4 doan DUNG YEN qua toc do di chuyen diem chan), CO CALIB RIENG cho
CA 2 CAMERA (khac ban P3 goc chi calib 1 cam roi muon cam kia).

Theo "Video demo/Boundary/Kịch bản.docx": nguoi dung yen 4 diem cua o vuong
canh 0.7m (KHAC 0.5m cua P3 goc), da xac nhan voi nguoi dung chieu cao
nguoi calib = 1.77m.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p2_homography"))
from homography import calibrate_from_points, foot_point_from_bbox, head_point_from_bbox, apply_homography  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Video demo" / "Boundary" / "Homography"
CALIB_VIDEOS = {
    "CAM 1": DEMO_DIR / "Cam 1- Scene 1.avi",
    "CAM 2": DEMO_DIR / "Cam 2- Scene 1.avi",
}
RESULTS_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "video_demo"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SQUARE_SIDE_M = 0.7
WORLD_POINTS = {
    0: (0.0, 0.0),
    1: (SQUARE_SIDE_M, 0.0),
    2: (SQUARE_SIDE_M, SQUARE_SIDE_M),
    3: (0.0, SQUARE_SIDE_M),
}
CALIBRATOR_HEIGHT_M = 1.77
DET_CONF = 0.3
SPEED_THRESHOLD_PX = 5.0
MIN_SEGMENT_FRAMES = 15


def extract_foot_head_trajectory(video_path, detector):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    foot_pts, head_pts = [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = detector(frame, classes=[0], conf=DET_CONF, verbose=False)[0]
        if results.boxes is not None and len(results.boxes) > 0:
            box = results.boxes[0].xyxy[0].cpu().numpy()
            foot_pts.append(foot_point_from_bbox(box))
            head_pts.append(head_point_from_bbox(box))
        else:
            foot_pts.append(None)
            head_pts.append(None)
    cap.release()
    return foot_pts, head_pts, fps


def median_point_in_range(pts, start, end):
    sub = [p for p in pts[start:end] if p is not None]
    arr = np.array(sub)
    return (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))


def find_stable_segments(foot_pts):
    valid_idx = [i for i, p in enumerate(foot_pts) if p is not None]
    speeds = {}
    for a, b in zip(valid_idx[:-1], valid_idx[1:]):
        pa, pb = np.array(foot_pts[a]), np.array(foot_pts[b])
        speeds[b] = np.linalg.norm(pb - pa)

    segments = []
    cur_start = None
    for i in valid_idx:
        is_slow = speeds.get(i, 0.0) < SPEED_THRESHOLD_PX
        if is_slow:
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None and i - cur_start >= MIN_SEGMENT_FRAMES:
                segments.append((cur_start, i))
            cur_start = None
    if cur_start is not None and valid_idx[-1] - cur_start >= MIN_SEGMENT_FRAMES:
        segments.append((cur_start, valid_idx[-1]))

    merged = []
    for s, e in segments:
        if merged:
            prev_s, prev_e = merged[-1]
            prev_pt = np.array(median_point_in_range(foot_pts, prev_s, prev_e))
            cur_pt = np.array(median_point_in_range(foot_pts, s, e))
            if np.linalg.norm(cur_pt - prev_pt) < 30.0:
                merged[-1] = (prev_s, e)
                continue
        merged.append((s, e))
    return merged


def calibrate_one_camera(cam_name, video_path, detector):
    print(f"\n=== Calib {cam_name} ({video_path.name}) ===")
    foot_pts, head_pts, fps = extract_foot_head_trajectory(video_path, detector)
    print(f"  {len(foot_pts)} frame, fps={fps}")

    segments = find_stable_segments(foot_pts)
    print(f"  Phat hien {len(segments)} doan dung yen:")
    for i, (s, e) in enumerate(segments):
        print(f"    Doan {i}: frame {s}-{e} ({s/fps:.1f}s-{e/fps:.1f}s), {e-s} frame")

    if len(segments) != 4:
        raise RuntimeError(
            f"{cam_name}: ky vong 4 doan dung yen, phat hien {len(segments)} -- kiem tra lai "
            f"SPEED_THRESHOLD_PX/MIN_SEGMENT_FRAMES hoac xem lai video truoc khi tin ket qua."
        )

    foot_pixel_points, head_pixel_points, world_points = [], [], []
    for i, (s, e) in enumerate(segments):
        foot_px = median_point_in_range(foot_pts, s, e)
        head_px = median_point_in_range(head_pts, s, e)
        foot_pixel_points.append(foot_px)
        head_pixel_points.append(head_px)
        world_points.append(WORLD_POINTS[i])
        print(f"    Doan {i}: foot_px={foot_px} head_px={head_px} -> world={WORLD_POINTS[i]}")

    H_floor = calibrate_from_points(foot_pixel_points, world_points)
    H_head = calibrate_from_points(head_pixel_points, world_points)

    errors = []
    for px, wd in zip(foot_pixel_points, world_points):
        proj = apply_homography(H_floor, px)
        err = np.linalg.norm(np.array(proj) - np.array(wd))
        errors.append(err)
    print(f"  Sai so chieu lai san (m): {[round(e, 4) for e in errors]} (trung binh {np.mean(errors):.4f}m)")
    if np.mean(errors) > 0.15:
        print(f"  CANH BAO: sai so trung binh {np.mean(errors):.4f}m kha lon (>0.15m) -- "
              f"kiem tra lai truoc khi tin ket qua.")

    return H_floor, H_head


def main():
    print("Nap detector YOLOv8n...")
    detector = YOLO(str(AI_ROOT / "yolov8n.pt"))

    all_H_floor, all_H_head = {}, {}
    for cam_name, video_path in CALIB_VIDEOS.items():
        H_floor, H_head = calibrate_one_camera(cam_name, video_path, detector)
        all_H_floor[cam_name] = H_floor.tolist()
        all_H_head[cam_name] = H_head.tolist()

    out_floor = RESULTS_DIR / "homography_matrices.json"
    with open(out_floor, "w", encoding="utf-8") as f:
        json.dump(all_H_floor, f, indent=2)
    out_head = RESULTS_DIR / "homography_head_matrices.json"
    with open(out_head, "w", encoding="utf-8") as f:
        json.dump({"height_m": CALIBRATOR_HEIGHT_M, "H": all_H_head}, f, indent=2)

    print(f"\nDa luu: {out_floor}")
    print(f"Da luu: {out_head}")


if __name__ == "__main__":
    main()
