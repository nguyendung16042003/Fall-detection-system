"""
P2 -- Calib homography tu video "P2 Data Calib" (4 canh, moi canh 1 nguoi
dung yen tai 1 trong 4 diem A/B/C/D cua hinh vuong 0.7m tren san, theo dung
"Kich ban P2.docx"). Moi file video da duoc cat rieng tung canh san (Scene N).

Thu tu diem theo kich ban: Scene 1=A, Scene 2=B, Scene 3=C, Scene 4=D.
Gia dinh hinh vuong chuan: A=(0,0), B=(0.7,0), C=(0.7,0.7), D=(0,0.7) (met).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from homography import calibrate_from_points, foot_point_from_bbox, head_point_from_bbox  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
CALIB_DIR = (AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "File Run Problem 2"
             / "P2 data test" / "P2 Data Calib")
OUT_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SQUARE_SIDE_M = 0.7
WORLD_POINTS = {
    1: (0.0, 0.0),
    2: (SQUARE_SIDE_M, 0.0),
    3: (SQUARE_SIDE_M, SQUARE_SIDE_M),
    4: (0.0, SQUARE_SIDE_M),
}
CAMS = ["CAM 1", "CAM 2"]
DET_CONF = 0.4
# Nguoi quay calib mac ao den = nguoi A (xac nhan qua anh, khop "A mac ao den"
# trong Kich ban P2/P1.docx), chieu cao A = 1.77m (cap nhat tu nguoi dung).
CALIBRATOR_HEIGHT_M = 1.77


def median_point(video_path, detector, point_fn):
    """Chay detector tren toan bo video (nguoi dung yen), tra ve 1 diem pixel
    trung vi (on dinh hon trung binh neu co vai frame detect loi).
    point_fn: foot_point_from_bbox hoac head_point_from_bbox."""
    cap = cv2.VideoCapture(str(video_path))
    points = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = detector(frame, classes=[0], conf=DET_CONF, verbose=False)[0]
        if results.boxes is not None and len(results.boxes) > 0:
            # nguoi dung yen 1 minh trong khung -> lay box tin cay nhat
            box = results.boxes[0].xyxy[0].cpu().numpy()
            points.append(point_fn(box))
    cap.release()
    if not points:
        return None
    arr = np.array(points)
    return (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))


def calibrate_camera(cam_name, detector):
    """Tra ve H_floor (mat san, Z=0) VA H_head (mat phang o do cao
    CALIBRATOR_HEIGHT_M) -- 2 mat phang doc lap, dung de noi suy H_by_height
    (xem homography_by_height.py) thay vi doan he so."""
    foot_pixel_points, head_pixel_points, world_points = [], [], []
    for scene_id in [1, 2, 3, 4]:
        video_path = CALIB_DIR / f"Scene {scene_id}-{cam_name}.mp4"
        foot_px = median_point(video_path, detector, foot_point_from_bbox)
        head_px = median_point(video_path, detector, head_point_from_bbox)
        if foot_px is None or head_px is None:
            print(f"  CANH BAO: khong detect duoc nguoi trong {video_path.name}")
            continue
        foot_pixel_points.append(foot_px)
        head_pixel_points.append(head_px)
        world_points.append(WORLD_POINTS[scene_id])
        print(f"  Scene {scene_id} ({cam_name}): foot_px={foot_px} head_px={head_px} -> world={WORLD_POINTS[scene_id]}")

    if len(foot_pixel_points) < 4:
        raise RuntimeError(f"{cam_name}: chi calib duoc {len(foot_pixel_points)}/4 diem, khong du de tinh homography")

    H_floor = calibrate_from_points(foot_pixel_points, world_points)
    H_head = calibrate_from_points(head_pixel_points, world_points)
    return H_floor, H_head, foot_pixel_points, world_points


def main():
    print("Dang khoi tao detector YOLOv8n...")
    detector = YOLO(str(AI_ROOT / "yolov8n.pt"))

    all_H_floor = {}
    all_H_head = {}
    for cam_name in CAMS:
        print(f"\n=== Calib {cam_name} ===")
        H_floor, H_head, pixel_pts, world_pts = calibrate_camera(cam_name, detector)
        all_H_floor[cam_name] = H_floor.tolist()
        all_H_head[cam_name] = H_head.tolist()
        print(f"  Homography san (Z=0) {cam_name}:\n{H_floor}")
        print(f"  Homography dinh dau (Z={CALIBRATOR_HEIGHT_M}m) {cam_name}:\n{H_head}")

        # Kiem tra sai so: chieu lai 4 diem pixel san qua H_floor, so voi world that
        from homography import apply_homography
        errors = []
        for px, wd in zip(pixel_pts, world_pts):
            proj = apply_homography(H_floor, px)
            err = np.linalg.norm(np.array(proj) - np.array(wd))
            errors.append(err)
        print(f"  Sai so chieu lai san (m): {[round(e, 4) for e in errors]} "
              f"(trung binh {np.mean(errors):.4f}m)")

    out_path = OUT_DIR / "homography_matrices.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_H_floor, f, indent=2)
    print(f"\nDa luu: {out_path}")

    out_path_head = OUT_DIR / "homography_head_matrices.json"
    with open(out_path_head, "w", encoding="utf-8") as f:
        json.dump({"height_m": CALIBRATOR_HEIGHT_M, "H": all_H_head}, f, indent=2)
    print(f"Da luu: {out_path_head}")


if __name__ == "__main__":
    main()
