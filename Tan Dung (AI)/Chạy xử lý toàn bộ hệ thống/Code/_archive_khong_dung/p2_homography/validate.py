"""
P2 -- Validate ghep cap tren du lieu that: "P2 Data 2" (1 nguoi, co canh nga)
va "P2 Data 3" (nhieu nguoi, Canh 4 la bai test quan trong nhat: A nga trong
khi B dung yen gan do -- kiem tra Hungarian co gan dung ID, khong lan giua
2 nguoi, dac biet dung luc A nga).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from homography import apply_homography, foot_point_from_bbox  # noqa: E402
from homography_matching import Track, match_two_cameras  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
P2_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "File Run Problem 2" / "P2 data test"
RESULTS_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p2"

DET_CONF = 0.2  # xac nhan qua debug: 0.4 loc mat nhieu detection that (conf~0.1-0.3
# van la nguoi that o goc quay P2, khac Data Calib nguoi dung gan/ro net hon)
DIST_THRESHOLD = 0.5
FRAME_SKIP = 3  # 15fps -> ~5fps xu ly, du cho validate (khong can real-time o day)


def load_homographies():
    with open(RESULTS_DIR / "homography_matrices.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {cam: np.array(H) for cam, H in raw.items()}


def detect_tracks(detector, frame, conf=DET_CONF):
    """Tra ve list dict {track_id, bbox_xyxy} cho tat ca nguoi trong frame."""
    results = detector.track(frame, classes=[0], conf=conf, persist=True, verbose=False)[0]
    out = []
    if results.boxes is None:
        return out
    for box in results.boxes:
        if box.id is None:
            continue
        out.append({
            "track_id": int(box.id[0]),
            "bbox_xyxy": tuple(box.xyxy[0].cpu().numpy().tolist()),
        })
    return out


def process_scene(scene_prefix, folder, detector1, detector2, H_by_cam, frame_skip=FRAME_SKIP):
    """Chay dong thoi CAM1+CAM2 cua 1 canh, tra ve list dict per-frame ket
    qua ghep cap (frame_idx, so_nguoi_cam1, so_nguoi_cam2, so_cap_ghep,
    chi_tiet_cap). DUNG 2 DETECTOR INSTANCE RIENG cho 2 cam -- neu dung
    chung 1 detector voi persist=True, trang thai tracker (Kalman, track
    history) se bi lan giua 2 luong video khac nhau khi goi xen ke frame
    tung cam, gay mat track/detect sai (loi da gap thuc te khi debug)."""
    cap1 = cv2.VideoCapture(str(folder / f"{scene_prefix}-CAM 1.mp4"))
    cap2 = cv2.VideoCapture(str(folder / f"{scene_prefix}-CAM 2.mp4"))

    rows = []
    frame_idx = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue

        dets1 = detect_tracks(detector1, frame1)
        dets2 = detect_tracks(detector2, frame2)

        tracks1 = [Track("CAM 1", d["track_id"], d["bbox_xyxy"],
                          ground_xy=apply_homography(H_by_cam["CAM 1"], foot_point_from_bbox(d["bbox_xyxy"])))
                   for d in dets1]
        tracks2 = [Track("CAM 2", d["track_id"], d["bbox_xyxy"],
                          ground_xy=apply_homography(H_by_cam["CAM 2"], foot_point_from_bbox(d["bbox_xyxy"])))
                   for d in dets2]

        pairs = match_two_cameras(tracks1, tracks2, distance_threshold=DIST_THRESHOLD)
        pair_str = ";".join(f"{p[0].track_id}<->{p[1].track_id}" for p in pairs)
        dist_str = ";".join(f"{np.linalg.norm(np.array(p[0].ground_xy)-np.array(p[1].ground_xy)):.3f}" for p in pairs)

        rows.append({
            "frame_idx": frame_idx,
            "n_cam1": len(tracks1),
            "n_cam2": len(tracks2),
            "n_pairs": len(pairs),
            "pairs": pair_str,
            "pair_distances_m": dist_str,
        })

    cap1.release()
    cap2.release()
    return rows


def main():
    print("Dang khoi tao 2 detector YOLOv8n rieng (1 cho moi cam) + nap homography...")
    detector1 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    detector2 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    H_by_cam = load_homographies()

    all_results = {}
    for data_name, scenes in [("P2 Data 2", [1, 2, 3, 4, 5]), ("P2 Data 3", [1, 2, 3, 4])]:
        folder = P2_DIR / data_name
        for scene_id in scenes:
            scene_prefix = f"Scene {scene_id}"
            print(f"\n=== {data_name} / {scene_prefix} ===")
            rows = process_scene(scene_prefix, folder, detector1, detector2, H_by_cam)
            df = pd.DataFrame(rows)
            key = f"{data_name}_{scene_prefix}".replace(" ", "_")
            all_results[key] = df
            out_csv = RESULTS_DIR / f"validate_{key}.csv"
            df.to_csv(out_csv, index=False, encoding="utf-8-sig")

            n_frames = len(df)
            n_with_pair = (df["n_pairs"] > 0).sum() if n_frames else 0
            n_multi_pair = (df["n_pairs"] > 1).sum() if n_frames else 0
            print(f"  {n_frames} frame xu ly | {n_with_pair} frame co >=1 cap ghep | "
                  f"{n_multi_pair} frame co >=2 cap ghep")
            if n_frames:
                print(f"  Vi du 3 dong dau: \n{df.head(3).to_string(index=False)}")

    print(f"\nDa luu tat ca CSV vao: {RESULTS_DIR}/validate_*.csv")


if __name__ == "__main__":
    main()
