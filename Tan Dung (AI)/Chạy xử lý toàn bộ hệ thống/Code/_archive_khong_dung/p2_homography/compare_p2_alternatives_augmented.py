"""
Ban mo rong cua compare_p2_alternatives.py -- CHI Canh 4 (A nga, B dung yen,
quan trong nhat), qua 6 bien the (goc + 4 augment an toan + dao camera) thay
vi chi 1 lan chay. Dung LAI dung phuong phap da kiem chung (agree-with-
baseline Hungarian+chan), KHONG dung heuristic ground-truth tu dong moi
(da phat hien loi voi track gay doan -- xem compare_p2_case_by_case.py).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from homography import apply_homography, foot_point_from_bbox, head_point_from_bbox  # noqa: E402
from homography_matching import Track, match_two_cameras  # noqa: E402
from greedy_matching import match_two_cameras_greedy  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
P2_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "File Run Problem 2" / "P2 data test"
RESULTS_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p2"
SRC_DIR = P2_DIR / "P2 Data 3"
AUG_DIR = P2_DIR / "P2 Data 3 (augmented)"
SCENE_ID = 4
DET_CONF = 0.2
DIST_THRESHOLD = 0.5

VARIANTS = ["original", "aug-bright_up", "aug-bright_down", "aug-noise", "aug-compress", "swap"]


def get_video_paths(variant):
    if variant == "original":
        return (SRC_DIR / f"Scene {SCENE_ID}-CAM 1.mp4", SRC_DIR / f"Scene {SCENE_ID}-CAM 2.mp4", False)
    if variant == "swap":
        return (SRC_DIR / f"Scene {SCENE_ID}-CAM 2.mp4", SRC_DIR / f"Scene {SCENE_ID}-CAM 1.mp4", True)
    return (AUG_DIR / f"Scene {SCENE_ID}-CAM 1_{variant}.mp4",
            AUG_DIR / f"Scene {SCENE_ID}-CAM 2_{variant}.mp4", False)


def load_H(name):
    with open(RESULTS_DIR / name, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if "H" in raw:
        raw = raw["H"]
    return {cam: np.array(H) for cam, H in raw.items()}


def detect_tracks(detector, frame, conf=DET_CONF):
    results = detector.track(frame, classes=[0], conf=conf, persist=True, verbose=False)[0]
    out = []
    if results.boxes is None:
        return out
    for box in results.boxes:
        if box.id is None:
            continue
        out.append({"track_id": int(box.id[0]),
                     "bbox_xyxy": tuple(box.xyxy[0].cpu().numpy().tolist())})
    return out


def pair_signature(pairs):
    return frozenset((p[0].track_id, p[1].track_id) for p in pairs)


def process_variant(variant, detector1, detector2, H_floor, H_head):
    v1, v2, is_swapped = get_video_paths(variant)
    if not v1.exists() or not v2.exists():
        print(f"  [{variant}] thieu file, bo qua")
        return []
    cam1_key, cam2_key = ("CAM 2", "CAM 1") if is_swapped else ("CAM 1", "CAM 2")

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

        dets1 = detect_tracks(detector1, frame1)
        dets2 = detect_tracks(detector2, frame2)
        if not dets1 or not dets2:
            continue

        t1_foot = [Track("C1", d["track_id"], d["bbox_xyxy"],
                          ground_xy=apply_homography(H_floor[cam1_key], foot_point_from_bbox(d["bbox_xyxy"])))
                   for d in dets1]
        t2_foot = [Track("C2", d["track_id"], d["bbox_xyxy"],
                          ground_xy=apply_homography(H_floor[cam2_key], foot_point_from_bbox(d["bbox_xyxy"])))
                   for d in dets2]
        pairs_A = match_two_cameras(t1_foot, t2_foot, distance_threshold=DIST_THRESHOLD)
        pairs_B = match_two_cameras_greedy(t1_foot, t2_foot, distance_threshold=DIST_THRESHOLD)

        t1_head = [Track("C1", d["track_id"], d["bbox_xyxy"],
                          ground_xy=apply_homography(H_head[cam1_key], head_point_from_bbox(d["bbox_xyxy"])))
                   for d in dets1]
        t2_head = [Track("C2", d["track_id"], d["bbox_xyxy"],
                          ground_xy=apply_homography(H_head[cam2_key], head_point_from_bbox(d["bbox_xyxy"])))
                   for d in dets2]
        pairs_C = match_two_cameras(t1_head, t2_head, distance_threshold=DIST_THRESHOLD)

        sig_A, sig_B, sig_C = pair_signature(pairs_A), pair_signature(pairs_B), pair_signature(pairs_C)
        rows.append({
            "variant": variant, "frame_idx": frame_idx,
            "B_agrees_with_A": sig_B == sig_A,
            "C_agrees_with_A": sig_C == sig_A,
        })
    cap1.release()
    cap2.release()
    return rows


def main():
    print("Nap 2 detector + H_floor + H_head...")
    detector1 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    detector2 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    H_floor = load_H("homography_matrices.json")
    H_head = load_H("homography_head_matrices.json")

    all_rows = []
    for variant in VARIANTS:
        rows = process_variant(variant, detector1, detector2, H_floor, H_head)
        if rows:
            df = pd.DataFrame(rows)
            agree_B = df["B_agrees_with_A"].mean()
            agree_C = df["C_agrees_with_A"].mean()
            print(f"  [{variant}] {len(df)} frame | Greedy+chan khop: {agree_B:.1%} | Hungarian+dau khop: {agree_C:.1%}")
        all_rows.extend(rows)

    df_all = pd.DataFrame(all_rows)
    out_csv = RESULTS_DIR / "compare_p2_alternatives_augmented.csv"
    df_all.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print(f"TONG KET Canh 4 -- {len(df_all)} frame qua {len(VARIANTS)} bien the "
          f"(moc tham chieu = Hungarian+chan da validate):")
    print(f"Greedy+chan khop: {df_all['B_agrees_with_A'].mean():.1%}")
    print(f"Hungarian+dau khop: {df_all['C_agrees_with_A'].mean():.1%}")
    print("=" * 70)
    print(f"\nDa luu: {out_csv}")


if __name__ == "__main__":
    main()
