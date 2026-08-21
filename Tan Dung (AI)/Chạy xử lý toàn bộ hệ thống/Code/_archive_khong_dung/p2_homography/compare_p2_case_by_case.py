"""
So sanh case-by-case P2 tren Canh 4 (A nga, B dung yen) qua 6 bien the
(goc + 4 augment an toan + dao camera). 3 phuong an: (A) Hungarian+chan
(da chon), (B) Greedy+chan, (C) Hungarian+dau (lay cam hung Eshel & Moses).

GROUND TRUTH DOC LAP (khong dung chinh 3 phuong an tren de tu lam chuan,
tranh thien vi): track "A" (nguoi nga) duoc xac dinh bang HEURISTIC HINH
HOC don gian -- ty le khung hinh (height/width) cua bbox sut giam manh
theo thoi gian (dung -> nam, tu cao-hep sang thap-rong) -- KHONG lien quan
gi den homography/thuat toan gan ghep dang so sanh, nen la chuan doc lap
hop ly. Track con lai (khong nga) la "B".

Case = 1 frame co ca 2 cam deu thay + co ghep cap -- tinh DUNG neu cap
duoc ghep khop DUNG voi ground truth (A-cam1 <-> A-cam2), SAI neu nguoc lai.
"""
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
import json  # noqa: E402

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


def load_H():
    with open(RESULTS_DIR / "homography_matrices.json", encoding="utf-8") as f:
        H_floor = {k: np.array(v) for k, v in json.load(f).items()}
    with open(RESULTS_DIR / "homography_head_matrices.json", encoding="utf-8") as f:
        raw = json.load(f)
    H_head = {k: np.array(v) for k, v in raw["H"].items()}
    return H_floor, H_head


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


def read_all_tracks(video_path, detector):
    cap = cv2.VideoCapture(str(video_path))
    frames_dets = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_dets.append(detect_tracks(detector, frame))
    cap.release()
    return frames_dets


def find_faller_track(frames_dets):
    """Ground truth doc lap: track co ty le height/width GIAM MANH NHAT
    (dung -> nam) qua thoi gian la nguoi nga (A)."""
    track_ratios = {}
    for dets in frames_dets:
        for d in dets:
            x1, y1, x2, y2 = d["bbox_xyxy"]
            w, h = x2 - x1, y2 - y1
            if w <= 0:
                continue
            ratio = h / w
            track_ratios.setdefault(d["track_id"], []).append(ratio)

    best_track, best_drop = None, -1.0
    for tid, ratios in track_ratios.items():
        if len(ratios) < 10:
            continue
        early = np.mean(ratios[:len(ratios) // 4]) if len(ratios) >= 4 else ratios[0]
        late = np.mean(ratios[-len(ratios) // 4:]) if len(ratios) >= 4 else ratios[-1]
        drop = early - late
        if drop > best_drop:
            best_drop = drop
            best_track = tid
    return best_track, best_drop


def project(H, bbox, point_fn):
    return apply_homography(H, point_fn(bbox))


def run_variant(variant, detector1, detector2, H_floor, H_head):
    v1, v2, is_swapped = get_video_paths(variant)
    if not v1.exists() or not v2.exists():
        return None
    cam1_key, cam2_key = ("CAM 2", "CAM 1") if is_swapped else ("CAM 1", "CAM 2")

    frames1 = read_all_tracks(v1, detector1)
    frames2 = read_all_tracks(v2, detector2)

    faller1, _ = find_faller_track(frames1)
    faller2, _ = find_faller_track(frames2)
    if faller1 is None or faller2 is None:
        print(f"  [{variant}] khong xac dinh duoc track nga (ground truth) -- bo qua")
        return None
    print(f"  [{variant}] ground truth (doc lap): track_A_cam1={faller1} track_A_cam2={faller2}")

    n = min(len(frames1), len(frames2))
    rows = []
    for i in range(n):
        dets1, dets2 = frames1[i], frames2[i]
        if not dets1 or not dets2:
            continue

        t1_foot = [Track("CAM 1", d["track_id"], d["bbox_xyxy"],
                          ground_xy=project(H_floor[cam1_key], d["bbox_xyxy"], foot_point_from_bbox))
                   for d in dets1]
        t2_foot = [Track("CAM 2", d["track_id"], d["bbox_xyxy"],
                          ground_xy=project(H_floor[cam2_key], d["bbox_xyxy"], foot_point_from_bbox))
                   for d in dets2]
        t1_head = [Track("CAM 1", d["track_id"], d["bbox_xyxy"],
                          ground_xy=project(H_head[cam1_key], d["bbox_xyxy"], head_point_from_bbox))
                   for d in dets1]
        t2_head = [Track("CAM 2", d["track_id"], d["bbox_xyxy"],
                          ground_xy=project(H_head[cam2_key], d["bbox_xyxy"], head_point_from_bbox))
                   for d in dets2]

        pairs_A = match_two_cameras(t1_foot, t2_foot, distance_threshold=DIST_THRESHOLD)
        pairs_B = match_two_cameras_greedy(t1_foot, t2_foot, distance_threshold=DIST_THRESHOLD)
        pairs_C = match_two_cameras(t1_head, t2_head, distance_threshold=DIST_THRESHOLD)

        def check_correct(pairs):
            for p in pairs:
                if p[0].track_id == faller1 and p[1].track_id == faller2:
                    return True
            # neu faller khong xuat hien trong ket qua ghep, coi la khong xac dinh (bo qua case nay)
            faller_present = any(p[0].track_id == faller1 for p in pairs) or \
                              any(p[1].track_id == faller2 for p in pairs)
            return None if not faller_present else False

        rows.append({
            "variant": variant, "frame": i,
            "A_hungarian_foot": check_correct(pairs_A),
            "B_greedy_foot": check_correct(pairs_B),
            "C_hungarian_head": check_correct(pairs_C),
        })
    return rows


def main():
    print("Nap 2 detector + homography...")
    detector1 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    detector2 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    H_floor, H_head = load_H()

    all_rows = []
    for variant in VARIANTS:
        rows = run_variant(variant, detector1, detector2, H_floor, H_head)
        if rows:
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_csv = RESULTS_DIR / "p2_case_by_case_augmented.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    for col, name in [("A_hungarian_foot", "A: Hungarian+chan (da chon)"),
                       ("B_greedy_foot", "B: Greedy+chan"),
                       ("C_hungarian_head", "C: Hungarian+dau (Eshel&Moses)")]:
        valid = df[df[col].notna()]
        n_correct = (valid[col] == True).sum()
        n_total = len(valid)
        print(f"{name}: {n_correct}/{n_total} frame ghep DUNG voi ground truth doc lap ({n_correct/n_total:.1%})")
    print("=" * 70)
    print(f"\nDa luu: {out_csv}")


if __name__ == "__main__":
    main()
