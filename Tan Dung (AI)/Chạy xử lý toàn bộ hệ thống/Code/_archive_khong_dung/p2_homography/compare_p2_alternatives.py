"""
P2 -- So sanh ky thuat (xem plan, muc "P2 CO so sanh ky thuat"). Tren CUNG 1
luot detect+track (P2 Data 3, ca 4 canh), tinh SONG SONG 3 phuong an gan ghep
tu cung cap track/bbox moi frame:

  (A) BASELINE -- Hungarian + diem CHAN (H_floor) -- chinh la homography_matching.py
      da validate (khong sua gi).
  (B) Greedy nearest-neighbor + diem CHAN (H_floor) -- thay THUAT TOAN gan ghep,
      giu nguyen mat phang chieu.
  (C) Hungarian + diem DAU (H_head) -- lay cam hung Eshel & Moses (CVPR 2008,
      dung mat phang DAU cho homography da camera dam dong) -- giu nguyen
      thuat toan Hungarian, thay MAT PHANG chieu.

Vi (A) da duoc xac nhan dung qua kiem tra truoc do (validate.py: khop ID on
dinh xuyen suot, khong lan luc nga -- vd cap track "23<->24" o Canh 4), dung
(A) LAM MOC THAM CHIEU -- do ty le (B)/(C) cho ra CUNG ket qua voi (A) moi
frame. Frame lech = nghi ngo loi (dang chu y nhat o Canh 2 "dung sat nhau" va
Canh 4 "A nga, B dung gan").
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

DET_CONF = 0.2
DIST_THRESHOLD = 0.5
FRAME_SKIP = 3

SCENE_NOTE = {
    1: "dung cach 1-1.5m",
    2: "dung/ngoi SAT nhau (kho nhat, tinh huong)",
    3: "A bi che khuat 1 cam",
    4: "A NGA, B dung yen gan do (quan trong nhat)",
}


def load_H(name):
    with open(RESULTS_DIR / name, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if "H" in raw:  # homography_head_matrices.json: {"height_m":..., "H": {cam: ...}}
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
    """Chu ky de so sanh 2 tap ghep cap co GIONG NHAU khong (khong quan tam
    thu tu) -- vd {(3,7),(4,9)}."""
    return frozenset((p[0].track_id, p[1].track_id) for p in pairs)


def process_scene(scene_id, detector1, detector2, H_floor, H_head):
    folder = P2_DIR / "P2 Data 3"
    cap1 = cv2.VideoCapture(str(folder / f"Scene {scene_id}-CAM 1.mp4"))
    cap2 = cv2.VideoCapture(str(folder / f"Scene {scene_id}-CAM 2.mp4"))

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

        dets1 = detect_tracks(detector1, frame1)
        dets2 = detect_tracks(detector2, frame2)
        if not dets1 or not dets2:
            continue

        # (A) Hungarian + chan (baseline da validate)
        t1_foot = [Track("CAM 1", d["track_id"], d["bbox_xyxy"],
                          ground_xy=apply_homography(H_floor["CAM 1"], foot_point_from_bbox(d["bbox_xyxy"])))
                   for d in dets1]
        t2_foot = [Track("CAM 2", d["track_id"], d["bbox_xyxy"],
                          ground_xy=apply_homography(H_floor["CAM 2"], foot_point_from_bbox(d["bbox_xyxy"])))
                   for d in dets2]
        pairs_A = match_two_cameras(t1_foot, t2_foot, distance_threshold=DIST_THRESHOLD)

        # (B) Greedy + chan
        pairs_B = match_two_cameras_greedy(t1_foot, t2_foot, distance_threshold=DIST_THRESHOLD)

        # (C) Hungarian + dau
        t1_head = [Track("CAM 1", d["track_id"], d["bbox_xyxy"],
                          ground_xy=apply_homography(H_head["CAM 1"], head_point_from_bbox(d["bbox_xyxy"])))
                   for d in dets1]
        t2_head = [Track("CAM 2", d["track_id"], d["bbox_xyxy"],
                          ground_xy=apply_homography(H_head["CAM 2"], head_point_from_bbox(d["bbox_xyxy"])))
                   for d in dets2]
        pairs_C = match_two_cameras(t1_head, t2_head, distance_threshold=DIST_THRESHOLD)

        sig_A, sig_B, sig_C = pair_signature(pairs_A), pair_signature(pairs_B), pair_signature(pairs_C)
        rows.append({
            "scene": scene_id, "frame_idx": frame_idx,
            "n_cam1": len(dets1), "n_cam2": len(dets2),
            "pairs_A_hungarian_foot": ";".join(f"{a}<->{b}" for a, b in sig_A),
            "pairs_B_greedy_foot": ";".join(f"{a}<->{b}" for a, b in sig_B),
            "pairs_C_hungarian_head": ";".join(f"{a}<->{b}" for a, b in sig_C),
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
    for scene_id in [1, 2, 3, 4]:
        print(f"\n=== P2 Data 3 / Scene {scene_id} ({SCENE_NOTE[scene_id]}) ===")
        rows = process_scene(scene_id, detector1, detector2, H_floor, H_head)
        df = pd.DataFrame(rows)
        if len(df):
            agree_B = df["B_agrees_with_A"].mean()
            agree_C = df["C_agrees_with_A"].mean()
            print(f"  {len(df)} frame | Greedy+chan khop Hungarian+chan: {agree_B:.1%} | "
                  f"Hungarian+dau khop Hungarian+chan: {agree_C:.1%}")
        all_rows.extend(rows)

    df_all = pd.DataFrame(all_rows)
    out_csv = RESULTS_DIR / "compare_p2_alternatives.csv"
    df_all.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("TONG KET (tren toan bo P2 Data 3, moc tham chieu = Hungarian+chan da validate):")
    summary_rows = []
    for scene_id in [1, 2, 3, 4]:
        sub = df_all[df_all["scene"] == scene_id]
        if len(sub) == 0:
            continue
        row = {
            "scene": scene_id, "note": SCENE_NOTE[scene_id], "n_frames": len(sub),
            "greedy_foot_agree_pct": round(sub["B_agrees_with_A"].mean() * 100, 1),
            "hungarian_head_agree_pct": round(sub["C_agrees_with_A"].mean() * 100, 1),
        }
        summary_rows.append(row)
        print(f"  Canh {scene_id} ({SCENE_NOTE[scene_id]}): "
              f"Greedy+chan khop {row['greedy_foot_agree_pct']}% | "
              f"Hungarian+dau khop {row['hungarian_head_agree_pct']}%")
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "compare_p2_alternatives_summary.csv",
                                       index=False, encoding="utf-8-sig")
    print("=" * 70)
    print(f"\nDa luu: {out_csv}")
    print(f"Da luu: {RESULTS_DIR / 'compare_p2_alternatives_summary.csv'}")


if __name__ == "__main__":
    main()
