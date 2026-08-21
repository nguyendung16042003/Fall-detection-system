"""
Xay dataset windowed (1 giay, chong lap 0.5 giay) tu MCFD cho Espinosa
replica -- dung CO DINH cam1+cam2 (khop setup "2 camera RGB co dinh" that
cua nhom, thay vi dung ca 8 cam cua MCFD).

Nhan window: Fall neu [window_start, window_end] giao voi khoang nga (theo
mcfd_manual_fall_windows.json -- gan tay qua contact sheet, xem file do va
annotate_mcfd_falls.py de biet ly do khong dung nhan tu dong), con lai ADL.
chute19 (khong xac dinh duoc nga) bi LOAI KHOI dataset (khong dung lam ADL
lan Fall, tranh nhiem nhan) -- theo dung tinh than "khong bia du lieu".

Chia TRAIN/TEST theo CHUTE (khong theo window) de tranh leakage -- xap xi
67/33 nhu paper goc.
"""
import json
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from optical_flow import precompute_flow_magnitudes, flow_window_from_magnitudes, combine_two_cams  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
MCFD_DIR = AI_ROOT / "Datasets" / "File Test 2" / "MCFD" / "dataset"
RESULTS_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p0_espinosa"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ANNOT_PATH = Path(__file__).resolve().parent / "mcfd_manual_fall_windows.json"

WINDOW_S = 1.0
STRIDE_S = 0.5
CAM1, CAM2 = "cam1", "cam2"

# Chia theo chute -- xap xi 67/33, tron deu de ca train/test co du chute
# "confidence high/medium/low" va chute23/24 (no_fall) o ca 2 phia.
TEST_CHUTES = {2, 5, 9, 13, 17, 21, 24, 8}  # 8/23 chute ~ 35%
# (chute19 tu dong bi loai vi unconfirmed, khong can liet ke rieng)


def load_annotations():
    with open(ANNOT_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw.pop("_meta", None)
    return raw


def read_all_frames(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def window_label(win_start_s, win_end_s, ann):
    if ann.get("no_fall"):
        return 0
    if ann.get("unconfirmed"):
        return None  # loai
    fs, fe = ann["fall_start_s"], ann["fall_end_s"]
    overlap = max(0.0, min(win_end_s, fe) - max(win_start_s, fs))
    return 1 if overlap > 0 else 0


def process_chute(chute_name, ann):
    if ann.get("unconfirmed"):
        print(f"[{chute_name}] unconfirmed -- bo qua")
        return []

    frames1, fps1 = read_all_frames(MCFD_DIR / chute_name / f"{CAM1}.avi")
    frames2, fps2 = read_all_frames(MCFD_DIR / chute_name / f"{CAM2}.avi")
    fps = fps1
    n = min(len(frames1), len(frames2))
    if n < int(WINDOW_S * fps):
        print(f"[{chute_name}] video qua ngan, bo qua")
        return []

    win_frames = int(WINDOW_S * fps)
    stride_frames = int(STRIDE_S * fps)

    # Tinh flow 1 LAN cho toan bo video (khong tinh lai theo tung cua so
    # chong lap) -- xem toi uu trong optical_flow.py.
    mags1 = precompute_flow_magnitudes(frames1[:n])
    mags2 = precompute_flow_magnitudes(frames2[:n])

    samples = []
    start = 0
    n_fall = 0
    n_adl = 0
    while start + win_frames <= n:
        end = start + win_frames
        win_start_s = start / fps
        win_end_s = end / fps
        label = window_label(win_start_s, win_end_s, ann)
        if label is not None:
            flow1 = flow_window_from_magnitudes(mags1, start, end - 1)
            flow2 = flow_window_from_magnitudes(mags2, start, end - 1)
            combined = combine_two_cams(flow1, flow2)  # (2,38,51)
            samples.append((combined, label, chute_name, round(win_start_s, 2)))
            n_fall += label
            n_adl += 1 - label
        start += stride_frames

    print(f"[{chute_name}] {len(samples)} window ({n_fall} Fall, {n_adl} ADL)")
    return samples


def main():
    annotations = load_annotations()
    train_X, train_y, train_meta = [], [], []
    test_X, test_y, test_meta = [], [], []

    for i in range(1, 25):
        chute_name = f"chute{i:02d}"
        if chute_name not in annotations:
            continue
        samples = process_chute(chute_name, annotations[chute_name])
        target_X, target_y, target_meta = (
            (test_X, test_y, test_meta) if i in TEST_CHUTES else (train_X, train_y, train_meta)
        )
        for combined, label, cname, wstart in samples:
            target_X.append(combined)
            target_y.append(label)
            target_meta.append((cname, wstart, label, "test" if i in TEST_CHUTES else "train"))

    train_X = np.stack(train_X).astype(np.float32)
    train_y = np.array(train_y, dtype=np.int64)
    test_X = np.stack(test_X).astype(np.float32)
    test_y = np.array(test_y, dtype=np.int64)

    np.save(RESULTS_DIR / "espinosa_train_X.npy", train_X)
    np.save(RESULTS_DIR / "espinosa_train_y.npy", train_y)
    np.save(RESULTS_DIR / "espinosa_test_X.npy", test_X)
    np.save(RESULTS_DIR / "espinosa_test_y.npy", test_y)

    import csv
    with open(RESULTS_DIR / "espinosa_dataset_manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["chute", "window_start_s", "label", "split"])
        for row in train_meta + test_meta:
            writer.writerow(row)

    print("\n" + "=" * 50)
    print(f"Train: {len(train_y)} window ({train_y.sum()} Fall, {len(train_y)-train_y.sum()} ADL) "
          f"tu {len(set(m[0] for m in train_meta))} chute")
    print(f"Test:  {len(test_y)} window ({test_y.sum()} Fall, {len(test_y)-test_y.sum()} ADL) "
          f"tu {len(set(m[0] for m in test_meta))} chute")
    print(f"Da luu vao: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
