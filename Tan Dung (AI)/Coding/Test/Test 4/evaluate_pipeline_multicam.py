"""
Ban sua cua evaluate_pipeline.py: danh gia MCFD theo dung cach he thong da
camera duoc thiet ke -- moi CHUTE la 1 don vi test, du doan "Fall" neu BAT KY
cam nao trong 8 cam phat hien duoc (OR-fusion), thay vi chi dung cam1.

Ly do: MCFD la dataset da camera (8 cam dong bo/chute), va he thong cua nhom
von duoc thiet ke de dung nhieu camera -- chi test bang 1 cam duy nhat (cam1)
la danh gia SAI thiet ke that cua he thong, khong phai loi model.

URFD ADL van giu nguyen (chi co 1 camera/video, khong doi duoc).
"""
import os
import random
import sys
import time
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Pipeline"))
from detect_classify_pipeline import DetectClassifyPipeline  # noqa: E402
from fall_rule import detect_fall_events  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent

MCFD_DIR = AI_ROOT / "Datasets" / "File Test 2" / "MCFD" / "dataset"
URFD_ADL_DIR = AI_ROOT / "Datasets" / "File Test 2" / "URFD" / "Cam" / "ADL"

N_CHUTES_TOTAL = 24
N_ADL_TOTAL = 40
N_CAMS_PER_CHUTE = 8
SAMPLE_RATIO = 0.3  # random 30%, khop cach PIFR tu chon tap test (khong test het 100%)
# Doi qua bien moi truong EVAL_SEED=<so> de chay lai voi mau random khac, kiem
# tra do on dinh cua ket qua (vd: EVAL_SEED=7 python evaluate_pipeline_multicam.py)
RANDOM_SEED = int(os.environ.get("EVAL_SEED", "42"))


def main():
    random.seed(RANDOM_SEED)
    all_chutes = list(range(1, N_CHUTES_TOTAL + 1))
    sample_chutes = sorted(random.sample(all_chutes, max(1, round(N_CHUTES_TOTAL * SAMPLE_RATIO))))

    all_adl = sorted(URFD_ADL_DIR.glob("*.mp4"))[:N_ADL_TOTAL]
    sample_adl = random.sample(all_adl, max(1, round(len(all_adl) * SAMPLE_RATIO)))

    print(f"Random sample {SAMPLE_RATIO:.0%} (seed={RANDOM_SEED}): "
          f"{len(sample_chutes)}/{N_CHUTES_TOTAL} chute MCFD, {len(sample_adl)}/{len(all_adl)} video ADL")
    print(f"Chute duoc chon: {sample_chutes}")

    pipeline = DetectClassifyPipeline()
    summary_rows = []

    # --- MCFD: moi chute (trong mau random) test qua ca 8 cam, OR-fusion ---
    for chute_idx in sample_chutes:
        chute_name = f"chute{chute_idx:02d}"
        any_fall = False
        cams_detected = []
        t0 = time.time()
        for cam_idx in range(1, N_CAMS_PER_CHUTE + 1):
            video_path = MCFD_DIR / chute_name / f"cam{cam_idx}.avi"
            if not video_path.exists():
                continue
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.release()
            records = pipeline.process_video(video_path, frame_skip=5)
            events = detect_fall_events(records, fps=fps)
            if events:
                any_fall = True
                cams_detected.append(cam_idx)
        elapsed = time.time() - t0
        prediction = "Fall" if any_fall else "NoFall"
        print(f"[MCFD chute{chute_idx:02d}] ground_truth=Fall | "
              f"prediction={prediction} | cam_phat_hien={cams_detected} | {elapsed:.1f}s")
        summary_rows.append({
            "video": chute_name,
            "ground_truth": "Fall",
            "prediction": prediction,
            "cam_phat_hien": ",".join(str(c) for c in cams_detected),
            "thoi_gian_xu_ly_s": round(elapsed, 1),
        })

    # --- URFD ADL: mau random (trong sample_adl), chi co 1 camera ---
    for idx, video_path in enumerate(sample_adl, start=1):
        t0 = time.time()
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        records = pipeline.process_video(video_path, frame_skip=5)
        events = detect_fall_events(records, fps=fps)
        prediction = "Fall" if events else "NoFall"
        elapsed = time.time() - t0
        video_id = f"ADL_{video_path.name}"
        print(f"[ADL {idx}/{len(sample_adl)}] {video_id} | ground_truth=NoFall | "
              f"prediction={prediction} | {elapsed:.1f}s")
        summary_rows.append({
            "video": video_id,
            "ground_truth": "NoFall",
            "prediction": prediction,
            "cam_phat_hien": "",
            "thoi_gian_xu_ly_s": round(elapsed, 1),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "tom_tat_ket_qua_multicam.csv", index=False, encoding="utf-8-sig")

    TP = ((summary_df.ground_truth == "Fall") & (summary_df.prediction == "Fall")).sum()
    FN = ((summary_df.ground_truth == "Fall") & (summary_df.prediction == "NoFall")).sum()
    TN = ((summary_df.ground_truth == "NoFall") & (summary_df.prediction == "NoFall")).sum()
    FP = ((summary_df.ground_truth == "NoFall") & (summary_df.prediction == "Fall")).sum()

    print("\n" + "=" * 60)
    print(f"TP={TP}  FN={FN}  TN={TN}  FP={FP}")
    total = TP + FN + TN + FP
    if total:
        acc = (TP + TN) / total
        recall = TP / (TP + FN) if (TP + FN) else 0
        precision = TP / (TP + FP) if (TP + FP) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        print(f"Accuracy={acc:.2%}  Precision={precision:.2%}  Recall={recall:.2%}  F1={f1:.4f}")
    print(f"\nDa luu: {OUT_DIR / 'tom_tat_ket_qua_multicam.csv'}")


if __name__ == "__main__":
    main()
