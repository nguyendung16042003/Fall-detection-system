"""
Chay thu pipeline MVP moi (detect+crop+classify+rule windowed) tren 1 mau nho
video MCFD (co nga) + URFD ADL (khong nga), luu ket qua ro rang vao Test 4/
de xem nhanh model + rule moi hoat dong dung huong hay chua.

Day la bai test nhanh dinh tinh (sanity check), KHONG phai benchmark day du
tren toan bo dataset (viec do se lam o buoc benchmark PIFR rieng).
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
POSE_DIR = OUT_DIR / "pose_sequences"
POSE_DIR.mkdir(exist_ok=True)

MCFD_DIR = AI_ROOT / "Datasets" / "File Test 2" / "MCFD" / "dataset"
URFD_ADL_DIR = AI_ROOT / "Datasets" / "File Test 2" / "URFD" / "Cam" / "ADL"

N_CHUTES_TOTAL = 24
N_CAMS_PER_CHUTE = 8
N_ADL_TOTAL = 40
SAMPLE_RATIO = 0.3  # random 30%, khong test het toan bo MCFD/URFD (giong PIFR)
RANDOM_SEED = int(os.environ.get("EVAL_SEED", "42"))


def pick_samples():
    random.seed(RANDOM_SEED)

    # Moi (chute, cam) la 1 video test DOC LAP -- khong nhom theo chute, khong
    # OR-fusion. Tong pool = 24 chute x 8 cam = 192 video 1-camera rieng biet.
    all_mcfd_videos = [
        MCFD_DIR / f"chute{c:02d}" / f"cam{cam}.avi"
        for c in range(1, N_CHUTES_TOTAL + 1) for cam in range(1, N_CAMS_PER_CHUTE + 1)
    ]
    all_mcfd_videos = [v for v in all_mcfd_videos if v.exists()]
    fall_videos = random.sample(all_mcfd_videos, max(1, round(len(all_mcfd_videos) * SAMPLE_RATIO)))

    all_adl = sorted(URFD_ADL_DIR.glob("*.mp4"))[:N_ADL_TOTAL]
    adl_videos = random.sample(all_adl, max(1, round(len(all_adl) * SAMPLE_RATIO)))

    print(f"Random sample {SAMPLE_RATIO:.0%} (seed={RANDOM_SEED}, moi (chute,cam) doc lap): "
          f"{len(fall_videos)}/{len(all_mcfd_videos)} video MCFD, {len(adl_videos)}/{len(all_adl)} video ADL")
    return [(v, "Fall") for v in fall_videos] + [(v, "NoFall") for v in adl_videos]


def main():
    samples = pick_samples()
    print(f"Se chay tren {len(samples)} video")

    pipeline = DetectClassifyPipeline()  # tai model 1 lan, dung lai cho tat ca video

    summary_rows = []
    all_events_rows = []

    for idx, (video_path, ground_truth) in enumerate(samples, start=1):
        t0 = time.time()
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        records = pipeline.process_video(video_path, frame_skip=5)
        events = detect_fall_events(records, fps=fps)

        video_name = video_path.name if video_path.parent.name != "cam1.avi" else video_path.name
        video_id = f"{video_path.parent.name}_{video_path.name}"  # vd: chute01_cam1.avi

        pd.DataFrame(records).to_csv(POSE_DIR / f"{video_id}.csv", index=False, encoding="utf-8-sig")

        prediction = "Fall" if events else "NoFall"
        elapsed = time.time() - t0
        print(f"[{idx}/{len(samples)}] {video_id} | ground_truth={ground_truth} | "
              f"prediction={prediction} | so_event={len(events)} | {elapsed:.1f}s")

        summary_rows.append({
            "video": video_id,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "so_su_kien_nga": len(events),
            "so_frame_xu_ly": len(records),
            "thoi_gian_xu_ly_s": round(elapsed, 1),
        })

        for e in events:
            all_events_rows.append({
                "video": video_id,
                "person_id": e["person_id"],
                "frame_id": e["frame_id"],
                "thoi_diem_ms": round(e["timestamp_ms"]),
                "tu_the_truoc": e["detection"]["class_before"],
                "do_tin_cay": round(e["detection"]["confidence"], 3),
                "kieu_chuyen_tiep": e["rule"]["trigger"],
                "thoi_gian_chuyen_tiep_ms": e["rule"]["transition_ms"],
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "tom_tat_ket_qua.csv", index=False, encoding="utf-8-sig")

    events_df = pd.DataFrame(all_events_rows)
    events_df.to_csv(OUT_DIR / "chi_tiet_su_kien_nga.csv", index=False, encoding="utf-8-sig")

    TP = ((summary_df.ground_truth == "Fall") & (summary_df.prediction == "Fall")).sum()
    FN = ((summary_df.ground_truth == "Fall") & (summary_df.prediction == "NoFall")).sum()
    TN = ((summary_df.ground_truth == "NoFall") & (summary_df.prediction == "NoFall")).sum()
    FP = ((summary_df.ground_truth == "NoFall") & (summary_df.prediction == "Fall")).sum()

    print("\n" + "=" * 50)
    print(f"TP={TP}  FN={FN}  TN={TN}  FP={FP}")
    total = TP + FN + TN + FP
    if total:
        acc = (TP + TN) / total
        recall = TP / (TP + FN) if (TP + FN) else 0
        precision = TP / (TP + FP) if (TP + FP) else 0
        print(f"Accuracy={acc:.2%}  Recall={recall:.2%}  Precision={precision:.2%}")
    print(f"\nDa luu: {OUT_DIR / 'tom_tat_ket_qua.csv'}")
    print(f"Da luu: {OUT_DIR / 'chi_tiet_su_kien_nga.csv'}")
    print(f"Pose sequence chi tiet tung video: {POSE_DIR}/")


if __name__ == "__main__":
    main()
