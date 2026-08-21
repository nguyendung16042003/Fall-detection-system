"""
Danh gia Espinosa replica (train tren MCFD) tren CHINH bo test P2/P3 nhom da
quay -- dung theo dinh huong trong plan ("chay tren cung bo test P2/P3 da
quay" de so sanh cong bang voi P3).

Nhan muc-canh (scene-level) lay THANG tu kich ban quay (Kich ban P2.docx/
Kich Ban P3.docx), KHONG phai doan:
  - Fall: P2 Data 2 Canh 5, P2 Data 3 Canh 4, P3 Data 1 (ca 4 canh -- "nga
    giua 2 cam", toan bo la du lieu nga).
  - ADL: P2 Data 2 Canh 1-4, P2 Data 3 Canh 1-3, P3 Data 2 (Canh 1 "cui nhat
    do", Canh 2 "ngoi xuong san" -- deu KHONG phai nga).

Voi moi canh: truot cua so 1s/0.5s overlap qua toan bo video, du doan tung
window, ket luan canh = Fall neu CO IT NHAT 1 window du doan Fall (logic OR
theo thoi gian -- giong cach evaluate_pipeline_multicam.py trong du an dang
dung cho toan he thong, de so sanh cung logic).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from espinosa_model import EspinosaCNN  # noqa: E402
from optical_flow import precompute_flow_magnitudes, flow_window_from_magnitudes, combine_two_cams  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p0_espinosa"
CKPT = RESULTS_DIR / "espinosa_best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WINDOW_S = 1.0
STRIDE_S = 0.5

P2_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "File Run Problem 2" / "P2 data test"
P3_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "File Run Problem 3" / "P3"

# (video1, video2, label, scene_id) -- label: 1=Fall, 0=ADL, tu kich ban quay
SCENES = [
    (P2_DIR / "P2 Data 2" / "Scene 1-CAM 1.mp4", P2_DIR / "P2 Data 2" / "Scene 1-CAM 2.mp4", 0, "P2D2_Scene1"),
    (P2_DIR / "P2 Data 2" / "Scene 2-CAM 1.mp4", P2_DIR / "P2 Data 2" / "Scene 2-CAM 2.mp4", 0, "P2D2_Scene2"),
    (P2_DIR / "P2 Data 2" / "Scene 3-CAM 1.mp4", P2_DIR / "P2 Data 2" / "Scene 3-CAM 2.mp4", 0, "P2D2_Scene3"),
    (P2_DIR / "P2 Data 2" / "Scene 4-CAM 1.mp4", P2_DIR / "P2 Data 2" / "Scene 4-CAM 2.mp4", 0, "P2D2_Scene4"),
    (P2_DIR / "P2 Data 2" / "Scene 5-CAM 1.mp4", P2_DIR / "P2 Data 2" / "Scene 5-CAM 2.mp4", 1, "P2D2_Scene5_FALL"),
    (P2_DIR / "P2 Data 3" / "Scene 1-CAM 1.mp4", P2_DIR / "P2 Data 3" / "Scene 1-CAM 2.mp4", 0, "P2D3_Scene1"),
    (P2_DIR / "P2 Data 3" / "Scene 2-CAM 1.mp4", P2_DIR / "P2 Data 3" / "Scene 2-CAM 2.mp4", 0, "P2D3_Scene2"),
    (P2_DIR / "P2 Data 3" / "Scene 3-CAM 1.mp4", P2_DIR / "P2 Data 3" / "Scene 3-CAM 2.mp4", 0, "P2D3_Scene3"),
    (P2_DIR / "P2 Data 3" / "Scene 4-CAM 1.mp4", P2_DIR / "P2 Data 3" / "Scene 4-CAM 2.mp4", 1, "P2D3_Scene4_FALL"),
    (P3_DIR / "P3 Data 1" / "Scene 1-Cam 1.mp4", P3_DIR / "P3 Data 1" / "Scene 1-Cam 2.mp4", 1, "P3D1_Scene1_FALL"),
    (P3_DIR / "P3 Data 1" / "Scene 2-Cam 1.mp4", P3_DIR / "P3 Data 1" / "Scene 2-Cam 2.mp4", 1, "P3D1_Scene2_FALL"),
    (P3_DIR / "P3 Data 1" / "Scene 3-Cam 1.mp4", P3_DIR / "P3 Data 1" / "Scene 3-Cam 2.mp4", 1, "P3D1_Scene3_FALL"),
    (P3_DIR / "P3 Data 1" / "Scene 4-Cam 1.mp4", P3_DIR / "P3 Data 1" / "Scene 4-Cam 2.mp4", 1, "P3D1_Scene4_FALL"),
    (P3_DIR / "P3 Data 2" / "Scene 1-Cam 1.mp4", P3_DIR / "P3 Data 2" / "Scene 1-Cam 2.mp4", 0, "P3D2_Scene1"),
    (P3_DIR / "P3 Data 2" / "Scene 2-Cam 1.mp4", P3_DIR / "P3 Data 2" / "Scene 2-Cam 2.mp4", 0, "P3D2_Scene2"),
]


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


def evaluate_scene(model, video1, video2):
    frames1, fps1 = read_all_frames(video1)
    frames2, fps2 = read_all_frames(video2)
    fps = fps1
    n = min(len(frames1), len(frames2))
    win_frames = int(WINDOW_S * fps)
    stride_frames = max(1, int(STRIDE_S * fps))

    mags1 = precompute_flow_magnitudes(frames1[:n])
    mags2 = precompute_flow_magnitudes(frames2[:n])

    window_preds = []
    start = 0
    while start + win_frames <= n:
        end = start + win_frames
        flow1 = flow_window_from_magnitudes(mags1, start, end - 1)
        flow2 = flow_window_from_magnitudes(mags2, start, end - 1)
        combined = combine_two_cams(flow1, flow2)
        x = torch.from_numpy(combined).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(x)
            pred = int(logits.argmax(dim=1).item())
            prob_fall = float(torch.softmax(logits, dim=1)[0, 1].item())
        window_preds.append((round(start / fps, 2), pred, prob_fall))
        start += stride_frames

    return window_preds


def main():
    model = EspinosaCNN().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()

    rows = []
    for video1, video2, label, scene_name in SCENES:
        if not video1.exists() or not video2.exists():
            print(f"[{scene_name}] thieu file, bo qua")
            continue
        window_preds = evaluate_scene(model, video1, video2)
        n_fall_windows = sum(1 for _, p, _ in window_preds if p == 1)
        video_pred = 1 if n_fall_windows > 0 else 0
        max_prob = max((p for _, _, p in window_preds), default=0.0)
        correct = "DUNG" if video_pred == label else "SAI"
        print(f"[{scene_name}] label={label} pred={video_pred} "
              f"({n_fall_windows}/{len(window_preds)} window Fall, max_prob={max_prob:.3f}) -- {correct}")
        rows.append({
            "scene": scene_name, "label": label, "pred": video_pred,
            "n_windows": len(window_preds), "n_fall_windows": n_fall_windows,
            "max_fall_prob": round(max_prob, 4), "correct": video_pred == label,
        })

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "espinosa_eval_p2p3.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    tp = ((df["pred"] == 1) & (df["label"] == 1)).sum()
    fn = ((df["pred"] == 0) & (df["label"] == 1)).sum()
    tn = ((df["pred"] == 0) & (df["label"] == 0)).sum()
    fp = ((df["pred"] == 1) & (df["label"] == 0)).sum()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    acc = (tp + tn) / len(df)

    print("\n" + "=" * 60)
    print(f"So canh: {len(df)} (Fall={df['label'].sum()}, ADL={len(df)-df['label'].sum()})")
    print(f"TP={tp} FN={fn} TN={tn} FP={fp}")
    print(f"Sensitivity={sens:.4f} Specificity={spec:.4f} Accuracy={acc:.4f}")
    print("=" * 60)
    print(f"Da luu: {out_csv}")


if __name__ == "__main__":
    main()
