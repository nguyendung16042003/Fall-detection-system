"""
Video minh hoa report cho folder 2 ("2. Doi thu (Espinosa 2019)"): ghep 2
camera canh nhau (P3 Data 1, Scene 2-Cam 1,2 -- canh Espinosa DU DOAN DUNG
theo espinosa_eval_p2p3.csv, pred=1=Fall dung voi label that), hien thi xac
suat Fall theo tung cua so 1s/0.5s overlap (dung dung tien xu ly + model cua
evaluate_p2p3.py). KHONG ve bbox: phuong phap goc cua Espinosa la optical-flow
toan khung hinh (early-fusion truoc CNN), khong co buoc detect nguoi/bbox --
ve bbox gia se sai voi ban chat phuong phap, giai thich ro trong Word.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from espinosa_model import EspinosaCNN  # noqa: E402
from optical_flow import precompute_flow_magnitudes, flow_window_from_magnitudes, combine_two_cams  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
P3_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "File Run Problem 3" / "P3"
RESULTS_ESP = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p0_espinosa"
CKPT_ESP = RESULTS_ESP / "espinosa_best.pt"
WINDOW_S = 1.0
STRIDE_S = 0.5
FALL_THRESHOLD = 0.5
TILE_H = 540
BANNER_H = 50

DATA_NAME = "P3 Data 1"
SCENE_ID = 2
CAM1_VIDEO = P3_DIR / DATA_NAME / f"Scene {SCENE_ID}-Cam 1.mp4"
CAM2_VIDEO = P3_DIR / DATA_NAME / f"Scene {SCENE_ID}-Cam 2.mp4"
OUT_DIR = (AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Số liệu thống kê"
           / "P3" / "2. Doi thu (Espinosa 2019)")


def analyze_espinosa(model, frames1, frames2, fps):
    n = min(len(frames1), len(frames2))
    win_frames = int(WINDOW_S * fps)
    stride_frames = max(1, int(STRIDE_S * fps))
    mags1 = precompute_flow_magnitudes(frames1[:n])
    mags2 = precompute_flow_magnitudes(frames2[:n])

    frame_prob = np.full(n, np.nan, dtype=np.float32)
    start = 0
    last_prob = 0.0
    while start + win_frames <= n:
        end = start + win_frames
        flow1 = flow_window_from_magnitudes(mags1, start, end - 1)
        flow2 = flow_window_from_magnitudes(mags2, start, end - 1)
        combined = combine_two_cams(flow1, flow2)
        x = torch.from_numpy(combined).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            prob_fall = float(torch.softmax(logits, dim=1)[0, 1].item())
        frame_prob[start:end] = prob_fall
        last_prob = prob_fall
        start += stride_frames
    for i in range(n):
        if np.isnan(frame_prob[i]):
            frame_prob[i] = last_prob if i > 0 else 0.0
    return frame_prob, n


def render_tile(frame, cam_label):
    h, w = frame.shape[:2]
    scale = TILE_H / h
    frame = cv2.resize(frame, (int(w * scale), TILE_H))
    cv2.putText(frame, cam_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return frame


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Nap model Espinosa...")
    esp_model = EspinosaCNN()
    esp_model.load_state_dict(torch.load(CKPT_ESP, map_location="cpu"))
    esp_model.eval()

    print(f"Doc {DATA_NAME} / Scene {SCENE_ID} (Cam 1 + Cam 2)...")
    cap1 = cv2.VideoCapture(str(CAM1_VIDEO))
    cap2 = cv2.VideoCapture(str(CAM2_VIDEO))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    frames1, frames2 = [], []
    while True:
        ret1, f1 = cap1.read()
        ret2, f2 = cap2.read()
        if not ret1 or not ret2:
            break
        frames1.append(f1)
        frames2.append(f2)
    cap1.release()
    cap2.release()

    print("Phan tich bang Espinosa (optical-flow theo cua so)...")
    frame_prob, n = analyze_espinosa(esp_model, frames1, frames2, fps)
    video_pred = "Fall" if frame_prob.max() >= FALL_THRESHOLD else "ADL"
    print(f"  max_fall_prob={frame_prob.max():.4f} -> ket luan ca canh: {video_pred} "
          f"(nhan that: Fall -- {'DUNG' if video_pred == 'Fall' else 'SAI'})")

    tile_w_sample = int(TILE_H * frames1[0].shape[1] / frames1[0].shape[0])
    out_w = tile_w_sample * 2 + 10
    out_h = TILE_H + BANNER_H

    out_path = OUT_DIR / f"Espinosa_video_Scene{SCENE_ID}_Cam1_Cam2_ghep.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
    assert writer.isOpened(), f"VideoWriter khong mo duoc: {out_path}"

    for i in range(n):
        tile1 = render_tile(frames1[i], "Cam 1")
        tile2 = render_tile(frames2[i], "Cam 2")
        if tile1.shape[1] != tile_w_sample:
            tile1 = cv2.resize(tile1, (tile_w_sample, TILE_H))
        if tile2.shape[1] != tile_w_sample:
            tile2 = cv2.resize(tile2, (tile_w_sample, TILE_H))
        sep = np.full((TILE_H, 10, 3), 255, dtype=np.uint8)
        top = np.hstack([tile1, sep, tile2])

        banner = np.full((BANNER_H, out_w, 3), 20, dtype=np.uint8)
        prob = float(frame_prob[i])
        state = "Fall" if prob >= FALL_THRESHOLD else "ADL"
        color = (80, 220, 80) if state == "Fall" else (60, 60, 230)
        text = f"Espinosa (optical-flow 2 cam, gop truoc CNN): p(Fall)={prob:.2f} -> {state}"
        cv2.putText(banner, text, (15, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        combined = np.vstack([top, banner])
        writer.write(combined)
    writer.release()
    assert out_path.exists() and out_path.stat().st_size > 0, f"Video khong duoc ghi ra: {out_path}"
    print(f"\nDa luu: {out_path}")


if __name__ == "__main__":
    main()
