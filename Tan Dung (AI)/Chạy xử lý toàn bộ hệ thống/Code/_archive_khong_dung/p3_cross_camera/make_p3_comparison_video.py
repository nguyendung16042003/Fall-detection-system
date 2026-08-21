"""
Video SO SANH cho folder 3 ("3. So sanh doi thu"): CUNG 1 canh nga that
(P3 Data 1, Scene 1-Cam 1,2) -- ghep 2 camera canh nhau, hien thi SONG SONG:
  - P3 (bai cua nhom): bbox + nhan tu the fused (dung pipeline that, giong
    make_p3_report_video.py).
  - Espinosa (doi thu): xac suat Fall theo tung cua so 1s/0.5s overlap (dung
    dung model + tien xu ly cua evaluate_p2p3.py), hien thi dang chu o duoi
    (khong co bbox vi phuong phap goc la optical-flow toan khung hinh, khong
    detect nguoi -- ghi ro trong Word, khong ve bbox gia cho doi thu).

Scene 1 la vi du RO NHAT cho so sanh: theo espinosa_eval_p2p3.csv, P3 nhan
dung "lie" 14/18 frame gop duoc, con Espinosa DU DOAN SAI ca canh nay (pred=0,
max_fall_prob chi 0.1544 -- khong vuot nguong) dan chung truc tiep cho ket qua
tong (P3 24/24 vs Espinosa 7/24 Sensitivity).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p2_homography"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p0_espinosa_baseline"))

from shared_backbone import SharedBackbone, preprocess_crop  # noqa: E402
from cross_camera_fuse import build_distance_matrices, sgie_forward  # noqa: E402
from homography import apply_homography, foot_point_from_bbox  # noqa: E402
from espinosa_model import EspinosaCNN  # noqa: E402
from optical_flow import precompute_flow_magnitudes, flow_window_from_magnitudes, combine_two_cams  # noqa: E402
from p3_fall_rule_adapter import detect_fall_events_fused  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
P3_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "File Run Problem 3" / "P3"
RESULTS_P3 = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p3"
RESULTS_ESP = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p0_espinosa"
CKPT_P3 = (AI_ROOT / "Coding" / "training" / "training 3" / "runs" / "classify"
           / "Fall_Detection_Advanced_Loss" / "YOLOv8n_AFCL_Balanced-8" / "weights" / "best.pt")
CKPT_ESP = RESULTS_ESP / "espinosa_best.pt"
CLASS_NAMES = ["bend", "exercise", "lie", "sit", "stand"]
DET_CONF = 0.2
DIST_THRESHOLD = 1.5
SIGMA = 2.0
TILE_H = 540
ESP_WINDOW_S = 1.0
ESP_STRIDE_S = 0.5
ESP_FALL_THRESHOLD = 0.5

# Khop dung make_p3_report_video.py: YOLOv8n COCO-pretrained bo lo nguoi nam det
# rat thuong xuyen tren Cam 2 -- da xac nhan qua debug (18/150 -> chi 12%). Ap
# dung lai dung co che tracking + giu tam bbox (~1.2s, giong het GRACE_FRAMES
# cua detect_classify_pipeline.py) de nhat quan voi video demo P3 va so lieu
# benchmark, khong con dung detect tung-frame-doc-lap nua.
GRACE_MS = 1200

DATA_NAME = "P3 Data 1"
SCENE_ID = 1
CAM1_VIDEO = P3_DIR / DATA_NAME / f"Scene {SCENE_ID}-Cam 1.mp4"
CAM2_VIDEO = P3_DIR / DATA_NAME / f"Scene {SCENE_ID}-Cam 2.mp4"
OUT_DIR = (AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Số liệu thống kê"
           / "P3" / "3. So sánh đối thủ")


def load_h_by_height():
    with open(RESULTS_P3 / "h_by_height_p3.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {cam: {float(h): np.array(H) for h, H in d.items()} for cam, d in raw.items()}


def load_h_floor():
    with open(RESULTS_P3 / "homography_matrices_p3.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {cam: np.array(H) for cam, H in raw.items()}


def detect_one_tracked(detector, frame, held_state, grace_frames, conf=DET_CONF):
    """Nhu detect_classify_pipeline.py: .track() de co ID on dinh; neu mat dau
    dot ngot thi giu tam bbox cu (ap len FRAME HIEN TAI) toi da grace_frames
    lien tiep truoc khi coi la that su mat nguoi."""
    results = detector.track(frame, classes=[0], conf=conf, persist=True, verbose=False)[0]
    if results.boxes is not None and len(results.boxes) > 0:
        box = results.boxes[0]
        bbox = tuple(box.xyxy[0].cpu().numpy().tolist())
        held_state["bbox"] = bbox
        held_state["missed"] = 0
        return bbox, False
    if held_state["bbox"] is not None and held_state["missed"] < grace_frames:
        held_state["missed"] += 1
        return held_state["bbox"], True
    held_state["bbox"] = None
    return None, False


def analyze_p3(backbone, detector1, detector2, H_floor, H_by_height, frames1, frames2, fps):
    grace_frames = max(1, round(GRACE_MS / 1000.0 * fps))
    held1 = {"bbox": None, "missed": 0}
    held2 = {"bbox": None, "missed": 0}
    n_held1 = n_held2 = 0
    per_frame = []
    for frame_idx, (frame1, frame2) in enumerate(zip(frames1, frames2), start=1):
        bbox1, is_held1 = detect_one_tracked(detector1, frame1, held1, grace_frames)
        bbox2, is_held2 = detect_one_tracked(detector2, frame2, held2, grace_frames)
        n_held1 += is_held1
        n_held2 += is_held2
        rec = {"frame_idx": frame_idx, "bbox1": bbox1, "bbox2": bbox2,
               "held1": is_held1, "held2": is_held2,
               "label": None, "conf": None, "fused": False}
        if bbox1 is not None and bbox2 is not None:
            foot1 = apply_homography(H_floor["P3 Cam1"], foot_point_from_bbox(bbox1))
            foot2 = apply_homography(H_floor["P3 Cam2"], foot_point_from_bbox(bbox2))
            dist = np.linalg.norm(np.array(foot1) - np.array(foot2))
            crop1 = frame1[int(bbox1[1]):int(bbox1[3]), int(bbox1[0]):int(bbox1[2])]
            crop2 = frame2[int(bbox2[1]):int(bbox2[3]), int(bbox2[0]):int(bbox2[2])]
            if dist <= DIST_THRESHOLD and crop1.size > 0 and crop2.size > 0:
                x1 = preprocess_crop(crop1)
                x2 = preprocess_crop(crop2)
                with torch.no_grad():
                    grid1 = backbone(x1)
                    h, w = grid1.shape[2], grid1.shape[3]
                    dist_1to2, dist_2to1 = build_distance_matrices(
                        (h, w), foot1, foot2, H_by_height["P3 Cam1"], H_by_height["P3 Cam2"],
                        bbox1, bbox2)
                    v_fused = sgie_forward(backbone, x1, x2, has_both_cams=True,
                                            dist_1to2=dist_1to2, dist_2to1=dist_2to1, sigma=SIGMA)
                    logits_fused = backbone.classify_from_vector(v_fused)
                    probs_fused = torch.softmax(logits_fused, dim=1)[0]
                rec["label"] = CLASS_NAMES[int(probs_fused.argmax())]
                rec["conf"] = float(probs_fused.max())
                rec["fused"] = True
        per_frame.append(rec)
    print(f"  grace_frames={grace_frames} (~{GRACE_MS}ms @ {fps:.1f}fps) "
          f"-- Cam1 held {n_held1} frame, Cam2 held {n_held2} frame")
    return per_frame


def analyze_espinosa(model, frames1, frames2, fps):
    n = min(len(frames1), len(frames2))
    win_frames = int(ESP_WINDOW_S * fps)
    stride_frames = max(1, int(ESP_STRIDE_S * fps))
    mags1 = precompute_flow_magnitudes(frames1[:n])
    mags2 = precompute_flow_magnitudes(frames2[:n])

    # Xac suat Fall gan cho TUNG FRAME: frame idx thuoc cua so [start,end) nao
    # thi lay prob cua so do; frame chua thuoc cua so nao -> lay prob cua so
    # gan nhat da tinh (giu nguyen hien thi, khong dat 0 gay hieu nham).
    frame_prob = np.zeros(n, dtype=np.float32)
    frame_prob[:] = np.nan
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
    # dien cac frame cuoi chua duoc cua so nao phu (do stride) bang gia tri gan nhat
    for i in range(n):
        if np.isnan(frame_prob[i]):
            frame_prob[i] = last_prob if i > 0 else 0.0
    return frame_prob


def render_tile(frame, bbox, label, conf, fused, cam_label, held=False):
    frame = frame.copy()
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        color = (46, 160, 46) if fused else (180, 180, 60)
        thickness = 4 if held else 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        if label is not None:
            text = f"{label} ({conf:.2f})" + (" [fused]" if fused else "") + (" (held)" if held else "")
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 12)), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    h, w = frame.shape[:2]
    scale = TILE_H / h
    frame = cv2.resize(frame, (int(w * scale), TILE_H))
    cv2.putText(frame, cam_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return frame


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Nap P3 (SharedBackbone + 2 detector + calib that) va Espinosa...")
    backbone = SharedBackbone(CKPT_P3).eval()
    detector1 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    detector2 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    H_by_height = load_h_by_height()
    H_floor = load_h_floor()
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
    n = min(len(frames1), len(frames2))
    frames1, frames2 = frames1[:n], frames2[:n]

    print("Phan tich bang P3 (fused pose)...")
    p3_per_frame = analyze_p3(backbone, detector1, detector2, H_floor, H_by_height, frames1, frames2, fps)
    n_fused = sum(1 for r in p3_per_frame if r["fused"])
    n_lie = sum(1 for r in p3_per_frame if r["fused"] and r["label"] == "lie")
    print(f"  P3: {n_fused} frame gop duoc, {n_lie}/{n_fused} nhan dung 'lie'")

    p3_events = detect_fall_events_fused(p3_per_frame, fps)
    print(f"  P3 RULE THAT: {len(p3_events)} su kien nga:")
    for e in p3_events:
        print(f"    -> t={e['timestamp_ms']/1000:.2f}s trigger={e['rule']['trigger']}")
    p3_event_windows = [(e["timestamp_ms"], e["timestamp_ms"] + 1500) for e in p3_events]

    print("Phan tich bang Espinosa (optical-flow theo cua so)...")
    esp_frame_prob = analyze_espinosa(esp_model, frames1, frames2, fps)
    esp_video_pred = "Fall" if esp_frame_prob.max() >= ESP_FALL_THRESHOLD else "ADL (bo sot)"
    print(f"  Espinosa: max_fall_prob={esp_frame_prob.max():.4f} -> ket luan ca canh: {esp_video_pred}")

    tile_w_sample = int(TILE_H * frames1[0].shape[1] / frames1[0].shape[0])
    out_w = tile_w_sample * 2 + 10
    banner_h = 70
    out_h = TILE_H + banner_h

    out_path = OUT_DIR / f"SoSanh_video_Scene{SCENE_ID}_Cam1_Cam2_P3_vs_Espinosa.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
    assert writer.isOpened(), f"VideoWriter khong mo duoc: {out_path}"

    for i in range(n):
        rec = p3_per_frame[i]
        tile1 = render_tile(frames1[i], rec["bbox1"], rec["label"], rec["conf"], rec["fused"], "Cam 1", held=rec["held1"])
        tile2 = render_tile(frames2[i], rec["bbox2"], rec["label"], rec["conf"], rec["fused"], "Cam 2", held=rec["held2"])
        if tile1.shape[1] != tile_w_sample:
            tile1 = cv2.resize(tile1, (tile_w_sample, TILE_H))
        if tile2.shape[1] != tile_w_sample:
            tile2 = cv2.resize(tile2, (tile_w_sample, TILE_H))
        sep = np.full((TILE_H, 10, 3), 255, dtype=np.uint8)
        top = np.hstack([tile1, sep, tile2])

        banner = np.full((banner_h, out_w, 3), 20, dtype=np.uint8)
        t_ms_p3 = rec["frame_idx"] / fps * 1000.0
        p3_fall_active = any(lo <= t_ms_p3 <= hi for lo, hi in p3_event_windows)
        if p3_fall_active:
            p3_text = "P3 (nhom): !!! FALL DETECTED !!! (rule that)"
            p3_color = (0, 0, 255)
        elif rec["fused"]:
            p3_text = f"P3 (nhom): {rec['label']} (fused)"
            p3_color = (80, 220, 80) if rec["label"] == "lie" else (220, 220, 220)
        else:
            p3_text = "P3 (nhom): chua gop du 2 cam"
            p3_color = (220, 220, 220)
        cv2.putText(banner, p3_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, p3_color, 2)

        prob = float(esp_frame_prob[i])
        esp_state = "Fall" if prob >= ESP_FALL_THRESHOLD else "ADL"
        esp_color = (60, 60, 230) if esp_state == "ADL" else (80, 220, 80)
        esp_text = f"Espinosa (doi thu): p(Fall)={prob:.2f} -> {esp_state}"
        cv2.putText(banner, esp_text, (15, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, esp_color, 2)

        combined = np.vstack([top, banner])
        writer.write(combined)
    writer.release()
    assert out_path.exists() and out_path.stat().st_size > 0, f"Video khong duoc ghi ra: {out_path}"
    print(f"\nDa luu: {out_path}")


if __name__ == "__main__":
    main()
