"""
Identity Association -- video kiem tra RULE THAT chay tren dung chuoi xu ly:
Hungarian that (p2_fall_rule_adapter.detect_tracks/match_two_cameras) -> gop
Boundary Feature Fusion khi ghep duoc ca 2 cam (co_both_cams=True) hoac phan
loai don-cam khi chi 1 cam thay (giong nhanh du phong da dung trong Boundary
Feature Fusion that) -> RULE THAT (Coding/Pipeline/fall_rule.py, qua global
person_id on dinh xuyen 2 camera).

2 canh muc tieu:
  - Data 2 Scene 5: nga don (1 nguoi) -- ky vong dung 1 event.
  - Data 3 Scene 4: A NGA, B dung yen gan do (2 nguoi) -- BAI TEST QUAN TRONG
    NHAT: phai bao dung cho A, TUYET DOI KHONG bao nham sang B.
"""
import sys
from pathlib import Path

import cv2
import json
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p3_cross_camera"))

from shared_backbone import SharedBackbone  # noqa: E402
from p2_fall_rule_adapter import (  # noqa: E402
    GlobalIdAssigner, process_frame_pair, detect_fall_events_ia,
)

AI_ROOT = Path(__file__).resolve().parents[3]
P2_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "File Run Problem 2" / "P2 data test"
RESULTS_P2 = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p2"
CKPT = (AI_ROOT / "Coding" / "training" / "training 3" / "runs" / "classify"
        / "Fall_Detection_Advanced_Loss" / "YOLOv8n_AFCL_Balanced-8" / "weights" / "best.pt")
TILE_H = 540
FALL_BANNER_MS = 1500
SIGMA = 2.0
DET_CONF = 0.4
DIST_THRESHOLD = 0.5

SCENES = [
    ("Data 2 Scene 5 (nga don)", P2_DIR / "P2 Data 2" / "Scene 5-CAM 1.mp4",
     P2_DIR / "P2 Data 2" / "Scene 5-CAM 2.mp4", "IA_Data2_Scene5_rule_ghep.mp4"),
    ("Data 3 Scene 4 (A nga, B dung gan)", P2_DIR / "P2 Data 3" / "Scene 4-CAM 1.mp4",
     P2_DIR / "P2 Data 3" / "Scene 4-CAM 2.mp4", "IA_Data3_Scene4_rule_ghep.mp4"),
]
OUT_DIR = (AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Số liệu thống kê"
           / "Identity Association" / "4. Posture + Rule check")


def load_H():
    with open(RESULTS_P2 / "homography_matrices.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {cam: np.array(H) for cam, H in raw.items()}


def load_h_by_height():
    with open(RESULTS_P2 / "h_by_height.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {cam: {float(h): np.array(H) for h, H in d.items()} for cam, d in raw.items()}


def render_tile(frame, items, cam_key, cam_label, fall_ids):
    """items: list frame_out entries co bbox cho cam_key ('bbox1' hoac 'bbox2')."""
    frame = frame.copy()
    bbox_field = "bbox1" if cam_key == "CAM 1" else "bbox2"
    for item in items:
        bbox = item[bbox_field]
        if bbox is None:
            continue
        gid = item["global_id"]
        is_fall = gid in fall_ids
        color = (0, 0, 255) if is_fall else ((46, 160, 46) if item["fused"] else (180, 180, 60))
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        text = f"ID{gid} {item['label']} ({item['conf']:.2f})" + ("" if item["fused"] else " [solo]")
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), color, -1)
        cv2.putText(frame, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    h, w = frame.shape[:2]
    scale = TILE_H / h
    frame = cv2.resize(frame, (int(w * scale), TILE_H))
    cv2.putText(frame, cam_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    if any(gid in fall_ids for item in items for gid in [item["global_id"]]):
        cv2.putText(frame, "!!! FALL DETECTED !!!", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 4)
    return frame


def run_scene(scene_name, cam1_video, cam2_video, out_name, backbone, detector1, detector2, H_floor, H_by_height):
    print(f"\n=== {scene_name} ===")
    cap1 = cv2.VideoCapture(str(cam1_video))
    cap2 = cv2.VideoCapture(str(cam2_video))
    fps = cap1.get(cv2.CAP_PROP_FPS)

    assigner = GlobalIdAssigner()
    all_frame_out = []
    frame_idx = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        frame_idx += 1
        frame_out = process_frame_pair(frame1, frame2, detector1, detector2, backbone,
                                        H_floor, H_by_height, assigner, sigma=SIGMA,
                                        det_conf=DET_CONF, dist_threshold=DIST_THRESHOLD)
        all_frame_out.append((frame_idx, frame_out))
    cap1.release()
    cap2.release()

    n_fused = sum(1 for _, fo in all_frame_out for item in fo if item["fused"])
    n_solo = sum(1 for _, fo in all_frame_out for item in fo if not item["fused"])
    all_gids = sorted(set(item["global_id"] for _, fo in all_frame_out for item in fo))
    print(f"  {frame_idx} frame, {n_fused} luot fused, {n_solo} luot solo, global_id xuat hien: {all_gids}")

    events = detect_fall_events_ia(all_frame_out, fps)
    print(f"  RULE THAT: {len(events)} su kien nga")
    for e in events:
        print(f"    -> t={e['timestamp_ms']/1000:.2f}s global_id={e['person_id']} trigger={e['rule']['trigger']}")
    fall_ids_by_time = [(e["person_id"], e["timestamp_ms"], e["timestamp_ms"] + FALL_BANNER_MS) for e in events]

    # ---------- ghi video ----------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap1 = cv2.VideoCapture(str(cam1_video))
    ret, f0 = cap1.read()
    cap1.release()
    tile_w_sample = int(TILE_H * f0.shape[1] / f0.shape[0])
    out_w = tile_w_sample * 2 + 10
    out_h = TILE_H
    out_path = OUT_DIR / out_name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
    assert writer.isOpened(), f"VideoWriter khong mo duoc: {out_path}"

    cap1 = cv2.VideoCapture(str(cam1_video))
    cap2 = cv2.VideoCapture(str(cam2_video))
    for frame_idx, frame_out in all_frame_out:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        t_ms = frame_idx / fps * 1000.0
        fall_ids = {gid for gid, lo, hi in fall_ids_by_time if lo <= t_ms <= hi}
        tile1 = render_tile(frame1, frame_out, "CAM 1", "CAM 1", fall_ids)
        tile2 = render_tile(frame2, frame_out, "CAM 2", "CAM 2", fall_ids)
        if tile1.shape[1] != tile_w_sample:
            tile1 = cv2.resize(tile1, (tile_w_sample, TILE_H))
        if tile2.shape[1] != tile_w_sample:
            tile2 = cv2.resize(tile2, (tile_w_sample, TILE_H))
        sep = np.full((TILE_H, 10, 3), 255, dtype=np.uint8)
        combined = np.hstack([tile1, sep, tile2])
        writer.write(combined)
    cap1.release()
    cap2.release()
    writer.release()
    assert out_path.exists() and out_path.stat().st_size > 0, f"Video khong duoc ghi ra: {out_path}"
    print(f"  Da luu: {out_path}")
    return events


def main():
    print("Nap SharedBackbone + 2 detector + calib Identity Association that...")
    backbone = SharedBackbone(CKPT).eval()
    H_floor = load_H()
    H_by_height = load_h_by_height()

    for scene_name, cam1_video, cam2_video, out_name in SCENES:
        detector1 = YOLO(str(AI_ROOT / "yolov8n.pt"))
        detector2 = YOLO(str(AI_ROOT / "yolov8n.pt"))
        run_scene(scene_name, cam1_video, cam2_video, out_name, backbone, detector1, detector2, H_floor, H_by_height)


if __name__ == "__main__":
    main()
