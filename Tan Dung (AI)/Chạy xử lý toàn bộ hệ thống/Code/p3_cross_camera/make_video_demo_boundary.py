"""Video demo 2/3 -- Boundary Feature Fusion + Identity Association, dung
DUNG giai phap CHINH THUC cua nhom (p2_fall_rule_adapter.py, KHONG sua) --
Hungarian that (match_two_cameras) + gop dac trung khi ghep duoc ca 2 cam +
RULE THAT (fall_rule.detect_fall_events) qua global_id on dinh xuyen 2 cam.

Du lieu: "Video demo/Boundary/Fall detect/" -- B dung yen o Cam 2, A NGA o
vung giua 2 camera. Muc dich: chung minh KHONG loan ID du co B dung gan --
chi global_id cua A duoc bao nga.

Khac make_p2_posture_rule_video.py (P2 Data 2/3): tren bbox CHI hien "ID{gid}"
(KHONG hien tu the/label) -- tu the/rule van chay NGAM ben trong nhu binh
thuong, chi dung de quyet dinh khi nao hien banner FALL DETECTED. Dung calib
homography MOI (Results/video_demo/, tu calibrate_video_demo_boundary.py),
KHONG dung calib P2/P3 cu.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p2_homography"))

from shared_backbone import SharedBackbone  # noqa: E402
from p2_fall_rule_adapter import (  # noqa: E402
    GlobalIdAssigner, process_frame_pair, detect_fall_events_ia,
)

AI_ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Video demo" / "Boundary" / "Fall detect"
SCENES = [
    # Nguoi dung chon giu lai DUY NHAT Scene 2 cho video cuoi cung (Scene 1/3 bo).
    ("Scene 2", DEMO_DIR / "Cam 1- Scene 2.avi", DEMO_DIR / "Cam 2- Scene 2.avi"),
]
RESULTS_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "video_demo"
CKPT = (AI_ROOT / "Coding" / "training" / "training 3" / "runs" / "classify"
        / "Fall_Detection_Advanced_Loss" / "YOLOv8n_AFCL_Balanced-8" / "weights" / "best.pt")
OUT_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Video demo" / "Video cuối cùng"
OUT_NAME = "2_Boundary_khong_loan_ID.mp4"

TILE_H = 540
FALL_BANNER_MS = 1500
SIGMA = 2.0
DET_CONF = 0.4
DIST_THRESHOLD = 0.5
MAX_PEOPLE = None  # DA THU gioi han 2 nguoi + gan lai theo khoang cach gan nhat --
# GAY LOI NANG HON: track YOLO cua nguoi dang nam (kho detect) hay bi dut gay
# thanh nhieu track_id ngan, moi lan gan lai co the doan NHAM sang nguoi kia
# (B bi bao nga 53-61 lan o Scene 1/2 khi bat cai nay). TAT han, chi dung
# grace_frames (giu tam bbox, ngan CAM mat dau ngay tu dau) de xu ly goc re.
GRACE_FRAMES = 0  # DA THU 45 (~3s) -- KHONG sua duoc loi tach ID (goc re la Hungarian
# khong ghep duoc 2 cam cho tu the nam, khong phai do het gio giu track), con lam
# XAU hon o Scene 1 (them 1 id bao nga moi). TAT, quay lai logic goc da xac nhan an
# toan (B khong bao gio bi bao nham).

# Preview/hien thi: nguoi dung muon XEM THU neu gop hien thi cac global_id "bi tach"
# (vi du ID3, ID4 la cung 1 nguoi A bi mat dau roi detect lai) thanh 1 ID duy nhat
# co trong ket qua co giong y muon khong -- CHI sua CACH HIEN THI (text tren video),
# KHONG dung logic gan ID/rule ngay ben duoi (global_id that, fall event that giu
# nguyen, van dam bao B khong bao gio bi bao nham).


def load_H():
    with open(RESULTS_DIR / "homography_matrices.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {cam: np.array(H) for cam, H in raw.items()}


def load_h_by_height():
    with open(RESULTS_DIR / "h_by_height.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {cam: {float(h): np.array(H) for h, H in d.items()} for cam, d in raw.items()}


def render_tile(frame, items, cam_key, cam_label, fall_ids, display_id_map=None):
    """items: list frame_out entries. CHI hien 'ID{gid}' -- tu the/label chay
    NGAM, khong ve len hinh (dung theo yeu cau video demo).
    display_id_map: {global_id_that -> global_id hien thi tren video}, CHI anh
    huong chu hien thi -- fall_ids/logic ben duoi van dung global_id THAT."""
    display_id_map = display_id_map or {}
    frame = frame.copy()
    bbox_field = "bbox1" if cam_key == "CAM 1" else "bbox2"
    held_field = "held1" if cam_key == "CAM 1" else "held2"
    for item in items:
        bbox = item[bbox_field]
        if bbox is None:
            continue
        gid = item["global_id"]
        disp_gid = display_id_map.get(gid, gid)
        is_fall = gid in fall_ids
        is_held = item.get(held_field, False)
        color = (0, 0, 255) if is_fall else ((46, 160, 46) if item["fused"] else (180, 180, 60))
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4 if is_held else 3)
        text = f"ID {disp_gid}" + (" (held)" if is_held else "")
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 12)), (x1 + tw + 10, y1), color, -1)
        cv2.putText(frame, text, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    h, w = frame.shape[:2]
    scale = TILE_H / h
    frame = cv2.resize(frame, (int(w * scale), TILE_H))
    cv2.putText(frame, cam_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    if any(item["global_id"] in fall_ids for item in items):
        cv2.putText(frame, "!!! FALL DETECTED !!!", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 4)
    return frame


def make_title_card(text, size, seconds, fps):
    w, h = size
    frame = np.full((h, w, 3), 30, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 1.0, 3
    lines = text.split("\n")
    heights = [cv2.getTextSize(line, font, scale, thick)[0][1] for line in lines]
    y0 = h // 2 - (len(lines) - 1) * (max(heights) + 20) // 2
    for i, line in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(line, font, scale, thick)
        cv2.putText(frame, line, ((w - tw) // 2, y0 + i * (th + 25)), font, scale, (255, 255, 255), thick)
    return [frame] * int(seconds * fps)


def main():
    print("Nap SharedBackbone + 2 detector + calib Boundary demo moi...")
    backbone = SharedBackbone(CKPT).eval()
    H_floor = load_H()
    H_by_height = load_h_by_height()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / OUT_NAME
    writer = None
    out_fps = 15.0

    for scene_name, cam1_video, cam2_video in SCENES:
        print(f"\n=== {scene_name} ({cam1_video.name} / {cam2_video.name}) ===")
        detector1 = YOLO(str(AI_ROOT / "yolov8n.pt"))
        detector2 = YOLO(str(AI_ROOT / "yolov8n.pt"))
        cap1 = cv2.VideoCapture(str(cam1_video))
        cap2 = cv2.VideoCapture(str(cam2_video))
        fps = cap1.get(cv2.CAP_PROP_FPS)
        out_fps = fps

        assigner = GlobalIdAssigner(max_people=MAX_PEOPLE)
        held_state1, held_state2 = {}, {}
        all_frame_out = []
        frame_idx = 0
        print("  Xu ly (am tham: detect + Hungarian + gop dac trung + phan loai tu the)...")
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if not ret1 or not ret2:
                break
            frame_idx += 1
            frame_out = process_frame_pair(frame1, frame2, detector1, detector2, backbone,
                                            H_floor, H_by_height, assigner, sigma=SIGMA,
                                            det_conf=DET_CONF, dist_threshold=DIST_THRESHOLD,
                                            held_state1=held_state1, held_state2=held_state2,
                                            grace_frames=GRACE_FRAMES)
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
        fall_gids = sorted(set(e["person_id"] for e in events))
        print(f"  global_id co bao nga: {fall_gids} (ky vong CHI 1 id, ung voi nguoi A)")
        fall_ids_by_time = [(e["person_id"], e["timestamp_ms"], e["timestamp_ms"] + FALL_BANNER_MS) for e in events]

        # Preview hien thi: gop cac global_id "bi tach" (cung bao nga -> gia dinh
        # cung 1 nguoi A) thanh 1 ID hien thi duy nhat = id nho nhat trong nhom.
        # CHI anh huong chu "ID.." tren video, KHONG doi global_id that/rule ben duoi.
        display_id_map = {gid: fall_gids[0] for gid in fall_gids} if fall_gids else {}
        if len(fall_gids) > 1:
            print(f"  [preview hien thi] gop {fall_gids} -> hien 'ID {fall_gids[0]}' tren video")

        cap1 = cv2.VideoCapture(str(cam1_video))
        ret, f0 = cap1.read()
        cap1.release()
        tile_w = int(TILE_H * f0.shape[1] / f0.shape[0])
        out_w, out_h = tile_w * 2 + 10, TILE_H

        if writer is None:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (out_w, out_h))
            assert writer.isOpened(), f"VideoWriter khong mo duoc: {out_path}"

        title = f"Boundary Feature Fusion - {scene_name}\nTest nga giua 2 camera - khong loan ID"
        for f in make_title_card(title, (out_w, out_h), 3.0, out_fps):
            writer.write(f)

        cap1 = cv2.VideoCapture(str(cam1_video))
        cap2 = cv2.VideoCapture(str(cam2_video))
        for frame_idx, frame_out in all_frame_out:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if not ret1 or not ret2:
                break
            t_ms = frame_idx / fps * 1000.0
            fall_ids = {gid for gid, lo, hi in fall_ids_by_time if lo <= t_ms <= hi}
            tile1 = render_tile(frame1, frame_out, "CAM 1", "CAM 1", fall_ids, display_id_map)
            tile2 = render_tile(frame2, frame_out, "CAM 2", "CAM 2", fall_ids, display_id_map)
            if tile1.shape[1] != tile_w:
                tile1 = cv2.resize(tile1, (tile_w, TILE_H))
            if tile2.shape[1] != tile_w:
                tile2 = cv2.resize(tile2, (tile_w, TILE_H))
            sep = np.full((TILE_H, 10, 3), 255, dtype=np.uint8)
            writer.write(np.hstack([tile1, sep, tile2]))
        cap1.release()
        cap2.release()

    writer.release()
    assert out_path.exists() and out_path.stat().st_size > 0, f"Video khong duoc ghi ra: {out_path}"
    print(f"\nDa luu: {out_path}")


if __name__ == "__main__":
    main()
