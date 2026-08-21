"""Video demo bo sung (data cu tu File Run Problem 1/2/3) them vao "Video cuoi
cung", dung DUNG giai phap CHINH THUC cua nhom (p2_fall_rule_adapter.py,
KHONG sua) -- giong make_video_demo_boundary.py nhung dung calib CU (P2/P3 da
co san, khong can calib lai).

3 canh:
  4_IA_nga_don.mp4        -- P2 Data 2 Scene 5: 1 nguoi nga, xuyen 2 cam
                             (Identity Association). Model phat hien nga +
                             rule binh thuong, tu the chay NGAM (khong hien).
  5_IA_di_khong_nga.mp4   -- P2 Data 2 Scene 4: 1 nguoi di qua lai giua 2 cam,
                             KHONG nga. Cung pipeline nhu tren.
  6_Boundary_cui_nguoi.mp4-- P3 Data 2 Scene 1: cui nguoi giua 2 cam cua
                             Boundary Feature Fusion. KHONG PHAI truong hop
                             nga -- KHONG goi model phan loai tu the / rule
                             nga o day (chi detect + Hungarian match + gan ID
                             xuyen 2 cam, dung Track/match_two_cameras that,
                             KHONG sua).

Ca 3 ghi thanh 3 file RIENG (giu nguyen quy uoc 1 file/1 kich ban nhu video
1/2/3), moi file co title card mo dau.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p3_cross_camera"))
PIPELINE_DIR = Path(__file__).resolve().parents[2] / "Coding" / "Pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

from shared_backbone import SharedBackbone  # noqa: E402
from p2_fall_rule_adapter import (  # noqa: E402
    GlobalIdAssigner, process_frame_pair, detect_fall_events_ia, detect_tracks,
)
from homography import apply_homography, foot_point_from_bbox  # noqa: E402
from homography_matching import Track, match_two_cameras  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
CHXL = AI_ROOT / "Chạy xử lý toàn bộ hệ thống"
P2_DIR = CHXL / "File Run Problem 2" / "P2 data test" / "P2 Data 2"
P3_DIR = CHXL / "File Run Problem 3" / "P3" / "P3 Data 2"
RESULTS_P2 = CHXL / "Results" / "p2"
RESULTS_P3 = CHXL / "Results" / "p3"
CKPT = (AI_ROOT / "Coding" / "training" / "training 3" / "runs" / "classify"
        / "Fall_Detection_Advanced_Loss" / "YOLOv8n_AFCL_Balanced-8" / "weights" / "best.pt")
OUT_DIR = CHXL / "Video demo" / "Video cuối cùng"

TILE_H = 540
FALL_BANNER_MS = 1500
SIGMA = 2.0
DET_CONF = 0.4
DIST_THRESHOLD = 0.5


def load_H_p2():
    with open(RESULTS_P2 / "homography_matrices.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {cam: np.array(H) for cam, H in raw.items()}


def load_h_by_height_p2():
    with open(RESULTS_P2 / "h_by_height.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {cam: {float(h): np.array(H) for h, H in d.items()} for cam, d in raw.items()}


_P3_KEY_REMAP = {"P3 Cam1": "CAM 1", "P3 Cam2": "CAM 2"}


def load_H_p3():
    with open(RESULTS_P3 / "homography_matrices_p3.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {_P3_KEY_REMAP[cam]: np.array(H) for cam, H in raw.items()}


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


def render_tile_id(frame, items, cam_key, cam_label, fall_ids):
    """Chi hien 'ID{gid}' + banner khi co fall_ids (dung cho canh CO fall rule)."""
    frame = frame.copy()
    bbox_field = "bbox1" if cam_key == "CAM 1" else "bbox2"
    held_field = "held1" if cam_key == "CAM 1" else "held2"
    for item in items:
        bbox = item[bbox_field]
        if bbox is None:
            continue
        gid = item["global_id"]
        is_fall = gid in fall_ids
        is_held = item.get(held_field, False)
        color = (0, 0, 255) if is_fall else ((46, 160, 46) if item["fused"] else (180, 180, 60))
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4 if is_held else 3)
        text = f"ID {gid}" + (" (held)" if is_held else "")
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


def process_frame_pair_geometry_only(frame1, frame2, detector1, detector2, H_floor,
                                      assigner, det_conf=DET_CONF, dist_threshold=DIST_THRESHOLD):
    """Ban RUT GON cua process_frame_pair: CHI detect + Hungarian match (P2) +
    gan global_id -- KHONG goi backbone/classifier, KHONG lien quan fall rule
    (dung cho canh 'cui nguoi' KHONG PHAI truong hop nga, theo yeu cau nguoi
    dung). Tai dung nguyen detect_tracks/Track/match_two_cameras/homography."""
    dets1 = detect_tracks(detector1, frame1, det_conf)
    dets2 = detect_tracks(detector2, frame2, det_conf)

    t1 = [Track("CAM 1", d["track_id"], d["bbox_xyxy"],
                 ground_xy=apply_homography(H_floor["CAM 1"], foot_point_from_bbox(d["bbox_xyxy"])))
          for d in dets1]
    t2 = [Track("CAM 2", d["track_id"], d["bbox_xyxy"],
                 ground_xy=apply_homography(H_floor["CAM 2"], foot_point_from_bbox(d["bbox_xyxy"])))
          for d in dets2]

    pairs = match_two_cameras(t1, t2, distance_threshold=dist_threshold)
    paired_ids1 = {p[0].track_id for p in pairs}
    paired_ids2 = {p[1].track_id for p in pairs}

    frame_out = []
    for track1, track2 in pairs:
        key1, key2 = ("CAM 1", track1.track_id), ("CAM 2", track2.track_id)
        gid = assigner.assign_pair(key1, key2, ground_xy=track1.ground_xy)
        frame_out.append({"global_id": gid, "bbox1": track1.bbox_xyxy, "bbox2": track2.bbox_xyxy, "fused": True})
    for d in dets1:
        if d["track_id"] in paired_ids1:
            continue
        key = ("CAM 1", d["track_id"])
        gxy = apply_homography(H_floor["CAM 1"], foot_point_from_bbox(d["bbox_xyxy"]))
        gid = assigner.assign_solo(key, ground_xy=gxy)
        frame_out.append({"global_id": gid, "bbox1": d["bbox_xyxy"], "bbox2": None, "fused": False})
    for d in dets2:
        if d["track_id"] in paired_ids2:
            continue
        key = ("CAM 2", d["track_id"])
        gxy = apply_homography(H_floor["CAM 2"], foot_point_from_bbox(d["bbox_xyxy"]))
        gid = assigner.assign_solo(key, ground_xy=gxy)
        frame_out.append({"global_id": gid, "bbox1": None, "bbox2": d["bbox_xyxy"], "fused": False})
    return frame_out


def render_tile_id_only(frame, items, cam_key, cam_label):
    """Nhu render_tile_id nhung KHONG co khai niem fall (khong banner, mau co
    dinh theo fused/solo) -- dung cho canh khong goi fall rule."""
    frame = frame.copy()
    bbox_field = "bbox1" if cam_key == "CAM 1" else "bbox2"
    for item in items:
        bbox = item[bbox_field]
        if bbox is None:
            continue
        gid = item["global_id"]
        color = (46, 160, 46) if item["fused"] else (180, 180, 60)
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        text = f"ID {gid}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 12)), (x1 + tw + 10, y1), color, -1)
        cv2.putText(frame, text, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    h, w = frame.shape[:2]
    scale = TILE_H / h
    frame = cv2.resize(frame, (int(w * scale), TILE_H))
    cv2.putText(frame, cam_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return frame


def write_scene_with_fall_rule(cam1_video, cam2_video, H_floor, H_by_height, backbone,
                                title, out_path, writer_state):
    detector1 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    detector2 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    cap1 = cv2.VideoCapture(str(cam1_video))
    cap2 = cv2.VideoCapture(str(cam2_video))
    fps = cap1.get(cv2.CAP_PROP_FPS)

    assigner = GlobalIdAssigner()
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

    cap1 = cv2.VideoCapture(str(cam1_video))
    ret, f0 = cap1.read()
    cap1.release()
    tile_w = int(TILE_H * f0.shape[1] / f0.shape[0])
    out_w, out_h = tile_w * 2 + 10, TILE_H

    if writer_state["writer"] is None:
        writer_state["writer"] = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
        assert writer_state["writer"].isOpened(), f"VideoWriter khong mo duoc: {out_path}"
    writer = writer_state["writer"]

    for f in make_title_card(title, (out_w, out_h), 3.0, fps):
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
        tile1 = render_tile_id(frame1, frame_out, "CAM 1", "CAM 1", fall_ids)
        tile2 = render_tile_id(frame2, frame_out, "CAM 2", "CAM 2", fall_ids)
        if tile1.shape[1] != tile_w:
            tile1 = cv2.resize(tile1, (tile_w, TILE_H))
        if tile2.shape[1] != tile_w:
            tile2 = cv2.resize(tile2, (tile_w, TILE_H))
        sep = np.full((TILE_H, 10, 3), 255, dtype=np.uint8)
        writer.write(np.hstack([tile1, sep, tile2]))
    cap1.release()
    cap2.release()


def write_scene_geometry_only(cam1_video, cam2_video, H_floor, title, out_path, writer_state):
    detector1 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    detector2 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    cap1 = cv2.VideoCapture(str(cam1_video))
    cap2 = cv2.VideoCapture(str(cam2_video))
    fps = cap1.get(cv2.CAP_PROP_FPS)

    assigner = GlobalIdAssigner()
    all_frame_out = []
    frame_idx = 0
    print("  Xu ly (am tham: detect + Hungarian, KHONG goi model tu the/rule nga)...")
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        frame_idx += 1
        frame_out = process_frame_pair_geometry_only(frame1, frame2, detector1, detector2,
                                                       H_floor, assigner, det_conf=DET_CONF,
                                                       dist_threshold=DIST_THRESHOLD)
        all_frame_out.append((frame_idx, frame_out))
    cap1.release()
    cap2.release()

    n_fused = sum(1 for _, fo in all_frame_out for item in fo if item["fused"])
    n_solo = sum(1 for _, fo in all_frame_out for item in fo if not item["fused"])
    all_gids = sorted(set(item["global_id"] for _, fo in all_frame_out for item in fo))
    print(f"  {frame_idx} frame, {n_fused} luot fused, {n_solo} luot solo, global_id xuat hien: {all_gids}")

    cap1 = cv2.VideoCapture(str(cam1_video))
    ret, f0 = cap1.read()
    cap1.release()
    tile_w = int(TILE_H * f0.shape[1] / f0.shape[0])
    out_w, out_h = tile_w * 2 + 10, TILE_H

    if writer_state["writer"] is None:
        writer_state["writer"] = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
        assert writer_state["writer"].isOpened(), f"VideoWriter khong mo duoc: {out_path}"
    writer = writer_state["writer"]

    for f in make_title_card(title, (out_w, out_h), 3.0, fps):
        writer.write(f)

    cap1 = cv2.VideoCapture(str(cam1_video))
    cap2 = cv2.VideoCapture(str(cam2_video))
    for frame_idx, frame_out in all_frame_out:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        tile1 = render_tile_id_only(frame1, frame_out, "CAM 1", "CAM 1")
        tile2 = render_tile_id_only(frame2, frame_out, "CAM 2", "CAM 2")
        if tile1.shape[1] != tile_w:
            tile1 = cv2.resize(tile1, (tile_w, TILE_H))
        if tile2.shape[1] != tile_w:
            tile2 = cv2.resize(tile2, (tile_w, TILE_H))
        sep = np.full((TILE_H, 10, 3), 255, dtype=np.uint8)
        writer.write(np.hstack([tile1, sep, tile2]))
    cap1.release()
    cap2.release()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Nap SharedBackbone + calib P2 (cu)...")
    backbone = SharedBackbone(CKPT).eval()
    H_floor_p2 = load_H_p2()
    H_by_height_p2 = load_h_by_height_p2()

    print("\n=== 4. Identity Association -- nga don (P2 Data 2 Scene 5) ===")
    writer_state = {"writer": None}
    out_path = OUT_DIR / "4_IA_nga_don.mp4"
    write_scene_with_fall_rule(
        P2_DIR / "Scene 5-CAM 1.mp4", P2_DIR / "Scene 5-CAM 2.mp4",
        H_floor_p2, H_by_height_p2, backbone,
        "Identity Association - Nga don\n1 nguoi nga, theo doi ID xuyen 2 camera",
        out_path, writer_state)
    writer_state["writer"].release()
    assert out_path.exists() and out_path.stat().st_size > 0
    print(f"  Da luu: {out_path}")

    print("\n=== 5. Identity Association -- di qua lai khong nga (P2 Data 2 Scene 4) ===")
    writer_state = {"writer": None}
    out_path = OUT_DIR / "5_IA_di_khong_nga.mp4"
    write_scene_with_fall_rule(
        P2_DIR / "Scene 4-CAM 1.mp4", P2_DIR / "Scene 4-CAM 2.mp4",
        H_floor_p2, H_by_height_p2, backbone,
        "Identity Association - Di qua lai giua 2 camera\nKhong nga -- ID giu on dinh xuyen camera",
        out_path, writer_state)
    writer_state["writer"].release()
    assert out_path.exists() and out_path.stat().st_size > 0
    print(f"  Da luu: {out_path}")

    print("\nNap calib P3 (cu)...")
    H_floor_p3 = load_H_p3()

    print("\n=== 6. Boundary Feature Fusion -- cui nguoi, khong phai truong hop nga (P3 Data 2 Scene 1) ===")
    writer_state = {"writer": None}
    out_path = OUT_DIR / "6_Boundary_cui_nguoi.mp4"
    write_scene_geometry_only(
        P3_DIR / "Scene 1-Cam 1.mp4", P3_DIR / "Scene 1-Cam 2.mp4",
        H_floor_p3,
        "Boundary Feature Fusion - Cui nguoi giua 2 camera\n(Khong phai truong hop nga -- chi theo doi ID)",
        out_path, writer_state)
    writer_state["writer"].release()
    assert out_path.exists() and out_path.stat().st_size > 0
    print(f"  Da luu: {out_path}")


if __name__ == "__main__":
    main()
