#!/usr/bin/env python3
"""Identity Association — ghép người cùng lúc xuất hiện ở 2 camera (vùng chồng
lấn), qua hình học THUẦN (không model, không train): chiếu điểm chân bbox lên
toạ độ sàn chung bằng homography, rồi Hungarian (scipy) tìm cặp ghép tối ưu.

Port trực tiếp từ `tan_dung_gui/Handoff_for_Edge/2_Multi_Camera_Modules/
pipeline_code/identity_association/{homography.py,homography_matching.py}` +
`GlobalIdAssigner` trong `p2_fall_rule_adapter.py` — CHỈ đổi cách lấy bbox/
track_id (đọc từ pyds NvDsObjectMeta của DeepStream thay vì ultralytics
.track()), giữ nguyên toán học.

⚠️ File calib mặc định (`configs/calibration/homography_matrices_SAMPLE.json`)
là calib PHÒNG TEST CỦA TẤN DŨNG (camera EZVIZ), KHÔNG PHẢI phòng lắp camera
thật hiện tại (Hikvision, 192.168.2.26/.19). Dùng tạm theo quyết định của
người dùng (2026-08-20: "cứ dùng đi, tôi đang cần luồng chạy hết, không quan
tâm độ chính xác") — CHỈ để verify luồng dữ liệu chạy thông, KHÔNG dùng số
liệu ghép cặp ra quyết định thật. Khi có calib phòng thật (quay 4 góc vuông
0.7m, chạy lại `recalibrate_own_room.py`), trỏ `IA_HOMOGRAPHY_PATH` sang file
mới, không cần sửa code.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

log = logging.getLogger("identity_association")

_DEFAULT_HOMOGRAPHY_PATH = (
    Path(__file__).resolve().parent.parent
    / "configs" / "calibration" / "homography_matrices_SAMPLE.json"
)
DEFAULT_DIST_THRESHOLD_M = 0.5  # ngưỡng khớp Identity Association (Handoff mục 4)


def foot_point_from_bbox(bbox_xyxy: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    return ((x1 + x2) / 2.0, y2)


def apply_homography(H: np.ndarray, point_xy: Tuple[float, float]) -> Tuple[float, float]:
    x, y = point_xy
    p = np.array([x, y, 1.0])
    p_proj = H @ p
    p_proj = p_proj / p_proj[2]
    return (float(p_proj[0]), float(p_proj[1]))


def project_bbox_foot(H: np.ndarray, bbox_xyxy) -> Tuple[float, float]:
    return apply_homography(H, foot_point_from_bbox(bbox_xyxy))


def load_homography(path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Đọc JSON {"CAM 1": 3x3, "CAM 2": 3x3} -> {"cam_1": np.array, "cam_2": np.array}."""
    p = Path(path) if path else _DEFAULT_HOMOGRAPHY_PATH
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for key, mat in raw.items():
        # "CAM 1" -> "cam_1" (khớp cam_id_from_pad() trong pipeline.py)
        cam_id = "cam_" + key.strip().split()[-1]
        out[cam_id] = np.array(mat, dtype=np.float64)
    log.info("Đã nạp homography calib (%s): %s", p.name, list(out.keys()))
    return out


class Track:
    __slots__ = ("cam_id", "track_id", "bbox_xyxy", "ground_xy")

    def __init__(self, cam_id, track_id, bbox_xyxy, ground_xy):
        self.cam_id = cam_id
        self.track_id = track_id
        self.bbox_xyxy = bbox_xyxy
        self.ground_xy = ground_xy


def match_two_cameras(
    cam1_tracks: List[Track], cam2_tracks: List[Track],
    distance_threshold: float = DEFAULT_DIST_THRESHOLD_M,
) -> List[Tuple[Track, Track]]:
    """Hungarian ghép cặp tối ưu toàn cục theo khoảng cách Euclid trên mặt sàn
    (mét), loại cặp có khoảng cách > distance_threshold."""
    if not cam1_tracks or not cam2_tracks:
        return []
    n1, n2 = len(cam1_tracks), len(cam2_tracks)
    cost = np.zeros((n1, n2))
    for i, t1 in enumerate(cam1_tracks):
        for j, t2 in enumerate(cam2_tracks):
            cost[i, j] = np.linalg.norm(np.array(t1.ground_xy) - np.array(t2.ground_xy))
    row_idx, col_idx = linear_sum_assignment(cost)
    pairs = []
    for i, j in zip(row_idx, col_idx):
        if cost[i, j] <= distance_threshold:
            pairs.append((cam1_tracks[i], cam2_tracks[j]))
    return pairs


class GlobalIdAssigner:
    """Giữ trạng thái xuyên suốt pipeline (1 instance / edge context).

    Khi Hungarian ghép được 1 cặp (cam1_track, cam2_track), gán 1 global_id
    DÙNG CHUNG cho cả 2. Nếu 1 trong 2 đã có global_id từ frame trước thì DÙNG
    LẠI (không mint mới) — tránh 1 người bị đổi ID liên tục qua các frame.
    Track lẻ (không ghép được, VD bị che khuất 1 cam) vẫn giữ global_id riêng.
    """

    def __init__(self):
        self.map: Dict[Tuple[str, int], int] = {}
        self.next_id = 1

    def _get_or_mint(self, key: Tuple[str, int]) -> int:
        if key not in self.map:
            self.map[key] = self.next_id
            self.next_id += 1
        return self.map[key]

    def assign_pair(self, key1: Tuple[str, int], key2: Tuple[str, int]) -> int:
        if key1 in self.map:
            gid = self.map[key1]
            self.map[key2] = gid
            return gid
        if key2 in self.map:
            gid = self.map[key2]
            self.map[key1] = gid
            return gid
        gid = self.next_id
        self.next_id += 1
        self.map[key1] = gid
        self.map[key2] = gid
        return gid

    def assign_solo(self, key: Tuple[str, int]) -> int:
        return self._get_or_mint(key)


class IdentityAssociator:
    """Tiện ích gộp cho pipeline.py: nạp homography + giữ GlobalIdAssigner,
    expose 1 hàm duy nhất `assign_batch()` xử lý 1 batch (nhiều cam, 1 lúc)."""

    def __init__(self, homography_path: Optional[str] = None,
                 distance_threshold: float = DEFAULT_DIST_THRESHOLD_M):
        self.H = load_homography(homography_path)
        self.distance_threshold = distance_threshold
        self.assigner = GlobalIdAssigner()
        self.last_pairs = []  # set lại mỗi assign_batch() — xem docstring ở đó

    def assign_batch(self, objs_by_cam: Dict[str, List[dict]]) -> Dict[Tuple[str, int], int]:
        """objs_by_cam: {"cam_1": [{"track_id":.., "bbox_xyxy":(x1,y1,x2,y2)}, ...],
        "cam_2": [...]}. Trả về dict (cam_id, track_id) -> global_id cho MỌI
        object đưa vào (cả ghép được lẫn không).

        Tiện thể ghi lại `self.last_pairs` (danh sách các cặp GHÉP ĐƯỢC trong
        batch này, kèm sẵn bbox + ground_xy đã chiếu 2 bên) — Boundary Feature
        Fusion (module 3) dùng lại trực tiếp, khỏi phải tính lại homography."""
        cam_ids = list(objs_by_cam.keys())
        # Hiện chỉ hỗ trợ đúng 2 cam (kiến trúc P2 gốc) — cam nào không có
        # homography calib thì coi như "solo" luôn (không ghép), không crash.
        tracks_by_cam: Dict[str, List[Track]] = {}
        for cam_id in cam_ids:
            H = self.H.get(cam_id)
            tracks = []
            for d in objs_by_cam[cam_id]:
                if H is None:
                    ground_xy = None
                else:
                    ground_xy = project_bbox_foot(H, d["bbox_xyxy"])
                tracks.append(Track(cam_id, d["track_id"], d["bbox_xyxy"], ground_xy))
            tracks_by_cam[cam_id] = tracks

        result: Dict[Tuple[str, int], int] = {}
        self.last_pairs = []

        pairable = [c for c in cam_ids if c in self.H]
        paired_track_ids = {c: set() for c in cam_ids}
        if len(pairable) >= 2:
            c1, c2 = pairable[0], pairable[1]
            pairs = match_two_cameras(
                tracks_by_cam[c1], tracks_by_cam[c2], self.distance_threshold)
            for t1, t2 in pairs:
                key1, key2 = (t1.cam_id, t1.track_id), (t2.cam_id, t2.track_id)
                gid = self.assigner.assign_pair(key1, key2)
                result[key1] = gid
                result[key2] = gid
                paired_track_ids[c1].add(t1.track_id)
                paired_track_ids[c2].add(t2.track_id)
                self.last_pairs.append({
                    "global_id": gid,
                    "cam_1": {"cam_id": t1.cam_id, "track_id": t1.track_id,
                               "bbox_xyxy": t1.bbox_xyxy, "ground_xy": t1.ground_xy},
                    "cam_2": {"cam_id": t2.cam_id, "track_id": t2.track_id,
                               "bbox_xyxy": t2.bbox_xyxy, "ground_xy": t2.ground_xy},
                })

        for cam_id, tracks in tracks_by_cam.items():
            for t in tracks:
                if t.track_id in paired_track_ids.get(cam_id, ()):
                    continue
                key = (cam_id, t.track_id)
                result[key] = self.assigner.assign_solo(key)

        return result
