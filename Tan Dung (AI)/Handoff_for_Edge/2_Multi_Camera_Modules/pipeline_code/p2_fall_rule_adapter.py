"""
Identity Association -- adapter noi Hungarian that (homography_matching.py,
KHONG sua) + co che gop cua Boundary Feature Fusion (cross_camera_fuse.py,
KHONG sua) vao RULE THAT (fall_rule.py, KHONG sua).

PHU THUOC Part 1 (1_Single_Camera_Pipeline/): can fall_rule.py cua Part 1 --
xem README.md muc "Phu thuoc Part 1" de biet cach tro dung duong dan neu di
chuyen 2 thu muc nay tach roi nhau (khong con la 2 thu muc con cung cap cha).

assign_global_person_id(): online, dict {(cam,track_id): global_id}. Khi 1
cap (cam1_track, cam2_track) duoc Hungarian ghep trong frame hien tai:
  - Neu 1 trong 2 da co global_id (tu frame truoc) -> ben kia dung LAI global_id do.
  - Neu ca 2 chua co -> mint global_id MOI, gan cho ca 2.
Track le (khong ghep duoc, vd bi che khuat 1 cam) giu global_id rieng theo
chinh no (mint moi neu la lan dau thay track_id nay).
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PART1_PIPELINE_CODE = HERE.parents[1] / "1_Single_Camera_Pipeline" / "pipeline_code"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "identity_association"))
sys.path.insert(0, str(HERE / "boundary_feature_fusion"))
sys.path.insert(0, str(PART1_PIPELINE_CODE))  # cho fall_rule.py (thuoc Part 1)

from homography import apply_homography, foot_point_from_bbox  # noqa: E402
from homography_matching import Track, match_two_cameras  # noqa: E402
from shared_backbone import preprocess_crop  # noqa: E402
from cross_camera_fuse import build_distance_matrices, sgie_forward  # noqa: E402
from fall_rule import detect_fall_events  # noqa: E402

CLASS_NAMES = ["bend", "exercise", "lie", "sit", "stand"]


class GlobalIdAssigner:
    """Giu trang thai xuyen video (1 instance / 1 video-pair)."""

    def __init__(self):
        self.map = {}  # (cam, track_id) -> global_id
        self.next_id = 1

    def _get_or_mint(self, key):
        if key not in self.map:
            self.map[key] = self.next_id
            self.next_id += 1
        return self.map[key]

    def assign_pair(self, key1, key2):
        """key1=("CAM 1",track_id), key2=("CAM 2",track_id). Tra ve global_id
        dung chung cho ca 2, uu tien tai su dung id da co cua 1 trong 2 ben."""
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

    def assign_solo(self, key):
        return self._get_or_mint(key)


def detect_tracks(detector, frame, det_conf=0.4):
    results = detector.track(frame, classes=[0], conf=det_conf, persist=True, verbose=False)[0]
    out = []
    if results.boxes is None:
        return out
    for box in results.boxes:
        if box.id is None:
            continue
        out.append({"track_id": int(box.id[0]), "bbox_xyxy": tuple(box.xyxy[0].cpu().numpy().tolist())})
    return out


def classify_solo(backbone, frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return None, None
    x = preprocess_crop(crop)
    with torch.no_grad():
        grid = backbone(x)
        logits = backbone.classify_from_grid(grid)
        probs = torch.softmax(logits, dim=1)[0]
    return CLASS_NAMES[int(probs.argmax())], float(probs.max())


def process_frame_pair(frame1, frame2, detector1, detector2, backbone,
                        H_floor, H_by_height, assigner, sigma=2.0,
                        det_conf=0.4, dist_threshold=0.5):
    """1 frame CUNG THOI DIEM ca 2 cam. Tra ve list record theo dung schema
    fall_rule.detect_fall_events (person_id = global_id), + thong tin ve/debug
    (bbox theo cam, held=False luon o day -- P2/Boundary Feature Fusion
    khong dung grace-hold, giu nguyen thiet ke goc)."""
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

    frame_out = []  # (global_id, bbox_cam1_or_none, bbox_cam2_or_none, label, conf, fused)

    for track1, track2 in pairs:
        key1, key2 = ("CAM 1", track1.track_id), ("CAM 2", track2.track_id)
        gid = assigner.assign_pair(key1, key2)
        crop1 = frame1[int(track1.bbox_xyxy[1]):int(track1.bbox_xyxy[3]),
                        int(track1.bbox_xyxy[0]):int(track1.bbox_xyxy[2])]
        crop2 = frame2[int(track2.bbox_xyxy[1]):int(track2.bbox_xyxy[3]),
                        int(track2.bbox_xyxy[0]):int(track2.bbox_xyxy[2])]
        if crop1.size == 0 or crop2.size == 0:
            continue
        x1 = preprocess_crop(crop1)
        x2 = preprocess_crop(crop2)
        with torch.no_grad():
            grid1 = backbone(x1)
            h, w = grid1.shape[2], grid1.shape[3]
            dist_1to2, dist_2to1 = build_distance_matrices(
                (h, w), track1.ground_xy, track2.ground_xy,
                H_by_height["CAM 1"], H_by_height["CAM 2"],
                track1.bbox_xyxy, track2.bbox_xyxy)
            v_fused = sgie_forward(backbone, x1, x2, has_both_cams=True,
                                    dist_1to2=dist_1to2, dist_2to1=dist_2to1, sigma=sigma)
            logits = backbone.classify_from_vector(v_fused)
            probs = torch.softmax(logits, dim=1)[0]
        label = CLASS_NAMES[int(probs.argmax())]
        conf = float(probs.max())
        frame_out.append({"global_id": gid, "bbox1": track1.bbox_xyxy, "bbox2": track2.bbox_xyxy,
                           "label": label, "conf": conf, "fused": True})

    for d in dets1:
        if d["track_id"] in paired_ids1:
            continue
        key = ("CAM 1", d["track_id"])
        gid = assigner.assign_solo(key)
        label, conf = classify_solo(backbone, frame1, d["bbox_xyxy"])
        if label is None:
            continue
        frame_out.append({"global_id": gid, "bbox1": d["bbox_xyxy"], "bbox2": None,
                           "label": label, "conf": conf, "fused": False})

    for d in dets2:
        if d["track_id"] in paired_ids2:
            continue
        key = ("CAM 2", d["track_id"])
        gid = assigner.assign_solo(key)
        label, conf = classify_solo(backbone, frame2, d["bbox_xyxy"])
        if label is None:
            continue
        frame_out.append({"global_id": gid, "bbox1": None, "bbox2": d["bbox_xyxy"],
                           "label": label, "conf": conf, "fused": False})

    return frame_out


def frame_out_to_records(frame_idx, frame_out):
    """Chuyen list frame_out (1 frame) -> list record dung schema fall_rule
    (dung bbox cam1 neu co, khong thi bbox cam2 -- rule chi can 1 bbox de tinh
    aspect ratio, khong quan trong tu cam nao)."""
    records = []
    for item in frame_out:
        bbox = item["bbox1"] if item["bbox1"] is not None else item["bbox2"]
        x1, y1, x2, y2 = bbox
        records.append({
            "frame_id": frame_idx, "person_id": item["global_id"],
            "pose": item["label"], "pose_confidence": item["conf"],
            "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
            "det_confidence": 1.0, "held": False,
        })
    return records


def detect_fall_events_ia(all_frame_out, fps):
    """all_frame_out: list[(frame_idx, frame_out)]. Goi thang detect_fall_events
    (KHONG sua) tren records gop tu MOI nguoi (global_id) cung luc -- ham do da
    tu tach rieng by_person ben trong."""
    records = []
    for frame_idx, frame_out in all_frame_out:
        records.extend(frame_out_to_records(frame_idx, frame_out))
    return detect_fall_events(records, fps=fps)
