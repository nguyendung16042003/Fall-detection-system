"""
P2 -- Buoc 2/3: ghep cap nguoi xuyen 2 camera CUNG THOI DIEM, dua tren khoang
cach toa do san sau khi chieu qua homography. Hungarian algorithm (scipy) +
nguong khoang cach -- KHONG dung model (dung dac ta "Context xu ly van de
2.docx": "Ban GNN da bi loai bo vi them 1 model rieng, khong khop rang buoc
'khong them model' tren Jetson Nano 4GB").
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from homography import project_bbox_foot


class Track(object):
    """KHONG dung dataclasses (Python 3.6 tren Jetson Nano khong co module nay,
    chi co tu 3.7) -- dung class thuong de tuong thich ca dev (3.10+) lan
    Jetson (3.6.9) khong can sua code."""

    def __init__(self, cam_id, track_id, bbox_xyxy, ground_xy=None):
        self.cam_id = cam_id
        self.track_id = track_id
        self.bbox_xyxy = bbox_xyxy
        self.ground_xy = ground_xy


def match_two_cameras(cam1_tracks, cam2_tracks, distance_threshold=0.5):
    """
    cam1_tracks, cam2_tracks: list[Track], moi Track da co ground_xy (toa do
    san, met) tu project_bbox_foot().
    Tra ve list cac cap (track_cam1, track_cam2) duoc ghep la CUNG 1 nguoi.
    """
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


def build_tracks_from_detections(cam_id, detections, H):
    """detections: list dict {track_id, bbox_xyxy}. Tra ve list Track da co
    ground_xy tinh san qua homography H cua camera do."""
    tracks = []
    for d in detections:
        ground_xy = project_bbox_foot(H, d["bbox_xyxy"])
        tracks.append(Track(cam_id=cam_id, track_id=d["track_id"],
                             bbox_xyxy=d["bbox_xyxy"], ground_xy=ground_xy))
    return tracks
