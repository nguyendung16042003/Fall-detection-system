"""
Tinh anh optical-flow xam 38x51 dai dien 1 cua so thoi gian (mac dinh 1 giay),
theo dung mo ta Espinosa et al. 2019 (xem gia dinh trong espinosa_model.py).

Voi 1 camera: doc cac frame trong cua so -> Farneback dense optical flow giua
cac cap frame lien tiep -> do lon (magnitude) moi cap -> trung binh cong qua
ca cua so -> resize ve 38x51 -> chuan hoa [0,1].

Toi uu quan trong: ha do phan giai xuong TRUOC khi chay Farneback (anh goc
720x480 nhung output chi can 38x51 -> tinh flow tren toan bo pixel goc la
lang phi nang). Downscale ve DOWNSCALE_SIZE truoc, Farneback tren anh nho
nhanh hon nhieu lan ma khong mat gi (van resize xuong 38x51 sau do).
"""
from typing import List

import cv2
import numpy as np

FLOW_H = 38
FLOW_W = 51
DOWNSCALE_SIZE = (160, 107)  # (w,h) -- du chi tiet chuyen dong, nhanh hon nhieu


def precompute_flow_magnitudes(frames: List[np.ndarray]) -> np.ndarray:
    """Tinh do lon flow giua MOI cap frame LIEN TIEP 1 LAN cho toan bo video
    (khong tinh lai cho tung cua so chong lap). Tra ve mang (n_frames-1, H, W)
    da downscale, chua trung binh theo cua so."""
    if len(frames) < 2:
        return np.zeros((0, DOWNSCALE_SIZE[1], DOWNSCALE_SIZE[0]), dtype=np.float32)

    grays_small = [
        cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), DOWNSCALE_SIZE)
        for f in frames
    ]
    mags = np.empty((len(grays_small) - 1, DOWNSCALE_SIZE[1], DOWNSCALE_SIZE[0]), dtype=np.float32)
    for i in range(len(grays_small) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            grays_small[i], grays_small[i + 1], None,
            pyr_scale=0.5, levels=2, winsize=13,
            iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
        )
        mags[i] = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return mags


def flow_window_from_magnitudes(mags: np.ndarray, start_pair_idx: int, end_pair_idx: int) -> np.ndarray:
    """Lay trung binh cac ban do do-lon flow (da tinh san) trong khoang cap
    frame [start_pair_idx, end_pair_idx) -- ung voi 1 cua so -- roi resize ve
    38x51, chuan hoa [0,1]."""
    if end_pair_idx <= start_pair_idx or start_pair_idx >= len(mags):
        return np.zeros((FLOW_H, FLOW_W), dtype=np.float32)
    end_pair_idx = min(end_pair_idx, len(mags))
    avg_mag = mags[start_pair_idx:end_pair_idx].mean(axis=0)
    resized = cv2.resize(avg_mag, (FLOW_W, FLOW_H), interpolation=cv2.INTER_AREA)
    max_val = resized.max()
    if max_val > 1e-6:
        resized = resized / max_val
    return resized.astype(np.float32)


def compute_flow_magnitude_image(frames: List[np.ndarray]) -> np.ndarray:
    """Ban don gian (khong cache) -- dung cho danh gia (evaluate_p2p3.py) noi
    moi cua so chi tinh 1 lan, khong can toi uu cache toan video."""
    mags = precompute_flow_magnitudes(frames)
    return flow_window_from_magnitudes(mags, 0, len(mags))


def combine_two_cams(flow_cam1: np.ndarray, flow_cam2: np.ndarray) -> np.ndarray:
    """Gop 2 anh flow 1-camera thanh input 2-kenh (channel-stack, xem gia dinh
    trong espinosa_model.py). Tra ve (2, 38, 51) float32."""
    return np.stack([flow_cam1, flow_cam2], axis=0).astype(np.float32)
