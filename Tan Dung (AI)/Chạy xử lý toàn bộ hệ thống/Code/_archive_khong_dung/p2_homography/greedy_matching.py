"""
So sanh A (xem plan): thuat toan gan ghep thay the cho Hungarian -- Greedy
nearest-neighbor (gan tung cap GAN NHAT truoc, khong toi uu toan cuc). Day la
lua chon "toan hoc co san" pho bien trong MOT/tracking lam baseline don gian
hon Hungarian. File RIENG, KHONG sua homography_matching.py da validate.
"""
import numpy as np


def match_two_cameras_greedy(cam1_tracks, cam2_tracks, distance_threshold=0.5):
    """Cung input/output format voi match_two_cameras() (Hungarian) de so
    sanh cong bang -- chi khac thuat toan gan ghep."""
    if not cam1_tracks or not cam2_tracks:
        return []

    n1, n2 = len(cam1_tracks), len(cam2_tracks)
    cost = np.zeros((n1, n2))
    for i, t1 in enumerate(cam1_tracks):
        for j, t2 in enumerate(cam2_tracks):
            cost[i, j] = np.linalg.norm(np.array(t1.ground_xy) - np.array(t2.ground_xy))

    candidates = sorted(
        ((cost[i, j], i, j) for i in range(n1) for j in range(n2)),
        key=lambda x: x[0],
    )

    used1, used2 = set(), set()
    pairs = []
    for d, i, j in candidates:
        if d > distance_threshold:
            break  # da sap xep tang dan -- con lai deu > nguong
        if i in used1 or j in used2:
            continue
        used1.add(i)
        used2.add(j)
        pairs.append((cam1_tracks[i], cam2_tracks[j]))
    return pairs
