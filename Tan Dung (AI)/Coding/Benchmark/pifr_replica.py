"""
Ban tai tao lai PIFR (Kong et al., PLOS One 2025) cho dung nhat co the, de lam
doi chung cong bang khi benchmark voi model cua nhom.

2 bug da sua so voi code cu trong Coding/training/training2/dataset.py:
1. calculate_angle_3points: norm_BC dung nham BA[1] thay vi BC[1] -> lam sai
   3/9 feature (shoulder_nose_angle, left_leg_angle, right_leg_angle).
2. SVM hyperparameter: code cu dung C=10, gamma='scale' -- bai bao PIFR (Bang 2)
   dung C=1.0, kernel=RBF, gamma=0.1. Sua lai dung theo bai bao.

Cung sua luon van de "train/test khong tach biet theo video" (code cu SVM.PY
train/test tren cung 1 file 654 dong) -- o day tach han video train rieng,
video test rieng (khong trung video nao).

Rule xac nhan nga: dung dung nhu bai bao mo ta (khong them heuristic
aspect_ratio/torso_angle nhu ban Test 2 cu cua nhom) -- chi dung sliding
window + bo phieu da so tren OUTPUT NHI PHAN cua SVM, giu window 1.5s +
>50% giong logic "criteria xac nhan nga" ma bai bao de xuat (khong cong bo
gia tri cu the nen day la lua chon hop ly nhat co the tai tao).
"""
import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

AI_ROOT = Path(__file__).resolve().parents[2]
POSE_MODEL_PATH = AI_ROOT / "yolo11n-pose.pt"

FEATURE_NAMES = [
    "com_x", "com_y", "shoulder_nose_angle", "torso_angle", "hip_angle",
    "shoulder_angle", "left_leg_angle", "right_leg_angle", "nose_to_ankle_angle",
]


def calculate_angle_3points(A, B, C):
    """FIXED: norm_BC dung dung BC (code cu o dataset.py dung nham BA[1])."""
    BA = (A[0] - B[0], A[1] - B[1])
    BC = (C[0] - B[0], C[1] - B[1])
    dot = BA[0] * BC[0] + BA[1] * BC[1]
    norm = math.sqrt(BA[0] ** 2 + BA[1] ** 2) * math.sqrt(BC[0] ** 2 + BC[1] ** 2)
    if norm == 0:
        return 0
    cos_angle = max(-1.0, min(1.0, dot / norm))
    return math.degrees(math.acos(cos_angle))


def calculate_angle_with_vertical(A, B):
    dx, dy = B[0] - A[0], B[1] - A[1]
    norm = math.sqrt(dx ** 2 + dy ** 2)
    return math.degrees(math.acos(dy / norm)) if norm != 0 else 0


def calculate_angle_with_horizontal(A, B):
    dx, dy = B[0] - A[0], B[1] - A[1]
    norm = math.sqrt(dx ** 2 + dy ** 2)
    return math.degrees(math.acos(dx / norm)) if norm != 0 else 0


def extract_biomechanic_features(pose_model, frame):
    results = pose_model(frame, verbose=False)
    if not results or results[0].keypoints is None or results[0].keypoints.xy.shape[0] == 0:
        return None
    kp = results[0].keypoints.xy[0].cpu().numpy()
    if len(kp) < 17 or np.all(kp == 0):
        return None

    nose, l_shoulder, r_shoulder = kp[0], kp[5], kp[6]
    l_hip, r_hip = kp[11], kp[12]
    l_knee, r_knee = kp[13], kp[14]
    l_ankle, r_ankle = kp[15], kp[16]
    mid_hip = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
    mid_ankle = ((l_ankle[0] + r_ankle[0]) / 2, (l_ankle[1] + r_ankle[1]) / 2)

    return [
        float(np.mean(kp[:, 0])), float(np.mean(kp[:, 1])),
        calculate_angle_3points(l_shoulder, nose, r_shoulder),
        calculate_angle_with_vertical(nose, mid_hip),
        calculate_angle_with_horizontal(l_hip, r_hip),
        calculate_angle_with_horizontal(l_shoulder, r_shoulder),
        calculate_angle_3points(l_hip, l_knee, l_ankle),
        calculate_angle_3points(r_hip, r_knee, r_ankle),
        calculate_angle_with_vertical(nose, mid_ankle),
    ]


def build_labeled_features(video_paths, pose_model, is_fall_flags, sample_every_s=1.0):
    """Trich feature + nhan (0=Lying,1=Standing) tu danh sach video, mau 1fps.
    Giu nguyen heuristic torso_angle>=45 cua nhom cu de gan nhan cho video Fall
    (KHONG phai gan tay nhu PIFR that -- day la gioi han da biet, ghi trong bao cao).
    """
    rows = []
    for video_path, is_fall in zip(video_paths, is_fall_flags):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(fps * sample_every_s))
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % step == 0:
                feats = extract_biomechanic_features(pose_model, frame)
                if feats is not None:
                    if is_fall:
                        label = 0 if feats[3] >= 45.0 else 1  # torso_angle
                    else:
                        label = 1
                    rows.append(feats + [label])
            count += 1
        cap.release()
    return rows


def evaluate_video_fall(video_path, pose_model, clf, scaler, window_s=1.5, vote_ratio=0.5):
    """Sliding window + bo phieu da so tren du doan nhi phan cua SVM -- KHONG co
    heuristic bo sung nao khac (dung nhu PIFR mo ta), tra ve True neu trigger Fall."""
    import pandas as pd

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    window_size = max(1, int(fps * window_s))
    buffer = []
    triggered = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        feats = extract_biomechanic_features(pose_model, frame)
        if feats is not None:
            X = pd.DataFrame([feats], columns=FEATURE_NAMES)
            Xs = scaler.transform(X)
            pred = clf.predict(Xs)[0]
            buffer.append(0 if pred == 0 else 1)
        else:
            buffer.append(buffer[-1] if buffer else 1)

        if len(buffer) > window_size:
            buffer.pop(0)
        if len(buffer) == window_size and buffer.count(0) / window_size > vote_ratio:
            triggered = True
            break

    cap.release()
    return triggered
