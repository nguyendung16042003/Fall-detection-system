"""
So sanh P3 (co gop) vs Espinosa replica CHI TREN DATA P3 that (P3 Data 1/2,
dung kich ban "nguoi bi cat doi giua 2 cam" -- KHONG lan P2, xem hoi thoai:
"day la toi bao rieng P3 thoi"). Mo rong tu 6 canh goc len 36 "lan chay"
bang augment AN TOAN voi homography (sang/toi/nhieu/nen, xem
make_p3_test_augmentations.py) + dao vai tro camera (tinh o code, khong tao
file rieng) -- 6 bien the/canh x 6 canh = 36.

QUAN TRONG khi doc ket qua: 36 la SO LAN CHAY (robustness), KHONG PHAI 36
su kien nga doc lap -- van chi co 4 su kien nga That + 2 su kien ADL that.
Bao cao ca 2 muc: (a) tren 6 canh GOC (doc lap, dang tin cay nhat cho
Sensitivity chinh thuc), (b) tren toan bo 36 lan chay (kiem tra do on dinh).

CAP NHAT: quyet dinh Fall/ADL cap scene gio dung RULE THAT
(Coding/Pipeline/fall_rule.py, qua p3_fall_rule_adapter.py) -- transition
<=2s HOAC sustained_lying >=1.5s -- THAY THE proxy cu "any(p=='lie')" (chi
can 1 frame la tinh Fall, khong co cua so thoi gian/yeu cau duy tri).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
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
CKPT_BACKBONE = (AI_ROOT / "Coding" / "training" / "training 3" / "runs" / "classify"
                  / "Fall_Detection_Advanced_Loss" / "YOLOv8n_AFCL_Balanced-8" / "weights" / "best.pt")
CKPT_ESPINOSA = RESULTS_ESP / "espinosa_best.pt"
CLASS_NAMES = ["bend", "exercise", "lie", "sit", "stand"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DET_CONF = 0.2
DIST_THRESHOLD = 1.5
FRAME_SKIP = 2
SIGMA = 2.0
WINDOW_S = 1.0
STRIDE_S = 0.5
# Khop dung make_p3_report_video.py: YOLOv8n COCO-pretrained bo lo nguoi nam
# det rat thuong xuyen tren mot so goc camera (da xac nhan qua debug tren Scene
# 1: 18/150 -> 12%). Ap dung lai co che giu tam bbox (~1.2s, giong het
# GRACE_FRAMES cua detect_classify_pipeline.py) de nhat quan voi video demo va
# khong danh gia thap sai P3 chi vi 1-2 frame detector "mu" thoang qua. Dung
# .predict() (khong dung .track()) vi detector1/detector2 dung lai xuyen 36
# video khac nhau trong vong lap ben duoi -- .track(persist=True) se giu
# trang thai tracker cu tu video truoc, gay nhieu/sai cho video sau.
GRACE_MS = 1200

# (data_dir_goc, data_dir_aug, scene_id, label, ten_canh)
BASE_SCENES = [
    (P3_DIR / "P3 Data 1", P3_DIR / "P3 Data 1 (augmented)", 1, 1, "P3D1_S1"),
    (P3_DIR / "P3 Data 1", P3_DIR / "P3 Data 1 (augmented)", 2, 1, "P3D1_S2"),
    (P3_DIR / "P3 Data 1", P3_DIR / "P3 Data 1 (augmented)", 3, 1, "P3D1_S3"),
    (P3_DIR / "P3 Data 1", P3_DIR / "P3 Data 1 (augmented)", 4, 1, "P3D1_S4"),
    (P3_DIR / "P3 Data 2", P3_DIR / "P3 Data 2 (augmented)", 1, 0, "P3D2_S1"),
    (P3_DIR / "P3 Data 2", P3_DIR / "P3 Data 2 (augmented)", 2, 0, "P3D2_S2"),
]
VARIANTS = ["original", "aug-bright_up", "aug-bright_down", "aug-noise", "aug-compress", "swap"]


def get_video_paths(data_dir, data_dir_aug, scene_id, variant):
    """Tra ve (v1, v2, is_swapped)."""
    if variant == "original":
        return (data_dir / f"Scene {scene_id}-Cam 1.mp4",
                data_dir / f"Scene {scene_id}-Cam 2.mp4", False)
    if variant == "swap":
        return (data_dir / f"Scene {scene_id}-Cam 2.mp4",
                data_dir / f"Scene {scene_id}-Cam 1.mp4", True)
    return (data_dir_aug / f"Scene {scene_id}-Cam 1_{variant}.mp4",
            data_dir_aug / f"Scene {scene_id}-Cam 2_{variant}.mp4", False)


def read_all_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def detect_one_held(detector, frame, held_state, grace_frames, conf=DET_CONF):
    """Nhu detect_classify_pipeline.py: neu detector mat dau dot ngot thi giu
    tam bbox cu (ap len FRAME HIEN TAI) toi da grace_frames LAN GOI HAM lien
    tiep (khong phai frame video -- ham nay chi duoc goi moi FRAME_SKIP frame)
    truoc khi coi la that su mat nguoi. Dung .predict(), khong .track(), vi
    detector dung lai xuyen nhieu video khac nhau (xem comment GRACE_MS)."""
    results = detector(frame, classes=[0], conf=conf, verbose=False)[0]
    if results.boxes is not None and len(results.boxes) > 0:
        bbox = tuple(results.boxes[0].xyxy[0].cpu().numpy().tolist())
        held_state["bbox"] = bbox
        held_state["missed"] = 0
        return bbox
    if held_state["bbox"] is not None and held_state["missed"] < grace_frames:
        held_state["missed"] += 1
        return held_state["bbox"]
    held_state["bbox"] = None
    return None


def run_p3_fused(backbone, detector1, detector2, v1, v2, is_swapped, H_floor, H_by_height):
    """is_swapped: neu True, v1 that ra la Cam2 vat ly va v2 la Cam1 -- dung
    DUNG homography theo NOI DUNG hinh anh (khong theo ten bien), tuc la
    hoan doi ca vai tro trong cong thuc gop (kiem tra tinh doi xung)."""
    cam1_key, cam2_key = ("P3 Cam2", "P3 Cam1") if is_swapped else ("P3 Cam1", "P3 Cam2")
    cap1 = cv2.VideoCapture(str(v1))
    cap2 = cv2.VideoCapture(str(v2))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    # cua so 1.2s tinh theo frame THAT SU DUOC XU LY (moi FRAME_SKIP frame goc)
    grace_frames = max(1, round(GRACE_MS / 1000.0 * fps / FRAME_SKIP))
    held1 = {"bbox": None, "missed": 0}
    held2 = {"bbox": None, "missed": 0}
    per_frame = []
    frame_idx = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue
        bbox1 = detect_one_held(detector1, frame1, held1, grace_frames)
        bbox2 = detect_one_held(detector2, frame2, held2, grace_frames)
        if bbox1 is None or bbox2 is None:
            continue
        foot1 = apply_homography(H_floor[cam1_key], foot_point_from_bbox(bbox1))
        foot2 = apply_homography(H_floor[cam2_key], foot_point_from_bbox(bbox2))
        dist = np.linalg.norm(np.array(foot1) - np.array(foot2))
        if dist > DIST_THRESHOLD:
            continue
        crop1 = frame1[int(bbox1[1]):int(bbox1[3]), int(bbox1[0]):int(bbox1[2])]
        crop2 = frame2[int(bbox2[1]):int(bbox2[3]), int(bbox2[0]):int(bbox2[2])]
        if crop1.size == 0 or crop2.size == 0:
            continue
        x1 = preprocess_crop(crop1)
        x2 = preprocess_crop(crop2)
        with torch.no_grad():
            grid1 = backbone(x1)
            grid2 = backbone(x2)
            h, w = grid1.shape[2], grid1.shape[3]
            dist_1to2, dist_2to1 = build_distance_matrices(
                (h, w), foot1, foot2, H_by_height[cam1_key], H_by_height[cam2_key], bbox1, bbox2)
            v_fused = sgie_forward(backbone, x1, x2, has_both_cams=True,
                                    dist_1to2=dist_1to2, dist_2to1=dist_2to1, sigma=SIGMA)
            logits_fused = backbone.classify_from_vector(v_fused)
            probs_fused = torch.softmax(logits_fused, dim=1)[0]
        per_frame.append({
            "frame_idx": frame_idx, "bbox1": bbox1, "bbox2": bbox2,
            "label": CLASS_NAMES[int(probs_fused.argmax())],
            "conf": float(probs_fused.max()), "fused": True,
        })
    cap1.release()
    cap2.release()
    events = detect_fall_events_fused(per_frame, fps)
    return len(events) > 0, len(per_frame)


def run_espinosa(model, v1, v2):
    frames1, fps1 = read_all_frames(v1)
    frames2, fps2 = read_all_frames(v2)
    n = min(len(frames1), len(frames2))
    win_frames = int(WINDOW_S * fps1)
    stride_frames = max(1, int(STRIDE_S * fps1))
    mags1 = precompute_flow_magnitudes(frames1[:n])
    mags2 = precompute_flow_magnitudes(frames2[:n])
    n_fall = 0
    n_windows = 0
    start = 0
    while start + win_frames <= n:
        end = start + win_frames
        flow1 = flow_window_from_magnitudes(mags1, start, end - 1)
        flow2 = flow_window_from_magnitudes(mags2, start, end - 1)
        combined = combine_two_cams(flow1, flow2)
        x = torch.from_numpy(combined).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(x)
            pred = int(logits.argmax(dim=1).item())
        n_fall += pred
        n_windows += 1
        start += stride_frames
    return n_fall > 0, n_windows


def load_p3_calib():
    with open(RESULTS_P3 / "homography_matrices_p3.json", encoding="utf-8") as f:
        H_floor = {k: np.array(v) for k, v in json.load(f).items()}
    with open(RESULTS_P3 / "h_by_height_p3.json", encoding="utf-8") as f:
        raw = json.load(f)
    H_by_height = {cam: {float(h): np.array(H) for h, H in d.items()} for cam, d in raw.items()}
    return H_floor, H_by_height


def main():
    print("Nap model P3 (SharedBackbone) + Espinosa + 2 detector + calib P3...")
    backbone = SharedBackbone(CKPT_BACKBONE).eval()
    detector1 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    detector2 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    H_floor, H_by_height = load_p3_calib()
    espinosa = EspinosaCNN().to(DEVICE)
    espinosa.load_state_dict(torch.load(CKPT_ESPINOSA, map_location=DEVICE))
    espinosa.eval()

    rows = []
    for data_dir, data_dir_aug, scene_id, label, scene_name in BASE_SCENES:
        for variant in VARIANTS:
            v1, v2, is_swapped = get_video_paths(data_dir, data_dir_aug, scene_id, variant)
            if not v1.exists() or not v2.exists():
                print(f"[{scene_name}/{variant}] thieu file, bo qua")
                continue

            p3_pred, p3_n = run_p3_fused(backbone, detector1, detector2, v1, v2, is_swapped,
                                          H_floor, H_by_height)
            esp_pred, esp_n = run_espinosa(espinosa, v1, v2)

            print(f"[{scene_name}/{variant}] label={label} | "
                  f"P3={int(p3_pred)}({'DUNG' if int(p3_pred)==label else 'SAI'}, {p3_n}f) | "
                  f"Espinosa={int(esp_pred)}({'DUNG' if int(esp_pred)==label else 'SAI'}, {esp_n}w)")
            rows.append({
                "scene": scene_name, "variant": variant, "label": label,
                "p3_pred": int(p3_pred), "p3_correct": int(p3_pred) == label,
                "espinosa_pred": int(esp_pred), "espinosa_correct": int(esp_pred) == label,
            })

    df = pd.DataFrame(rows)
    out_csv = RESULTS_P3 / "p3_vs_espinosa_augmented.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    def report(sub_df, tag):
        tp_p3 = ((sub_df.p3_pred == 1) & (sub_df.label == 1)).sum()
        fn_p3 = ((sub_df.p3_pred == 0) & (sub_df.label == 1)).sum()
        tn_p3 = ((sub_df.p3_pred == 0) & (sub_df.label == 0)).sum()
        fp_p3 = ((sub_df.p3_pred == 1) & (sub_df.label == 0)).sum()
        tp_e = ((sub_df.espinosa_pred == 1) & (sub_df.label == 1)).sum()
        fn_e = ((sub_df.espinosa_pred == 0) & (sub_df.label == 1)).sum()
        tn_e = ((sub_df.espinosa_pred == 0) & (sub_df.label == 0)).sum()
        fp_e = ((sub_df.espinosa_pred == 1) & (sub_df.label == 0)).sum()
        print(f"\n--- {tag} (n={len(sub_df)}, Fall={sub_df.label.sum()}, ADL={len(sub_df)-sub_df.label.sum()}) ---")
        print(f"P3       : bat nga {tp_p3}/{sub_df.label.sum()} | Sens={tp_p3/(tp_p3+fn_p3):.2%} "
              f"Spec={tn_p3/(tn_p3+fp_p3+1e-9):.2%} Acc={(tp_p3+tn_p3)/len(sub_df):.2%}")
        print(f"Espinosa : bat nga {tp_e}/{sub_df.label.sum()} | Sens={tp_e/(tp_e+fn_e):.2%} "
              f"Spec={tn_e/(tn_e+fp_e+1e-9):.2%} Acc={(tp_e+tn_e)/len(sub_df):.2%}")

    print("\n" + "=" * 70)
    report(df[df.variant == "original"], "6 CANH GOC (doc lap, dang tin cay nhat)")
    report(df, "TOAN BO 36 LAN CHAY (robustness -- KHONG phai 36 su kien doc lap)")
    print("=" * 70)
    print(f"\nDa luu: {out_csv}")


if __name__ == "__main__":
    main()
