"""
Calib homography cho P3 Cam1 (huong ra hanh lang tim) -- video LIEN TUC 1
file (khac P2 Data Calib da cat san tung canh), nen tu dong phat hien 4 doan
DUNG YEN (ung voi 4 goc vuong A/B/C/D) qua toc do di chuyen diem chan giua
cac frame lien tiep.

Theo "Kich Ban P3.docx" (Data Calib Bo sung): nguoi calib cao 1.46m, o vuong
0.5m x 0.5m. Diem goc (0,0) DUOC DO QUY CHIEU CHUNG voi diem goc calib P2
(bep, Cam2 cua P3 = CAM 1 cua P2) -- nguoi dung xac nhan truc tiep -- nen
toa do san 2 camera P3 GHEP DUOC TRUC TIEP vao 1 he quy chieu, dung cho
epipolar cross-camera cua P3.

P3 Cam2 (bep) trong VI DU GOC cua nhom KHONG can calib lai -- dung nguyen
homography cua P2 CAM 1 (2 camera nay xac nhan la CUNG 1 camera vat ly,
khong doi vi tri). *** GIA DINH NAY CHI DUNG CHO PHONG SETUP CU CUA NHOM --
KIEM TRA LAI THUC TE PHONG MOI, neu 2 camera cua ban KHONG trung nhau thi
phai calib CA 2 camera rieng cho Boundary Feature Fusion (bo qua doan doc
lai RESULTS_IA ben duoi, tu tao ca H_floor va H_head cho camera thu 2 giong
cach lam voi camera thu 1 trong file nay). ***

*** SUA CAC DUONG DAN CO DANH DAU TODO BEN DUOI CHO DUNG PHONG THAT TRUOC
KHI CHAY. ***
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "identity_association"))
from homography import calibrate_from_points, foot_point_from_bbox, head_point_from_bbox, apply_homography  # noqa: E402

PERSON_DETECTOR = HERE.parents[2] / "1_Single_Camera_Pipeline" / "models" / "person_detector.pt"
CALIB_VIDEO = HERE / "calib_video_own_room.avi"  # TODO: doi thanh video calib that (1 file lien tuc, 4 doan dung yen)
# TODO: neu ap dung gia dinh "Cam2 = P2/Identity Association CAM 1" (xem doc
# tren) thi tro dung ve thu muc output cua identity_association/recalibrate_own_room.py:
RESULTS_IA = HERE.parents[1] / "identity_association" / "calib_output_own_room"
OUT_DIR = HERE / "calib_output_own_room"  # TODO: doi neu muon luu noi khac
OUT_DIR.mkdir(parents=True, exist_ok=True)

SQUARE_SIDE_M = 0.5
WORLD_POINTS = {
    0: (0.0, 0.0),
    1: (SQUARE_SIDE_M, 0.0),
    2: (SQUARE_SIDE_M, SQUARE_SIDE_M),
    3: (0.0, SQUARE_SIDE_M),
}
CALIBRATOR_HEIGHT_M = 1.46
DET_CONF = 0.3
SPEED_THRESHOLD_PX = 5.0  # duoi nguong nay coi la "dung yen"
MIN_SEGMENT_FRAMES = 15  # ~1s @ 15fps, loai doan qua ngan (nhieu/qua canh)


def extract_foot_head_trajectory(video_path, detector):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    foot_pts, head_pts = [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = detector(frame, classes=[0], conf=DET_CONF, verbose=False)[0]
        if results.boxes is not None and len(results.boxes) > 0:
            box = results.boxes[0].xyxy[0].cpu().numpy()
            foot_pts.append(foot_point_from_bbox(box))
            head_pts.append(head_point_from_bbox(box))
        else:
            foot_pts.append(None)
            head_pts.append(None)
    cap.release()
    return foot_pts, head_pts, fps


def find_stable_segments(foot_pts, n_expected=4):
    """Tra ve list (start_idx, end_idx) cho cac doan dung yen, theo dung THU
    TU xuat hien trong video (gia dinh khop thu tu A->B->C->D cua kich ban)."""
    valid_idx = [i for i, p in enumerate(foot_pts) if p is not None]
    speeds = {}
    for a, b in zip(valid_idx[:-1], valid_idx[1:]):
        pa, pb = np.array(foot_pts[a]), np.array(foot_pts[b])
        speeds[b] = np.linalg.norm(pb - pa)

    segments = []
    cur_start = None
    for i in valid_idx:
        is_slow = speeds.get(i, 0.0) < SPEED_THRESHOLD_PX
        if is_slow:
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None and i - cur_start >= MIN_SEGMENT_FRAMES:
                segments.append((cur_start, i))
            cur_start = None
    if cur_start is not None and valid_idx[-1] - cur_start >= MIN_SEGMENT_FRAMES:
        segments.append((cur_start, valid_idx[-1]))

    # Gop cac doan lien tiep co vi tri chan GAN GIONG NHAU (rung nhe khi dung
    # yen co the vuot nguong toc do trong 1-2 frame, tach nham 1 canh thanh 2)
    merged = []
    for s, e in segments:
        if merged:
            prev_s, prev_e = merged[-1]
            prev_pt = np.array(median_point_in_range(foot_pts, prev_s, prev_e))
            cur_pt = np.array(median_point_in_range(foot_pts, s, e))
            if np.linalg.norm(cur_pt - prev_pt) < 30.0:
                merged[-1] = (prev_s, e)
                continue
        merged.append((s, e))
    return merged


def median_point_in_range(pts, start, end):
    sub = [p for p in pts[start:end] if p is not None]
    arr = np.array(sub)
    return (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))


def main():
    print("Nap detector YOLOv8n...")
    detector = YOLO(str(PERSON_DETECTOR))

    print(f"Trich quy dao chan/dau tu {CALIB_VIDEO.name}...")
    foot_pts, head_pts, fps = extract_foot_head_trajectory(CALIB_VIDEO, detector)
    print(f"  {len(foot_pts)} frame, fps={fps}")

    segments = find_stable_segments(foot_pts, n_expected=4)
    print(f"\nPhat hien {len(segments)} doan dung yen:")
    for i, (s, e) in enumerate(segments):
        print(f"  Doan {i}: frame {s}-{e} ({s/fps:.1f}s - {e/fps:.1f}s), {e-s} frame")

    if len(segments) != 4:
        print(f"\nCANH BAO: ky vong 4 doan, phat hien {len(segments)} -- kiem tra lai "
              f"SPEED_THRESHOLD_PX/MIN_SEGMENT_FRAMES hoac xem lai video truoc khi tin ket qua.")
        return

    foot_pixel_points, head_pixel_points, world_points = [], [], []
    for i, (s, e) in enumerate(segments):
        foot_px = median_point_in_range(foot_pts, s, e)
        head_px = median_point_in_range(head_pts, s, e)
        foot_pixel_points.append(foot_px)
        head_pixel_points.append(head_px)
        world_points.append(WORLD_POINTS[i])
        print(f"  Doan {i}: foot_px={foot_px} head_px={head_px} -> world={WORLD_POINTS[i]}")

    H_floor = calibrate_from_points(foot_pixel_points, world_points)
    H_head = calibrate_from_points(head_pixel_points, world_points)

    print(f"\nHomography san (Z=0) P3 Cam1 (hanh lang):\n{H_floor}")
    print(f"Homography dinh dau (Z={CALIBRATOR_HEIGHT_M}m) P3 Cam1:\n{H_head}")

    errors = []
    for px, wd in zip(foot_pixel_points, world_points):
        proj = apply_homography(H_floor, px)
        err = np.linalg.norm(np.array(proj) - np.array(wd))
        errors.append(err)
    print(f"Sai so chieu lai san (m): {[round(e, 4) for e in errors]} (trung binh {np.mean(errors):.4f}m)")

    # Ghep voi P2 CAM 1 (P3 Cam2, cung 1 camera vat ly, khong calib lai --
    # nguoi dung xac nhan cung he quy chieu voi P3 Cam1)
    with open(RESULTS_IA / "homography_matrices.json", encoding="utf-8") as f:
        p2_floor = json.load(f)
    with open(RESULTS_IA / "homography_head_matrices.json", encoding="utf-8") as f:
        p2_head = json.load(f)

    out_floor = {"P3 Cam1": H_floor.tolist(), "P3 Cam2": p2_floor["CAM 1"]}
    out_head = {
        "P3 Cam1": {"height_m": CALIBRATOR_HEIGHT_M, "H": H_head.tolist()},
        "P3 Cam2": {"height_m": p2_head["height_m"], "H": p2_head["H"]["CAM 1"]},
    }

    with open(OUT_DIR / "homography_matrices_p3.json", "w", encoding="utf-8") as f:
        json.dump(out_floor, f, indent=2)
    with open(OUT_DIR / "homography_head_matrices_p3.json", "w", encoding="utf-8") as f:
        json.dump(out_head, f, indent=2)
    print(f"\nDa luu: {OUT_DIR / 'homography_matrices_p3.json'}")
    print(f"Da luu: {OUT_DIR / 'homography_head_matrices_p3.json'}")


if __name__ == "__main__":
    main()
