"""
Bo sung cho P3 (chua co san trong thiet ke P2 goc): ho homography theo nhieu
muc chieu cao gia dinh (0-2m), dung de xap xi duong epipolar trong
"Context xu ly van de 3 (no_training).md" muc 3.1/4.

Cach lam: KHONG doan he so tuy tien. Dung 2 mat phang DA CALIB THAT tu chinh
video "P2 Data Calib":
  - H_floor (Z=0m): tu diem cham chan cua nguoi dung o 4 diem A/B/C/D.
  - H_head  (Z=1.77m): tu diem dinh dau CUNG nguoi do tai CUNG 4 diem --
    nguoi quay calib la "A" (ao den), chieu cao thuc te 1.77m (xac nhan qua
    anh + Kich ban P1/P2.docx).
Voi 2 mat phang that o 2 do cao khac nhau, NOI SUY TUYEN TINH tung phan tu
ma tran 3x3 giua H_floor va H_head cho cac muc h trong (0, 1.77), va NGOAI
SUY tuyen tinh (cung huong) cho h > 1.77m (toi da 2.0m theo ASSUMED_HEIGHTS_M).
Day la xap xi hop ly hon nhieu so voi doan 1 hang so "scale_per_meter" khong
co can cu -- van la xap xi (khong phai fundamental matrix that), nhung dua
tren 2 diem du lieu THAT thay vi 1 gia dinh tuy y.
"""
import json
from pathlib import Path

import numpy as np

ASSUMED_HEIGHTS_M = np.linspace(0.0, 2.0, num=8)


def interpolate_homography(H_floor, H_head, height_floor, height_head, target_h):
    """Noi suy/ngoai suy tuyen tinh tung phan tu ma tran giua 2 homography
    da calib that o 2 do cao khac nhau."""
    t = (target_h - height_floor) / (height_head - height_floor)
    return H_floor + t * (H_head - H_floor)


def build_h_by_height(H_floor, H_head, calibrator_height_m, assumed_heights_m=ASSUMED_HEIGHTS_M):
    """
    H_floor, H_head: np.array 3x3, da calib that (xem calibrate.py).
    calibrator_height_m: do cao that cua H_head (nguoi calib, vd 1.77m).
    Tra ve dict {height_m: H_h (3x3 np.array)}.
    """
    H_by_height = {}
    for h in assumed_heights_m:
        h = float(h)
        if h == 0.0:
            H_by_height[h] = H_floor.copy()
        elif h == calibrator_height_m:
            H_by_height[h] = H_head.copy()
        else:
            H_by_height[h] = interpolate_homography(H_floor, H_head, 0.0, calibrator_height_m, h)
    return H_by_height


def load_and_build(results_dir: Path, cam_name: str):
    """Tien ich: doc truc tiep tu homography_matrices.json +
    homography_head_matrices.json (output cua recalibrate_own_room.py),
    tra ve H_by_height cho 1 camera."""
    with open(results_dir / "homography_matrices.json", "r", encoding="utf-8") as f:
        H_floor_all = json.load(f)
    with open(results_dir / "homography_head_matrices.json", "r", encoding="utf-8") as f:
        head_data = json.load(f)

    H_floor = np.array(H_floor_all[cam_name])
    H_head = np.array(head_data["H"][cam_name])
    calibrator_height_m = head_data["height_m"]
    return build_h_by_height(H_floor, H_head, calibrator_height_m)


if __name__ == "__main__":
    # Doc dung thu muc output cua recalibrate_own_room.py (chay file do TRUOC).
    RESULTS_DIR = Path(__file__).resolve().parent / "calib_output_own_room"

    all_cams_H_by_height = {}
    for cam_name in ["CAM 1", "CAM 2"]:
        H_by_height = load_and_build(RESULTS_DIR, cam_name)
        all_cams_H_by_height[cam_name] = {str(h): H.tolist() for h, H in H_by_height.items()}
        print(f"{cam_name}: da tinh H_by_height cho {len(H_by_height)} muc chieu cao "
              f"({list(H_by_height.keys())})")

    out_path = RESULTS_DIR / "h_by_height.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_cams_H_by_height, f, indent=2)
    print(f"Da luu: {out_path}")
