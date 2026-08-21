"""
Dung H_by_height cho DUNG 2 camera cua P3 (Cam1 hanh lang + Cam2 bep) -- khac
homography_by_height.py cua P2 o cho: 2 camera co CALIBRATOR_HEIGHT_M KHAC
NHAU (Cam1=1.46m nguoi calib rieng, Cam2=1.77m nguoi calib P2) nen khong
dung chung 1 "height_m" o cap cao nhat nhu file P2 -- doc rieng tung camera.
"""
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p2_homography"))
from homography_by_height import build_h_by_height  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
RESULTS_P3 = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p3"


def main():
    with open(RESULTS_P3 / "homography_matrices_p3.json", encoding="utf-8") as f:
        H_floor_all = json.load(f)
    with open(RESULTS_P3 / "homography_head_matrices_p3.json", encoding="utf-8") as f:
        head_all = json.load(f)

    all_cams_H_by_height = {}
    for cam_name in ["P3 Cam1", "P3 Cam2"]:
        H_floor = np.array(H_floor_all[cam_name])
        H_head = np.array(head_all[cam_name]["H"])
        calibrator_height_m = head_all[cam_name]["height_m"]
        H_by_height = build_h_by_height(H_floor, H_head, calibrator_height_m)
        all_cams_H_by_height[cam_name] = {str(h): H.tolist() for h, H in H_by_height.items()}
        print(f"{cam_name}: H_by_height cho {len(H_by_height)} muc chieu cao "
              f"(calibrator_height={calibrator_height_m}m)")

    out_path = RESULTS_P3 / "h_by_height_p3.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_cams_H_by_height, f, indent=2)
    print(f"Da luu: {out_path}")


if __name__ == "__main__":
    main()
