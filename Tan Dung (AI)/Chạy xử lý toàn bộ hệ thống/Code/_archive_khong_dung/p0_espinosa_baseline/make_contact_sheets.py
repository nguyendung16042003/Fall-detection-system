"""
Vi khong tim duoc annotation chinh thuc (frame bat dau/ket thuc cu nga) cong
khai cho MCFD, va heuristic tu dong (dinh nang luong chuyen dong) cho ket qua
sai nhieu (kiem tra bang mat thay dinh thuong roi vao luc di lai/don do, khong
phai luc nga -- xem annotate_mcfd_falls.py), chuyen sang GAN NHAN THU CONG
bang mat qua contact sheet (luoi anh nho co so frame) -- moi chute 1 anh de
xem nhanh, xac dinh khoang frame co canh nga.

Theo tai lieu chinh thuc (technicalReport.pdf, Auvinet et al.): chute 1-22 co
nga + confounding events, chute 23-24 CHI co confounding events (KHONG nga).
"""
from pathlib import Path

import cv2
import numpy as np

AI_ROOT = Path(__file__).resolve().parents[3]
MCFD_DIR = AI_ROOT / "Datasets" / "File Test 2" / "MCFD" / "dataset"
OUT_DIR = Path(
    r"C:\Users\ADMIN\AppData\Local\Temp\claude\d--DOWLOAD-FileTaiLieuHocTapCuaDung-Ki9-----n-Fall-detection-system-Fall-detection-system-Tan-Dung--AI-\c85b318a-c9ee-44d9-921e-d40c735bfdbb\scratchpad\mcfd_contact_sheets"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_TILES = 24  # 6x4 grid
GRID_COLS = 6
TILE_W, TILE_H = 160, 107


def make_sheet(chute: str, cam: str = "cam1"):
    video_path = MCFD_DIR / chute / f"{cam}.avi"
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_ids = np.linspace(0, total - 1, N_TILES).astype(int)
    tiles = []
    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (TILE_W, TILE_H))
        t_s = fid / fps
        cv2.putText(frame, f"f{fid} {t_s:.1f}s", (3, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        tiles.append(frame)
    cap.release()

    rows = []
    for r in range(0, N_TILES, GRID_COLS):
        rows.append(np.hstack(tiles[r:r + GRID_COLS]))
    grid = np.vstack(rows)

    out_path = OUT_DIR / f"{chute}_{cam}_sheet.png"
    cv2.imwrite(str(out_path), grid)
    return out_path, total, fps


def main():
    for i in range(1, 23):  # chi chute 1-22 (co nga theo tai lieu chinh thuc)
        chute = f"chute{i:02d}"
        out_path, total, fps = make_sheet(chute)
        print(f"[{chute}] {total} frame @ {fps}fps -> {out_path}")


if __name__ == "__main__":
    main()
