"""
MCFD (Multiple Cameras Fall Dataset) trong repo nay CHI co nhan o muc "ca
chute co xay ra 1 lan nga" (theo Note.docx + cach dung truoc do trong
evaluate_pipeline_multicam.py), KHONG co moc thoi gian nga chinh xac (frame
bat dau/ket thuc) -- ma Espinosa can nhan THEO TUNG CUA SO 1 GIAY.

De co nhan cua so ma KHONG bia dat va KHONG dung chinh model cua nhom (tranh
thien vi vong khi dung lam doi thu so sanh), script nay tu dong do "nang
luong chuyen dong" (frame differencing thuan tuy, khong phai optical flow day
du -- nhanh hon nhieu de quet toan bo 24 chute) tren cam1 cua moi chute, tim
diem dinh (spike) chuyen dong manh nhat -- dac trung cua 1 cu nga (chuyen
dong manh dot ngot roi dung yen), gan nhan cua so [dinh-0.5s, dinh+0.5s] la
"Fall", con lai la "ADL".

Day la GIA DINH/UOC LUONG (khong phai nhan tay chuan xac tu paper goc MCFD) --
ghi ro trong bao cao, va co luu anh dinh diem chuyen dong de kiem tra bang
mat (sanity check) truoc khi dung de train.
"""
import json
from pathlib import Path

import cv2
import numpy as np

AI_ROOT = Path(__file__).resolve().parents[3]
MCFD_DIR = AI_ROOT / "Datasets" / "File Test 2" / "MCFD" / "dataset"
RESULTS_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p0_espinosa"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_CHUTES = 24
CAM_FOR_DETECTION = "cam1"
SCAN_STRIDE = 2  # doc 1/2 frame de quet nhanh hon (120fps du day du)
WINDOW_S = 1.0
MARGIN_FRAC = 0.05  # bo qua 5% dau/cuoi video (tranh nhieu luc bat dau quay)


def motion_energy_series(video_path: Path, stride: int = SCAN_STRIDE):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    energies = []
    frame_indices = []
    prev_gray = None
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 107))
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                energies.append(float(diff.mean()))
                frame_indices.append(idx)
            prev_gray = gray
        idx += 1
    cap.release()
    return np.array(energies), np.array(frame_indices), fps, idx


def find_fall_window(energies, frame_indices, fps, total_frames):
    n = len(energies)
    lo = int(n * MARGIN_FRAC)
    hi = int(n * (1 - MARGIN_FRAC))
    if hi <= lo:
        lo, hi = 0, n
    peak_local = int(np.argmax(energies[lo:hi]))
    peak_idx = lo + peak_local
    peak_frame = int(frame_indices[peak_idx])
    peak_energy = float(energies[peak_idx])

    baseline = float(np.median(energies))
    # kiem tra "sau dinh la yen tinh hon truoc dinh nhieu" -- dac trung cua nga
    after = energies[peak_idx:min(peak_idx + n // 6, n)]
    before = energies[max(0, peak_idx - n // 6):peak_idx]
    after_mean = float(after.mean()) if len(after) else 0.0
    before_mean = float(before.mean()) if len(before) else 0.0
    fall_signature = after_mean < before_mean  # yen di sau dinh

    half_win_frames = int(WINDOW_S * fps / 2)
    win_start = max(0, peak_frame - half_win_frames)
    win_end = min(total_frames - 1, peak_frame + half_win_frames)

    return {
        "peak_frame": peak_frame,
        "peak_time_s": round(peak_frame / fps, 2),
        "peak_energy": round(peak_energy, 3),
        "baseline_energy": round(baseline, 3),
        "peak_to_baseline_ratio": round(peak_energy / (baseline + 1e-6), 2),
        "fall_signature_ok": bool(fall_signature),
        "window_start_frame": win_start,
        "window_end_frame": win_end,
        "fps": fps,
        "total_frames": total_frames,
    }


def main():
    all_annotations = {}
    for i in range(1, N_CHUTES + 1):
        chute = f"chute{i:02d}"
        video_path = MCFD_DIR / chute / f"{CAM_FOR_DETECTION}.avi"
        if not video_path.exists():
            print(f"[{chute}] KHONG thay video, bo qua")
            continue
        energies, frame_indices, fps, total_frames = motion_energy_series(video_path)
        if len(energies) == 0:
            print(f"[{chute}] khong doc duoc frame")
            continue
        info = find_fall_window(energies, frame_indices, fps, total_frames)
        all_annotations[chute] = info
        flag = "OK" if info["fall_signature_ok"] else "NGHI NGO (khong yen di sau dinh)"
        print(f"[{chute}] peak={info['peak_time_s']}s ratio={info['peak_to_baseline_ratio']}x "
              f"tong={total_frames}f @{fps}fps -- {flag}")

    out_path = RESULTS_DIR / "mcfd_fall_annotations.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_annotations, f, ensure_ascii=False, indent=2)
    print(f"\nDa luu: {out_path}")

    n_ok = sum(1 for v in all_annotations.values() if v["fall_signature_ok"])
    print(f"So chute co dac trung nga ro rang (yen di sau dinh): {n_ok}/{len(all_annotations)}")


if __name__ == "__main__":
    main()
