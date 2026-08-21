"""So sanh THAT: 'Round 1' -- ban classifier DAU TIEN nhom tung train (tren
v2_split_flat cu, 6 lop ke ca half_person, loss thuong, 120 epoch) vs ban
DANG DEPLOY (fine-tune tren v4_split_flat, 5 lop tu the, loss AFCL) -- do %
dung TREN CUNG bo anh val cua v4_split_flat (Round 1 CHUA TUNG thay bo data
nay vi no train tren v2 khac han -- van la test/held-out that, cong bang)."""
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

AI_ROOT = Path(__file__).resolve().parents[3]
VAL_DIR = AI_ROOT / "Datasets" / "File Traning" / "v4_split_flat" / "val"
DEPLOYED_CKPT = (AI_ROOT / "Coding" / "training" / "training 3" / "runs" / "classify"
                 / "Fall_Detection_Advanced_Loss" / "YOLOv8n_AFCL_Balanced-8" / "weights" / "best.pt")
ROUND1_CKPT = (AI_ROOT / "Coding" / "runs" / "classify" / "runs" / "classify"
               / "clsv2_1-14" / "weights" / "best.pt")

CLASSES = ["bend", "exercise", "lie", "sit", "stand"]


def long_path_imread(path: Path):
    """cv2.imread/PIL fail tren Windows voi duong dan > 260 ky tu (MAX_PATH) --
    van de da biet trong du an (ten file dataset qua dai). Mo file qua prefix
    \\\\?\\ (extended-length path, bo qua gioi han MAX_PATH cua Win32 API kieu
    cu) roi decode bang cv2, khong phu thuoc tinh nang Long Path he thong."""
    abs_path = str(path.resolve())
    win_long_path = "\\\\?\\" + abs_path if not abs_path.startswith("\\\\?\\") else abs_path
    with open(win_long_path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    return img


def evaluate(model, name):
    print(f"\n=== {name} ===")
    total, correct = 0, 0
    per_class = {c: {"n": 0, "correct": 0} for c in CLASSES}
    t0 = time.time()
    for cls_name in CLASSES:
        cls_dir = VAL_DIR / cls_name
        img_paths = sorted(cls_dir.glob("*"))
        for i in range(0, len(img_paths), 64):
            batch_paths = img_paths[i:i + 64]
            batch_imgs = [long_path_imread(p) for p in batch_paths]
            batch_imgs = [im for im in batch_imgs if im is not None]
            if not batch_imgs:
                continue
            results = model.predict(batch_imgs, verbose=False)
            for r in results:
                pred_name = r.names[int(r.probs.top1)]
                is_correct = pred_name == cls_name
                total += 1
                per_class[cls_name]["n"] += 1
                if is_correct:
                    correct += 1
                    per_class[cls_name]["correct"] += 1
        print(f"  {cls_name}: {per_class[cls_name]['correct']}/{per_class[cls_name]['n']} dung "
              f"({100*per_class[cls_name]['correct']/max(1,per_class[cls_name]['n']):.2f}%)")
    elapsed = time.time() - t0
    acc = 100 * correct / total
    print(f"  TONG: {correct}/{total} dung = {acc:.2f}% (het {elapsed:.0f}s)")
    return acc, total, correct


def main():
    print(f"Val dir: {VAL_DIR}")
    assert VAL_DIR.exists(), f"Khong tim thay: {VAL_DIR}"

    print("\nNap ban DANG DEPLOY (fine-tuned, AFCL, 5 lop)...")
    deployed = YOLO(str(DEPLOYED_CKPT))
    acc_deployed, n_deployed, c_deployed = evaluate(deployed, "DEPLOYED (fine-tuned, AFCL)")

    print("\nNap ban ROUND 1 (train tren v2_split_flat cu, loss thuong, 120 epoch)...")
    round1 = YOLO(str(ROUND1_CKPT))
    print(f"  Ten 6 lop cua Round 1 (theo dung thu tu index): {round1.names}")
    acc_r1, n_r1, c_r1 = evaluate(round1, "ROUND 1 (v2_split_flat, loss thuong)")

    print("\n" + "=" * 60)
    print("KET QUA CUOI CUNG (cung bo anh val v4_split_flat, "
          f"{n_deployed} anh):")
    print(f"  Round 1 (v2_split_flat cu): {acc_r1:.2f}% ({c_r1}/{n_r1})")
    print(f"  Deployed (fine-tuned AFCL): {acc_deployed:.2f}% ({c_deployed}/{n_deployed})")
    print("=" * 60)


if __name__ == "__main__":
    main()
