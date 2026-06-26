from pathlib import Path
import pandas as pd

# ===== ĐỔI ĐƯỜNG DẪN =====
DATASET_ROOT = Path(
    r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tấn Dũng (AI)\Datasets\File Traning\v2_split_flat"
)

results = []

for split in ["train", "val"]:

    split_path = DATASET_ROOT / split

    for cls in sorted(split_path.iterdir()):

        if cls.is_dir():

            count = len(
                [
                    f
                    for f in cls.iterdir()
                    if f.suffix.lower()
                    in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
                ]
            )

            results.append(
                {
                    "Split": split,
                    "Class": cls.name,
                    "Images": count,
                }
            )

df = pd.DataFrame(results)

print("\n========== CLASS DISTRIBUTION ==========\n")
print(df)

print("\n========== TOTAL ==========\n")
print(df.groupby("Split")["Images"].sum())

df.to_csv("class_distribution.csv", index=False)

print("\nSaved: class_distribution.csv")