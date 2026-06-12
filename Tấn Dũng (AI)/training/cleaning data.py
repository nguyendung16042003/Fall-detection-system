from pathlib import Path
from PIL import Image

# ==========================
# ĐƯỜNG DẪN DATASET
# ==========================

DATASET_ROOT = Path(
    r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Datasets\File Traning\v2_split_flat"
)

bad_files = []
total_images = 0

VALID_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
]

for split in ["train", "val"]:

    split_path = DATASET_ROOT / split

    for cls in split_path.iterdir():

        if not cls.is_dir():
            continue

        print(f"\nChecking {split}/{cls.name}")

        for img_path in cls.iterdir():

            if img_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            total_images += 1

            try:
                with Image.open(img_path) as img:

                    img.verify()

                    w, h = img.size

                    if w < 50 or h < 50:
                        bad_files.append(
                            (str(img_path), f"Too small ({w}x{h})")
                        )

            except Exception as e:

                bad_files.append(
                    (str(img_path), str(e))
                )

print("\n==========================")
print("AUDIT RESULT")
print("==========================")

print(f"Total images checked: {total_images}")
print(f"Bad files found: {len(bad_files)}")

if bad_files:

    print("\nBad files list:\n")

    for path, reason in bad_files:
        print(reason)
        print(path)
        print("-" * 50)

else:

    print("\nNo bad images found!")