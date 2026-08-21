"""
Do that Rank-1/mAP cua OSNet_x1.0 qua DUNG bo query/gallery MSMT17 -- de co
con so doi chung that (khong chi trich dan literature) truoc khi chot OSNet
la giai phap thay the (xem "Xu ly van de 1 moi.md" cua Dung). Dung lai
eval_rank1_map_chunked.py de tinh Rank-1/mAP tu embedding da trich.

QUAN TRONG: FeatureExtractor KHONG dua model_path se CHI nap trong so
ImageNet (khong phai model Re-ID that) -- da phat hien loi nay khi debug.
Dung dung checkpoint osnet_x1_0 DA TRAIN+TEST TREN MSMT17 (same-domain,
Rank-1 74.9%/mAP 43.8% theo MODEL_ZOO chinh thuc cua deep-person-reid),
tai ve osnet_x1_0_msmt17.pth (link Google Drive tu MODEL_ZOO.md).
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchreid.reid.utils import FeatureExtractor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from msmt17_dataset import MSMT17_DIR  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p1"
import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--variant", default="osnet_x1_0")
_args = _parser.parse_args()
OSNET_NAME = _args.variant
OSNET_CKPT = Path(__file__).resolve().parent / f"{OSNET_NAME}_msmt17.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256


class MSMT17RawDataset(Dataset):
    """Anh RGB tho (resize 256x128 chuan ReID, KHONG qua classify_transforms
    cua Ultralytics vi OSNet dung transform rieng cua no)."""

    def __init__(self, list_file, img_root):
        self.samples = []
        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel_path, pid = line.split(" ")
                camid = int(rel_path.split("_")[2]) - 1
                self.samples.append((str(img_root / rel_path), int(pid), camid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def main():
    query_ds = MSMT17RawDataset(MSMT17_DIR / "list_query.txt", MSMT17_DIR / "test")
    gallery_ds = MSMT17RawDataset(MSMT17_DIR / "list_gallery.txt", MSMT17_DIR / "test")
    print(f"Query: {len(query_ds)} anh | Gallery: {len(gallery_ds)} anh")

    print(f"Nap {OSNET_NAME} DA TRAIN TREN MSMT17 ({OSNET_CKPT.name})...")
    extractor = FeatureExtractor(model_name=OSNET_NAME, model_path=str(OSNET_CKPT), device=DEVICE)

    for ds, tag in [(query_ds, "query"), (gallery_ds, "gallery")]:
        paths = [s[0] for s in ds.samples]
        pids = np.array([s[1] for s in ds.samples])
        camids = np.array([s[2] for s in ds.samples])

        embeds = []
        for start in range(0, len(paths), BATCH_SIZE):
            batch_paths = paths[start:start + BATCH_SIZE]
            emb = extractor(batch_paths)  # (B, 512), da tren DEVICE
            emb = torch.nn.functional.normalize(emb, dim=1)
            embeds.append(emb.cpu().numpy())
            if (start // BATCH_SIZE) % 10 == 0:
                print(f"  [{tag}] {start + len(batch_paths)}/{len(paths)} anh...")
        embeds = np.concatenate(embeds).astype(np.float32)

        np.save(RESULTS_DIR / f"msmt17_{tag}_embeds_{OSNET_NAME}.npy", embeds)

    print(f"\nDa luu embedding OSNet (suffix={OSNET_NAME}). "
          f"Chay eval_rank1_map_chunked.py --suffix {OSNET_NAME} de tinh Rank-1/mAP.")


if __name__ == "__main__":
    main()
