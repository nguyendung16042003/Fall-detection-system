"""
Do Rank-1/mAP THAT cua Student OSNet_x0.25 (sau Knowledge Distillation tu
Teacher OSNet_x1.0, xem train_reid_kd_osnet025.py) tren DUNG bo query/gallery
MSMT17 chuan (11659/82161 anh) -- CUNG protocol da dung cho OSNet_x0.5
(Rank-1 71.14%, mAP 41.99%, xem msmt17_rank1_map_osnet_x0_5.txt) de so sanh
cong bang. Dung lai eval_rank1_map_chunked.py cho buoc xep hang (--suffix
osnet_x0_25_kd).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchreid.reid.utils import FeatureExtractor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from msmt17_dataset import MSMT17_DIR  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256

_parser = argparse.ArgumentParser()
_parser.add_argument("--ckpt", default="osnet_x0_25_kd_msmt17_best.pt")
_parser.add_argument("--suffix", default="osnet_x0_25_kd")
_parser.add_argument("--model-name", default="osnet_x0_25")
_args = _parser.parse_args()
CKPT_PATH = Path(__file__).resolve().parent / _args.ckpt


class MSMT17RawDataset(Dataset):
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

    print(f"Nap Student osnet_x0_25 (sau KD, {CKPT_PATH.name})...")
    extractor = FeatureExtractor(model_name=_args.model_name, model_path=str(CKPT_PATH), device=DEVICE)

    for ds, tag in [(query_ds, "query"), (gallery_ds, "gallery")]:
        paths = [s[0] for s in ds.samples]
        embeds = []
        for start in range(0, len(paths), BATCH_SIZE):
            batch_paths = paths[start:start + BATCH_SIZE]
            emb = extractor(batch_paths)
            emb = torch.nn.functional.normalize(emb, dim=1)
            embeds.append(emb.cpu().numpy())
            if (start // BATCH_SIZE) % 10 == 0:
                print(f"  [{tag}] {start + len(batch_paths)}/{len(paths)} anh...")
        embeds = np.concatenate(embeds).astype(np.float32)
        np.save(RESULTS_DIR / f"msmt17_{tag}_embeds_{_args.suffix}.npy", embeds)

    print(f"\nDa luu embedding (suffix={_args.suffix}). "
          f"Chay eval_rank1_map_chunked.py --suffix {_args.suffix} de tinh Rank-1/mAP.")


if __name__ == "__main__":
    main()
