"""Do Rank-1/mAP THAT cua MSINet (checkpoint pretrained tai ve, KHONG tu
train) tren DUNG bo query/gallery MSMT17 chuan (11659/82161 anh) -- CUNG
protocol da dung cho cac model khac, de MSINet co so lieu do THAT (khong
phai trich dan paper) khi dua vao so sanh "Phuong phap 4 va backbone va
MSI"."""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from msmt17_dataset import MSMT17_DIR  # noqa: E402
from msinet_model import load_msinet, MSINET_TRANSFORM  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64


class MSMT17ImageDataset(Dataset):
    def __init__(self, list_file, img_root):
        self.samples = []
        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel_path, pid = line.split(" ")
                self.samples.append(str(img_root / rel_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        return MSINET_TRANSFORM(img)


def extract(model, dataset, tag):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    embeds = []
    n_done = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            feat = model(batch)
            feat = torch.nn.functional.normalize(feat, dim=1)
            embeds.append(feat.cpu().numpy())
            n_done += batch.shape[0]
            if (n_done // BATCH_SIZE) % 20 == 0:
                print(f"  [{tag}] {n_done}/{len(dataset)} anh...")
    embeds = np.concatenate(embeds).astype(np.float32)
    return embeds


def main():
    print(f"Device: {DEVICE}")
    query_ds = MSMT17ImageDataset(MSMT17_DIR / "list_query.txt", MSMT17_DIR / "test")
    gallery_ds = MSMT17ImageDataset(MSMT17_DIR / "list_gallery.txt", MSMT17_DIR / "test")
    print(f"Query: {len(query_ds)} anh | Gallery: {len(gallery_ds)} anh")

    print("Nap MSINet (checkpoint pretrained tai ve)...")
    model = load_msinet(device=DEVICE)
    model.eval()

    for ds, tag in [(query_ds, "query"), (gallery_ds, "gallery")]:
        embeds = extract(model, ds, tag)
        np.save(RESULTS_DIR / f"msmt17_{tag}_embeds_msinet.npy", embeds)
        print(f"  Da luu msmt17_{tag}_embeds_msinet.npy, shape={embeds.shape}")

    print("\nDa luu embedding (suffix=msinet). Chay eval_rank1_map_chunked.py --suffix msinet de tinh Rank-1/mAP.")


if __name__ == "__main__":
    main()
