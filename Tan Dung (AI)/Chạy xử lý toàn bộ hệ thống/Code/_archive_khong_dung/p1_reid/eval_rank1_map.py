"""
P1 Test Tang 1 -- Rank-1/mAP tren MSMT17 test split (query/gallery chuan),
doi chieu moc literature OSNet_x1.0 ~74-78% Rank-1, ~52-55% mAP (theo
"Huong dan train test.pdf" muc 4). Day la benchmark TACH BIET hoan toan
khoi nha (closed-set voi hang nghin danh tinh la), KHONG so truc tiep voi
True Positive Rate cua Test Tang 2.

camid lay theo dung quy uoc torchreid.data.datasets.image.msmt17.MSMT17:
camid = int(path.split('_')[2]) - 1.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reid_model import ReIDModel  # noqa: E402
from msmt17_dataset import get_eval_transform, MSMT17_DIR  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
CKPT = (AI_ROOT / "Coding" / "training" / "training 3" / "runs" / "classify"
        / "Fall_Detection_Advanced_Loss" / "YOLOv8n_AFCL_Balanced-8" / "weights" / "best.pt")
RESULTS_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256

parser = argparse.ArgumentParser()
parser.add_argument("--reid-ckpt", default="reid_head_msmt17_frozen_backbone.pt")
parser.add_argument("--freeze-backbone", action="store_true")
parser.add_argument("--suffix", default="frozen_backbone")
args = parser.parse_args()


class MSMT17EvalDataset(Dataset):
    def __init__(self, list_file, img_root, transform):
        self.samples = []  # (path, pid, camid)
        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel_path, pid = line.split(" ")
                camid = int(rel_path.split("_")[2]) - 1
                self.samples.append((img_root / rel_path, int(pid), camid))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, pid, camid = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return tensor, pid, camid


@torch.no_grad()
def extract_all(model, loader, tag):
    embeds, pids, camids = [], [], []
    n_done = 0
    for imgs, p, c in loader:
        imgs = imgs.to(DEVICE)
        emb = model.inference_embedding(imgs)
        embeds.append(emb.cpu().numpy())
        pids.append(p.numpy())
        camids.append(c.numpy())
        n_done += len(p)
        if n_done % 2560 == 0:
            print(f"  [{tag}] {n_done} anh...")
    return np.concatenate(embeds), np.concatenate(pids), np.concatenate(camids)


def main():
    transform = get_eval_transform(224)
    query_ds = MSMT17EvalDataset(MSMT17_DIR / "list_query.txt", MSMT17_DIR / "test", transform)
    gallery_ds = MSMT17EvalDataset(MSMT17_DIR / "list_gallery.txt", MSMT17_DIR / "test", transform)
    print(f"Query: {len(query_ds)} anh | Gallery: {len(gallery_ds)} anh")

    num_train_classes = 1041  # phai khop voi luc train de load state_dict dung shape
    model = ReIDModel(CKPT, num_classes=num_train_classes,
                       freeze_backbone=args.freeze_backbone).to(DEVICE)
    state = torch.load(RESULTS_DIR / args.reid_ckpt, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    query_loader = DataLoader(query_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=8, pin_memory=True)
    gallery_loader = DataLoader(gallery_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=8, pin_memory=True)

    print("Trich embedding Query...")
    q_embeds, q_pids, q_camids = extract_all(model, query_loader, "query")
    print("Trich embedding Gallery...")
    g_embeds, g_pids, g_camids = extract_all(model, gallery_loader, "gallery")

    np.save(RESULTS_DIR / f"msmt17_query_embeds_{args.suffix}.npy", q_embeds)
    np.save(RESULTS_DIR / f"msmt17_gallery_embeds_{args.suffix}.npy", g_embeds)
    print(f"\nDa luu embedding (suffix={args.suffix}). "
          f"Chay eval_rank1_map_chunked.py --suffix {args.suffix} de tinh Rank-1/mAP "
          f"(tranh loi tran RAM cua torchreid.evaluate_rank ban day du).")


if __name__ == "__main__":
    main()
