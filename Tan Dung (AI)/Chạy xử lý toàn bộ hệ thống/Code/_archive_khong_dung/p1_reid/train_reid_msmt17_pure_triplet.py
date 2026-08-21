"""
Phuong an con lai chua thu (xem cuoc trao doi truoc): CE loss (phan loai
1041 lop) co the dang lam embedding overfit theo huong phan loai thay vi
khoang cach -- thu BO HAN CE loss, CHI dung Triplet Loss thuan (WEIGHT_X=0).
Dung backbone DONG BANG (nhanh nhat de thu, ~55 phut, va la ban co Rank-1
cao nhat trong 3 thi nghiem truoc -- 4.88%).
"""
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchreid.reid.losses import TripletLoss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reid_model import ReIDModel  # noqa: E402
from msmt17_dataset import MSMT17Dataset, PKSampler, get_train_transform, MSMT17_DIR  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
CKPT = (AI_ROOT / "Coding" / "training" / "training 3" / "runs" / "classify"
        / "Fall_Detection_Advanced_Loss" / "YOLOv8n_AFCL_Balanced-8" / "weights" / "best.pt")
RESULTS_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

P, K = 16, 4
EPOCHS = 60
LR = 3e-4
MARGIN = 0.3
STEP_SIZE = 20
OUT_CKPT = RESULTS_DIR / "reid_head_pure_triplet.pt"


def main():
    print("Nap dataset MSMT17 train...")
    transform = get_train_transform(224)
    train_ds = MSMT17Dataset(MSMT17_DIR / "list_train.txt", MSMT17_DIR / "train", transform)
    num_classes = len(set(train_ds.labels))

    sampler = PKSampler(train_ds.labels, P=P, K=K)
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=8, pin_memory=True,
                         persistent_workers=True, prefetch_factor=4)

    print("Nap model (backbone dong bang) -- CHI Triplet Loss, KHONG Cross-Entropy...")
    model = ReIDModel(CKPT, num_classes=num_classes, freeze_backbone=True).to(DEVICE)
    # bo qua nhanh classifier -- khong dua vao optimizer (tiet kiem, du forward
    # van chay qua no nhung khong anh huong gradient cua embed_fc/bn)
    trainable = [p for n, p in model.named_parameters()
                 if p.requires_grad and not n.startswith("classifier.")]

    optimizer = torch.optim.Adam(trainable, lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=0.1)
    triplet_loss = TripletLoss(margin=MARGIN)

    model.train()
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        total_loss_t, n_batches = 0.0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            embed, _logits = model(imgs)
            loss = triplet_loss(embed, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_t += loss.item()
            n_batches += 1

        scheduler.step()
        elapsed = time.time() - t_start
        print(f"Epoch {epoch:3d}/{EPOCHS}  loss_triplet={total_loss_t/n_batches:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  elapsed={elapsed/60:.1f}min")

        if epoch % 10 == 0 or epoch == EPOCHS:
            torch.save(model.state_dict(), OUT_CKPT)

    torch.save(model.state_dict(), OUT_CKPT)
    print(f"\nDa luu: {OUT_CKPT}")
    print(f"Tong thoi gian: {(time.time()-t_start)/60:.1f} phut")


if __name__ == "__main__":
    main()
