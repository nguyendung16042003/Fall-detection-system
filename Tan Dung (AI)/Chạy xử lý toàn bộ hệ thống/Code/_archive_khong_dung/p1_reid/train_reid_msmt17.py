"""
P1 Giai doan A -- Train Re-ID Head tren MSMT17 (Triplet + Cross-Entropy,
P-K sampling), theo "Huong dan train test.pdf" muc 3, Cach 1 muc 5 (dung
thang backbone YOLOv8n-cls that, KHONG DCNv2/EMA, input 224x224).

Backbone DONG BANG (xem ly do trong reid_model.py) -- chi Re-ID head
(embed_fc + BNNeck + classifier) duoc train.
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchreid.reid.losses import TripletLoss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reid_model import ReIDModel  # noqa: E402
from msmt17_dataset import MSMT17Dataset, PKSampler, get_train_transform, MSMT17_DIR  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
CKPT = (AI_ROOT / "Coding" / "training" / "training 3" / "runs" / "classify"
        / "Fall_Detection_Advanced_Loss" / "YOLOv8n_AFCL_Balanced-8" / "weights" / "best.pt")
RESULTS_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

P, K = 16, 4
BATCH_SIZE = P * K
EPOCHS = 60
LR = 3e-4
MARGIN = 0.3
WEIGHT_T = 1.0
WEIGHT_X = 1.0
STEP_SIZE = 20


def main():
    print("Nap dataset MSMT17 train...")
    transform = get_train_transform(224)
    train_ds = MSMT17Dataset(MSMT17_DIR / "list_train.txt", MSMT17_DIR / "train", transform)
    num_classes = len(set(train_ds.labels))
    print(f"  {len(train_ds)} anh, {num_classes} danh tinh")

    sampler = PKSampler(train_ds.labels, P=P, K=K)
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=8, pin_memory=True,
                         persistent_workers=True, prefetch_factor=4)

    print("Nap model (backbone dong bang, tu checkpoint Pose hien tai)...")
    model = ReIDModel(CKPT, num_classes=num_classes, freeze_backbone=True).to(DEVICE)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  So tham so Re-ID head duoc train: {n_trainable:,}")

    optimizer = torch.optim.Adam(trainable, lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=0.1)
    triplet_loss = TripletLoss(margin=MARGIN)
    ce_loss = nn.CrossEntropyLoss()

    model.train()
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        total_loss_t, total_loss_x, n_batches = 0.0, 0.0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            embed, logits = model(imgs)
            loss_t = triplet_loss(embed, labels)
            loss_x = ce_loss(logits, labels)
            loss = WEIGHT_T * loss_t + WEIGHT_X * loss_x

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_t += loss_t.item()
            total_loss_x += loss_x.item()
            n_batches += 1

        scheduler.step()
        elapsed = time.time() - t_start
        print(f"Epoch {epoch:3d}/{EPOCHS}  loss_triplet={total_loss_t/n_batches:.4f}  "
              f"loss_ce={total_loss_x/n_batches:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
              f"elapsed={elapsed/60:.1f}min")

        if epoch % 10 == 0 or epoch == EPOCHS:
            torch.save(model.state_dict(), RESULTS_DIR / "reid_head_msmt17.pt")

    torch.save(model.state_dict(), RESULTS_DIR / "reid_head_msmt17.pt")
    print(f"\nDa luu: {RESULTS_DIR / 'reid_head_msmt17.pt'}")
    print(f"Tong thoi gian: {(time.time()-t_start)/60:.1f} phut")


if __name__ == "__main__":
    main()
