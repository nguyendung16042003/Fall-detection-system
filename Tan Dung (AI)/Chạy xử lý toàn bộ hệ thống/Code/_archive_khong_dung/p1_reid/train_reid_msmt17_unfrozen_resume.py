"""
Train TIEP tu checkpoint reid_backbone_finetuned.pt (20 epoch dau, CE loss
con cao 1.26 -- chua hoi tu du, xem train_reid_msmt17_unfrozen.py va ket qua
Rank-1 3.72% thap hon ca ban dong-bang). LR schedule cu da giam sau (3e-6/
3e-7 o epoch 20) nen RESET ve LR ban dau thay vi tiep tuc giam them -- coi
day la "vong 2" cua cung 1 qua trinh train, khong phai fine-tune tiep voi LR
qua nho (se khong hoc them duoc gi dang ke).
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

P, K = 16, 4
EXTRA_EPOCHS = 28
LR_HEAD = 3e-4
LR_BACKBONE = 3e-5
MARGIN = 0.3
STEP_SIZE = 12
IN_CKPT = RESULTS_DIR / "reid_backbone_finetuned.pt"
OUT_CKPT = RESULTS_DIR / "reid_backbone_finetuned.pt"  # ghi de -- tiep tuc cung 1 checkpoint


def main():
    print("Nap dataset MSMT17 train...")
    transform = get_train_transform(224)
    train_ds = MSMT17Dataset(MSMT17_DIR / "list_train.txt", MSMT17_DIR / "train", transform)
    num_classes = len(set(train_ds.labels))

    sampler = PKSampler(train_ds.labels, P=P, K=K)
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=8, pin_memory=True,
                         persistent_workers=True, prefetch_factor=4)

    print(f"Nap tiep checkpoint: {IN_CKPT}")
    model = ReIDModel(CKPT, num_classes=num_classes, freeze_backbone=False).to(DEVICE)
    model.load_state_dict(torch.load(IN_CKPT, map_location=DEVICE))

    optimizer = torch.optim.Adam([
        {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("backbone.")], "lr": LR_HEAD},
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=0.1)
    triplet_loss = TripletLoss(margin=MARGIN)
    ce_loss = nn.CrossEntropyLoss()
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    t_start = time.time()
    for epoch in range(1, EXTRA_EPOCHS + 1):
        total_loss_t, total_loss_x, n_batches = 0.0, 0.0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=use_amp):
                embed, logits = model(imgs)
                loss_t = triplet_loss(embed.float(), labels)
                loss_x = ce_loss(logits.float(), labels)
                loss = loss_t + loss_x

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss_t += loss_t.item()
            total_loss_x += loss_x.item()
            n_batches += 1

        scheduler.step()
        elapsed = time.time() - t_start
        print(f"Epoch {epoch:3d}/{EXTRA_EPOCHS} (tong ~{20+epoch})  "
              f"loss_triplet={total_loss_t/n_batches:.4f}  loss_ce={total_loss_x/n_batches:.4f}  "
              f"lr_head={scheduler.get_last_lr()[1]:.2e}  elapsed={elapsed/60:.1f}min")

        if epoch % 10 == 0 or epoch == EXTRA_EPOCHS:
            torch.save(model.state_dict(), OUT_CKPT)

    torch.save(model.state_dict(), OUT_CKPT)
    print(f"\nDa luu: {OUT_CKPT}")
    print(f"Tong thoi gian vong nay: {(time.time()-t_start)/60:.1f} phut")


if __name__ == "__main__":
    main()
