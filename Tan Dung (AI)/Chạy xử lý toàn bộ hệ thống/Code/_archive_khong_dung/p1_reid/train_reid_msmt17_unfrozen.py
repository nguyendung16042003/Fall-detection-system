"""
P1 Giai doan A -- BAN 2, backbone MO KHOA (dung dung thiet ke goc trong
"Huong dan train test.pdf" muc 1+3: Giai doan A train CA backbone LAN Re-ID
Head tren MSMT17). Chay sau khi ban dong-bang-backbone (train_reid_msmt17.py)
cho Rank-1 qua thap (4.88%, xem Results/p1/msmt17_rank1_map_frozen_backbone.txt)
-- nguoi dung xac nhan chon huong nay.

QUAN TRONG: KHONG ghi de checkpoint Pose goc (best.pt) ma P3 dang dung/da
validate -- model nay khoi tao TU checkpoint do (transfer learning) nhung
luu ra file RIENG (reid_backbone_finetuned.pt). Neu sau nay muon dung 1
backbone DUY NHAT cho ca Pose+Re-ID that su (dung nhu kien truc ly tuong),
can chay tiep Giai doan B (fine-tune lai Pose Head tren backbone MOI nay) --
CHUA lam trong script nay, de quyet dinh rieng sau khi thay Rank-1 cai
thien co dang lam tiep khong.

LR backbone THAP HON head 10 lan (3e-5 vs 3e-4) -- backbone da co diem khoi
dau tot (khong phai random init), tranh "quen" dot ngot cac dac trung thap
tang da hoc duoc.
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
EPOCHS = 60  # DUNG THEO thiet ke goc "Huong dan train test.pdf" muc 3 (Giai
# doan A train ca backbone lan Re-ID Head, 60 epoch) -- ban rut gon 20 epoch
# truoc do hoi tu chua du (CE loss con 1.26, Rank-1 3.72% < ban dong bang),
# nguoi dung xac nhan uu tien dung thiet ke goc, chap nhan thoi gian dai hon.
# Co AMP (torch.autocast+GradScaler) de giam thoi gian thuc te (~0.25s/batch
# thay vi 0.41s/batch khong AMP) ma khong doi so epoch/cau hinh so voi goc.
LR_HEAD = 3e-4
LR_BACKBONE = 3e-5
MARGIN = 0.3
WEIGHT_T = 1.0
WEIGHT_X = 1.0
STEP_SIZE = 20
OUT_CKPT = RESULTS_DIR / "reid_backbone_finetuned.pt"


def main():
    print("Nap dataset MSMT17 train...")
    transform = get_train_transform(224)
    train_ds = MSMT17Dataset(MSMT17_DIR / "list_train.txt", MSMT17_DIR / "train", transform)
    num_classes = len(set(train_ds.labels))
    print(f"  {len(train_ds)} anh, {num_classes} danh tinh")

    sampler = PKSampler(train_ds.labels, P=P, K=K)
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=8, pin_memory=True,
                         persistent_workers=True, prefetch_factor=4)

    print("Nap model (backbone MO KHOA, khoi tao tu checkpoint Pose hien tai)...")
    model = ReIDModel(CKPT, num_classes=num_classes, freeze_backbone=False).to(DEVICE)
    n_backbone = sum(p.numel() for p in model.backbone.parameters())
    n_head = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("backbone."))
    print(f"  Backbone (mo khoa): {n_backbone:,} tham so | Head: {n_head:,} tham so")

    optimizer = torch.optim.Adam([
        {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("backbone.")], "lr": LR_HEAD},
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=0.1)
    triplet_loss = TripletLoss(margin=MARGIN)
    ce_loss = nn.CrossEntropyLoss()
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"  Mixed precision (AMP): {use_amp}")

    model.train()
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        total_loss_t, total_loss_x, n_batches = 0.0, 0.0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=use_amp):
                embed, logits = model(imgs)
                loss_t = triplet_loss(embed.float(), labels)
                loss_x = ce_loss(logits.float(), labels)
                loss = WEIGHT_T * loss_t + WEIGHT_X * loss_x

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss_t += loss_t.item()
            total_loss_x += loss_x.item()
            n_batches += 1

        scheduler.step()
        elapsed = time.time() - t_start
        print(f"Epoch {epoch:3d}/{EPOCHS}  loss_triplet={total_loss_t/n_batches:.4f}  "
              f"loss_ce={total_loss_x/n_batches:.4f}  lr_head={scheduler.get_last_lr()[1]:.2e}  "
              f"elapsed={elapsed/60:.1f}min")

        if epoch % 10 == 0 or epoch == EPOCHS:
            torch.save(model.state_dict(), OUT_CKPT)

    torch.save(model.state_dict(), OUT_CKPT)
    print(f"\nDa luu: {OUT_CKPT}")
    print(f"Tong thoi gian: {(time.time()-t_start)/60:.1f} phut")


if __name__ == "__main__":
    main()
