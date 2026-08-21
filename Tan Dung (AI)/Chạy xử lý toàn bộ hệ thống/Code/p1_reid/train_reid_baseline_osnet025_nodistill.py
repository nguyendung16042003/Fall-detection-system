"""
P1 -- Baseline OSNet_x0.25 KHONG Knowledge Distillation, dung DUNG quy trinh
trong "File Run Problem 1/Quy_trinh_OSNet_x025_Baseline_KhongDistill.docx":
  L_Total = 1.0*L_ID (CrossEntropy) + 1.0*L_triplet (margin=0.3)
  (KHONG co Teacher, KHONG co L_KD -- day la BIEN DUY NHAT khac voi
  train_reid_kd_osnet025.py, de phep so sanh co gia tri khoa hoc).

Moi thu khac GIU NGUYEN 100% so voi ban Distill (dataset/split, transform,
kien truc osnet_x0_25, khoi tao ImageNet pretrained, optimizer/scheduler,
early-stop, so epoch) -- dung theo checklist "so sanh cong bang" muc 11 cua
tai lieu guide.
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchreid.reid.data.transforms import RandomErasing
from torchreid.reid.losses import TripletLoss
from torchreid.reid.models import build_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from msmt17_dataset import MSMT17Dataset, PKSampler, MSMT17_DIR  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_osnet_train_transform():
    """Giong het train_reid_kd_osnet025.py."""
    return T.Compose([
        T.Resize((256, 128)),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((256, 128)),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        RandomErasing(probability=0.5, mean=IMAGENET_MEAN),
    ])


def get_osnet_eval_transform():
    return T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


HERE = Path(__file__).resolve().parent
AI_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STUDENT_OUT = HERE / "osnet_x0_25_baseline_msmt17.pth"
RESUME_CKPT = RESULTS_DIR / "osnet_x0_25_baseline_checkpoint_latest.pth"

P, K = 16, 4
BATCH_SIZE = P * K
EPOCHS = 120
LR = 0.00035
WEIGHT_DECAY = 5e-4
MARGIN = 0.3
W_ID, W_TRIPLET = 1.0, 1.0        # KHONG co W_KD -- khac duy nhat so voi ban Distill
MILESTONES = [40, 70]
GAMMA = 0.1
CKPT_EVERY = 5

PATIENCE = 10
MIN_EPOCHS = 30
BEST_STUDENT_OUT = HERE / "osnet_x0_25_baseline_msmt17_best.pt"


@torch.no_grad()
def eval_val_rank1(model, val_loader):
    model.eval()
    embeds, labels = [], []
    for imgs, lbls in val_loader:
        imgs = imgs.to(DEVICE)
        feat = model(imgs)
        feat = F.normalize(feat, p=2, dim=1)
        embeds.append(feat.cpu())
        labels.append(lbls)
    model.train()
    embeds = torch.cat(embeds)
    labels = torch.cat(labels)
    sim = embeds @ embeds.t()
    sim.fill_diagonal_(-1.0)
    nn_idx = sim.argmax(dim=1)
    correct = (labels[nn_idx] == labels).sum().item()
    return correct / len(labels)


def build_baseline(num_classes):
    model = build_model("osnet_x0_25", num_classes=num_classes, loss="triplet", pretrained=True)
    return model.to(DEVICE)


def main():
    print("Nap dataset MSMT17 train (giong het ban Distill: 256x128 + ImageNet normalize)...")
    train_ds = MSMT17Dataset(MSMT17_DIR / "list_train.txt", MSMT17_DIR / "train", get_osnet_train_transform())
    num_classes = len(set(train_ds.labels))
    print(f"  {len(train_ds)} anh, {num_classes} danh tinh")

    sampler = PKSampler(train_ds.labels, P=P, K=K)
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=2, pin_memory=True)

    print("Nap MSMT17 val split (list_val.txt, cho early stopping)...")
    val_ds = MSMT17Dataset(MSMT17_DIR / "list_val.txt", MSMT17_DIR / "train", get_osnet_eval_transform())
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    print(f"  {len(val_ds)} anh val")

    print("Nap model osnet_x0_25 (khoi tao ImageNet pretrained, KHONG Teacher)...")
    model = build_baseline(num_classes)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {n_params:,} tham so (trainable)")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_triplet = TripletLoss(margin=MARGIN)
    # AMP tat -- giong het ban Distill (da xac nhan crash CUBLAS tren GPU nay khi bat AMP)
    use_amp = False
    print(f"  Mixed precision (AMP): {use_amp} (tat, giong ban Distill de dong nhat dieu kien)")

    start_epoch = 1
    best_rank1 = -1.0
    patience_left = PATIENCE
    if RESUME_CKPT.exists():
        print(f"Tim thay checkpoint resume: {RESUME_CKPT} -- tiep tuc training...")
        ckpt = torch.load(RESUME_CKPT, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_rank1 = ckpt.get("best_rank1", -1.0)
        patience_left = ckpt.get("patience_left", PATIENCE)
        print(f"  Resume tu epoch {start_epoch}, best_rank1={best_rank1:.4f}, patience_left={patience_left}")

    model.train()
    t_start = time.time()
    for epoch in range(start_epoch, EPOCHS + 1):
        total_ce, total_tri, n_batches = 0.0, 0.0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            logits, feat = model(imgs)  # train mode, loss="triplet" -> (y, v)

            loss_ce = criterion_ce(logits, labels)
            loss_tri = criterion_triplet(feat, labels)
            total_loss = W_ID * loss_ce + W_TRIPLET * loss_tri

            total_loss.backward()
            optimizer.step()

            total_ce += loss_ce.item()
            total_tri += loss_tri.item()
            n_batches += 1

        scheduler.step()
        elapsed = time.time() - t_start
        avg_per_epoch = elapsed / (epoch - start_epoch + 1)
        eta_min = avg_per_epoch * (EPOCHS - epoch) / 60

        val_rank1 = eval_val_rank1(model, val_loader)
        improved = val_rank1 > best_rank1
        if improved:
            best_rank1 = val_rank1
            patience_left = PATIENCE
            torch.save(model.state_dict(), BEST_STUDENT_OUT)
        else:
            patience_left -= 1

        print(f"Epoch {epoch:3d}/{EPOCHS}  loss_ce={total_ce/n_batches:.4f}  "
              f"loss_triplet={total_tri/n_batches:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  val_rank1={val_rank1:.4f}"
              f"{' (BEST, da luu)' if improved else f' (khong cai thien, patience={patience_left}/{PATIENCE})'}"
              f"  elapsed={elapsed/60:.1f}min  ETA={eta_min:.1f}min")

        if epoch % CKPT_EVERY == 0 or epoch == EPOCHS:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_rank1": best_rank1,
                "patience_left": patience_left,
            }, RESUME_CKPT)
            torch.save(model.state_dict(), STUDENT_OUT)
            print(f"  [checkpoint] da luu {RESUME_CKPT.name} va {STUDENT_OUT.name}")

        if patience_left <= 0 and epoch >= MIN_EPOCHS:
            print(f"\nDUNG SOM: {PATIENCE} epoch lien tiep khong cai thien val_rank1 "
                  f"(tot nhat={best_rank1:.4f}, tai epoch {epoch - PATIENCE}), "
                  f"da qua MIN_EPOCHS={MIN_EPOCHS}.")
            break
        elif patience_left <= 0:
            print(f"  (early-stop dieu kien du nhung chua qua MIN_EPOCHS={MIN_EPOCHS}, tiep tuc train)")

    torch.save(model.state_dict(), STUDENT_OUT)
    print(f"\nDa luu model epoch cuoi: {STUDENT_OUT}")
    print(f"Da luu model TOT NHAT (val_rank1={best_rank1:.4f}): {BEST_STUDENT_OUT}")
    print(f"Tong thoi gian: {(time.time()-t_start)/60:.1f} phut")


if __name__ == "__main__":
    main()
