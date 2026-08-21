"""
P1 -- Phuong an 4 (Teacher Assistant), GIAI DOAN 1: distill OSNet_x1.0
(Teacher, dong bang, checkpoint co san) -> OSNet_x0.5 (Teacher Assistant,
TA) -- dung DUNG code mau trong
"File Run Problem 1/PhuongAn4_Teacher_Assistant.docx" muc 2: kd_weight=0.5
CO DINH (khong warm-up), giong het cau truc train_reid_kd_osnet025.py nhung
Student la osnet_x0_5 thay vi osnet_x0_25.

Capacity gap giai doan nay (x1.0 -> x0.5, ~2 lan) nho hon nhieu so voi di
thang x1.0 -> x0.25 (~4 lan) -- muc dich la de KD phat huy tac dung tot hon
o buoc trung gian nay, roi dung TA lam Teacher moi cho Giai doan 2
(train_reid_kd_pa4_stage2_final.py).
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
from torchreid.reid.utils import load_pretrained_weights

sys.path.insert(0, str(Path(__file__).resolve().parent))
from msmt17_dataset import MSMT17Dataset, PKSampler, MSMT17_DIR  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_osnet_train_transform():
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

TEACHER_CKPT = HERE / "osnet_x1_0_msmt17.pth"
TA_OUT = HERE / "teacher_assistant_osnet_x0_5_msmt17.pth"
RESUME_CKPT = RESULTS_DIR / "teacher_assistant_osnet_x0_5_checkpoint_latest.pth"
BEST_TA_OUT = HERE / "teacher_assistant_osnet_x0_5_msmt17_best.pt"

P, K = 16, 4
BATCH_SIZE = P * K
EPOCHS = 120
LR = 0.00035
WEIGHT_DECAY = 5e-4
MARGIN = 0.3
W_ID, W_TRIPLET, W_KD = 1.0, 1.0, 0.5   # kd_weight=0.5 co dinh, dung code mau guide
MILESTONES = [40, 70]
GAMMA = 0.1
CKPT_EVERY = 5

PATIENCE = 10
MIN_EPOCHS = 30


@torch.no_grad()
def eval_val_rank1(student, val_loader):
    student.eval()
    embeds, labels = [], []
    for imgs, lbls in val_loader:
        imgs = imgs.to(DEVICE)
        feat = student(imgs)
        feat = F.normalize(feat, p=2, dim=1)
        embeds.append(feat.cpu())
        labels.append(lbls)
    student.train()
    embeds = torch.cat(embeds)
    labels = torch.cat(labels)
    sim = embeds @ embeds.t()
    sim.fill_diagonal_(-1.0)
    nn_idx = sim.argmax(dim=1)
    correct = (labels[nn_idx] == labels).sum().item()
    return correct / len(labels)


def build_teacher(num_classes):
    teacher = build_model("osnet_x1_0", num_classes=num_classes, loss="softmax", pretrained=False)
    load_pretrained_weights(teacher, str(TEACHER_CKPT))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher.to(DEVICE)


def build_ta_student(num_classes):
    ta = build_model("osnet_x0_5", num_classes=num_classes, loss="triplet", pretrained=True)
    return ta.to(DEVICE)


def main():
    print("Nap dataset MSMT17 train...")
    train_ds = MSMT17Dataset(MSMT17_DIR / "list_train.txt", MSMT17_DIR / "train", get_osnet_train_transform())
    num_classes = len(set(train_ds.labels))
    print(f"  {len(train_ds)} anh, {num_classes} danh tinh")

    sampler = PKSampler(train_ds.labels, P=P, K=K)
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=2, pin_memory=True)

    print("Nap MSMT17 val split...")
    val_ds = MSMT17Dataset(MSMT17_DIR / "list_val.txt", MSMT17_DIR / "train", get_osnet_eval_transform())
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    print(f"  {len(val_ds)} anh val")

    print("Nap Teacher (osnet_x1_0, dong bang)...")
    teacher = build_teacher(num_classes)

    print("Nap Teacher Assistant Student (osnet_x0_5, khoi tao ImageNet pretrained)...")
    ta = build_ta_student(num_classes)
    n_ta = sum(p.numel() for p in ta.parameters() if p.requires_grad)
    print(f"  TA (x0.5): {n_ta:,} tham so (trainable), KD_WEIGHT={W_KD} co dinh")

    optimizer = torch.optim.Adam(ta.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_triplet = TripletLoss(margin=MARGIN)

    start_epoch = 1
    best_rank1 = -1.0
    patience_left = PATIENCE
    if RESUME_CKPT.exists():
        print(f"Tim thay checkpoint resume: {RESUME_CKPT} -- tiep tuc training...")
        ckpt = torch.load(RESUME_CKPT, map_location=DEVICE)
        ta.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_rank1 = ckpt.get("best_rank1", -1.0)
        patience_left = ckpt.get("patience_left", PATIENCE)
        print(f"  Resume tu epoch {start_epoch}, best_rank1={best_rank1:.4f}, patience_left={patience_left}")

    ta.train()
    t_start = time.time()
    for epoch in range(start_epoch, EPOCHS + 1):
        total_ce, total_tri, total_kd, n_batches = 0.0, 0.0, 0.0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with torch.no_grad():
                feat_teacher = teacher(imgs)

            logits_ta, feat_ta = ta(imgs)

            loss_ce = criterion_ce(logits_ta, labels)
            loss_tri = criterion_triplet(feat_ta, labels)

            feat_t_norm = F.normalize(feat_teacher, p=2, dim=1)
            feat_ta_norm = F.normalize(feat_ta, p=2, dim=1)
            loss_kd = F.mse_loss(feat_ta_norm, feat_t_norm)

            total_loss = W_ID * loss_ce + W_TRIPLET * loss_tri + W_KD * loss_kd

            total_loss.backward()
            optimizer.step()

            total_ce += loss_ce.item()
            total_tri += loss_tri.item()
            total_kd += loss_kd.item()
            n_batches += 1

        scheduler.step()
        elapsed = time.time() - t_start
        avg_per_epoch = elapsed / (epoch - start_epoch + 1)
        eta_min = avg_per_epoch * (EPOCHS - epoch) / 60

        val_rank1 = eval_val_rank1(ta, val_loader)
        improved = val_rank1 > best_rank1
        if improved:
            best_rank1 = val_rank1
            patience_left = PATIENCE
            torch.save(ta.state_dict(), BEST_TA_OUT)
        else:
            patience_left -= 1

        print(f"Epoch {epoch:3d}/{EPOCHS}  loss_ce={total_ce/n_batches:.4f}  "
              f"loss_triplet={total_tri/n_batches:.4f}  loss_kd={total_kd/n_batches:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  val_rank1={val_rank1:.4f}"
              f"{' (BEST, da luu)' if improved else f' (khong cai thien, patience={patience_left}/{PATIENCE})'}"
              f"  elapsed={elapsed/60:.1f}min  ETA={eta_min:.1f}min")

        if epoch % CKPT_EVERY == 0 or epoch == EPOCHS:
            torch.save({
                "epoch": epoch,
                "model_state_dict": ta.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_rank1": best_rank1,
                "patience_left": patience_left,
            }, RESUME_CKPT)
            torch.save(ta.state_dict(), TA_OUT)
            print(f"  [checkpoint] da luu {RESUME_CKPT.name} va {TA_OUT.name}")

        if patience_left <= 0 and epoch >= MIN_EPOCHS:
            print(f"\nDUNG SOM: {PATIENCE} epoch lien tiep khong cai thien val_rank1 "
                  f"(tot nhat={best_rank1:.4f}), da qua MIN_EPOCHS={MIN_EPOCHS}.")
            break
        elif patience_left <= 0:
            print(f"  (early-stop dieu kien du nhung chua qua MIN_EPOCHS={MIN_EPOCHS}, tiep tuc train)")

    torch.save(ta.state_dict(), TA_OUT)
    print(f"\nDa luu TA epoch cuoi: {TA_OUT}")
    print(f"Da luu TA TOT NHAT (val_rank1={best_rank1:.4f}): {BEST_TA_OUT}")
    print(f"Tong thoi gian: {(time.time()-t_start)/60:.1f} phut")


if __name__ == "__main__":
    main()
