"""
P1 -- Knowledge Distillation OSNet_x1.0 (Teacher, dong bang) -> OSNet_x0.25
(Student), dung DUNG quy trinh trong
"File Run Problem 1/Quy_trinh_Knowledge_Distillation_OSNet 1x - 0.25x.docx":
  L_Total = 1.0*L_ID (CrossEntropy) + 1.0*L_triplet (margin=0.3)
          + 0.5*L_KD (MSE giua embedding 512-d da L2-normalize cua Teacher/Student)
Optimizer Adam(lr=3.5e-4, weight_decay=5e-4), 120 epoch, LR giam o epoch 40/70
(MultiStepLR gamma=0.1) -- dung DUNG so nhu tai lieu de xuat.

Teacher: osnet_x1_0_msmt17.pth (co san trong repo, DA train MSMT17 -- dung lai,
KHONG train lai Teacher, dung nhu muc 3 tai lieu). Student: osnet_x0_25, khoi
tao pretrained=True (ImageNet, tu torchreid model zoo) roi finetune qua KD --
dung nhu muc 4 tai lieu ("Khoi tao tu dau hoac ImageNet weights").

Dataset: MSMT17 (list_train.txt, 30248 anh) -- dung LAI class MSMT17Dataset/
PKSampler cua msmt17_dataset.py (chi phan doc list+sampler, khong lien quan
transform). KHONG dung get_train_transform/get_eval_transform co san trong
file do -- 2 ham do dung classify_transforms() cua Ultralytics (224x224
vuong, KHONG chuan hoa mean/std), thiet ke rieng cho backbone YOLOv8n-cls
(train_reid_msmt17_unfrozen.py), SAI hoan toan voi OSNet chuan (can
Resize(256,128) + Normalize ImageNet). Dung DUNG transform theo tai lieu KD
muc 9 (T.Resize((256,128)), Pad+RandomCrop, ColorJitter, Normalize,
RandomErasing) va DUNG khop preprocessing mac dinh cua torchreid
FeatureExtractor (dung khi eval Rank-1/mAP that va khi trien khai P1 that) --
neu lech transform train/eval, embedding hoc duoc se KHONG dung nghia khi
dung qua FeatureExtractor (da xac nhan bang thuc nghiem: ban dau dung nham
classify_transforms() cho ra val_rank1 noi bo 72% nhung Rank-1 THAT tren bo
query/gallery chuan chi 0.24% -- gan random, vi FeatureExtractor luc infer
dung transform KHAC hoan toan luc train).
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
    """Dung DUNG tai lieu KD muc 9."""
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
    """Khop DUNG preprocessing mac dinh cua torchreid FeatureExtractor
    (Resize((256,128)) + ToTensor + Normalize, khong augment)."""
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
STUDENT_OUT = HERE / "osnet_x0_25_kd_msmt17.pth"          # state_dict sach, dung cho FeatureExtractor
RESUME_CKPT = RESULTS_DIR / "osnet_x0_25_kd_checkpoint_latest.pth"  # full checkpoint (resume)

P, K = 16, 4                 # batch = 64, dung theo tai lieu muc 8
BATCH_SIZE = P * K
EPOCHS = 120                 # dung theo tai lieu muc 8 (khong rut gon)
LR = 0.00035
WEIGHT_DECAY = 5e-4
MARGIN = 0.3
W_ID, W_TRIPLET, W_KD = 1.0, 1.0, 0.5   # dung DUNG cong thuc tai lieu muc 5
MILESTONES = [40, 70]        # dung theo tai lieu muc 8 ("giam dan LR o epoch 40, 70")
GAMMA = 0.1
CKPT_EVERY = 5

# Early stopping: MOI epoch do Rank-1 leave-one-out tren list_val.txt (2373
# anh, tach rieng khoi train/test, cung khong gian nhan train nen dung duoc
# ngay -- KHONG can bo query/gallery day du 94k anh moi lan, du nhanh de check
# moi epoch). Neu 10 epoch LIEN TIEP khong cai thien Rank-1 tot nhat -> dung
# som, luu lai checkpoint TOT NHAT (khong phai epoch cuoi). MIN_EPOCHS=30 --
# nguoi dung yeu cau train it nhat 30 epoch du early-stop co kich hoat som
# hon (KD can du epoch de Student "duoi kip" Teacher, dung early-stop qua som).
PATIENCE = 10
MIN_EPOCHS = 30
BEST_STUDENT_OUT = HERE / "osnet_x0_25_kd_msmt17_best.pt"


@torch.no_grad()
def eval_val_rank1(student, val_loader):
    """Leave-one-out Rank-1 tren list_val.txt: voi moi anh, tim anh gan nhat
    (cosine) trong SO CON LAI cua tap val, dung neu cung nhan."""
    student.eval()
    embeds, labels = [], []
    for imgs, lbls in val_loader:
        imgs = imgs.to(DEVICE)
        feat = student(imgs)  # eval mode -> (B,512)
        feat = F.normalize(feat, p=2, dim=1)
        embeds.append(feat.cpu())
        labels.append(lbls)
    student.train()
    embeds = torch.cat(embeds)          # (N,512)
    labels = torch.cat(labels)          # (N,)
    sim = embeds @ embeds.t()           # (N,N) cosine (da normalize)
    sim.fill_diagonal_(-1.0)            # loai chinh no
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


def build_student(num_classes):
    student = build_model("osnet_x0_25", num_classes=num_classes, loss="triplet", pretrained=True)
    return student.to(DEVICE)


def main():
    print("Nap dataset MSMT17 train (transform chuan OSNet: 256x128 + ImageNet normalize)...")
    train_ds = MSMT17Dataset(MSMT17_DIR / "list_train.txt", MSMT17_DIR / "train", get_osnet_train_transform())
    num_classes = len(set(train_ds.labels))
    print(f"  {len(train_ds)} anh, {num_classes} danh tinh")

    sampler = PKSampler(train_ds.labels, P=P, K=K)
    # num_workers giam tu 8 xuong 2 (+ tat persistent_workers/prefetch cao) --
    # may chi con ~3GB RAM trong, 8 worker x prefetch=4 vuot gioi han commit
    # cua Windows ("Couldn't open shared file mapping", error code 1455).
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=2, pin_memory=True)

    print("Nap MSMT17 val split (list_val.txt, cho early stopping)...")
    val_ds = MSMT17Dataset(MSMT17_DIR / "list_val.txt", MSMT17_DIR / "train", get_osnet_eval_transform())
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    print(f"  {len(val_ds)} anh val")

    print("Nap Teacher (osnet_x1_0, dong bang, dung checkpoint MSMT17 co san)...")
    teacher = build_teacher(num_classes)
    n_teacher = sum(p.numel() for p in teacher.parameters())
    print(f"  Teacher: {n_teacher:,} tham so (dong bang, khong train)")

    print("Nap Student (osnet_x0_25, khoi tao ImageNet pretrained)...")
    student = build_student(num_classes)
    n_student = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"  Student: {n_student:,} tham so (trainable)")

    optimizer = torch.optim.Adam(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_triplet = TripletLoss(margin=MARGIN)
    # AMP (fp16) TAT: da xac nhan 2 lan crash CUBLAS_STATUS_INTERNAL_ERROR /
    # "CUDA error: unknown error" deu xay ra trong duong fp16 tensor-core
    # (cublasLtMatmul/cublasGemmEx) tren GPU nay (RTX 5060 Laptop, kien truc
    # rat moi) -- khong on dinh voi ban PyTorch/cuBLAS hien cai. Chay fp32
    # thuan cham hon nhung on dinh; model nho (~740K tham so, batch 64,
    # anh 256x128) nen khong lo thieu VRAM.
    use_amp = False
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"  Mixed precision (AMP): {use_amp} (da tat vi khong on dinh tren GPU nay)")

    start_epoch = 1
    best_rank1 = -1.0
    patience_left = PATIENCE
    if RESUME_CKPT.exists():
        print(f"Tim thay checkpoint resume: {RESUME_CKPT} -- tiep tuc training...")
        ckpt = torch.load(RESUME_CKPT, map_location=DEVICE)
        student.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_rank1 = ckpt.get("best_rank1", -1.0)
        patience_left = ckpt.get("patience_left", PATIENCE)
        print(f"  Resume tu epoch {start_epoch}, best_rank1={best_rank1:.4f}, patience_left={patience_left}")

    student.train()
    t_start = time.time()
    for epoch in range(start_epoch, EPOCHS + 1):
        total_ce, total_tri, total_kd, n_batches = 0.0, 0.0, 0.0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with torch.no_grad():
                feat_teacher = teacher(imgs)  # eval mode -> (B,512)

            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits_student, feat_student = student(imgs)  # train mode, loss="triplet" -> (y, v)

                loss_ce = criterion_ce(logits_student.float(), labels)
                loss_tri = criterion_triplet(feat_student.float(), labels)

                feat_t_norm = F.normalize(feat_teacher.float(), p=2, dim=1)
                feat_s_norm = F.normalize(feat_student.float(), p=2, dim=1)
                loss_kd = F.mse_loss(feat_s_norm, feat_t_norm)

                total_loss = W_ID * loss_ce + W_TRIPLET * loss_tri + W_KD * loss_kd

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_ce += loss_ce.item()
            total_tri += loss_tri.item()
            total_kd += loss_kd.item()
            n_batches += 1

        scheduler.step()
        elapsed = time.time() - t_start
        avg_per_epoch = elapsed / (epoch - start_epoch + 1)
        eta_min = avg_per_epoch * (EPOCHS - epoch) / 60

        val_rank1 = eval_val_rank1(student, val_loader)
        improved = val_rank1 > best_rank1
        if improved:
            best_rank1 = val_rank1
            patience_left = PATIENCE
            torch.save(student.state_dict(), BEST_STUDENT_OUT)
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
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_rank1": best_rank1,
                "patience_left": patience_left,
            }, RESUME_CKPT)
            torch.save(student.state_dict(), STUDENT_OUT)
            print(f"  [checkpoint] da luu {RESUME_CKPT.name} va {STUDENT_OUT.name}")

        if patience_left <= 0 and epoch >= MIN_EPOCHS:
            print(f"\nDUNG SOM: {PATIENCE} epoch lien tiep khong cai thien val_rank1 "
                  f"(tot nhat={best_rank1:.4f}, tai epoch {epoch - PATIENCE}), "
                  f"da qua MIN_EPOCHS={MIN_EPOCHS}.")
            break
        elif patience_left <= 0:
            print(f"  (early-stop dieu kien du nhung chua qua MIN_EPOCHS={MIN_EPOCHS}, tiep tuc train)")

    torch.save(student.state_dict(), STUDENT_OUT)
    print(f"\nDa luu Student epoch cuoi: {STUDENT_OUT}")
    print(f"Da luu Student TOT NHAT (val_rank1={best_rank1:.4f}): {BEST_STUDENT_OUT}")
    print(f"Tong thoi gian: {(time.time()-t_start)/60:.1f} phut")


if __name__ == "__main__":
    main()
