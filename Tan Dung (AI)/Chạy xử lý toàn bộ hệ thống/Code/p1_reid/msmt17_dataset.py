"""
Dataset + P-K sampler cho MSMT17 (list_train.txt: "path label" moi dong, label
da la so nguyen 0-indexed lien tuc -- xem kiem tra truoc). Dung transform
CUNG kieu voi luc infer that (classify_transforms cua Ultralytics, xem
shared_backbone.preprocess_crop) de dam bao dac trung nhat quan giua
train/test/khi trien khai.
"""
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

AI_ROOT = Path(__file__).resolve().parents[3]
MSMT17_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "File Run Problem 1" / "Data train" / "MSMT17_V1"


class MSMT17Dataset(Dataset):
    def __init__(self, list_file, img_root, transform):
        self.samples = []  # (path, label)
        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel_path, label = line.rsplit(" ", 1)
                self.samples.append((img_root / rel_path, int(label)))
        self.transform = transform
        self.labels = [lb for _, lb in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return tensor, label


class PKSampler(Sampler):
    """Moi batch: P danh tinh, moi danh tinh K anh (batch_size = P*K). Bat
    buoc cho Triplet Loss co du cap positive/negative trong 1 batch."""

    def __init__(self, labels, P=16, K=4, n_batches=None):
        self.P = P
        self.K = K
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)
        self.labels_unique = list(self.label_to_indices.keys())
        self.n_batches = n_batches or (len(labels) // (P * K))

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            chosen_labels = random.sample(
                self.labels_unique, min(self.P, len(self.labels_unique)))
            for label in chosen_labels:
                indices = self.label_to_indices[label]
                if len(indices) >= self.K:
                    chosen = random.sample(indices, self.K)
                else:
                    chosen = [random.choice(indices) for _ in range(self.K)]
                batch.extend(chosen)
            yield batch

    def __len__(self):
        return self.n_batches


def get_train_transform(imgsz=224):
    from ultralytics.data.augment import classify_transforms
    import torchvision.transforms as T
    base = classify_transforms(size=imgsz)
    # them augmentation nhe (flip) truoc base transform -- Re-ID chuan can
    # random_flip toi thieu (theo huong dan muc 2), giu resize/crop nhu infer that
    return T.Compose([T.RandomHorizontalFlip(p=0.5), base])


def get_eval_transform(imgsz=224):
    from ultralytics.data.augment import classify_transforms
    return classify_transforms(size=imgsz)
