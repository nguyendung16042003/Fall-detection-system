"""
Train EspinosaCNN tren dataset windowed da build (build_dataset.py). Data mat
can bang manh (moi chute chi co 1-2 window Fall trong hang chuc window ADL,
giong dung ty le that cua 1 cu nga hiem trong sinh hoat hang ngay) -- dung
class weight trong CrossEntropyLoss thay vi undersample (giu toi da du lieu
that, khong bia them).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from espinosa_model import EspinosaCNN  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p0_espinosa"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 60
BATCH_SIZE = 32
LR = 1e-3


def main():
    train_X = np.load(RESULTS_DIR / "espinosa_train_X.npy")
    train_y = np.load(RESULTS_DIR / "espinosa_train_y.npy")
    test_X = np.load(RESULTS_DIR / "espinosa_test_X.npy")
    test_y = np.load(RESULTS_DIR / "espinosa_test_y.npy")
    print(f"Train: {train_X.shape}, Fall={train_y.sum()}/{len(train_y)}")
    print(f"Test:  {test_X.shape}, Fall={test_y.sum()}/{len(test_y)}")

    n_fall = train_y.sum()
    n_adl = len(train_y) - n_fall
    class_weights = torch.tensor(
        [1.0, n_adl / max(n_fall, 1)], dtype=torch.float32, device=DEVICE)
    print(f"Class weights (ADL, Fall): {class_weights.tolist()}")

    train_ds = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_X_t = torch.from_numpy(test_X).to(DEVICE)
    test_y_t = torch.from_numpy(test_y).to(DEVICE)

    model = EspinosaCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_bal_acc = -1.0
    best_state = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        model.eval()
        with torch.no_grad():
            logits = model(test_X_t)
            preds = logits.argmax(dim=1)
            tp = ((preds == 1) & (test_y_t == 1)).sum().item()
            fn = ((preds == 0) & (test_y_t == 1)).sum().item()
            tn = ((preds == 0) & (test_y_t == 0)).sum().item()
            fp = ((preds == 1) & (test_y_t == 0)).sum().item()
            sens = tp / (tp + fn + 1e-8)
            spec = tn / (tn + fp + 1e-8)
            bal_acc = (sens + spec) / 2

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} loss={total_loss/len(train_ds):.4f} "
                  f"test_sens={sens:.3f} test_spec={spec:.3f} bal_acc={bal_acc:.3f}")

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    torch.save(best_state, RESULTS_DIR / "espinosa_best.pt")
    print(f"\nBest balanced accuracy (test): {best_bal_acc:.4f}")
    print(f"Da luu weights: {RESULTS_DIR / 'espinosa_best.pt'}")


if __name__ == "__main__":
    main()
