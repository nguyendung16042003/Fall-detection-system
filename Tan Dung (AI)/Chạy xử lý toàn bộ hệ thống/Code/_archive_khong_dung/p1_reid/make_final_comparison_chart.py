"""Bieu do tong ket P1: Re-ID Head tu train (backbone dung chung) vs OSNet
(checkpoint MSMT17 THAT, da sua loi ImageNet-only), tren CUNG data test that
(Test Tang 2, Gallery A/B, video Scene 5-CAM 2). Diem la diem KHOP CAO NHAT
(co the la voi A hoac B) -- mau xanh la/do the hien dung/sai nguoi."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

AI_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p1"

track_ids = ["track_id=11\n(A quay lại lần 1)", "track_id=14\n(A quay lại lần 2)"]
own_scores = [0.4493, 0.4382]
own_correct = [False, False]
osnet_scores = [0.7939, 0.7775]
osnet_predicted = ["A", "B"]
osnet_correct = [True, False]
threshold = 0.6

x = np.arange(len(track_ids))
width = 0.32

fig, ax = plt.subplots(figsize=(8, 5.2))
GREEN, RED = "#2E7D4F", "#C4562B"
b1 = ax.bar(x - width / 2, own_scores, width,
            color=[GREEN if c else RED for c in own_correct])
b2 = ax.bar(x + width / 2, osnet_scores, width,
            color=[GREEN if c else RED for c in osnet_correct])

for bars, preds in ((b1, ["NEW", "NEW"]), (b2, osnet_predicted)):
    for bar, pred in zip(bars, preds):
        h = bar.get_height()
        ax.annotate(f"{h:.3f}\n→{pred}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

ax.axhline(threshold, color="#475569", linewidth=1.5, linestyle="--")
ax.annotate(f"Ngưỡng khớp = {threshold}", xy=(1.35, threshold), xytext=(0, 4),
            textcoords="offset points", fontsize=9, color="#475569")

from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor=GREEN, label="Đoán ĐÚNG (A)"),
    Patch(facecolor=RED, label="Đoán SAI"),
]

ax.set_ylabel("Cosine similarity (điểm khớp cao nhất)")
ax.set_ylim(0, 1.0)
ax.set_xticks(x)
ax.set_xticklabels(track_ids)
ax.set_title("P1 Test Tầng 2 — cùng data thật (Enrollment A/B + Scene 5-CAM 2)\n"
              "Trái: Re-ID Head tự train (backbone dùng chung) — Phải: OSNet (checkpoint MSMT17 thật)\n"
              "True Positive Rate: model tự train 0% (0/2)  |  OSNet 50% (1/2)")
ax.legend(handles=legend_elems, loc="upper left", frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()

out_path = RESULTS_DIR / "p1_tang2_own_vs_osnet_chart.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print("Da luu:", out_path)
