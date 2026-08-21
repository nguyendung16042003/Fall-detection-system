"""Bieu do tong ket 4 thi nghiem Giai doan A (Rank-1/mAP tren MSMT17) --
cho thay ca 4 deu thap hon nhieu moc tham khao OSNet, khong phai do thieu
epoch/cau hinh sai (da thu du huong hop ly)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

AI_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p1"

experiments = [
    "Đóng băng backbone\n(CE+Triplet, 60ep)",
    "Mở khoá backbone\n(20 epoch)",
    "Mở khoá backbone\n(60 epoch, đúng thiết kế gốc)",
    "Đóng băng backbone\n(Pure Triplet, 60ep)",
    "OSNet_x1.0\n(model Re-ID riêng,\ncheckpoint MSMT17 thật)",
]
rank1 = [4.88, 3.72, 3.48, 4.25, 75.90]
colors = ["#C4562B"] * 4 + ["#2E6F9E"]

x = np.arange(len(experiments))
fig, ax = plt.subplots(figsize=(9.5, 5.2))
bars = ax.bar(x, rank1, width=0.55, color=colors)
for bar, v in zip(bars, rank1):
    ax.annotate(f"{v:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, v),
                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)

ax.set_ylabel("Rank-1 (%) trên MSMT17 test split")
ax.set_ylim(0, 85)
ax.set_xticks(x)
ax.set_xticklabels(experiments, fontsize=9)
ax.set_title("P1 — 4 thí nghiệm backbone dùng chung vs. OSNet model Re-ID riêng\n"
              "(đo thật qua đúng 1 pipeline Rank-1/mAP, không trích dẫn literature)")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()

out_path = RESULTS_DIR / "p1_rank1_all_experiments_chart.png"
fig.savefig(out_path, dpi=150)
print("Da luu:", out_path)
