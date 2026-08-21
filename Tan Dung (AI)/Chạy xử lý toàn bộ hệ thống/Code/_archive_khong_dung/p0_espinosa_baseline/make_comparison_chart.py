"""Ve bieu do so sanh Espinosa et al. 2019 (paper goc, ban tai tao cung-domain
MCFD, ban tai tao khac-domain P2/P3) -- dung cho Muc 6 report."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

AI_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p0_espinosa"

groups = ["Sensitivity", "Specificity", "Accuracy"]
paper_goc = [97.95, 83.08, 95.64]        # Espinosa et al. 2019, tren UP-Fall (paper goc)
replica_mcfd = [71.4, 84.5, 82.8]        # ban tai tao cua nhom, test tren MCFD (cung domain train)
replica_p2p3 = [50.0, 44.4, 46.7]        # ban tai tao, test tren P2/P3 nhom tu quay (khac domain)

x = np.arange(len(groups))
width = 0.26

COLOR_PAPER = "#94A3B8"     # xam trung tinh -- so sanh tham khao, khong phai ket qua tu lam
COLOR_MCFD = "#2E6F9E"      # xanh -- ket qua tu train/test (cung domain)
COLOR_P2P3 = "#C4562B"      # cam -- ket qua tren data that cua nhom (khac domain, quan trong nhat)

fig, ax = plt.subplots(figsize=(8.5, 5.2))
b1 = ax.bar(x - width, paper_goc, width, label="Paper gốc (UP-Fall, 2 cam)", color=COLOR_PAPER)
b2 = ax.bar(x, replica_mcfd, width, label="Bản tái tạo -- test trên MCFD\n(cùng domain train)", color=COLOR_MCFD)
b3 = ax.bar(x + width, replica_p2p3, width, label="Bản tái tạo -- test trên P2/P3\n(data thật của nhóm)", color=COLOR_P2P3)

for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9, color="#1f2937")

ax.set_ylabel("%")
ax.set_ylim(0, 110)
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_title("Espinosa et al. 2019 -- paper gốc vs. bản tái tạo (Farneback + channel-stack,\n"
              "train trên MCFD, giả định do thiếu code/weight gốc)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False, fontsize=8.5)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()

out_path = RESULTS_DIR / "espinosa_comparison_chart.png"
fig.savefig(out_path, dpi=150)
print("Da luu:", out_path)
