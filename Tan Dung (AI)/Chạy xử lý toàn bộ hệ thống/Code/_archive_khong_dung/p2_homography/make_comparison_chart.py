"""Ve bieu do so sanh P2: Greedy+chan vs Hungarian+dau, doi chieu voi
Hungarian+chan (da validate) qua 4 canh P2 Data 3."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AI_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p2"

df = pd.read_csv(RESULTS_DIR / "compare_p2_alternatives_summary.csv")
scene_labels = ["Cảnh 1\n(cách 1-1.5m)", "Cảnh 2\n(sát nhau)", "Cảnh 3\n(che khuất)",
                "Cảnh 4\n(A ngã, quan trọng nhất)"]

x = np.arange(len(df))
width = 0.32

COLOR_GREEDY = "#2E6F9E"
COLOR_HEAD = "#C4562B"

fig, ax = plt.subplots(figsize=(8.5, 5.2))
b1 = ax.bar(x - width / 2, df["greedy_foot_agree_pct"], width,
            label="(B) Greedy + điểm chân", color=COLOR_GREEDY)
b2 = ax.bar(x + width / 2, df["hungarian_head_agree_pct"], width,
            label="(C) Hungarian + điểm đầu\n(lấy cảm hứng Eshel & Moses 2008)", color=COLOR_HEAD)

for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9, color="#1f2937")

ax.axhline(100, color="#94A3B8", linewidth=1, linestyle="--")
ax.set_ylabel("% khớp với (A) Hungarian + điểm chân (đã validate)")
ax.set_ylim(0, 112)
ax.set_xticks(x)
ax.set_xticklabels(scene_labels)
ax.set_title("P2 — So sánh kỹ thuật thay thế trên P2 Data 3\n"
              "(mốc tham chiếu 100% = Hungarian + điểm chân, phương án đã chọn)")
ax.legend(loc="lower left", frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()

out_path = RESULTS_DIR / "compare_p2_alternatives_chart.png"
fig.savefig(out_path, dpi=150)
print("Da luu:", out_path)
