"""Ve bieu do tong hop ket qua (thay cho bang CSV) -- luu PNG vao Test 4/charts/."""
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "charts"
OUT_DIR.mkdir(exist_ok=True)

# Mau categorical (dat theo thu tu co dinh, khong doi mau theo entity)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d6cf"


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors=MUTED, labelsize=10)


def bar_with_labels(ax, x, values, color, width, offset=0, fmt="{:.0f}%"):
    bars = ax.bar([xi + offset for xi in x], values, width=width, color=color, zorder=3)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, fmt.format(v),
                 ha="center", va="bottom", fontsize=9.5, color=TEXT)
    return bars


# --- Chart 1: So sanh Model cua nhom vs PIFR replica (32 video) ---
metrics = ["Accuracy", "Precision", "Recall", "F1"]
model_vals = [84.4, 60.0, 50.0, 54.5]
pifr_vals = [75.0, 40.0, 66.7, 50.0]

fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
x = range(len(metrics))
width = 0.35
bar_with_labels(ax, x, model_vals, BLUE, width, offset=-width / 2)
bar_with_labels(ax, x, pifr_vals, ORANGE, width, offset=width / 2)
ax.set_xticks(list(x))
ax.set_xticklabels(metrics)
ax.set_ylim(0, 100)
ax.set_ylabel("%", color=MUTED)
style_ax(ax)
ax.set_title("Model của nhóm vs PIFR replica (32 video: 6 Fall MCFD + 26 NoFall URFD)",
             fontsize=11, color=TEXT, pad=14)
handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
ax.legend(handles, ["Model của nhóm", "PIFR replica"], frameon=False, loc="upper right", fontsize=9.5)
fig.tight_layout()
fig.savefig(OUT_DIR / "so_sanh_model_vs_pifr.png", facecolor="white")
plt.close(fig)

# --- Chart 2: Recall 1 camera vs Da camera (OR-fusion) tren MCFD ---
labels = ["1 camera\n(cam1 duy nhất)", "Đa camera\n(OR-fusion, seed 42)", "Đa camera\n(OR-fusion, seed 7)"]
recall_vals = [50.0, 100.0, 100.0]

fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
colors = [BLUE, AQUA, AQUA]
bars = ax.bar(labels, recall_vals, color=colors, width=0.55, zorder=3)
for b, v in zip(bars, recall_vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%",
             ha="center", va="bottom", fontsize=10.5, color=TEXT, fontweight="bold")
ax.set_ylim(0, 110)
ax.set_ylabel("Recall (%)", color=MUTED)
style_ax(ax)
ax.set_title("Recall phát hiện ngã: 1 camera vs đa camera (MCFD)", fontsize=11.5, color=TEXT, pad=14)
fig.tight_layout()
fig.savefig(OUT_DIR / "recall_1cam_vs_dacam.png", facecolor="white")
plt.close(fig)

print("Da luu 2 bieu do vao", OUT_DIR)
