"""Sinh cac hinh minh hoa nho (PNG, nen trong suot) cho slide Marp -- tranh hoan
toan viec nhung raw SVG/HTML phuc tap vao markdown (markdown-it escape mat, da
xac nhan qua thu nghiem). Mau sac khop dung palette lay tu Slide chinh.pptx
(xem _slide_preview/, sample qua PIL)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

NAVY = "#13284a"
ORANGE = "#f5a623"
BLUE = "#3e6aa8"
GRAY = "#888888"
INK = "#1a1a2e"

OUT = Path(__file__).resolve().parent


def floor_projection_diagram():
    fig, ax = plt.subplots(figsize=(5.6, 1.5), dpi=200)
    ax.set_xlim(0, 560); ax.set_ylim(0, 150); ax.axis("off")
    ax.invert_yaxis()

    for x, label in [(60, "Camera 1"), (460, "Camera 2")]:
        ax.add_patch(plt.Rectangle((x, 20), 34, 60, fill=False, edgecolor=NAVY, linewidth=1.8))
        ax.text(x - 30, 35, label, fontsize=9, color=NAVY, fontweight="bold")

    ax.plot([20, 540], [120, 120], color=GRAY, linewidth=1.2)
    ax.text(545, 124, "ground plane", fontsize=8, color=GRAY, style="italic")

    ax.plot(230, 120, "o", color=ORANGE, markersize=6)
    ax.text(222, 138, "$g_1$", fontsize=9, color=INK)
    ax.plot(300, 120, "o", color=ORANGE, markersize=6)
    ax.text(292, 138, "$g_2$", fontsize=9, color=INK)

    ax.plot([77, 230], [80, 118], color=BLUE, linewidth=1.2, linestyle="--")
    ax.plot([477, 300], [80, 118], color=BLUE, linewidth=1.2, linestyle="--")
    ax.annotate("", xy=(295, 112), xytext=(235, 112),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, linewidth=1.4))
    ax.text(215, 100, r"$C[i,j] = \|g_1 - g_2\|$", fontsize=9, color=NAVY)

    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "diagram_floor_projection.png", transparent=True)
    plt.close(fig)


def classifier_bench_fps_only():
    """Ban rut gon: CHI bieu do FPS (bo bieu do params rieng), so tham so ghi
    kem ngay duoi truc x -- gon hon cho slide da bi phan anh 'qua dai'."""
    import pandas as pd
    df = pd.read_csv(OUT / "benchmark_lightweight_classifiers.csv")
    short_names = {
        "YOLOv8n-cls (deployed, fine-tuned)": "YOLOv8n-cls\n1.44M params",
        "MobileNetV3-Small (ImageNet, chua fine-tune)": "MobileNetV3-Small\n2.54M params",
        "ShuffleNetV2 x1.0 (ImageNet, chua fine-tune)": "ShuffleNetV2 x1.0\n2.28M params",
        "EfficientNet-B0 (ImageNet, chua fine-tune)": "EfficientNet-B0\n5.29M params",
        "SqueezeNet1.1 (ImageNet, chua fine-tune)": "SqueezeNet1.1\n1.24M params",
    }
    df["short"] = df["model"].map(short_names)
    colors = [ORANGE if "YOLOv8n" in m else NAVY for m in df["model"]]

    fig, ax = plt.subplots(figsize=(6.2, 3.0), dpi=200)
    bars = ax.bar(df["short"], df["fps_cpu_devmachine"], color=colors, width=0.6)
    ax.set_ylabel("FPS (CPU, dev machine)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.tick_params(axis="y", labelsize=8)
    for b, v in zip(bars, df["fps_cpu_devmachine"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.0f}", ha="center", fontsize=9,
                fontweight="bold", color=INK)
    fig.tight_layout(pad=0.5)
    fig.savefig(OUT / "chart_classifier_fps_only.png", transparent=True)
    plt.close(fig)


def classifier_bench_charts():
    import pandas as pd
    df = pd.read_csv(OUT / "benchmark_lightweight_classifiers.csv")
    short_names = {
        "YOLOv8n-cls (deployed, fine-tuned)": "YOLOv8n-cls",
        "MobileNetV3-Small (ImageNet, chua fine-tune)": "MobileNetV3-Small",
        "ShuffleNetV2 x1.0 (ImageNet, chua fine-tune)": "ShuffleNetV2 x1.0",
        "EfficientNet-B0 (ImageNet, chua fine-tune)": "EfficientNet-B0",
        "SqueezeNet1.1 (ImageNet, chua fine-tune)": "SqueezeNet1.1",
    }
    df["short"] = df["model"].map(short_names)
    colors = [ORANGE if "YOLOv8n" in m else NAVY for m in df["model"]]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), dpi=200)

    ax = axes[0]
    bars = ax.bar(df["short"], df["fps_cpu_devmachine"], color=colors, width=0.6)
    ax.set_title("Inference speed (FPS, CPU, dev machine)", fontsize=10, color=INK, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8, rotation=28)
    ax.tick_params(axis="y", labelsize=8)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    for b, v in zip(bars, df["fps_cpu_devmachine"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}", ha="center", fontsize=8, color=INK)

    ax = axes[1]
    bars = ax.bar(df["short"], df["params_M"], color=colors, width=0.6)
    ax.set_title("Model size (million parameters)", fontsize=10, color=INK, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8, rotation=28)
    ax.tick_params(axis="y", labelsize=8)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    for b, v in zip(bars, df["params_M"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}M", ha="center", fontsize=8, color=INK)

    fig.tight_layout(pad=0.6)
    fig.savefig(OUT / "chart_classifier_benchmark.png", transparent=True)
    plt.close(fig)


def stock_vs_trained_chart():
    # So sanh THAT (chay that, xem compare_classifier_default_vs_trained.py):
    # YOLOv8n-cls GOC (pretrained ImageNet-1k, chua fine-tune) vs ban DANG
    # DEPLOY (fine-tuned AFCL) -- do TREN CUNG 12,460 anh val v4_split_flat.
    labels = ["Stock\n(not fine-tuned)", "Deployed\n(fine-tuned, AFCL)"]
    acc = [0.0, 97.98]
    colors = [GRAY, ORANGE]

    fig, ax = plt.subplots(figsize=(4.6, 3.0), dpi=200)
    bars = ax.bar(labels, acc, color=colors, width=0.5)
    ax.set_ylabel("Top-1 accuracy on v4_split_flat val", fontsize=8.5)
    ax.set_ylim(0, 112)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=8)
    for b, v in zip(bars, acc):
        ax.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.2f}%", ha="center",
                fontsize=10, fontweight="bold", color=INK)
    fig.tight_layout(pad=0.5)
    fig.savefig(OUT / "chart_stock_vs_trained.png", transparent=True)
    plt.close(fig)


def training_improvement_chart():
    import numpy as np
    runs = ["Run 1", "Run 2 (deployed)"]
    epoch1 = [89.86, 91.57]
    final = [96.81, 97.90]

    fig, ax = plt.subplots(figsize=(5.6, 3.0), dpi=200)
    x = np.arange(len(runs))
    w = 0.32
    b1 = ax.bar(x - w / 2, epoch1, width=w, color="#b9c2d6", label="Epoch 1")
    b2 = ax.bar(x + w / 2, final, width=w, color=ORANGE, label="Epoch 40 (final)")
    ax.set_xticks(x); ax.set_xticklabels(runs, fontsize=9)
    ax.set_ylim(0, 122)
    ax.set_ylabel("Top-1 accuracy (%)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.16))
    for bars in [b1, b2]:
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.2f}%",
                    ha="center", fontsize=8, color=INK)
    fig.tight_layout(pad=0.6)
    fig.savefig(OUT / "chart_training_improvement.png", transparent=True)
    plt.close(fig)


def gnn_cca_results_chart():
    # Chi giu ARI/AMI/V-measure -- day la 3 chi so "an toan" (khong the danh lua
    # bang cach chia cum cuc doan). Homogeneity/Completeness rieng le de gay
    # hieu lam (vd Homogeneity = 100% neu tach moi nguoi thanh 1 cum), phai luon
    # doc CUNG NHAU -- ma V-measure da la trung binh dieu hoa cua chinh 2 cai do
    # roi nen bo qua khong mat thong tin quan trong.
    # CHI 1 GNN-CCA (do that tren data cua nhom) -- bo hang literature/EPFL
    # khoi bieu do nay (chuyen sang bang "so lieu ho cong bo" o slide 32, cho
    # gon va dung cho ke hoach "meet the competitor" -> "results" tach biet).
    import numpy as np
    metrics = ["ARI", "AMI", "V-measure"]
    hungarian = [85.45, 85.46, 95.01]
    gnn_own = [77.56, 77.71, 87.44]

    fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=200)
    x = np.arange(len(metrics))
    w = 0.32
    bars = [
        (x - w / 2, hungarian, ORANGE, "Hungarian"),
        (x + w / 2, gnn_own, NAVY, "GNN-CCA"),
    ]
    for pos, vals, color, label in bars:
        b = ax.bar(pos, vals, width=w, color=color, label=label)
        for rect, v in zip(b, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 1.5, f"{v:.2f}",
                    ha="center", fontsize=8, color=INK)

    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Score", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout(pad=0.6)
    fig.savefig(OUT / "chart_gnn_cca_results.png", transparent=True)
    plt.close(fig)


def espinosa_results_chart():
    # So lieu MOI NHAT trong Report chinh Sec 6.3.3 (P328) -- dung production
    # decision rule that, 36 canh (24 Fall/12 ADL). Thay cho so cu "0/24 missed"
    # da loi thoi tren slide 34/35 goc.
    import numpy as np
    metrics = ["Sensitivity", "Specificity", "Accuracy"]
    ours = [87.50, 91.67, 88.89]
    espinosa = [29.17, 83.3, 47.2]

    fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=200)
    x = np.arange(len(metrics))
    w = 0.32
    b1 = ax.bar(x - w / 2, ours, width=w, color=ORANGE, label="Boundary Feature Fusion (ours)")
    b2 = ax.bar(x + w / 2, espinosa, width=w, color=NAVY, label="Espinosa et al.")
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 112)
    ax.set_ylabel("%", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=1, bbox_to_anchor=(1.28, 0.85))
    for bars in [b1, b2]:
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.2f}",
                    ha="center", fontsize=8, color=INK)
    fig.tight_layout(pad=0.6)
    fig.savefig(OUT / "chart_espinosa_results.png", transparent=True)
    plt.close(fig)


def v4_split_flat_chart():
    # Train = 20,000 anh/lop, GIONG HET nhau ca 5 lop (da can bang) -- ve rieng
    # thanh 1 dong chu thich, KHONG ve chung 1 truc voi Val (Val chi ~2-4k, se
    # bi nen phang neu dung chung truc voi Train ~20k, chenh lech ~10 lan).
    classes = ["bend", "exercise", "lie", "sit", "stand"]
    val = [1916, 1901, 2029, 2675, 3939]

    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=200)
    bars = ax.bar(classes, val, color=NAVY, width=0.55)
    ax.set_ylabel("Validation images", fontsize=9)
    ax.set_ylim(0, max(val) * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=8)
    for b, v in zip(bars, val):
        ax.text(b.get_x() + b.get_width() / 2, v + max(val) * 0.02, f"{v:,}",
                ha="center", fontsize=8.5, color=INK)
    ax.set_title("Train: 20,000 images/class (balanced)", fontsize=9.5, color="#5a5f75", loc="left")
    fig.tight_layout(pad=0.5)
    fig.savefig(OUT / "chart_v4_split_flat.png", transparent=True)
    plt.close(fig)


def msmt17_split_chart():
    splits = ["Train", "Validation", "Query", "Gallery"]
    images = [30248, 2373, 11659, 82161]
    ids = [1041, 1041, 3060, 3060]

    fig, ax = plt.subplots(figsize=(5.6, 3.0), dpi=200)
    bars = ax.bar(splits, images, color=NAVY, width=0.55)
    ax.set_ylabel("Images", fontsize=9)
    ax.set_ylim(0, max(images) * 1.28)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=8)
    for b, v, n_id in zip(bars, images, ids):
        ax.text(b.get_x() + b.get_width() / 2, v + max(images) * 0.03, f"{v:,}",
                ha="center", fontsize=8.5, color=INK, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, v + max(images) * 0.10, f"{n_id:,} IDs",
                ha="center", fontsize=7.5, color="#5a5f75")
    fig.tight_layout(pad=0.5)
    fig.savefig(OUT / "chart_msmt17_split.png", transparent=True)
    plt.close(fig)


def espinosa_generalization_chart():
    conditions = ["UP-Fall Dataset", "MCFD", "PFDD-Test"]
    acc = [95.64, 82.84, 46.7]
    fig, ax = plt.subplots(figsize=(5.4, 2.6), dpi=200)
    bars = ax.bar(conditions, acc, color=[BLUE, "#8a97b8", NAVY], width=0.55)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Accuracy (%)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.tick_params(axis="y", labelsize=8)
    for b, v in zip(bars, acc):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.2f}%", ha="center", fontsize=8, color=INK)
    fig.tight_layout(pad=0.6)
    fig.savefig(OUT / "chart_espinosa_generalization.png", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    floor_projection_diagram()
    print("Da luu: diagram_floor_projection.png")
    classifier_bench_charts()
    print("Da luu: chart_classifier_benchmark.png")
    training_improvement_chart()
    print("Da luu: chart_training_improvement.png")
    gnn_cca_results_chart()
    print("Da luu: chart_gnn_cca_results.png")
    espinosa_results_chart()
    print("Da luu: chart_espinosa_results.png")
    espinosa_generalization_chart()
    print("Da luu: chart_espinosa_generalization.png")
    print("Da luu: chart_gnn_cca_results.png")
