"""Tao report so sanh Phuong an 4 (Teacher Assistant, model tot nhat trong
cac ban KD da thu) voi Shared Backbone (doi thu 1, dung lai backbone Pose
Head) -- folder "Phuong phap 4 va backbone" cua Re-Identification. 1 docx +
2 chart, bang/chart tieu de ngan gon tieng Anh, noi dung van ban tieng
Viet."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Inches

sys.stdout.reconfigure(encoding="utf-8")

AI_ROOT = Path(r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)")
BASE = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Số liệu thống kê" / "Re-Identification"
DIR_OUT = BASE / "Phương pháp 4 và backbone"

COLOR_PA4 = "#10B981"
COLOR_BACKBONE = "#EF4444"


def add_heading(doc, text, level=1):
    doc.add_heading(text, level=level)


def add_table(doc, headers, rows):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        for p in hdr[i].paragraphs:
            for r in p.runs:
                r.bold = True
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
    doc.add_paragraph()


def add_image(doc, path, width_in=5.8):
    if path.exists():
        doc.add_picture(str(path), width=Inches(width_in))


def add_file_catalog(doc, entries):
    add_heading(doc, "Danh mục file trong thư mục này", level=2)
    for name, desc in entries:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(name)
        run.bold = True
        p.add_run(f" — {desc}")


def make_chart_rank1_map():
    metrics = ["Rank-1", "Rank-5", "mAP"]
    pa4 = [52.94, 70.79, 28.23]
    backbone = [4.88, 10.71, 1.11]
    x = range(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - w / 2 for i in x], pa4, width=w, label="PA4 (Teacher Assistant)", color=COLOR_PA4)
    ax.bar([i + w / 2 for i in x], backbone, width=w, label="Shared Backbone", color=COLOR_BACKBONE)
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylabel("%")
    ax.set_title("Rank-1 / Rank-5 / mAP (MSMT17)")
    ax.legend()
    for i, (a, b) in enumerate(zip(pa4, backbone)):
        ax.text(i - w / 2, a + 1.2, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 1.2, f"{b:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    out = DIR_OUT / "chart_rank1_map.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def make_chart_tpr():
    labels = ["4 base cases", "25 full cases"]
    pa4 = [75.0, 72.0]
    backbone = [0.0, 0.0]
    x = range(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - w / 2 for i in x], pa4, width=w, label="PA4 (Teacher Assistant)", color=COLOR_PA4)
    ax.bar([i + w / 2 for i in x], backbone, width=w, label="Shared Backbone", color=COLOR_BACKBONE)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.set_title("Tier-2 TPR (Own Video Test)")
    ax.legend()
    for i, (a, b) in enumerate(zip(pa4, backbone)):
        ax.text(i - w / 2, a + 1.5, f"{a:.1f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 1.5, f"{b:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    out = DIR_OUT / "chart_tpr.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def build_doc():
    doc = Document()
    doc.add_heading("Re-Identification — Phương án 4 (Teacher Assistant) so với Shared Backbone (Đối thủ 1)", level=0)

    add_heading(doc, "1. Mục đích")
    doc.add_paragraph(
        "So sánh trực tiếp model tốt nhất trong các phiên bản Knowledge Distillation đã thử "
        "(Phương án 4 — Teacher Assistant, xem folder \"Phương pháp 1 và 4...\") với Shared "
        "Backbone (\"Đối thủ 1\", xem folder \"2. Mô phỏng Đối thủ (Backbone dùng chung)\") — "
        "thiết kế ban đầu của nhóm dùng lại backbone đã train cho Pose Head (YOLOv8n-cls, phân "
        "loại 5 tư thế) thay vì model Re-ID riêng, nhằm tiết kiệm tài nguyên Jetson Nano 4GB. Số "
        "liệu Shared Backbone lấy từ cấu hình TỐT NHẤT trong 4 cấu hình đã thử (Frozen backbone, "
        "CE + Triplet Loss)."
    )
    doc.add_paragraph(
        "Phương án 4 (Teacher Assistant): distill 2 bước OSNet_x1.0 → x0.5 (Teacher Assistant) → "
        "x0.25 (Final Student, kd_weight=0.3), thu hẹp capacity gap so với distill trực tiếp — "
        "đạt Rank-1/mAP cao nhất trong mọi bản KD/Baseline x0.25 đã thử."
    )

    add_heading(doc, "2. Rank-1 / Rank-5 / mAP (MSMT17, đo chính thức)")
    add_table(doc, ["Model", "Rank-1", "Rank-5", "mAP"], [
        ["Phương án 4 (Teacher Assistant)", "52.94%", "70.79%", "28.23%"],
        ["Shared Backbone (frozen, cấu hình tốt nhất)", "4.88%", "10.71%", "1.11%"],
    ])
    chart1 = make_chart_rank1_map()
    add_image(doc, chart1)
    doc.add_paragraph(
        "Chênh lệch RẤT LỚN, nhất quán với toàn bộ số liệu đã ghi nhận trước đây cho Shared "
        "Backbone (kể cả khi thử đủ 4 cấu hình train khác nhau, Rank-1 cao nhất chỉ 4.88%). "
        "Nguyên nhân gốc (đã phân tích trong folder Đối thủ 1): backbone Pose Head được train "
        "CHUYÊN BIỆT để phân loại tư thế thô (đứng/ngồi/nằm...), các lớp lọc có xu hướng bỏ qua "
        "đúng những chi tiết ngoại hình (màu áo, hoạ tiết, hình dáng riêng) mà Re-Identification "
        "cần."
    )

    add_heading(doc, "3. Tier-2 TPR (Own Video Test)")
    add_table(doc, ["Test set", "Phương án 4", "Shared Backbone"], [
        ["4 base cases (no augment)", "3/4 (75.0%)", "0/4 (0.0%)"],
        ["25 full cases (2 cams x 6 variants)", "18/25 (72.0%)", "0/25 (0.0%)"],
    ])
    chart2 = make_chart_tpr()
    add_image(doc, chart2)
    doc.add_paragraph(
        "Shared Backbone thất bại TUYỆT ĐỐI (0/25, mọi biến thể) — đã xác minh trước đây KHÔNG "
        "phải bug (embedding vẫn mang tín hiệu nhận dạng thật, cosine similarity cùng người "
        "~0.10-0.12 so với khác người ~0.002, nhưng tín hiệu quá yếu để vượt ngưỡng khớp 0.6). "
        "Phương án 4 dù có 1 case sai riêng (xem giải thích track_id=51, CAM 2/aug-flip — không "
        "gian embedding qua 2 bước distill tổ chức khác đi ở đúng case khó này) vẫn nhận đúng "
        "72-75% — vượt trội hoàn toàn so với Shared Backbone."
    )

    add_heading(doc, "4. Kết luận")
    doc.add_paragraph(
        "Kết quả củng cố mạnh mẽ quyết định kiến trúc ban đầu của nhóm: dùng OSNet làm model "
        "Re-Identification ĐỘC LẬP, tách khỏi backbone dùng chung với Pose Head. Dù Phương án 4 "
        "(Teacher Assistant) chưa phải cải thiện lớn so với Baseline/Distill gốc của chính OSNet "
        "(xem folder \"Phương pháp 1 và 4...\"), nó vẫn vượt Shared Backbone ở mức chênh lệch "
        "rất lớn (Rank-1 gấp ~10.8 lần, TPR 72% so với 0%) — xác nhận hướng kiến trúc OSNet độc "
        "lập là đúng đắn, bất kể biến thể KD nào được chọn."
    )

    add_file_catalog(doc, [
        ("chart_rank1_map.png", "Biểu đồ cột Rank-1/Rank-5/mAP: PA4 vs Shared Backbone."),
        ("chart_tpr.png", "Biểu đồ cột Tier-2 TPR (4 base, 25 full): PA4 vs Shared Backbone."),
        ("Noi_dung.docx", "Chính file này."),
    ])

    DIR_OUT.mkdir(parents=True, exist_ok=True)
    doc.save(str(DIR_OUT / "Noi_dung.docx"))
    print(f"Da luu: {DIR_OUT / 'Noi_dung.docx'}")


if __name__ == "__main__":
    DIR_OUT.mkdir(parents=True, exist_ok=True)
    build_doc()
