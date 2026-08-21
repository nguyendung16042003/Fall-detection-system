"""Tao noi dung cho subfolder "0. Bai cua nhom (PA4)" ben trong folder
"Phuong phap 4 va backbone va MSI" -- gioi thieu rieng ve Phuong an 4
(Teacher Assistant), model tot nhat cua nhom trong cac ban KD da thu, dung
lam "Bai cua nhom" khi so sanh voi 2 doi thu (Shared Backbone, MSINet)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Inches

sys.stdout.reconfigure(encoding="utf-8")

AI_ROOT = Path(r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)")
PARENT = (AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Số liệu thống kê" / "Re-Identification"
          / "Phương pháp 4 và backbone và MSI")
DIR_OUT = PARENT / "0. Bai cua nhom (PA4)"

COLOR = "#10B981"


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


def make_chart():
    metrics = ["Rank-1", "Rank-5", "mAP"]
    vals = [52.94, 70.79, 28.23]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(metrics, vals, color=COLOR, width=0.5)
    ax.set_ylabel("%")
    ax.set_title("Rank-1 / Rank-5 / mAP (MSMT17) — PA4 (Ours)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    out = DIR_OUT / "chart_rank1_map.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def build_doc():
    doc = Document()
    doc.add_heading("Re-Identification — Bài của nhóm (Phương án 4, Teacher Assistant)", level=0)

    add_heading(doc, "1. Giới thiệu")
    doc.add_paragraph(
        "Đây là model \"của nhóm\" trong bối cảnh so sánh với 2 đối thủ ở folder cha (Shared "
        "Backbone — Đối thủ 1, MSINet — Đối thủ 2): Phương án 4 (Teacher Assistant), kết quả "
        "TỐT NHẤT trong toàn bộ các thí nghiệm Knowledge Distillation nhóm đã thử cho OSNet "
        "x0.25 (xem chi tiết quy trình train trong folder \"Phương pháp 1 và 4, so sánh với "
        "0.25 không distill\")."
    )
    doc.add_paragraph(
        "Cách train: distill 2 bước thay vì 1 bước trực tiếp — Giai đoạn 1 distill OSNet_x1.0 "
        "(Teacher gốc) → OSNet_x0.5 (Teacher Assistant, TA), thu hẹp capacity gap từ ~4 lần "
        "xuống ~2 lần mỗi bước; Giai đoạn 2 dùng TA (đã đóng băng) làm Teacher mới, distill "
        "sang OSNet_x0.25 (Final Student, kd_weight=0.3). Checkpoint dùng trong video/số liệu "
        "dưới đây: `osnet_x0_25_kd_pa4_via_ta_msmt17_best.pt`."
    )
    doc.add_paragraph(
        "Lưu ý phạm vi: đây KHÔNG phải model đang chạy sản xuất thật trong hệ thống (model đó "
        "là bản Distill gốc, kd_weight=0.5, xem folder \"1. Bài của nhóm (OSNet x0.25 KD)\") — "
        "Phương án 4 là kết quả thực nghiệm cải thiện, giữ lại để đối chiếu."
    )

    add_heading(doc, "2. Rank-1 / Rank-5 / mAP (MSMT17, đo chính thức)")
    add_table(doc, ["Metric", "Giá trị"], [
        ["Rank-1", "52.94%"],
        ["Rank-5", "70.79%"],
        ["mAP", "28.23%"],
    ])
    chart = make_chart()
    add_image(doc, chart)

    add_heading(doc, "3. Tier-2 TPR (Own Video Test)")
    add_table(doc, ["Test set", "Kết quả"], [
        ["4 base cases (no augment)", "3/4 (75.0%)"],
        ["25 full cases (2 cams x 6 variants)", "18/25 (72.0%)"],
    ])
    doc.add_paragraph(
        "1 case sai đã phân tích riêng (track_id=51, CAM 2/aug-flip) — không gian embedding "
        "qua 2 bước distill tổ chức khác đi ở đúng case khó này (điểm khớp 0.76, tự tin nhưng "
        "tự tin sai, không phải sát ngưỡng)."
    )

    add_heading(doc, "4. Video demo (test thật, Scene 5)")
    doc.add_paragraph(
        "Video mới, chạy trực tiếp trên checkpoint Phương án 4 (chưa từng có video minh hoạ "
        "trước đây — các lần trước chỉ đo bằng số liệu CSV)."
    )

    add_file_catalog(doc, [
        ("Scene5_ghep.mp4",
         "VIDEO CHÍNH — ghép 2 camera Scene 5 (CAM 1 trái, CAM 2 phải) chạy song song, mỗi "
         "camera tự vẽ khung + nhãn 'Person N'. Kết quả: 3/4 đúng (CAM 2 track_id=6 sai — đúng "
         "case đã biết, xem Mục 3)."),
        ("dangky_test.mp4",
         "Video phụ, 2 đoạn nối tiếp: (1) Đăng ký Person 1 (Enrollment), (2) Test CHỈ CAM 2."),
        ("chart_rank1_map.png", "Biểu đồ cột Rank-1/Rank-5/mAP của Phương án 4."),
        ("Noi_dung.docx", "Chính file này."),
    ])

    DIR_OUT.mkdir(parents=True, exist_ok=True)
    doc.save(str(DIR_OUT / "Noi_dung.docx"))
    print(f"Da luu: {DIR_OUT / 'Noi_dung.docx'}")


if __name__ == "__main__":
    DIR_OUT.mkdir(parents=True, exist_ok=True)
    build_doc()
