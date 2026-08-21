"""Tao noi dung cho subfolder "0. Bai cua nhom (0.25 distill)" ben trong
folder "Phuong phap 1 va 4, so sanh voi 0.25 khong distill" -- gioi thieu lai
model dang dung SAN XUAT THAT (OSNet x0.25 KD goc, kd_weight=0.5) de nguoi
doc co day du 4 muc so sanh (Baseline/Distill goc/PA1/PA4) trong CUNG 1 noi.
Video test tai su dung nguyen video da co (that, da validate) tu folder
"1. Bai cua nhom (OSNet x0.25 KD)"."""
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
          / "Phương pháp 1 và 4, so sánh với 0.25 không distill")
DIR_OUT = PARENT / "0. Nhom"  # ten CUC NGAN -- path goc da sau 218 ky tu, chi con ~12 ky tu ngan sach truoc khi cham MAX_PATH (260)

COLOR = "#3B82F6"


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
    vals = [52.00, 70.05, 27.46]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(metrics, vals, color=COLOR, width=0.5)
    ax.set_ylabel("%")
    ax.set_title("Rank-1 / Rank-5 / mAP (MSMT17) — Current Production Model")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    out = DIR_OUT / "chart_rank1_map.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def build_doc():
    doc = Document()
    doc.add_heading("Re-Identification — Bài của nhóm (OSNet x0.25, Distill gốc — model đang dùng thật)", level=0)

    add_heading(doc, "1. Giới thiệu")
    doc.add_paragraph(
        "Đây chính là model OSNet x0.25 (Knowledge Distillation từ OSNet x1.0, kd_weight=0.5) "
        "đang được DÙNG THẬT trong hệ thống (checkpoint `reid_osnet_x0_25_kd.pt` trong gói bàn "
        "giao Handoff_for_Edge) — đưa vào folder này để người đọc có đủ 4 mốc so sánh (Baseline "
        "không-distill, chính bản Distill gốc này, Phương án 1, Phương án 4) trong cùng 1 nơi, "
        "không phải lật lại folder \"1. Bài của nhóm (OSNet x0.25 KD)\" ở ngoài."
    )
    doc.add_paragraph(
        "Chi tiết đầy đủ về quy trình train, phát hiện & sửa lỗi transform, so sánh với OSNet "
        "x0.5 cũ... xem folder gốc \"1. Bài của nhóm (OSNet x0.25 KD)\". Nội dung ở đây chỉ tóm "
        "tắt số liệu cốt lõi để tiện đối chiếu trực tiếp với Baseline/PA1/PA4."
    )

    add_heading(doc, "2. Rank-1 / Rank-5 / mAP (MSMT17, đo chính thức)")
    add_table(doc, ["Metric", "Giá trị"], [
        ["Rank-1", "52.00%"],
        ["Rank-5", "70.05%"],
        ["mAP", "27.46%"],
    ])
    chart = make_chart()
    add_image(doc, chart)

    add_heading(doc, "3. Tier-2 TPR (Own Video Test)")
    add_table(doc, ["Test set", "Kết quả"], [
        ["4 base cases (no augment)", "3/4 (75.0%)"],
        ["25 full cases (2 cams x 6 variants)", "19/25 (76.0%)"],
    ])

    add_heading(doc, "4. So sánh nhanh với 3 mốc còn lại trong folder cha")
    add_table(doc, ["Model", "Rank-1", "mAP", "TPR (25 case)"], [
        ["Baseline (không distill)", "52.44%", "27.60%", "19/25 (76.0%)"],
        ["Distill gốc (bản này)", "52.00%", "27.46%", "19/25 (76.0%)"],
        ["Phương án 1 (kd=0.2+warmup)", "52.49%", "27.43%", "19/25 (76.0%)"],
        ["Phương án 4 (qua Teacher Assistant)", "52.94%", "28.23%", "18/25 (72.0%)"],
    ])
    doc.add_paragraph(
        "Cả 4 bản gần như tương đương ở Tier-2 TPR (72-76%) và Rank-1/mAP (chênh lệch trong "
        "khoảng 1 điểm phần trăm) — xem phân tích chi tiết trong Noi_dung.docx của thư mục cha."
    )

    add_file_catalog(doc, [
        ("Scene5_ghep.mp4",
         "VIDEO CHÍNH — ghép 2 camera Scene 5 (CAM 1 trái, CAM 2 phải) chạy song song, mỗi "
         "camera tự vẽ khung + nhãn 'Person N'. Bản sao y hệt video (đổi tên ngắn gọn hơn để "
         "tránh lỗi đường dẫn quá dài của Windows) trong folder gốc \"1. Bài của nhóm (OSNet "
         "x0.25 KD)\" — cùng checkpoint, cùng test thật."),
        ("dangky_test.mp4",
         "Video phụ, 2 đoạn nối tiếp: (1) Đăng ký Person 1 (Enrollment), (2) Test CHỈ CAM 2."),
        ("chart_rank1_map.png", "Biểu đồ cột Rank-1/Rank-5/mAP của bản Distill gốc."),
        ("Noi_dung.docx", "Chính file này."),
    ])

    DIR_OUT.mkdir(parents=True, exist_ok=True)
    doc.save(str(DIR_OUT / "Noi_dung.docx"))
    print(f"Da luu: {DIR_OUT / 'Noi_dung.docx'}")


if __name__ == "__main__":
    DIR_OUT.mkdir(parents=True, exist_ok=True)
    build_doc()
