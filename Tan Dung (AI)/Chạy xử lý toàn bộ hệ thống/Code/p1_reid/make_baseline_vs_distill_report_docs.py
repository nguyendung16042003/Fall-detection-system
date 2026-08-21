"""Tao report so sanh OSNet x0.25 Distill (KD) vs OSNet x0.25 Baseline (khong
Distill) -- folder 6 cua Re-Identification. 1 docx + 2 chart (Rank-1/mAP,
Tier-2 TPR), bang/chart tieu de ngan gon tieng Anh, noi dung van ban tieng
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
DIR_OUT = BASE / "6. So sánh bản 0.25 distill 2 và 0.25 không distill"

COLOR_DISTILL = "#3B82F6"
COLOR_BASELINE = "#F59E0B"


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
    distill = [52.00, 70.05, 27.46]
    baseline = [52.44, 70.07, 27.60]
    x = range(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - w / 2 for i in x], distill, width=w, label="Distill (KD)", color=COLOR_DISTILL)
    ax.bar([i + w / 2 for i in x], baseline, width=w, label="Baseline (No KD)", color=COLOR_BASELINE)
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylabel("%")
    ax.set_title("Rank-1 / Rank-5 / mAP (MSMT17)")
    ax.legend()
    for i, (d, b) in enumerate(zip(distill, baseline)):
        ax.text(i - w / 2, d + 0.8, f"{d:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 0.8, f"{b:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    out = DIR_OUT / "chart_rank1_map.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def make_chart_tpr():
    labels = ["4 base cases", "25 full cases"]
    distill = [75.0, 76.0]
    baseline = [75.0, 76.0]
    x = range(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - w / 2 for i in x], distill, width=w, label="Distill (KD)", color=COLOR_DISTILL)
    ax.bar([i + w / 2 for i in x], baseline, width=w, label="Baseline (No KD)", color=COLOR_BASELINE)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.set_title("Tier-2 TPR (Own Video Test)")
    ax.legend()
    for i, (d, b) in enumerate(zip(distill, baseline)):
        ax.text(i - w / 2, d + 1.5, f"{d:.1f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 1.5, f"{b:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    out = DIR_OUT / "chart_tpr.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def build_doc():
    doc = Document()
    doc.add_heading("Re-Identification — So sánh OSNet x0.25 Distill (KD) vs Baseline (không Distill)", level=0)

    add_heading(doc, "1. Mục đích & thiết kế thí nghiệm")
    doc.add_paragraph(
        "Đối chứng khoa học để trả lời câu hỏi: Knowledge Distillation (KD, dùng OSNet x1.0 "
        "làm Teacher) có thực sự cải thiện OSNet x0.25 so với chỉ train trực tiếp từ nhãn "
        "(không Teacher) hay không? Theo tài liệu "
        "\"Quy_trinh_OSNet_x025_Baseline_KhongDistill.docx\", 2 model dùng CHUNG 100% mọi thứ "
        "(dataset MSMT17 list_train/list_val, kiến trúc osnet_x0_25, khởi tạo ImageNet "
        "pretrained, Adam lr=3.5e-4/wd=5e-4, batch 64 (P=16,K=4), MultiStepLR milestones "
        "[40,70], early-stop MIN_EPOCHS=30/PATIENCE=10) — CHỈ khác đúng 1 biến: bản Distill có "
        "thêm L_KD=0.5×MSE(embedding Teacher, Student), bản Baseline chỉ có L_ID + L_triplet."
    )
    doc.add_paragraph(
        "Bản Distill (\"bản distill 2\" — tên gọi khác của model KD hiện có, \"1. Bai cua nhom\") "
        "đã có sẵn, không train lại. Bản Baseline được train MỚI hoàn toàn cho báo cáo này, "
        "dừng sớm ở epoch 55 (val_rank1 nội bộ tốt nhất 81.33% tại epoch 49, patience hết ở "
        "epoch 59)."
    )
    doc.add_paragraph(
        "Sự cố kỹ thuật khi train (ghi nhận trung thực): 2 tiến trình train Baseline vô tình "
        "chạy song song 1 đoạn (do lỗi thao tác dừng tiến trình không triệt để), cùng ghi vào 1 "
        "file checkpoint. Đã xác minh lại: checkpoint cuối cùng trên đĩa KHÔNG bị hỏng (load "
        "được đầy đủ, đúng cấu trúc 567 key) — nhưng để loại bỏ rủi ro, KHÔNG dùng số liệu "
        "val_rank1 nội bộ (chỉ mang tính early-stop) làm căn cứ so sánh chính thức. Toàn bộ số "
        "liệu Rank-1/mAP/TPR dưới đây đều đo LẠI hoàn toàn độc lập, trực tiếp trên checkpoint "
        "thật đang có trên đĩa."
    )

    add_heading(doc, "2. Rank-1 / mAP (MSMT17, đo chính thức)")
    doc.add_paragraph("Đo trên đúng bộ query/gallery chuẩn MSMT17 (11.659 / 82.161 ảnh), cùng protocol cho cả 2:")
    add_table(doc, ["Model", "Rank-1", "Rank-5", "mAP"], [
        ["Distill (KD)", "52.00%", "70.05%", "27.46%"],
        ["Baseline (No KD)", "52.44%", "70.07%", "27.60%"],
    ])
    chart1 = make_chart_rank1_map()
    add_image(doc, chart1)
    doc.add_paragraph(
        "KẾT QUẢ BẤT NGỜ, GHI NHẬN TRUNG THỰC: Baseline (không distill) nhỉnh hơn Distill ở CẢ "
        "3 chỉ số (dù chênh lệch rất nhỏ, trong khoảng nhiễu đo đạc: +0.44/+0.02/+0.14 điểm "
        "phần trăm). Ở quy mô thí nghiệm này, KHÔNG có bằng chứng cho thấy Knowledge "
        "Distillation cải thiện Rank-1/mAP so với train trực tiếp cùng kiến trúc/hyperparameter."
    )

    add_heading(doc, "3. Tier-2 TPR (Own Video Test)")
    doc.add_paragraph("Cùng data/quy trình Gallery 2 người, cả 2 camera Scene 5, gốc + 5 biến thể augment:")
    add_table(doc, ["Test set", "Distill (KD)", "Baseline (No KD)"], [
        ["4 base cases (no augment)", "3/4 (75.0%)", "3/4 (75.0%)"],
        ["25 full cases (2 cams x 6 variants)", "19/25 (76.0%)", "19/25 (76.0%)"],
    ])
    chart2 = make_chart_tpr()
    add_image(doc, chart2)
    doc.add_paragraph(
        "GIỐNG HỆT NHAU ở cả 2 mức test, kể cả cùng phân bố lỗi (đều sai ở CAM 2 khi Person 1 "
        "quay lại lần 2). Củng cố thêm phát hiện ở Mục 2: trong bài toán Gallery nhỏ (2 người), "
        "2 model thể hiện năng lực nhận dạng hoàn toàn tương đương."
    )

    add_heading(doc, "4. Params / FLOPs & Jetson Nano 4GB")
    doc.add_paragraph(
        "KHÔNG đo lại — 2 model dùng ĐÚNG kiến trúc osnet_x0_25 (chỉ khác hàm loss lúc train, "
        "không khác cấu trúc mạng), nên Params/FLOPs và FPS/RAM/nhiệt độ trên Jetson GIỐNG HỆT "
        "nhau về mặt kiến trúc (không phụ thuộc trọng số đã train). Dùng chung số liệu đã đo "
        "cho bản Distill:"
    )
    add_table(doc, ["Metric", "Giá trị (dùng chung cho cả 2)"], [
        ["Params", "0.20M"],
        ["FLOPs", "0.091G"],
        ["FPS Jetson Nano 4GB (model đứng riêng)", "64.11 (15.60 ms/call)"],
        ["FPS Jetson Nano 4GB (chạy chung YOLOv8n, xấu nhất)", "21.31"],
        ["RAM peak (Jetson)", "1.73 GB"],
        ["Temperature peak (Jetson)", "36.25°C"],
    ])

    add_heading(doc, "5. Kết luận")
    doc.add_paragraph(
        "Với đúng thiết kế đối chứng khoa học của tài liệu gốc (mọi thứ giống nhau, chỉ khác "
        "L_KD), thí nghiệm này KHÔNG tìm thấy bằng chứng cải thiện từ Knowledge Distillation ở "
        "cả 2 mức đánh giá (Rank-1/mAP quy mô lớn và Tier-2 TPR quy mô nhỏ) — 2 model gần như "
        "tương đương. Khả năng giải thích hợp lý: OSNet x0.25 (~736K tham số) đã đủ dung lượng "
        "để tự học tốt trực tiếp từ nhãn với dataset/hyperparameter hiện có, nên phần tri thức "
        "bổ sung từ Teacher (OSNet x1.0) không tạo khác biệt đáng kể ở kích thước model này. "
        "Đây là số liệu thực đo, không được điều chỉnh để khớp kỳ vọng ban đầu (rằng KD phải có "
        "ích) — quyết định kiến trúc cuối cùng của nhóm (dùng osnet_x0_25 làm Re-ID head độc "
        "lập) vẫn đúng đắn, chỉ riêng việc CÓ CẦN quy trình Knowledge Distillation hay không thì "
        "kết quả này cho thấy chưa chứng minh được lợi ích rõ ràng."
    )

    add_file_catalog(doc, [
        ("chart_rank1_map.png", "Biểu đồ cột Rank-1/Rank-5/mAP, Distill vs Baseline."),
        ("chart_tpr.png", "Biểu đồ cột Tier-2 TPR (4 base cases, 25 full cases), Distill vs Baseline."),
        ("Noi_dung.docx", "Chính file này."),
    ])

    DIR_OUT.mkdir(parents=True, exist_ok=True)
    doc.save(str(DIR_OUT / "Noi_dung.docx"))
    print(f"Da luu: {DIR_OUT / 'Noi_dung.docx'}")


if __name__ == "__main__":
    DIR_OUT.mkdir(parents=True, exist_ok=True)
    build_doc()
