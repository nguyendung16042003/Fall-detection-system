"""Tao report so sanh Phuong an 1 (tune hyperparameter KD) va Phuong an 4
(Teacher Assistant) voi Baseline (khong distill) -- folder moi cua
Re-Identification. 1 docx + 2 chart, bang/chart tieu de ngan gon tieng Anh,
noi dung van ban tieng Viet."""
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
DIR_OUT = BASE / "Phương pháp 1 và 4, so sánh với 0.25 không distill"

COLORS = {"Baseline": "#94A3B8", "PA1": "#3B82F6", "PA4": "#10B981", "TA x0.5": "#F59E0B"}


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
    data = {
        "Baseline": [52.44, 70.07, 27.60],
        "PA1": [52.49, 69.69, 27.43],
        "PA4": [52.94, 70.79, 28.23],
    }
    x = range(len(metrics))
    w = 0.25
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, (name, vals) in enumerate(data.items()):
        offset = (i - 1) * w
        ax.bar([j + offset for j in x], vals, width=w, label=name, color=COLORS[name])
        for j, v in enumerate(vals):
            ax.text(j + offset, v + 0.6, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylabel("%")
    ax.set_title("Rank-1 / Rank-5 / mAP (MSMT17)")
    ax.legend()
    fig.tight_layout()
    out = DIR_OUT / "chart_rank1_map.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def make_chart_tpr():
    labels = ["4 base cases", "25 full cases"]
    data = {
        "Baseline": [75.0, 76.0],
        "PA1": [75.0, 76.0],
        "PA4": [75.0, 72.0],
    }
    x = range(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, (name, vals) in enumerate(data.items()):
        offset = (i - 1) * w
        ax.bar([j + offset for j in x], vals, width=w, label=name, color=COLORS[name])
        for j, v in enumerate(vals):
            ax.text(j + offset, v + 1.2, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.set_title("Tier-2 TPR (Own Video Test)")
    ax.legend()
    fig.tight_layout()
    out = DIR_OUT / "chart_tpr.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def build_doc():
    doc = Document()
    doc.add_heading("Re-Identification — Phương án 1 & 4 (cải thiện KD) so với Baseline (không Distill)", level=0)

    add_heading(doc, "1. Bối cảnh & thiết kế thí nghiệm")
    doc.add_paragraph(
        "Sau khi phát hiện bản Distill gốc (kd_weight=0.5, distill trực tiếp OSNet_x1.0 → x0.25) "
        "KHÔNG vượt Baseline (không distill) — xem folder \"6. So sánh...\" — đã thử 2 hướng cải "
        "thiện độc lập theo 2 tài liệu guide mới:"
    )
    doc.add_paragraph(
        "• Phương án 1 — Điều chỉnh Hyperparameter KD: chọn 1 cấu hình đại diện mạnh nhất guide "
        "đề xuất (kết hợp mục 1+2): kd_weight giảm từ 0.5 xuống 0.2, thêm warm-up 25 epoch đầu "
        "(kd_weight=0 trong warm-up, coi như đang train Baseline), giữ nguyên MSE loss cho L_KD. "
        "Mọi thứ khác (dataset, kiến trúc, optimizer, early-stop) giữ nguyên 100%."
    )
    doc.add_paragraph(
        "• Phương án 4 — Teacher Assistant (TA): chèn model trung gian OSNet_x0.5 để thu hẹp "
        "capacity gap thành 2 bước nhỏ hơn thay vì 1 bước lớn (x1.0→x0.25, chênh ~4 lần). Giai "
        "đoạn 1: distill x1.0→x0.5 (TA, kd_weight=0.5 cố định) — TA đạt val_rank1 nội bộ tốt "
        "nhất 84.28%, RẤT cao so với mọi bản x0.25 (~80-82%), đúng như kỳ vọng vì capacity lớn "
        "hơn. Giai đoạn 2: dùng TA (đã đóng băng) làm Teacher mới, distill sang Final Student "
        "x0.25 (kd_weight=0.3 cố định, KHÔNG kết hợp thêm Phương án 1 để giữ 2 phương án tách "
        "biệt rõ ràng)."
    )
    doc.add_paragraph(
        "Cả 2 phương án train MỚI hoàn toàn (không dùng lại checkpoint cũ), mỗi lần chạy 1 seed "
        "(không sweep nhiều seed như guide đề xuất đầy đủ, do giới hạn thời gian — người dùng đã "
        "xác nhận trước khi chạy)."
    )

    add_heading(doc, "2. Rank-1 / Rank-5 / mAP (MSMT17, đo chính thức)")
    add_table(doc, ["Model", "Rank-1", "Rank-5", "mAP"], [
        ["Baseline (không distill)", "52.44%", "70.07%", "27.60%"],
        ["Distill gốc (kd=0.5, tham khảo)", "52.00%", "70.05%", "27.46%"],
        ["Phương án 1 (kd=0.2 + warmup)", "52.49%", "69.69%", "27.43%"],
        ["Phương án 4 — Teacher Assistant x0.5 (Giai đoạn 1, tham khảo)", "68.28%", "82.20%", "41.62%"],
        ["Phương án 4 Final (qua TA, kd=0.3)", "52.94%", "70.79%", "28.23%"],
    ])
    chart1 = make_chart_rank1_map()
    add_image(doc, chart1)
    doc.add_paragraph(
        "Phương án 4 Final đạt CAO NHẤT ở cả 3 chỉ số trong các bản x0.25 (Rank-1 52.94%, Rank-5 "
        "70.79%, mAP 28.23%) — vượt Baseline (+0.50/+0.72/+0.63 điểm phần trăm). Đây là cải thiện "
        "thật nhưng vẫn khiêm tốn về độ lớn. Phương án 1 chỉ nhỉnh hơn Baseline ở Rank-1 (+0.05), "
        "còn Rank-5/mAP thấp hơn — không có cải thiện rõ ràng. TA (x0.5, kết quả trung gian) cho "
        "thấy KD hoạt động RẤT tốt khi capacity gap nhỏ (~2 lần) — củng cố giả thuyết gốc của "
        "Phương án 4 rằng capacity gap là nguyên nhân chính khiến distill trực tiếp x1.0→x0.25 "
        "thất bại."
    )

    add_heading(doc, "3. Tier-2 TPR (Own Video Test)")
    add_table(doc, ["Test set", "Baseline", "Phương án 1", "Phương án 4 Final"], [
        ["4 base cases (no augment)", "3/4 (75.0%)", "3/4 (75.0%)", "3/4 (75.0%)"],
        ["25 full cases (2 cams x 6 variants)", "19/25 (76.0%)", "19/25 (76.0%)", "18/25 (72.0%)"],
    ])
    chart2 = make_chart_tpr()
    add_image(doc, chart2)
    doc.add_paragraph(
        "KẾT QUẢ TRÁI CHIỀU, ghi nhận trung thực: ở bài test Gallery nhỏ (2 người) này, Phương án "
        "4 Final — dù thắng rõ ở Rank-1/mAP quy mô lớn — lại THẤP HƠN Baseline (72.0% so với "
        "76.0%, thêm 1 lỗi ở biến thể lật ảnh CAM 2). Phương án 1 giữ nguyên y hệt Baseline "
        "(76.0%). Cho thấy 2 phép đo (Rank-1/mAP quy mô lớn vs TPR Gallery nhỏ) không nhất thiết "
        "đồng thuận — cải thiện ở benchmark lớn không đảm bảo cải thiện ở tình huống triển khai "
        "thực tế cụ thể."
    )

    add_heading(doc, "4. Params / FLOPs & Jetson Nano 4GB")
    doc.add_paragraph(
        "KHÔNG đo lại cho Phương án 1 và Phương án 4 Final — cả 2 đều dùng ĐÚNG kiến trúc "
        "osnet_x0_25 giống Baseline/Distill gốc (chỉ khác hàm loss/nguồn Teacher lúc train), nên "
        "Params/FLOPs/FPS/RAM/nhiệt độ trên Jetson giống hệt nhau về mặt kiến trúc:"
    )
    add_table(doc, ["Metric", "Giá trị (dùng chung cho mọi bản x0.25)"], [
        ["Params", "0.20M"],
        ["FLOPs", "0.091G"],
        ["FPS Jetson Nano 4GB (model đứng riêng)", "64.11 (15.60 ms/call)"],
        ["FPS Jetson Nano 4GB (chạy chung YOLOv8n, xấu nhất)", "21.31"],
        ["RAM peak (Jetson)", "1.73 GB"],
        ["Temperature peak (Jetson)", "36.25°C"],
    ])
    doc.add_paragraph(
        "TA (x0.5, chỉ dùng làm Teacher trung gian, KHÔNG triển khai thật) nặng hơn — không đo "
        "riêng vì không phải model đưa vào pipeline cuối cùng."
    )

    add_heading(doc, "5. Kết luận")
    doc.add_paragraph(
        "Phương án 4 (Teacher Assistant) là hướng cải thiện có cơ sở khoa học rõ ràng nhất trong "
        "2 phương án đã thử — capacity gap thực sự là một phần nguyên nhân khiến distill trực "
        "tiếp thất bại, và thu hẹp gap qua bước trung gian x0.5 giúp cải thiện thật ở Rank-1/mAP "
        "quy mô lớn (dù khiêm tốn, +0.5 điểm Rank-1) — nhưng KHÔNG cải thiện (thậm chí giảm nhẹ) "
        "ở bài test TPR Gallery nhỏ. Phương án 1 (tune hyperparameter) không cho thấy cải thiện "
        "đáng kể ở bất kỳ chỉ số nào so với Baseline. Với chi phí bổ sung của Phương án 4 (thêm "
        "1 lần train TA ~120 epoch) so với lợi ích thực tế còn khiêm tốn và không nhất quán giữa "
        "2 phép đo, CHƯA đủ bằng chứng thuyết phục để thay thế Baseline/Distill gốc hiện đang "
        "dùng trong hệ thống — số liệu được ghi nhận đầy đủ, trung thực làm tài liệu tham khảo."
    )

    add_file_catalog(doc, [
        ("chart_rank1_map.png", "Biểu đồ cột Rank-1/Rank-5/mAP: Baseline vs PA1 vs PA4 Final."),
        ("chart_tpr.png", "Biểu đồ cột Tier-2 TPR (4 base, 25 full): Baseline vs PA1 vs PA4 Final."),
        ("Noi_dung.docx", "Chính file này."),
    ])

    DIR_OUT.mkdir(parents=True, exist_ok=True)
    doc.save(str(DIR_OUT / "Noi_dung.docx"))
    print(f"Da luu: {DIR_OUT / 'Noi_dung.docx'}")


if __name__ == "__main__":
    DIR_OUT.mkdir(parents=True, exist_ok=True)
    build_doc()
