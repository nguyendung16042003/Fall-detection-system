"""Tao 3 file Word cho report Re-Identification (ten hoc thuat thay P1 --
xem Report chinh.docx muc 1.2/Part 1), voi model MOI OSNet_x0.25 (Knowledge
Distillation tu OSNet_x1.0) thay cho OSNet_x0.5."""
import sys
from pathlib import Path

from docx import Document
from docx.shared import Inches

sys.stdout.reconfigure(encoding="utf-8")

AI_ROOT = Path(r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)")
BASE = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Số liệu thống kê" / "Re-Identification"
DIR_OWN = BASE / "1. Bai cua nhom (OSNet x0.25 KD)"
DIR_COMP = BASE / "2. Mô phỏng Doi thu (Backbone dung chung)"
DIR_CMP = BASE / "3. So sanh"


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


# ============================================================
# 1. BAI CUA NHOM (OSNet x0.25 KD)
# ============================================================
def build_doc_own():
    doc = Document()
    doc.add_heading("Re-Identification — Bài của nhóm (OSNet x0.25, Knowledge Distillation)", level=0)

    add_heading(doc, "1. Cách triển khai")
    doc.add_paragraph(
        "Kiến trúc: OSNet là 1 model Re-Identification ĐỘC LẬP, tách hẳn khỏi backbone dùng "
        "chung với Pose Head. OSNet nhận trực tiếp ảnh crop RGB của người (từ bbox do YOLOv8n "
        "phát hiện), không đi qua Fused Feature Vector của Boundary Feature Fusion. Quyết định "
        "này đưa ra sau khi 4 thí nghiệm với backbone dùng chung đều thất bại rõ rệt (xem "
        "folder '2. Mô phỏng Đối thủ')."
    )
    doc.add_paragraph(
        "Biến thể hiện dùng: osnet_x0_25 (~0.2M tham số, nhẹ nhất trong 4 bản đã thử: x1_0/"
        "x0_75/x0_5/x0_25) — chuyển từ x0.5 sang x0.25 qua Knowledge Distillation (Teacher = "
        "osnet_x1_0 đã train MSMT17 thật, đóng băng; Student = osnet_x0_25, khởi tạo ImageNet "
        "pretrained). Loss tổng hợp: L = 1.0×L_ID (Cross-Entropy) + 1.0×L_triplet (margin=0.3) "
        "+ 0.5×L_KD (MSE giữa embedding 512-d đã L2-normalize của Teacher/Student). Optimizer "
        "Adam (lr=3.5e-4, weight_decay=5e-4), tối thiểu 30 epoch, early-stop nếu 10 epoch liên "
        "tiếp không cải thiện Rank-1 nội bộ (dừng thật tại epoch 56, tốt nhất epoch 46)."
    )
    doc.add_paragraph(
        "Lưu ý phát hiện & sửa lỗi khi train KD: lần train đầu tiên dùng nhầm tiền xử lý ảnh "
        "kiểu Ultralytics classify_transforms (224×224 vuông, không chuẩn hoá mean/std) — khác "
        "hẳn tiền xử lý chuẩn OSNet (Resize 256×128 + chuẩn hoá ImageNet) mà torchreid "
        "FeatureExtractor dùng lúc triển khai thật. Hậu quả: Rank-1 nội bộ báo 72% nhưng Rank-1 "
        "THẬT trên bộ query/gallery chuẩn chỉ 0.24% (gần ngẫu nhiên). Đã sửa đúng transform "
        "chuẩn OSNet và train lại từ đầu — số liệu trong tài liệu này là SAU khi sửa lỗi."
    )
    doc.add_paragraph(
        "Checkpoint Teacher (osnet_x1_0): dùng nguyên trạng đã train sẵn trên MSMT17 (nguồn: "
        "MODEL_ZOO chính thức của thư viện deep-person-reid/torchreid) — không tự train lại."
    )
    doc.add_paragraph(
        "Quy trình test gồm 2 tầng độc lập (không gộp chung số liệu):\n"
        "— Tầng 1: đo Rank-1/mAP trên bộ test chuẩn MSMT17 (closed-set, hàng nghìn danh tính "
        "lạ) — xác nhận model có học tổng quát tốt không.\n"
        "— Tầng 2: chạy đúng pipeline 4 lớp (trích đặc trưng → buffer → Gallery → so khớp) "
        "trên data tự quay thật (Gallery 2 người, Enrollment + video test), tính True "
        "Positive Rate — đo khả năng hoạt động thực tế tại nhà."
    )

    add_heading(doc, "2. Số liệu khi test")
    doc.add_paragraph("Tầng 1 — Rank-1/mAP trên MSMT17 (11.659 ảnh query / 82.161 ảnh gallery), "
                       "so với model x0.5 trước đó (đã thay thế):")
    add_table(doc, ["Metric", "OSNet x0.25 (KD, current)", "OSNet x0.5 (old, replaced)"], [
        ["Rank-1", "52.00%", "71.14%"],
        ["Rank-5", "70.05%", "82.99%"],
        ["mAP", "27.46%", "41.99%"],
    ])
    doc.add_paragraph(
        "Đánh đổi thật: x0.25 (KD) thấp hơn x0.5 rõ rệt ở benchmark quy mô lớn (gallery 82k "
        "ảnh, hàng nghìn danh tính lạ — bài toán khó, đòi hỏi đặc trưng phân biệt tinh vi). "
        "Knowledge Distillation giúp cải thiện đáng kể so với train từ đầu không có Teacher "
        "(Rank-1 nội bộ tăng dần từ ImageNet-init đến 80.87% qua các epoch), nhưng model nhỏ "
        "hơn 4 lần vẫn không thể tiệm cận hoàn toàn Teacher — điều này phù hợp với kỳ vọng "
        "trong tài liệu Knowledge Distillation (mục tiêu là 'tiệm cận', không phải 'bằng')."
    )

    doc.add_paragraph("Tầng 2 — True Positive Rate trên data thật (Gallery: Person 1 + Person 2), "
                       "CẢ 2 camera Scene 5 (CAM 1 + CAM 2):")
    add_table(doc, ["Test set", "OSNet x0.25 KD", "OSNet x0.5 (old)"], [
        ["4 base cases (no augment)", "3/4 (75.0%)", "3/4 (75.0%)"],
        ["24 full cases (2 cams x 6 variants)", "19/25 (76.0%)", "21/25 (84.0%)"],
    ])
    doc.add_paragraph(
        "Điểm đáng chú ý: ở bài toán Tầng 2 (Gallery chỉ 2 người, dễ hơn nhiều so với benchmark "
        "82k ảnh), khoảng cách giữa 2 model THU HẸP đáng kể (75%/76% so với 75%/84%) — không "
        "còn chênh lệch lớn như ở Tầng 1. Đây là bằng chứng thực nghiệm quan trọng: với bài "
        "toán triển khai thực tế tại nhà (Gallery nhỏ, ít người), model nhỏ hơn 4 lần vẫn giữ "
        "được phần lớn năng lực nhận dạng."
    )

    doc.add_paragraph("FPS/RAM/nhiệt độ THẬT trên Jetson Nano 4GB (chạy CHUNG YOLOv8n 320×320 + "
                       "OSNet, kịch bản xấu nhất — Re-ID chạy mỗi frame), đo lại cho model mới:")
    add_table(doc, ["Metric", "OSNet x0.25 KD", "OSNet x0.5 (old)", "Target"], [
        ["FPS (worst-case, both models)", "21.31", "18.15", ">= 12"],
        ["RAM peak", "1.73 GB", "2.31 GB", "< 3.5 GB"],
        ["Temperature (thermal zone) peak", "36.25°C", "41.8°C", "< 80°C"],
        ["OSNet only (latency)", "15.60 ms/call (64.11 FPS)", "not measured separately", "—"],
    ])
    doc.add_paragraph(
        "OSNet x0.25 (KD) NHANH HƠN, NHẸ HƠN, MÁT HƠN x0.5 ở mọi chỉ số — hợp lý vì số tham số "
        "giảm ~4 lần (osnet_x0_5 → osnet_x0_25). Đây chính là lợi ích cốt lõi của việc chuyển "
        "sang model nhỏ hơn qua Knowledge Distillation: đổi lấy tốc độ/tài nguyên rõ rệt, chấp "
        "nhận giảm độ chính xác Tầng 1 (không giảm nhiều ở Tầng 2, xem trên)."
    )

    add_heading(doc, "3. Các trường hợp sai — phân tích lý do")
    doc.add_paragraph(
        "Toàn bộ case sai còn lại (6/25, 24%) đều rơi vào CAM 2 — lần Person 1 quay lại thứ 2 "
        "trên CAM 2 bị đoán nhầm thành 'Person 2' ở hầu hết biến thể (điểm ~0.75-0.80 — vượt "
        "ngưỡng 0.6 nhưng khớp sai người). Đây là CÙNG loại lỗi đã ghi nhận với x0.5 (cùng "
        "track_id, cùng vị trí trong video) — không phải lỗi mới do model nhỏ hơn gây ra, mà là "
        "hạn chế cố hữu của bài toán tại đúng khoảnh khắc đó (góc quay/tư thế/ánh sáng CAM 2)."
    )
    doc.add_paragraph(
        "So với x0.5, x0.25 (KD) có thêm 2 case sai nữa trong 25 case đầy đủ (19/25 so với "
        "21/25) — nhưng vẫn giữ đúng 3/4 ở 4 case gốc đáng tin cậy nhất. Cho thấy model nhỏ "
        "hơn nhạy cảm hơn 1 chút với biến thể augment (sáng/tối/nhiễu/nén/lật) ở đúng case khó "
        "sẵn có, không phải suy giảm toàn diện."
    )

    add_heading(doc, "4. Gợi ý dùng số liệu/hình ảnh này khi viết Mục 6")
    doc.add_paragraph(
        "Nên trích: bảng Tầng 1 + Tầng 2 + bảng FPS/RAM/nhiệt độ làm bảng số liệu chính, nhấn "
        "mạnh đánh đổi tốc độ/độ chính xác rõ ràng, có số đo thật cả 2 model để so sánh trực "
        "tiếp; đoạn video minh hoạ dùng làm hình ảnh/screenshot minh hoạ pipeline hoạt động "
        "thật."
    )

    add_file_catalog(doc, [
        ("OSNet_x025KD_Scene5_ghep.mp4",
         "VIDEO CHÍNH — ghép 2 camera Scene 5 (CAM 1 bên trái, CAM 2 bên phải) chạy SONG "
         "SONG cùng lúc thật, mỗi camera tự vẽ khung + nhãn 'Person N' theo kết quả hệ thống "
         "thật (model OSNet_x0.25 sau Knowledge Distillation). 3/4 case đúng."),
        ("OSNet_x025KD_dangky_test.mp4",
         "Video phụ, 2 đoạn nối tiếp: (1) đoạn Đăng ký Person 1 (Enrollment, quay từ CAM 2), "
         "(2) đoạn Test CHỈ CAM 2. Giữ lại để đối chiếu, không phải bằng chứng chính."),
        ("Noi_dung.docx", "Chính file này."),
    ])

    doc.save(str(DIR_OWN / "Noi_dung.docx"))
    print(f"Da luu: {DIR_OWN / 'Noi_dung.docx'}")


# ============================================================
# 2. DOI THU (Backbone dung chung) -- KHONG DOI SO LIEU, chi sua ten P1
# ============================================================
def build_doc_competitor():
    doc = Document()
    doc.add_heading("Re-Identification — Đối thủ tự-so-sánh (Backbone dùng chung)", level=0)

    add_heading(doc, "1. Cách triển khai")
    doc.add_paragraph(
        "Đây là thiết kế BAN ĐẦU của nhóm (trước khi chuyển sang OSNet độc lập) — dùng LẠI "
        "chính backbone đã train cho Pose Head (YOLOv8n-cls, phân loại 5 tư thế: bend/"
        "exercise/lie/sit/stand), gắn thêm 1 Re-Identification Head mới (Linear + BNNeck + "
        "classifier) lên trên, train trên MSMT17. Lý do ban đầu chọn hướng này: tiết kiệm tài "
        "nguyên Jetson Nano 4GB (không cần chạy thêm 1 model riêng)."
    )
    doc.add_paragraph("Đã thử đủ 4 cấu hình train khác nhau để đảm bảo không phải do cấu hình sai:")
    add_table(doc, ["#", "Configuration"], [
        ["1", "Frozen backbone, train Re-ID head (CE + Triplet Loss), 60 epochs"],
        ["2", "Unfrozen backbone, 20 epochs (quick test)"],
        ["3", "Unfrozen backbone, 60 epochs (original design)"],
        ["4", "Frozen backbone, Pure Triplet Loss only (no Cross-Entropy)"],
    ])

    add_heading(doc, "2. Số liệu khi test")
    add_table(doc, ["Configuration", "Rank-1 (MSMT17)", "mAP"], [
        ["1. Frozen + CE/Triplet", "4.88%", "1.11%"],
        ["2. Unfrozen, 20 epochs", "3.72%", "0.87%"],
        ["3. Unfrozen, 60 epochs (original design)", "3.48%", "0.78%"],
        ["4. Frozen + Pure Triplet", "4.25%", "0.90%"],
    ])
    doc.add_paragraph("Test Tầng 2 (True Positive Rate, dùng checkpoint tốt nhất — cấu hình 1), "
                       "cả 2 camera Scene 5:")
    add_table(doc, ["Test set", "Correct / Total", "Rate"], [
        ["4 base cases (no augment)", "0/4", "0.0%"],
        ["24 full cases (2 cams x 6 variants)", "0/25", "0.0%"],
    ])

    add_heading(doc, "3. Các trường hợp sai — phân tích lý do")
    doc.add_paragraph(
        "Thất bại TUYỆT ĐỐI ở Tầng 2 (0/25, cả 2 camera, mọi biến thể ánh sáng/nhiễu/nén/lật) "
        "— không phải ngẫu nhiên. Đã xác minh KHÔNG PHẢI bug qua 2 cách độc lập: (a) đối chiếu code "
        "tính Rank-1/mAP khớp chính xác với hàm gốc của thư viện torchreid; (b) kiểm tra "
        "embedding vẫn mang tín hiệu nhận dạng thật (cosine similarity giữa ảnh CÙNG người "
        "trung bình ~0.10-0.12, cao hơn rõ rệt so với KHÁC người ~0.002) — tức là model không "
        "hỏng, chỉ là tín hiệu quá yếu để phân biệt chính xác."
    )
    doc.add_paragraph(
        "Nguyên nhân gốc: backbone vốn được train CHUYÊN BIỆT cho phân loại tư thế (5 lớp "
        "thô: đứng/ngồi/nằm...) — các lớp lọc có xu hướng bỏ qua đúng những chi tiết ngoại "
        "hình (màu áo, hoạ tiết, hình dáng riêng) mà Re-Identification cần, vì chúng không "
        "giúp ích gì cho việc phân biệt tư thế. Càng train sâu hơn (cấu hình 2, 3) kết quả "
        "càng KÉM đi (overfit vào 1041 danh tính train, không tổng quát hoá được sang người "
        "lạ) — không phải do thiếu epoch."
    )

    add_file_catalog(doc, [
        ("Shared_backbone_video_Scene5_CAM1_CAM2_ghep.mp4",
         "VIDEO CHÍNH — cùng cấu trúc ghép 2 camera như bài của nhóm (Mục 1), chạy trên "
         "CÙNG data (Scene 5, cả CAM 1 + CAM 2). Cả 4/4 case đều bị hệ thống này nhận NHẦM."),
        ("Shared_backbone_video_dang_ky_va_test.mp4",
         "Video phụ, chỉ CAM 2 (2 case, phiên bản đo lần đầu). Giữ lại để đối chiếu."),
        ("Noi_dung.docx", "Chính file này."),
    ])

    doc.save(str(DIR_COMP / "Noi_dung.docx"))
    print(f"Da luu: {DIR_COMP / 'Noi_dung.docx'}")


# ============================================================
# 3. SO SANH
# ============================================================
def build_doc_compare():
    doc = Document()
    doc.add_heading("Re-Identification — So sánh Bài của nhóm vs Đối thủ", level=0)

    add_heading(doc, "1. Bảng so sánh trực tiếp")
    add_table(doc, ["Metric", "OSNet x0.25 KD (Ours)", "Shared Backbone (Baseline)"], [
        ["Rank-1 (MSMT17)", "52.00%", "3.48% - 4.88% (4 configs)"],
        ["mAP (MSMT17)", "27.46%", "0.78% - 1.11%"],
        ["TPR - Tier 2 (4 base cases, 2 cams)", "3/4 (75%)", "0/4 (0%)"],
        ["TPR - Tier 2 (25 full cases, 2 cams x 6 variants)", "19/25 (76.0%)", "0/25 (0%)"],
        ["FPS on Jetson Nano 4GB (with YOLOv8n)", "21.31 (target >=12 met)", "not measured (rejected before this step)"],
        ["RAM peak on Jetson Nano 4GB", "1.73 GB", "not measured"],
    ])
    add_image(doc, DIR_CMP / "chart_rank1.png")
    add_image(doc, DIR_CMP / "chart_tpr.png")

    add_heading(doc, "2. Kết luận")
    doc.add_paragraph(
        "OSNet x0.25 (Knowledge Distillation) thắng áp đảo Backbone dùng chung ở MỌI chỉ số, "
        "kể cả sau khi đối thủ đã thử đủ 4 cấu hình train khác nhau (không phải do đối thủ "
        "thiếu cố gắng): Rank-1 52.00% vs 3.48-4.88%, mAP 27.46% vs 0.78-1.11%, True Positive "
        "Rate Tầng 2 75-76% vs 0% tuyệt đối ở mọi biến thể. Đây là bằng chứng thực nghiệm rõ "
        "ràng cho quyết định kiến trúc dùng OSNet làm model Re-Identification độc lập, tách "
        "khỏi backbone dùng chung với Pose Head."
    )
    doc.add_paragraph(
        "Lưu ý phạm vi: bảng và biểu đồ trên đây so sánh với Đối thủ (Backbone dùng chung) — "
        "riêng phần OSNet x0.25 (KD) đã đổi từ phiên bản OSNet x0.5 dùng trước đó (đổi lấy tốc "
        "độ/tài nguyên trên Jetson, xem chi tiết so sánh 2 phiên bản OSNet trong Mục 2 file "
        "'1. Bài của nhóm') — đây là 2 phép so sánh khác nhau, không gộp chung."
    )

    add_heading(doc, "3. Biểu đồ đính kèm")
    doc.add_paragraph("chart_rank1.png — so Rank-1 cả 4 cấu hình đối thủ + OSNet x0.25 KD trên cùng 1 biểu đồ.")
    doc.add_paragraph("chart_tpr.png — so điểm khớp Gallery giữa 2 hệ thống trên 4 case gốc (2 camera), có đánh dấu đúng/sai bằng màu.")

    add_file_catalog(doc, [
        ("chart_rank1.png",
         "Biểu đồ cột: Rank-1 của 4 cấu hình đối thủ (3.48-4.88%) so với OSNet x0.25 KD "
         "(52.00%) trên cùng 1 trục."),
        ("chart_tpr.png",
         "Biểu đồ cột: điểm số khớp Gallery của 2 hệ thống trên 4 case gốc — màu xanh=đúng, "
         "đỏ=sai, có vẽ ngưỡng khớp 0.6 tham chiếu."),
        ("Noi_dung.docx", "Chính file này."),
    ])

    doc.save(str(DIR_CMP / "Noi_dung.docx"))
    print(f"Da luu: {DIR_CMP / 'Noi_dung.docx'}")


def main():
    DIR_OWN.mkdir(parents=True, exist_ok=True)
    DIR_COMP.mkdir(parents=True, exist_ok=True)
    DIR_CMP.mkdir(parents=True, exist_ok=True)
    build_doc_own()
    build_doc_competitor()
    build_doc_compare()


if __name__ == "__main__":
    main()
