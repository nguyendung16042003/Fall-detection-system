"""Tao 3 file Word (Bai cua nhom / Doi thu / So sanh) cho report P1, moi file
co muc "Danh muc file trong thu muc" giai thich tung file dinh kem."""
import sys
from pathlib import Path

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

sys.stdout.reconfigure(encoding="utf-8")

AI_ROOT = Path(r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)")
BASE = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Số liệu thống kê" / "P1"
DIR_OWN = BASE / "1. Bai cua nhom (OSNet x0.5)"
DIR_COMP = BASE / "2. Mô phỏng Doi thu (Backbone dung chung)"
DIR_CMP = BASE / "3. So sanh đối thủ"


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


def add_file_catalog(doc, entries):
    """entries: list (ten_file, giai_thich)."""
    add_heading(doc, "Danh mục file trong thư mục này", level=2)
    for name, desc in entries:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(name)
        run.bold = True
        p.add_run(f" — {desc}")


# ============================================================
# 1. BAI CUA NHOM (OSNet x0.5)
# ============================================================
def build_doc_own():
    doc = Document()
    doc.add_heading("P1 — Cross-Camera Re-ID: Bài của nhóm (OSNet x0.5)", level=0)

    add_heading(doc, "1. Cách triển khai")
    doc.add_paragraph(
        "Kiến trúc: OSNet là 1 model Re-ID ĐỘC LẬP, tách hẳn khỏi backbone dùng chung với "
        "Pose Head (khác thiết kế ban đầu). OSNet nhận trực tiếp ảnh crop RGB của người (từ "
        "bbox do YOLOv8n phát hiện), không đi qua Fused Feature Vector của P3. Quyết định "
        "này đưa ra sau khi 4 thí nghiệm với backbone dùng chung đều thất bại rõ rệt "
        "(xem folder '2. Đối thủ')."
    )
    doc.add_paragraph(
        "OSNet (Omni-Scale Network, Zhou et al. ICCV 2019) là kiến trúc CNN thiết kế chuyên "
        "biệt cho bài toán Re-ID — mỗi block có nhiều nhánh với receptive field khác nhau "
        "(bắt chi tiết nhỏ như hoạ tiết áo lẫn dáng người tổng thể), khác hẳn 1 backbone phân "
        "loại thông thường."
    )
    doc.add_paragraph(
        "Biến thể đã chọn: osnet_x0_5 (~0.6M tham số, nhẹ nhất trong 3 bản đã thử: x1_0/"
        "x0_75/x0_5) — chọn sau khi đo FPS/RAM/nhiệt độ THẬT trên Jetson Nano 4GB, chạy CHUNG "
        "với YOLOv8n (input 320×320, cũng đã giảm từ 640×640 mặc định vì lý do hiệu năng)."
    )
    doc.add_paragraph(
        "Checkpoint: dùng nguyên trạng osnet_x0_5 đã train sẵn trên MSMT17 (nguồn: MODEL_ZOO "
        "chính thức của thư viện deep-person-reid/torchreid) — KHÔNG tự train lại, vì mục "
        "tiêu là dùng 1 model Re-ID đã được kiểm chứng, thay vì tự huấn luyện lại từ đầu."
    )
    doc.add_paragraph(
        "Quy trình test gồm 2 tầng độc lập (không gộp chung số liệu):\n"
        "— Tầng 1: đo Rank-1/mAP trên bộ test chuẩn MSMT17 (closed-set, hàng nghìn danh tính "
        "lạ) — xác nhận model có học tổng quát tốt không, so được với mốc literature.\n"
        "— Tầng 2: chạy đúng pipeline 4 lớp (trích đặc trưng → buffer → Gallery → so khớp) "
        "trên data tự quay thật (Gallery 2 người, Enrollment + video test), tính True "
        "Positive Rate — đo khả năng hoạt động thực tế tại nhà."
    )

    add_heading(doc, "2. Số liệu khi test")
    doc.add_paragraph("Tầng 1 — Rank-1/mAP trên MSMT17 (11.659 ảnh query / 82.161 ảnh gallery):")
    add_table(doc, ["Chỉ số", "Giá trị"], [
        ["Rank-1", "71.14%"],
        ["Rank-5", "82.99%"],
        ["mAP", "41.99%"],
        ["Mốc công bố gốc (MODEL_ZOO deep-person-reid)", "Rank-1 69.7% / mAP 37.5%"],
    ])
    doc.add_paragraph(
        "Số đo được cao hơn nhẹ so với mốc công bố gốc — hợp lý, sai khác nhỏ do biến động "
        "khi đo lại độc lập."
    )

    doc.add_paragraph("Tầng 2 — True Positive Rate trên data thật (Gallery: Person 1 + Person 2), "
                       "CẢ 2 camera Scene 5 (CAM 1 + CAM 2):")
    add_table(doc, ["Bộ test", "Số case đúng / tổng", "Tỷ lệ"], [
        ["4 case gốc (chưa augment)", "3/4", "75.0%"],
        ["25 case đầy đủ (2 camera × 6 biến thể: gốc + sáng/tối/nhiễu/nén/lật)", "21/25", "84.0%"],
    ])
    doc.add_paragraph(
        "Đây là số liệu ĐẦY ĐỦ NHẤT, dùng cả 2 camera Scene 5 (lần đo đầu tiên chỉ dùng CAM 2 "
        "nên chỉ có 2 case gốc/13 case mở rộng — đã bổ sung CAM 1 để có bức tranh đầy đủ). "
        "CAM 1 riêng: 2/2 đúng ở mọi biến thể (rất ổn định); CAM 2 riêng dao động 1-2/2-3 "
        "đúng tuỳ biến thể (đây là camera có case khó — track_id quay lại lần 2 hay bị nhầm)."
    )

    doc.add_paragraph("FPS/RAM/nhiệt độ thật trên Jetson Nano 4GB (chạy CHUNG YOLOv8n 320×320 + OSNet x0.5, "
                       "kịch bản xấu nhất — Re-ID chạy mỗi frame):")
    add_table(doc, ["Chỉ số", "Đo được", "Mục tiêu", "Đạt?"], [
        ["FPS", "18.15", "≥ 12", "Đạt"],
        ["RAM peak", "2.31 GB", "< 3.5 GB", "Đạt"],
        ["Nhiệt độ", "41.8°C", "< 80°C", "Đạt"],
    ])

    add_heading(doc, "3. Các trường hợp sai — phân tích lý do")
    doc.add_paragraph(
        "Toàn bộ case sai (4/25, 16%) đều rơi vào CAM 2 — lần Person 1 quay lại thứ 2 trên "
        "CAM 2 bị đoán nhầm thành 'Person 2' ở hầu hết biến thể (điểm ~0.76 — vượt ngưỡng "
        "0.6 nhưng khớp sai người). CAM 1 hoàn toàn ổn định (12/12 đúng ở mọi biến thể). "
        "Nguyên nhân khả dĩ: góc quay/tư thế/ánh sáng của CAM 2 tại đúng khoảnh khắc đó "
        "khiến đặc trưng ngoại hình trích được tình cờ gần với Gallery của Person 2 hơn — "
        "rủi ro cố hữu của Re-ID dựa hoàn toàn vào ngoại hình (không có thông tin bổ trợ như "
        "thời gian/vị trí)."
    )
    doc.add_paragraph(
        "16% sai cho thấy còn dư địa cải thiện: ngưỡng khớp (0.6) có thể chưa tối ưu cho đúng "
        "phân bố điểm của OSNet x0.5, số frame buffer (tối đa 10) có thể chưa đủ để trung "
        "bình hoá nhiễu, hoặc chất lượng crop khi người ở xa/góc nghiêng tại CAM 2."
    )

    add_heading(doc, "4. Gợi ý dùng số liệu/hình ảnh này khi viết Mục 6")
    doc.add_paragraph(
        "Nên trích: bảng Tầng 1 + Tầng 2 làm bảng số liệu chính; đoạn video minh hoạ dùng làm "
        "hình ảnh/screenshot minh hoạ pipeline hoạt động thật (có thể cắt 2-3 khung hình từ "
        "video làm ảnh tĩnh chèn vào Word nếu cần, thay vì nhúng cả video); bảng FPS/RAM dùng "
        "cho phần đánh giá khả năng triển khai trên Jetson Nano 4GB."
    )

    add_file_catalog(doc, [
        ("OSNet_x0.5_video_Scene5_CAM1_CAM2_ghep.mp4",
         "VIDEO CHÍNH — ghép 2 camera Scene 5 (CAM 1 bên trái, CAM 2 bên phải) chạy SONG "
         "SONG cùng lúc thật (2 camera quay đồng bộ, xem Mục 1), mỗi camera tự vẽ khung + "
         "nhãn 'Person N' theo kết quả hệ thống thật. Đây là bằng chứng ĐẦY ĐỦ NHẤT (4 case, "
         "3/4 đúng) — nên dùng ảnh cắt từ video này làm hình minh hoạ chính trong report."),
        ("OSNet_x0.5_video_dang_ky_va_test.mp4",
         "Video phụ, 2 đoạn nối tiếp: (1) đoạn Đăng ký Person 1 (Enrollment, quay từ CAM 2), "
         "(2) đoạn Test CHỈ CAM 2 (2 case, phiên bản đo lần đầu trước khi bổ sung CAM 1). Giữ "
         "lại để đối chiếu, không phải bằng chứng chính."),
        ("Noi_dung.docx", "Chính file này."),
    ])

    doc.save(str(DIR_OWN / "Noi_dung.docx"))
    print(f"Da luu: {DIR_OWN / 'Noi_dung.docx'}")


# ============================================================
# 2. DOI THU (Backbone dung chung)
# ============================================================
def build_doc_competitor():
    doc = Document()
    doc.add_heading("P1 — Cross-Camera Re-ID: Đối thủ tự-so-sánh (Backbone dùng chung)", level=0)

    add_heading(doc, "1. Cách triển khai")
    doc.add_paragraph(
        "Đây là thiết kế BAN ĐẦU của nhóm (trước khi chuyển sang OSNet độc lập) — dùng LẠI "
        "chính backbone đã train cho Pose Head (YOLOv8n-cls, phân loại 5 tư thế: bend/"
        "exercise/lie/sit/stand), gắn thêm 1 Re-ID Head mới (Linear + BNNeck + classifier) "
        "lên trên, train trên MSMT17. Lý do ban đầu chọn hướng này: tiết kiệm tài nguyên "
        "Jetson Nano 4GB (không cần chạy thêm 1 model riêng)."
    )
    doc.add_paragraph("Đã thử đủ 4 cấu hình train khác nhau để đảm bảo không phải do cấu hình sai:")
    add_table(doc, ["#", "Cấu hình"], [
        ["1", "Đóng băng backbone, train Re-ID head (CE + Triplet Loss), 60 epoch"],
        ["2", "Mở khoá backbone, train 20 epoch (thử nhanh)"],
        ["3", "Mở khoá backbone, train 60 epoch (đúng thiết kế gốc trong tài liệu hướng dẫn)"],
        ["4", "Đóng băng backbone, chỉ dùng Triplet Loss thuần (bỏ Cross-Entropy)"],
    ])

    add_heading(doc, "2. Số liệu khi test")
    add_table(doc, ["Cấu hình", "Rank-1 (MSMT17)", "mAP"], [
        ["1. Đóng băng + CE/Triplet", "4.88%", "1.11%"],
        ["2. Mở khoá, 20 epoch", "3.72%", "0.87%"],
        ["3. Mở khoá, 60 epoch (đúng thiết kế gốc)", "3.48%", "0.78%"],
        ["4. Đóng băng + Pure Triplet", "4.25%", "0.90%"],
    ])
    doc.add_paragraph("Test Tầng 2 (True Positive Rate, dùng checkpoint tốt nhất — cấu hình 1), "
                       "cả 2 camera Scene 5:")
    add_table(doc, ["Bộ test", "Số case đúng / tổng", "Tỷ lệ"], [
        ["4 case gốc (chưa augment)", "0/4", "0.0%"],
        ["25 case đầy đủ (2 camera × 6 biến thể)", "0/25", "0.0%"],
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
        "hình (màu áo, hoạ tiết, hình dáng riêng) mà Re-ID cần, vì chúng không giúp ích gì "
        "cho việc phân biệt tư thế. Càng train sâu hơn (cấu hình 2, 3) kết quả càng KÉM đi "
        "(overfit vào 1041 danh tính train, không tổng quát hoá được sang người lạ) — không "
        "phải do thiếu epoch."
    )

    add_file_catalog(doc, [
        ("Shared_backbone_video_Scene5_CAM1_CAM2_ghep.mp4",
         "VIDEO CHÍNH — cùng cấu trúc ghép 2 camera như bài của nhóm (Mục 1), chạy trên "
         "CÙNG data (Scene 5, cả CAM 1 + CAM 2). Cả 4/4 case đều bị hệ thống này nhận NHẦM "
         "(nhãn hiển thị Person 2/3/4/5 — không có case nào ra đúng 'Person 1')."),
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
    doc.add_heading("P1 — Cross-Camera Re-ID: So sánh Bài của nhóm vs Đối thủ", level=0)

    add_heading(doc, "1. Bảng so sánh trực tiếp")
    add_table(doc, ["Metric", "OSNet x0.5 (Ours)", "Shared Backbone (Baseline)"], [
        ["Rank-1 (MSMT17)", "71.14%", "3.48% - 4.88% (4 configs)"],
        ["mAP (MSMT17)", "41.99%", "0.78% - 1.11%"],
        ["TPR - Tier 2 (4 base cases, 2 cams)", "3/4 (75%)", "0/4 (0%)"],
        ["TPR - Tier 2 (25 full cases, 2 cams x 6 variants)", "21/25 (84.0%)", "0/25 (0%)"],
        ["FPS on Jetson Nano 4GB (with YOLOv8n)", "18.15 (target >=12 met)", "not measured (rejected before this step)"],
    ])

    add_heading(doc, "2. Kết luận")
    doc.add_paragraph(
        "OSNet x0.5 thắng áp đảo ở MỌI chỉ số, kể cả sau khi đối thủ đã thử đủ 4 cấu hình "
        "train khác nhau (không phải do đối thủ thiếu cố gắng). Backbone dùng chung với Pose "
        "Head — dù tiết kiệm tài nguyên hơn về lý thuyết — thất bại tuyệt đối trên bài toán "
        "thực tế. Quyết định chuyển sang OSNet làm model Re-ID độc lập (đề xuất kiến trúc "
        "mới) là đúng đắn, có bằng chứng thực nghiệm đầy đủ, không phải suy đoán."
    )
    doc.add_paragraph(
        "Đánh đổi thật cần ghi nhận: OSNet là model bổ sung (không dùng chung tài nguyên với "
        "Pose Head) — tốn thêm ~18ms/lần suy luận trên Jetson Nano 4GB so với thiết kế cũ "
        "(lý thuyết rẻ hơn nhưng không hoạt động). Đã đo và xác nhận vẫn đạt mọi tiêu chí "
        "hiệu năng (FPS/RAM/nhiệt độ) khi chạy chung với YOLOv8n thật trên thiết bị thật."
    )

    add_heading(doc, "3. Biểu đồ đính kèm")
    doc.add_paragraph(
        "chart_rank1.png — so Rank-1 cả 4 cấu hình đối thủ + OSNet x0.5 trên cùng 1 biểu đồ."
    )
    doc.add_paragraph(
        "chart_tpr.png — so điểm khớp Gallery giữa 2 hệ thống trên 2 case gốc, có đánh dấu "
        "đúng/sai bằng màu."
    )

    add_file_catalog(doc, [
        ("chart_rank1.png",
         "Biểu đồ cột: Rank-1 của 4 cấu hình đối thủ (3.48-4.88%) so với OSNet x0.5 (71.14%) "
         "trên cùng 1 trục — thấy rõ khoảng cách."),
        ("chart_tpr.png",
         "Biểu đồ cột: điểm số khớp Gallery của 2 hệ thống trên 2 case gốc (track_id=11, 14) "
         "— màu xanh=đúng, đỏ=sai, có vẽ ngưỡng khớp 0.6 tham chiếu."),
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
