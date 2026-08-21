"""Tao 3 file Word (Bai cua nhom / Doi thu / So sanh) cho report P3, cung cau
truc voi make_p1_report_docs.py (co "Danh muc file trong thu muc")."""
import sys
from pathlib import Path

from docx import Document
from docx.shared import Inches

sys.stdout.reconfigure(encoding="utf-8")

AI_ROOT = Path(r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)")
BASE = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Số liệu thống kê" / "P3"
DIR_OWN = BASE / "1. Bai cua nhom (P3 Fusion)"
DIR_COMP = BASE / "2. Doi thu (Espinosa 2019)"
DIR_CMP = BASE / "3. So sánh đối thủ"


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
# 1. BAI CUA NHOM (P3 Fusion)
# ============================================================
def build_doc_own():
    doc = Document()
    doc.add_heading("P3 — Cross-Camera Attention Fusion: Bài của nhóm", level=0)

    add_heading(doc, "1. Cách triển khai")
    doc.add_paragraph(
        "Theo đúng \"Context xử lý vấn đề 3 (no_training).md\": toàn bộ cơ chế gộp đặc trưng "
        "là CÔNG THỨC TOÁN CỐ ĐỊNH, KHÔNG có tham số nào cần học (không Wq/Wk/Wv, không "
        "Modality Dropout). Dùng lại NGUYÊN TRẠNG Backbone + Pose Head (YOLOv8n-cls, 5 lớp "
        "bend/exercise/lie/sit/stand) đã train sẵn từ Giai đoạn A — không train lại gì thêm."
    )
    doc.add_paragraph(
        "Pipeline cho 1 cặp người đã ghép (kế thừa bước ghép cặp bằng khoảng cách sàn qua "
        "homography của P2 — nếu > ngưỡng DIST_THRESHOLD=1.5m thì coi là chưa khớp, không "
        "gộp; ngưỡng này lớn hơn ngưỡng P2 dùng cho 1 phòng, vì P3 Cam1/Cam2 nhìn 2 khu vực "
        "sàn khác nhau — hành lang/bếp — nên sai số tự nhiên khi ghép qua 2 mặt phẳng sàn "
        "riêng biệt lớn hơn):\n"
        "— Tính đường epipolar xấp xỉ tới camera kia, dùng họ homography H_by_height (8 mức "
        "chiều cao giả định 0–2m, tính lại homography theo giả định camera pinhole cố định).\n"
        "— Trọng số Gaussian softmax-chuẩn hoá theo khoảng cách tới đường epipolar (sigma="
        "2.0) — KHÔNG học, hằng số chọn tay qua sigma sweep (xem Mục 2).\n"
        "— Cộng trực tiếp đặc trưng camera kia (đã nhân trọng số) vào đặc trưng camera mình; "
        "trung bình cộng đơn giản 2 camera thành 1 vector — KHÔNG có nn.Linear.\n"
        "— Đưa vector gộp qua Pose Head có sẵn để phân loại 5 lớp."
    )
    doc.add_paragraph(
        "Calib THẬT: dùng 8 file video hiệu chỉnh (4 điểm vuông 0.5m), tự động phát hiện 4 "
        "đoạn đứng yên qua ngưỡng tốc độ điểm chân + gộp đoạn gần nhau "
        "(calibrate_p3_corridor.py). Camera 2 của P3 dùng LẠI nguyên homography CAM 1 của P2 "
        "— đã xác nhận đây là cùng 1 camera vật lý, không calib lại."
    )
    doc.add_paragraph(
        "Nhánh dự phòng (không thể hiện trong video minh hoạ): chỉ 1 camera thấy người → "
        "dùng thẳng đặc trưng trung bình (GAP) của camera đó, bỏ qua toàn bộ cơ chế gộp."
    )
    doc.add_paragraph(
        "Quyết định Fall/ADL cuối cùng dùng ĐÚNG RULE THẬT của hệ thống — "
        "Coding/Pipeline/fall_rule.py, hàm detect_fall_events() — KHÔNG viết lại, chỉ nối "
        "chuỗi nhãn tư thế đã gộp của P3 vào (qua adapter p3_fall_rule_adapter.py). Rule có "
        "2 tiêu chí độc lập (OR): (1) chuyển tiếp đứng/ngồi→nằm trong vòng 2 giây; (2) nằm "
        "LIÊN TỤC ≥1.5 giây bất kể trạng thái trước đó. Trước đây phần đánh giá tự động dùng "
        "1 proxy đơn giản hơn nhiều (chỉ cần 1 frame ra 'lie' là tính cả cảnh là Fall) — đã "
        "thay bằng rule thật, số liệu Mục 2c dưới đây là số liệu ĐÃ CẬP NHẬT theo rule thật."
    )

    add_heading(doc, "2. Số liệu khi test")
    doc.add_paragraph(
        "a) Chọn sigma — đo rủi ro Pose Head (mục 5 đặc tả): Pose Head chỉ được train trên "
        "đặc trưng đơn-camera, chưa từng thấy input đã 'trộn' qua công thức gộp. Đo TRÊN P2 "
        "Data 2 (PROXY, dùng TRƯỚC KHI có data calib P3 thật) — thử 4 giá trị sigma:"
    )
    add_table(doc, ["Sigma", "Avg. confidence (fused)", "Delta vs baseline", "Strong-drop rate"], [
        ["0.5", "0.6260", "+0.1039", "11.94%"],
        ["1.0", "0.6844", "+0.1622", "9.70%"],
        ["2.0 (selected)", "0.7303", "+0.2081", "4.48%"],
        ["5.0", "0.7409", "+0.2188", "2.99%"],
    ])
    doc.add_paragraph(
        "Baseline (chưa gộp, camera yếu hơn) trên bộ proxy này: 0.5222. Chọn sigma=2.0 — "
        "tăng confidence rõ rệt (+0.208) trong khi tỷ lệ 'tụt mạnh' đã thấp (4.48%, gần mức "
        "bão hoà của sigma=5.0 nhưng không đánh đổi quá nhiều độ 'cục bộ' của trọng số epipolar)."
    )
    add_image(doc, DIR_OWN / "chart_sigma_sweep.png")

    doc.add_paragraph(
        "b) Xác nhận lại TRÊN DATA P3 THẬT (calib thật, P3 Data 1 — 4 cảnh ngã + P3 Data 2 — "
        "2 cảnh ADL, n=165 frame khớp được cả 2 camera):"
    )
    add_table(doc, ["Metric", "Value"], [
        ["Avg. confidence — weaker camera (before fusion)", "0.6069"],
        ["Avg. confidence — after fusion", "0.6566"],
        ["Delta", "+0.0497 (+4.97%)"],
        ["Agreement rate fused_top1 == cam1_top1", "29.09%"],
    ])
    add_image(doc, DIR_OWN / "chart_confidence_gain.png")
    doc.add_paragraph(
        "Xác nhận đúng như kỳ vọng ở bước (a): gộp giúp tăng confidence trên data thật, không "
        "chỉ trên proxy. Tỷ lệ nhãn thay đổi sau gộp (29%) cho thấy cơ chế gộp có tác động "
        "thực sự, không phải cộng nhiễu vô nghĩa."
    )

    doc.add_paragraph(
        "c) Sensitivity/Specificity/Accuracy — ĐO Ở 2 MỨC KHÁC NHAU (không cùng ý nghĩa, "
        "không gộp chung 1 số):"
    )
    add_table(doc, ["Evaluation level", "Sensitivity", "Specificity", "Accuracy", "n"], [
        ["FRAME (per frame, 'lie' = Fall)", "48.57%", "97.69%", "87.27%", "165"],
        ["SCENE (rule thật: transition ≤2s OR sustained_lying ≥1.5s)",
         "87.50%", "91.67%", "88.89%", "36 (24 Fall + 12 ADL)"],
    ])
    add_image(doc, DIR_OWN / "chart_frame_vs_scene_sensitivity.png")
    doc.add_paragraph(
        "Mức SCENE giờ dùng ĐÚNG rule thật của hệ thống (Coding/Pipeline/fall_rule.py), không "
        "còn là proxy OR đơn giản như bản trước — đây là số chính khi so sánh với Espinosa "
        "(xem folder 3). Bước detect người dùng lại đúng cơ chế giữ tạm bbox khi mất dấu đột "
        "ngột (grace-hold ~1.2 giây, giống hệt GRACE_FRAMES đã validate trong "
        "Coding/Pipeline/detect_classify_pipeline.py) — cần thiết vì YOLOv8n gốc (COCO-"
        "pretrained) bỏ lỡ người nằm đất khá thường xuyên ở một số góc camera; nếu không có "
        "cơ chế này, chuỗi 'lie' liên tục bị ngắt quãng giả tạo bởi lỗi detector chứ không "
        "phải lỗi phân loại thật, làm Sensitivity mức scene bị đánh giá thấp hơn thực tế."
    )

    doc.add_paragraph(
        "d) FPS thật trên Jetson Nano 4GB — CHỈ phần P3 (backbone×2 + gộp + phân loại, "
        "TorchScript, thuần PyTorch — KHÔNG gồm bước detect người YOLOv8n, vì đó là chi phí "
        "dùng CHUNG với các bài toán khác, đã đo riêng ở P1: 18.15 FPS khi chạy chung "
        "YOLOv8n 320×320 + OSNet x0.5):"
    )
    add_table(doc, ["Metric", "Measured"], [
        ["FPS (P3 scope only)", "37.31"],
        ["Time per call", "26.8 ms"],
    ])
    doc.add_paragraph(
        "Lưu ý phạm vi đo: lần đầu thử đo 'full pipeline' gộp cả detect 2 camera (TensorRT/"
        "pycuda) VÀO CÙNG phần P3 (TorchScript/PyTorch) trong 1 tiến trình — gặp xung đột "
        "CUDA context giữa pycuda và PyTorch (lỗi 'invalid resource handle' hoặc treo tiến "
        "trình). Sau khi rà lại đúng phạm vi trách nhiệm: bước detect người KHÔNG thuộc P3 "
        "(là bước dùng chung, thuộc phạm vi đã đo ở P1) — con số 37.31 FPS ở trên là phạm vi "
        "đúng và đã đo SẠCH (không lỗi, không cần pycuda)."
    )

    add_heading(doc, "3. Các trường hợp sai — phân tích lý do")
    doc.add_paragraph(
        "Lịch sử số liệu Sensitivity mức SCENE, 3 phiên bản: proxy đơn giản 'chỉ cần 1 frame "
        "lie' cho 100% (đáng ngờ, không đáng tin — 1 hệ thống thật hiếm khi đạt 100% mà không "
        "đánh đổi Specificity); sau khi thay bằng rule thật nhưng detect người vẫn còn dùng "
        "kiểu từng-frame-độc-lập, Sensitivity tụt xuống 66.67% (16/24) — tụt SAI, không phải "
        "vì phân loại tư thế kém đi, mà vì YOLOv8n gốc bỏ lỡ người nằm đất khá thường xuyên ở "
        "một số góc camera (ví dụ Cam 2 Scene 1: chỉ detect được 18/150 frame — 12%), làm "
        "GIÁN ĐOẠN giả tạo chuỗi 'lie' liên tục, khiến tiêu chí sustained_lying (cần ≥1.5s "
        "KHÔNG đứt quãng) khó đạt dù người vẫn đang nằm yên. Sau khi thêm cơ chế giữ tạm bbox "
        "khi mất dấu đột ngột (grace-hold ~1.2s, đúng cơ chế đã validate trong "
        "detect_classify_pipeline.py — GRACE_FRAMES), Sensitivity phục hồi lên 87.50% (21/24) "
        "— đây là số liệu đáng tin cậy nhất, vì đã tách được rõ 2 nguồn lỗi khác nhau (lỗi "
        "detector tạm thời vs lỗi phân loại tư thế thật)."
    )
    doc.add_paragraph(
        "3/24 ca ngã còn bị bỏ sót (Sensitivity 87.50%, chưa phải 100%) là hạn chế THẬT của "
        "chất lượng phân loại frame-level (xem hàng FRAME, 48.57%) — trong lúc rơi, tư thế "
        "trung gian (nửa đứng/nửa nằm) hay bị phân loại nhầm 'bend'/'sit' ở nhiều frame liên "
        "tiếp, hoặc cảnh bắt đầu giữa lúc đã ngã (class_before='unknown') khiến tiêu chí "
        "transition không áp dụng được — đây là giới hạn của Pose Head, không phải lỗi rule "
        "hay lỗi detect người."
    )
    doc.add_paragraph(
        "Specificity mức SCENE đạt 91.67% (11/12, chỉ còn 1 báo động giả) — cao hơn hẳn 41.7% "
        "ở bản proxy cũ, và KHÔNG bị ảnh hưởng bởi thay đổi grace-hold (vẫn giữ nguyên 11/12): "
        "rule thật đòi hỏi tính liên tục theo thời gian nên lọc được hầu hết các trường hợp "
        "1-2 frame lẻ tẻ bị nhận nhầm 'lie'."
    )
    doc.add_paragraph(
        "Kết luận từ số liệu rule thật (sau khi sửa detect người): hệ thống vượt trội Espinosa "
        "rõ rệt ở CẢ 2 chỉ số cùng lúc (87.50% vs 29.17% Sensitivity, 91.67% vs 83.33% "
        "Specificity, xem folder 3) — không còn hiện tượng 100% đáng ngờ của proxy cũ, cũng "
        "không còn bị đánh giá thấp giả tạo do lỗi detector như bản rule-thật đầu tiên; đây là "
        "con số phản ánh đúng nhất năng lực thật của hệ thống."
    )

    add_file_catalog(doc, [
        ("P3_video_Scene1_Cam1_Cam2_ghep.mp4",
         "VIDEO CHÍNH — Scene 1 (P3 Data 1, cảnh ngã), ghép Cam 1 + Cam 2 chạy song song "
         "thật, vẽ bbox + nhãn tư thế FUSED theo đúng pipeline thật (không phải Re-ID, nhãn "
         "là kết quả phân loại tư thế; bbox dày hơn + nhãn '(held)' khi đang giữ tạm vị trí "
         "do detector mất dấu thoáng qua). 59/150 frame gộp được cả 2 cam (38 ra 'lie'). Có "
         "banner đỏ '!!! FALL DETECTED !!!' xuất hiện đúng lúc rule thật (fall_rule.py) "
         "trigger (t≈4.47s, tiêu chí sustained_lying)."),
        ("chart_sigma_sweep.png", "Biểu đồ chọn sigma, đo trên P2 Data 2 (proxy)."),
        ("chart_confidence_gain.png", "Biểu đồ confidence trước/sau gộp trên data P3 thật."),
        ("chart_frame_vs_scene_sensitivity.png",
         "Biểu đồ Sensitivity ở 2 mức đo (frame vs scene) — tránh hiểu nhầm khi đọc 2 số "
         "khác ý nghĩa cạnh nhau."),
        ("Noi_dung.docx", "Chính file này."),
    ])

    doc.save(str(DIR_OWN / "Noi_dung.docx"))
    print(f"Da luu: {DIR_OWN / 'Noi_dung.docx'}")


# ============================================================
# 2. DOI THU (Espinosa 2019)
# ============================================================
def build_doc_competitor():
    doc = Document()
    doc.add_heading("P3 — Cross-Camera Attention Fusion: Đối thủ (Espinosa et al. 2019)", level=0)

    add_heading(doc, "1. Cách triển khai")
    doc.add_paragraph(
        "Espinosa et al. (2019), \"A vision-based approach for fall detection using multiple "
        "cameras and convolutional neural networks\", Computers in Biology and Medicine — "
        "cùng trường phái 'gộp trước' (early fusion) với P3, khác các bài gộp mức quyết định "
        "(decision-level) nên so sánh công bằng hơn."
    )
    doc.add_paragraph(
        "Paper gốc bị chặn quyền truy cập (403 trên ScienceDirect/ACM/ResearchGate/"
        "Academia.edu) — kiến trúc dưới đây suy lại từ Hình 11 của bài khảo sát Alam et al. "
        "2022 (arXiv:2207.10952), tái hiện sơ đồ khối của Espinosa et al. Không tìm được "
        "code/checkpoint chính thức công khai."
    )
    doc.add_paragraph(
        "Kiến trúc: Conv2D(128) → Pool → Conv2D(128) → Pool → Conv2D(64) → Pool → "
        "Dense(64) → Dense(128) → Dense(256) → Softmax(2 lớp). Input: 1 ảnh optical-flow "
        "xám 38×51 gộp từ 2 camera. 2 điểm KHÔNG xác nhận được từ paper gốc (giả định ghi "
        "rõ): (1) thuật toán optical flow cụ thể — giả định Farneback (lựa chọn mặc định phổ "
        "biến của OpenCV); (2) toán tử gộp 2 camera — giả định channel-stack (mỗi camera 1 "
        "kênh xám, gộp thành input 2 kênh)."
    )
    doc.add_paragraph(
        "Train trên MCFD (Multiple Cameras Fall Dataset) — 22 chute (video), gán nhãn cửa sổ "
        "thời gian (Fall/ADL) THỦ CÔNG qua contact sheet (vì bộ motion-heuristic tự động ban "
        "đầu không đáng tin cậy). Cửa sổ trượt 1 giây / 0.5 giây overlap → 491 window (60 "
        "Fall / 431 ADL, mất cân bằng nặng — đúng bản chất dữ liệu ngã hiếm)."
    )

    add_heading(doc, "2. Số liệu khi test")
    doc.add_paragraph("a) Trên chính test-set MCFD (169 window, 21 Fall / 148 ADL, giữ riêng khi train):")
    add_table(doc, ["Metric", "Value"], [
        ["Accuracy", "82.84%"],
        ["Sensitivity", "71.43%"],
        ["Specificity", "84.46%"],
        ["F1", "50.85%"],
        ["Original paper (UP-Fall dataset, DIFFERENT from MCFD)", "Accuracy 95.64% / F1 97.43%"],
    ])
    doc.add_paragraph(
        "Lưu ý: mốc gốc đo trên UP-Fall (dataset khác), không so 1-1 trực tiếp được với MCFD "
        "— nhưng cho thấy bản tái tạo (kiến trúc suy lại từ sơ đồ khối, không có checkpoint "
        "gốc) chưa đạt mức lý tưởng của bài báo, đặc biệt F1 thấp do MCFD mất cân bằng nặng."
    )
    doc.add_paragraph("b) Trên chính bộ test P2/P3 của nhóm (15 cảnh, scene-level, logic OR theo cửa sổ):")
    add_table(doc, ["Metric", "Value"], [
        ["Accuracy", "46.7% (7/15 correct)"],
    ])
    doc.add_paragraph("c) Trên bộ test P3 riêng, có augment (36 case: 6 cảnh × 6 biến thể — gốc/sáng/tối/"
                       "nhiễu/nén/hoán đổi camera):")
    add_table(doc, ["Metric", "Value"], [
        ["Sensitivity (fall detection)", "29.17% (7/24)"],
        ["Specificity (no false alarm)", "83.3% (10/12)"],
        ["Accuracy", "47.2%"],
    ])
    add_image(doc, DIR_COMP / "chart_espinosa_domain_shift.png")
    doc.add_paragraph(
        "Domain shift rõ rệt qua 3 bối cảnh: 95.64% (paper gốc, UP-Fall) → 82.84% (tái tạo, "
        "test MCFD — cùng phân bố train) → 46.7% (tái tạo, data P2/P3 của nhóm — hoàn toàn "
        "khác điều kiện quay: góc camera, ánh sáng, khoảng cách). Đây là bằng chứng thực "
        "nghiệm cho lý do KHÔNG dùng optical-flow toàn khung hình làm giải pháp chính — độ "
        "nhạy cao với điều kiện quay cụ thể lúc train."
    )

    add_heading(doc, "3. Các trường hợp sai — phân tích lý do")
    doc.add_paragraph(
        "Bỏ sót phần lớn ca ngã thật trên data P3 (17/24 = 70.8%, Sensitivity chỉ 29.17%) — "
        "optical-flow toàn khung hình (38×51, hạ độ phân giải mạnh) nhạy với CHUYỂN ĐỘNG NÓI "
        "CHUNG, không phân biệt được đặc trưng riêng của cú ngã nếu: (a) người ở xa camera "
        "(chuyển động chiếm ít pixel), (b) tốc độ rơi trong khung P3 không đủ khác biệt so "
        "với các chuyển động ADL khác trong tập train MCFD."
    )
    doc.add_paragraph(
        "Ngược lại, Specificity trên data P3 khá tốt (83.3%, ít báo động giả hơn P3 của nhóm "
        "— 41.7%) — vì ngưỡng quyết định của model học được từ MCFD (mất cân bằng nặng, 431 "
        "ADL/60 Fall) thiên về dự đoán ADL — model 'thận trọng', trả giá bằng việc bỏ sót "
        "nhiều ca ngã thật hơn."
    )

    add_file_catalog(doc, [
        ("Espinosa_video_Scene2_Cam1_Cam2_ghep.mp4",
         "VIDEO CHÍNH — Scene 2 (P3 Data 1, cảnh ngã Espinosa dự đoán ĐÚNG), ghép Cam 1 + "
         "Cam 2, hiển thị xác suất Fall theo từng cửa sổ thời gian dạng chữ (KHÔNG có bbox — "
         "phương pháp gốc là optical-flow toàn khung hình, không detect người, vẽ bbox giả sẽ "
         "sai bản chất phương pháp)."),
        ("chart_espinosa_domain_shift.png",
         "Biểu đồ accuracy tụt dần: paper gốc → test MCFD (tái tạo) → data P2/P3 (tái tạo)."),
        ("Noi_dung.docx", "Chính file này."),
    ])

    doc.save(str(DIR_COMP / "Noi_dung.docx"))
    print(f"Da luu: {DIR_COMP / 'Noi_dung.docx'}")


# ============================================================
# 3. SO SANH
# ============================================================
def build_doc_compare():
    doc = Document()
    doc.add_heading("P3 — Cross-Camera Attention Fusion: So sánh Bài của nhóm vs Đối thủ", level=0)

    add_heading(doc, "1. Bảng so sánh trực tiếp")
    doc.add_paragraph(
        "Cả 2 hệ thống test TRÊN CÙNG bộ data P3 (6 cảnh × 6 biến thể = 36 case: 24 Fall/12 "
        "ADL), mức SCENE — P3 dùng ĐÚNG rule thật của hệ thống (Coding/Pipeline/fall_rule.py: "
        "transition ≤2s HOẶC sustained_lying ≥1.5s, không còn là proxy đơn giản hoá):"
    )
    add_table(doc, ["Metric", "P3 (Ours)", "Espinosa (Baseline)"], [
        ["Sensitivity (fall detection, n=24)", "87.50% (21/24)", "29.17% (7/24)"],
        ["Specificity (no false alarm, n=12)", "91.67% (11/12)", "83.3% (10/12)"],
        ["Accuracy (n=36)", "88.89%", "47.2%"],
        ["FPS on Jetson Nano 4GB (model scope, pure PyTorch)", "37.31", "not measured (CPU-based, not a Jetson-focus baseline)"],
    ])
    add_image(doc, DIR_CMP / "chart_scene_level_comparison.png")

    add_heading(doc, "2. Kết luận")
    doc.add_paragraph(
        "Với rule thật VÀ detect người có cơ chế giữ tạm bbox khi mất dấu (grace-hold ~1.2s, "
        "đúng cơ chế đã validate trong detect_classify_pipeline.py), P3 vượt Espinosa ở CẢ 2 "
        "chỉ số cùng lúc: Sensitivity 87.50% vs 29.17% (bắt được hầu hết ca ngã thật), VÀ "
        "Specificity 91.67% vs 83.3% (ít báo động giả hơn, không phải đánh đổi ngược lại). "
        "Đây là kết quả đáng tin cậy hơn 2 bản trước: proxy cũ ('chỉ cần 1 frame lie') cho "
        "100% đáng ngờ; bản rule-thật đầu tiên (chưa sửa detect người) tụt xuống 66.67% do lỗi "
        "detector làm gián đoạn giả tạo chuỗi 'lie', không phản ánh đúng năng lực phân loại "
        "thật. 87.50%/91.67% là con số đã tách rõ 2 nguồn lỗi và đáng tin cậy nhất."
    )
    doc.add_paragraph(
        "Giới hạn thật cần nêu rõ: Sensitivity 87.50% (3/24 ca ngã còn bị bỏ sót) cho thấy "
        "chất lượng phân loại frame-level (48.57% Sensitivity mức frame, xem folder 1 Mục 2c) "
        "vẫn là nút thắt còn lại — khi tư thế trung gian lúc rơi bị phân loại nhầm liên tục "
        "hoặc cảnh bắt đầu giữa lúc đã ngã (class_before không xác định được), cả 2 tiêu chí "
        "của rule (transition/sustained) đều khó kích hoạt. Cải thiện độ chính xác phân loại "
        "frame-level (vd thêm data train tư thế trung gian lúc ngã) sẽ trực tiếp nâng "
        "Sensitivity mức scene, không cần sửa rule."
    )
    doc.add_paragraph(
        "Về mặt phương pháp: P3 dùng công thức toán cố định (không train), tận dụng lại toàn "
        "bộ Backbone + Pose Head sẵn có từ Giai đoạn A — không tốn thêm chi phí train hay "
        "model riêng, trong khi Espinosa cần train 1 CNN riêng trên data optical-flow. Về "
        "hiệu năng: 37.31 FPS (phần model P3) đo thật trên Jetson Nano 4GB — Espinosa không "
        "phải trọng tâm benchmark Jetson (dùng CPU Farneback, không phải đối thủ Jetson-first)."
    )

    add_heading(doc, "3. Biểu đồ và video đính kèm")
    doc.add_paragraph(
        "SoSanh_video_Scene1_Cam1_Cam2_P3_vs_Espinosa.mp4 — VIDEO SO SÁNH TRỰC TIẾP trên CÙNG "
        "1 cảnh ngã thật (P3 Data 1, Scene 1): P3 nhận đúng 'lie' (ngã) ở 38/59 frame gộp "
        "được (59/150 frame gộp được cả 2 cam), RULE THẬT trigger đúng lúc (t≈4.47s, "
        "sustained_lying) hiển thị banner đỏ trên video; Espinosa dự đoán SAI cả cảnh "
        "(pred=ADL, xác suất ngã cao nhất chỉ đạt 0.1544 — không vượt ngưỡng 0.5). Đây là "
        "cảnh nằm trong nhóm 21/24 ca P3 phát hiện đúng bằng rule thật (xem Mục 1)."
    )
    doc.add_paragraph("chart_scene_level_comparison.png — biểu đồ cột nhóm Sensitivity/Specificity/Accuracy, 2 hệ thống.")

    add_file_catalog(doc, [
        ("SoSanh_video_Scene1_Cam1_Cam2_P3_vs_Espinosa.mp4",
         "VIDEO CHÍNH — cùng 1 cảnh ngã thật, hiển thị song song kết quả P3 (đúng) và "
         "Espinosa (sai, bỏ sót) — dẫn chứng trực quan cho bảng số liệu Mục 1."),
        ("chart_scene_level_comparison.png",
         "Biểu đồ cột nhóm: Sensitivity/Specificity/Accuracy của P3 vs Espinosa, mức scene."),
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
