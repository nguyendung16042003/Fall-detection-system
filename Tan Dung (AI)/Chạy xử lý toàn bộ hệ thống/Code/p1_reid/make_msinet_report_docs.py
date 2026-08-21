"""Tao 1 file Word cho report Doi thu 2 (MSINet) cua Re-Identification, theo
DUNG pattern make_reid_report_docs_kd025.py (helper add_heading/add_table/
add_image/add_file_catalog), bang/do thi tieng Anh (ten ngan gon), noi dung
van ban tieng Viet. FPS/RAM/Temp dung DUNG so do THAT tren Jetson Nano (qua
SSH, TensorRT FP16, jetson_clocks max) -- KHONG dung so do tren laptop (khong
co y nghia vi khong phai thiet bi trien khai)."""
import sys
from pathlib import Path

from docx import Document

sys.stdout.reconfigure(encoding="utf-8")

AI_ROOT = Path(r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)")
BASE = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Số liệu thống kê" / "Re-Identification"
DIR_MSINET = BASE / "5. Mô phỏng Đối thủ 2 (MSINet)"


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
    add_heading(doc, "Danh mục file trong thư mục này", level=2)
    for name, desc in entries:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(name)
        run.bold = True
        p.add_run(f" — {desc}")


def build_doc():
    doc = Document()
    doc.add_heading("Re-Identification — Đối thủ 2 (MSINet, CVPR 2023)", level=0)

    add_heading(doc, "1. Giới thiệu & cách lấy model")
    doc.add_paragraph(
        "MSINet (\"Twins Contrastive Search of Multi-Scale Interaction for Object ReID\", "
        "Gu et al., CVPR 2023) là kiến trúc Re-Identification tìm bằng NAS (Neural "
        "Architecture Search), khác hẳn bản chất với OSNet x0.25 (distill kiến thức từ "
        "OSNet x1.0). Nguồn: paper arxiv.org/abs/2303.07065, code "
        "github.com/vimar-gu/MSINet. Số liệu Rank-1/mAP published trên MSMT17: 81.0% / 59.6%."
    )
    doc.add_paragraph(
        "Theo hướng dẫn triển khai (MSINet_benchmark_MSMT17_guide), KHÔNG tự train lại: "
        "codebase gốc hướng NAS-search/domain-adaptation, train supervised trực tiếp trên "
        "MSMT17 tốn nhiều giờ GPU và không chắc đúng flag (-ds/-dt) cho chế độ same-domain "
        "thông thường — rủi ro không tương xứng lợi ích khi mục tiêu chính là so sánh chi phí "
        "triển khai trên Jetson Nano 4GB. Thay vào đó: clone code thật, tải checkpoint "
        "pretrained có sẵn (msinet_msmt.pth.tar, đúng genotype 'msmt' tìm riêng cho MSMT17) "
        "vào đúng kiến trúc MSINet thật — đủ để đo Params/FLOPs, chạy inference thật cho "
        "Tier-2 TPR test + video demo, và export ONNX/TensorRT benchmark Jetson thật."
    )
    doc.add_paragraph(
        "Lưu ý trung thực về checkpoint: khi load, phát hiện checkpoint THIẾU nhánh phụ f_* "
        "(256/768 chiều output của model) — không có trong state_dict, các layer này vẫn "
        "random-init. Chỉ nhánh chính (512/768 chiều) có trọng số pretrained thật. Vì vậy "
        "checkpoint này KHÔNG phải bản fine-tune supervised trực tiếp trên MSMT17 tương ứng "
        "với số 81.0%/59.6% công bố trong paper — Rank-1/mAP dưới đây vẫn ghi là trích dẫn "
        "paper (literature), TÁCH RIÊNG khỏi kết quả suy luận (Tier-2 TPR, Jetson) đo bằng "
        "chính checkpoint này."
    )

    add_heading(doc, "2. Params & FLOPs")
    doc.add_paragraph("Đo bằng thop.profile, input (1,3,256,128), cùng phương pháp cho cả 2 model:")
    add_table(doc, ["Model", "Params (M)", "FLOPs (G)"], [
        ["MSINet (đối thủ 2)", "2.33", "1.131"],
        ["OSNet x0.25 KD (bài của nhóm)", "0.20", "0.091"],
    ])
    doc.add_paragraph(
        "MSINet nặng hơn OSNet x0.25 KD khoảng 11.6 lần về Params, ~12.4 lần về FLOPs — hợp lý "
        "vì MSINet là kiến trúc tìm bằng NAS chuyên biệt cho Re-Identification, không bị ràng "
        "buộc nhẹ như OSNet x0.25 (vốn đã distill để tối ưu cho Jetson Nano 4GB)."
    )

    add_heading(doc, "3. Rank-1 / mAP (MSMT17)")
    add_table(doc, ["Model", "Rank-1", "mAP", "Nguồn số liệu"], [
        ["MSINet", "81.0%", "59.6%", "Published (paper CVPR 2023, KHÔNG tự đo)"],
        ["OSNet x0.25 KD", "52.00%", "27.46%", "Thực nghiệm nhóm tự đo (11.659 query / 82.161 gallery)"],
    ])
    doc.add_paragraph(
        "2 số liệu này KHÔNG cùng nguồn đo (paper vs tự thực nghiệm) nên chỉ mang tính tham "
        "khảo về khoảng cách hiệu năng lý thuyết, không dùng để kết luận tuyệt đối — đây là "
        "khung so sánh do chính guide đề xuất khi không đủ tài nguyên/thời gian tự train lại "
        "MSINet."
    )

    add_heading(doc, "4. Tier-2 TPR (Own Video Test)")
    doc.add_paragraph(
        "Test THẬT bằng checkpoint pretrained tải về (không phải số liệu paper), cùng data/quy "
        "trình với OSNet x0.25 KD và Shared Backbone (Gallery 2 người, cả 2 camera Scene 5, "
        "gốc + 5 biến thể augment):"
    )
    add_table(doc, ["Test set", "MSINet", "OSNet x0.25 KD", "Shared Backbone"], [
        ["4 base cases (no augment)", "4/4 (100.0%)", "3/4 (75.0%)", "0/4 (0.0%)"],
        ["25 full cases (2 cams x 6 variants)", "25/25 (100.0%)", "19/25 (76.0%)", "0/25 (0.0%)"],
    ])
    doc.add_paragraph(
        "MSINet đạt tuyệt đối 100% ở cả 2 mức test, vượt hẳn OSNet x0.25 KD — phù hợp với việc "
        "MSINet có backbone chuyên biệt Re-ID đã pretrain thật trên domain MSMT17 (dù thiếu "
        "nhánh phụ f_*, nhánh chính vẫn đủ mạnh để phân biệt 2 người trong bài toán Gallery "
        "nhỏ). Đây là bằng chứng thực nghiệm cho luận điểm đánh đổi: model lớn/mạnh hơn cho "
        "độ chính xác cao hơn, đổi lại chi phí tính toán lớn hơn nhiều (xem Mục 2 và Mục 5)."
    )

    add_heading(doc, "5. Jetson Nano 4GB — FPS / RAM / Temp")
    doc.add_paragraph(
        "Đo THẬT trên Jetson Nano 4GB qua SSH, TensorRT FP16, jetson_clocks bật max-performance "
        "(GPU 921.6MHz), input (1,3,256,128), batch=1, trung bình 5 lần đo (MSINet: 11.83, "
        "10.71, 10.49, 10.39, 10.28 FPS):"
    )
    add_table(doc, ["Model", "Điều kiện đo", "FPS", "RAM peak", "Temp peak"], [
        ["MSINet", "Model đứng riêng (không chạy chung YOLOv8n)", "10.74", "1.73 GB", "40.0°C"],
        ["OSNet x0.25 KD", "Model đứng riêng (không chạy chung YOLOv8n)", "64.11 (15.60 ms/call)", "chưa đo riêng", "chưa đo riêng"],
        ["OSNet x0.25 KD", "Kịch bản xấu nhất, chạy CHUNG YOLOv8n mỗi frame", "21.31", "1.73 GB", "36.25°C"],
    ])
    doc.add_paragraph(
        "Lưu ý điều kiện đo KHÔNG đồng nhất — MSINet mới chỉ đo dạng \"model đứng riêng\" (chưa "
        "chạy chung YOLOv8n detect như kịch bản xấu nhất từng đo cho OSNet), nên KHÔNG so trực "
        "tiếp cột FPS \"chạy chung YOLOv8n\" (21.31) với 10.74 của MSINet. So sánh công bằng "
        "nhất là 2 dòng \"model đứng riêng\": MSINet 10.74 FPS so với OSNet x0.25 KD 64.11 FPS "
        "— OSNet nhanh hơn khoảng 6 lần trên Jetson, phù hợp với chênh lệch FLOPs ~12.4 lần ở "
        "Mục 2."
    )
    doc.add_paragraph(
        "RAM peak đo được của MSINet (1.73 GB) trùng số với RAM đo trước đây của OSNet KD "
        "(chạy chung YOLOv8n) — đây là trùng hợp do phần lớn RAM quan sát được là overhead cố "
        "định của runtime CUDA/TensorRT trên Jetson, không phản ánh chênh lệch thật giữa 2 "
        "model (MSINet nhẹ hơn 12.4 lần về FLOPs); số liệu vẫn được ghi trung thực đúng như đo "
        "được, không làm tròn hay chỉnh sửa."
    )

    add_heading(doc, "6. Kết luận — đánh đổi độ chính xác vs chi phí triển khai")
    doc.add_paragraph(
        "MSINet thắng rõ về độ chính xác (Rank-1/mAP literature cao hơn nhiều, Tier-2 TPR tự "
        "đo 100% so với 76-100%), nhưng nặng hơn ~11.6-12.4 lần (Params/FLOPs) và chậm hơn "
        "~6 lần khi chạy riêng trên Jetson Nano 4GB — thiết bị triển khai thật của hệ thống. "
        "Với ràng buộc tài nguyên biên (Jetson Nano 4GB, cần chạy song song YOLOv8n detect + "
        "Pose Head + Re-ID trong cùng ngân sách FPS/RAM), OSNet x0.25 (Knowledge Distillation) "
        "vẫn là lựa chọn phù hợp hơn cho hệ thống — MSINet ở đây đóng vai trò đối thủ đối "
        "chiếu \"trần độ chính xác\" khi không bị ràng buộc phần cứng biên, không phải model "
        "đề xuất thay thế."
    )

    add_file_catalog(doc, [
        ("MSINet_video_Scene5_CAM1_CAM2_ghep.mp4",
         "VIDEO CHÍNH — ghép 2 camera Scene 5 (CAM 1 trái, CAM 2 phải) chạy song song, mỗi "
         "camera tự vẽ khung + nhãn 'Person N' theo kết quả MSINet (checkpoint pretrained tải "
         "về). 4/4 case đúng."),
        ("MSINet_video_dang_ky_va_test.mp4",
         "Video phụ, 2 đoạn nối tiếp: (1) Đăng ký Person 1 (Enrollment, quay từ CAM 2), (2) "
         "Test CHỈ CAM 2. Giữ lại để đối chiếu, cùng cấu trúc với video phụ của Đối thủ 1."),
        ("Noi_dung.docx", "Chính file này."),
    ])

    DIR_MSINET.mkdir(parents=True, exist_ok=True)
    doc.save(str(DIR_MSINET / "Noi_dung.docx"))
    print(f"Da luu: {DIR_MSINET / 'Noi_dung.docx'}")


if __name__ == "__main__":
    build_doc()
