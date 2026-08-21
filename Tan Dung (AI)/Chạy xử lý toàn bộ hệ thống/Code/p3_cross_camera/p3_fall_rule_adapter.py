"""
Adapter: noi chuoi phan loai tu the DA FUSE cua P3 (dau ra analyze() trong
make_p3_report_video.py / make_p3_comparison_video.py) vao dung RULE THAT cua
he thong -- Coding/Pipeline/fall_rule.py, ham detect_fall_events(). KHONG
viet lai rule, chi tai dung nguyen ban (dung ban trong Coding/Pipeline/, MOI
hon va co du 2 tieu chi -- KHONG dung ban dong goi trong Handoff_for_Edge/,
ban do CU hon va thieu tieu chi "sustained_lying").

Ly do can adapter: fall_rule.detect_fall_events() thiet ke cho pipeline 1
camera (DetectClassifyPipeline.process_video tra ve record moi FRAME cua 1
video), trong khi P3 gop 2 camera truoc khi phan loai -- moi "frame" cua P3
la 1 thoi diem DA FUSE, dai dien cho 1 nguoi DUY NHAT (khong con tach theo
tung cam). Kien truc goc (Context toan bo he thong.docx) xac nhan dung thiet
ke nay: "Temporal Rule Detector chay SAU buoc gop, tren nhan da hop nhat theo
dinh danh nguoi -- neu chay truoc khi gop, ca nga dung vung giao 2 cam se
KHONG bao gio tao duoc chuoi stand/sit->lie".
"""
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[3] / "Coding" / "Pipeline"
sys.path.insert(0, str(PIPELINE_DIR))
from fall_rule import detect_fall_events  # noqa: E402


def fused_frames_to_records(per_frame, person_id=1):
    """per_frame: list dict {frame_idx, bbox1, bbox2, label, conf, fused} --
    dung dinh dang da co san tu analyze() trong make_p3_report_video.py /
    make_p3_comparison_video.py. Chi emit record cho frame da fuse THANH CONG
    (fused=True) -- dung bbox CAMERA 1 lam bbox dai dien de tinh aspect ratio
    trong rule (gia dinh ghi ro: rule can 1 bbox duy nhat/nguoi, chon cam1 lam
    dai dien vi khong co "bbox gop" tu nhien khi 2 camera nhin 2 goc khac
    nhau -- day la xap xi hop ly, KHONG anh huong logic transition/sustained
    vi ca 2 tieu chi chi phu thuoc CLASS "lie" + thoi gian, aspect ratio chi
    la 1 bo loc phu chong bao gia).

    Tra ve records dung schema DetectClassifyPipeline.process_video():
    {frame_id, person_id, pose, pose_confidence, bbox_x1..y2, det_confidence, held}.
    """
    records = []
    for rec in per_frame:
        if not rec["fused"]:
            continue
        x1, y1, x2, y2 = rec["bbox1"]
        records.append({
            "frame_id": rec["frame_idx"],
            "person_id": person_id,
            "pose": rec["label"],
            "pose_confidence": rec["conf"],
            "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
            "det_confidence": 1.0,
            "held": False,
        })
    return records


def detect_fall_events_fused(per_frame, fps, person_id=1, **rule_kwargs):
    """Tien ich goi thang: per_frame (dau ra analyze()) -> records -> rule
    that -> list fall event (dung schema mqtt_schema_v2.json, xem fall_rule.py)."""
    records = fused_frames_to_records(per_frame, person_id=person_id)
    return detect_fall_events(records, fps=fps, **rule_kwargs)
