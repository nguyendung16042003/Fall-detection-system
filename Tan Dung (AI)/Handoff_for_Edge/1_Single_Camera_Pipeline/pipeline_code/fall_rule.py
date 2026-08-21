"""
Rule Fall/NoFall theo cua so thoi gian (windowed), thay the rule cu
(temporal_fall_detector.py: LIE_RATIO_THRESHOLD=0.40 tren TOAN BO video --
khong phai temporal that, giai thich recall thap 64.9% cua rule cu).

Rule moi: theo doi TUNG NGUOI rieng (qua person_id tu tracker), phat hien
dung khoanh khac chuyen tu dung/ngoi -> nam trong 1 khung thoi gian ngan
(window_ms), khop dung field ma Dung can trong mqtt_schema_v2.json:
detection.class_before, detection.final_class, rule.trigger, rule.transition_ms.

Cau hinh duoi day la ban DA KIEM CHUNG tot nhat tren 64 video (24 MCFD Fall +
40 URFD ADL): Accuracy 84.4%, Precision 81.8%, Recall 75.0%, F1=0.783.

Da thu them 2 huong mo rong khi test tren File Test 3 (video ghep nhieu doan
nga nhanh, nhieu goc camera) nhung KHONG giu lai vi lam giam chat luong tren
tap 64 video lon hon:
  - Ha nguong aspect ratio tuyet doi 0.7 -> 0.5 + ha held_confidence 0.6 -> 0.4:
    Recall tren File Test 3 tang nhung Precision tren 64-video rot tu 81.8%
    xuong 61.8% (FP tang tu 4 len 13/64).
  - Doi aspect ratio tuyet doi sang TUONG DOI (so voi baseline luc dung/ngoi
    cua chinh nguoi do, nhan he so): phat hien 2 van de can ban (1) frame
    "held" khong doi bbox nen aspect luon = aspect luc mat dau, so sanh tuong
    doi voi chinh no vo nghia; (2) ty le thay doi cua true positive (1.64-1.88x)
    va false positive (1.72-1.98x) TREN CHINH DU LIEU THAT chong lan nhau,
    khong co nguong nao tach duoc. Ket luan: aspect ratio (tuyet doi hay tuong
    doi) co GIOI HAN THAT trong viec phan biet "nam that" vs "ngoi nga bi
    nham" -- can data train da dang hon cho classifier hoac kien truc manh
    hon (Cross-Camera Attention), khong phai dieu chinh rule them.
"""

# 5 lop classifier -> ten enum trong mqtt_schema_v2.json
POSE_TO_SCHEMA = {
    "stand": "standing",
    "sit": "sitting",
    "lie": "lying",
    "bend": "bending",
    "exercise": "other",
}
UPRIGHT_STATES = {"standing", "sitting"}
DEFAULT_WINDOW_MS = 2000  # khop "window_ms" trong mqtt_schema_v2.json

# Frame "held" (bbox giu tam khi detector mat dau, xem detect_classify_pipeline.py)
# doi khi cho ket qua "lie" nhieu (crop khong con dung nguoi that su nua vi ho da
# di noi khac). Da kiem chung tren du lieu that: bo qua het frame held se mat lai
# ca ngan chan nga that (chute02), nhung tin het se ra bao gia (URFD adl-03/04).
# Danh doi: chi tin frame held la "lie" khi confidence du cao.
MIN_HELD_LIE_CONFIDENCE = 0.6

# Phat hien tren tap 64 video: classifier nham "ngoi nga lung/ngoi thap" thanh
# "lie" -- xay ra CA o frame detect that (khong chi frame held), gay bao gia
# tren gan 50% video ADL (URFD hay co canh ngoi xuong ghe/sofa thap). Do dac
# hinh hoc: nguoi NAM DET THAT co bbox RONG hon CAO (aspect > ~1.0-1.6 o giua
# chuoi nga that trong chute01), con nguoi NGOI NGA (dung nham) van CAO hon
# RONG (aspect ~0.88-1.01). Dung aspect ratio bbox lam dieu kien loc bo sung,
# doc lap voi confidence cua classifier. Da thu 6 gia tri (0.5-1.0), 0.7 la
# tot nhat (F1=0.783); day la nguong TUYET DOI, co gioi han da ghi o docstring
# tren (khong quat cho moi goc camera) -- chap nhan gioi han nay.
MIN_LYING_ASPECT_RATIO = 0.7

# Tieu chi thu 2, chay SONG SONG voi tieu chi chuyen tiep o tren (giong PIFR --
# ho goi day la "time threshold", 1 trong 3 tieu chi xac nhan nga ma bai bao de
# xuat, nhom truoc gio moi lam tieu chi chuyen tiep). Y nghia: neu 1 nguoi nam
# LIEN TUC du lau, bao nga luon -- KHONG can biet truoc do ho dung hay ngoi.
# Giai quyet 2 truong hop tieu chi chuyen tiep bo sot: (1) nga qua nhanh/track
# bi mat dung luc chuyen tiep nen khong xac dinh duoc class_before; (2) video
# ghep nhieu doan, track ID moi xuat hien da nam san, khong co lich su truoc do.
MIN_SUSTAINED_LYING_MS = 1500  # nam lien tuc >= 1.5 giay -> bao nga du khong ro truoc do la gi
# (giam tu 3000ms xuong 1500ms theo yeu cau nguoi dung sau khi xem truc tiep
# video File Test 3 voi yolov8x: 3s qua lau, bo sot ca truong hop nam that su
# nhung video ket thuc/doi canh truoc khi du 3s)


def detect_fall_events(records, fps, window_ms=DEFAULT_WINDOW_MS,
                        min_held_lie_confidence=MIN_HELD_LIE_CONFIDENCE,
                        min_lying_aspect_ratio=MIN_LYING_ASPECT_RATIO,
                        min_sustained_lying_ms=MIN_SUSTAINED_LYING_MS):
    """
    records: list dict {frame_id, person_id, pose, pose_confidence, bbox_x1..y2,
              det_confidence, held} (output cua DetectClassifyPipeline.process_video)
    fps: fps that cua video, dung de doi frame_id sang mili-giay
    Tra ve list fall event, moi event khop cau truc detection/rule trong mqtt_schema_v2.json.

    2 tieu chi doc lap, tieu chi nao dat truoc thi trigger truoc (OR logic):
    1. "transition" -- dung/ngoi -> nam trong vong window_ms (nhu cu).
    2. "sustained_lying" -- nam lien tuc >= min_sustained_lying_ms, khong can
       biet truoc do la gi (moi, bo sung theo yeu cau xu ly case nga nhanh/
       track bi ngat quang).
    """
    events = []
    by_person = {}
    for r in records:
        by_person.setdefault(r["person_id"], []).append(r)

    for person_id, seq in by_person.items():
        seq.sort(key=lambda x: x["frame_id"])
        last_upright = None  # (pose_schema, time_ms)
        triggered = False  # tranh bao lien tuc khi nguoi van dang nam
        lying_streak_start = None  # thoi diem bat dau chuoi "nam" lien tuc hien tai

        for r in seq:
            pose_schema = POSE_TO_SCHEMA.get(r["pose"], "other")
            time_ms = r["frame_id"] / fps * 1000.0

            if pose_schema in UPRIGHT_STATES:
                last_upright = (pose_schema, time_ms)
                triggered = False  # nguoi da dung/ngoi lai -> san sang bat lan nga tiep theo
                lying_streak_start = None  # dung/ngoi lai -> cat dut chuoi nam truoc do
            elif pose_schema == "lying":
                if r.get("held") and r["pose_confidence"] < min_held_lie_confidence:
                    continue  # frame giu tam + confidence thap -> khong du tin tuong de trigger
                w = r["bbox_x2"] - r["bbox_x1"]
                h = r["bbox_y2"] - r["bbox_y1"]
                aspect = w / h if h > 0 else 0
                if aspect < min_lying_aspect_ratio:
                    continue  # bbox van cao hon rong -> nhieu kha nang la ngoi nga, khong phai nam that

                if lying_streak_start is None:
                    lying_streak_start = time_ms

                if not triggered:
                    # Tieu chi 1: chuyen tiep dung/ngoi -> nam trong window_ms
                    if last_upright is not None and (time_ms - last_upright[1]) <= window_ms:
                        transition_ms = time_ms - last_upright[1]
                        events.append({
                            "person_id": person_id,
                            "frame_id": r["frame_id"],
                            "timestamp_ms": time_ms,
                            "detection": {
                                "class_before": last_upright[0],
                                "final_class": "lying",
                                "confidence": r["pose_confidence"],
                                "bbox_xyxy": [r["bbox_x1"], r["bbox_y1"], r["bbox_x2"], r["bbox_y2"]],
                            },
                            "rule": {
                                "trigger": f"{last_upright[0]}_to_lying",
                                "transition_ms": round(transition_ms),
                                "window_ms": window_ms,
                            },
                        })
                        triggered = True
                    # Tieu chi 2: nam lien tuc du lau, bat ke truoc do la gi
                    elif (time_ms - lying_streak_start) >= min_sustained_lying_ms:
                        events.append({
                            "person_id": person_id,
                            "frame_id": r["frame_id"],
                            "timestamp_ms": time_ms,
                            "detection": {
                                "class_before": last_upright[0] if last_upright else "unknown",
                                "final_class": "lying",
                                "confidence": r["pose_confidence"],
                                "bbox_xyxy": [r["bbox_x1"], r["bbox_y1"], r["bbox_x2"], r["bbox_y2"]],
                            },
                            "rule": {
                                "trigger": "sustained_lying",
                                "transition_ms": None,
                                "window_ms": window_ms,
                                "sustained_ms": round(time_ms - lying_streak_start),
                            },
                        })
                        triggered = True
            # bend/other: tu the trung gian, khong reset last_upright/chuoi nam,
            # khong trigger (giu nguyen de tinh transition/sustained khi thuc su
            # nam xuong ro rang tro lai)

    return events
