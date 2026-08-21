"""
Xuat video co ve bbox + nhan tu the + banner "FALL DETECTED" luc rule trigger,
de xem truc tiep bang mat thay he thong dang doan gi tren File Test 3.
"""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Pipeline"))
from detect_classify_pipeline import DetectClassifyPipeline  # noqa: E402
from fall_rule import detect_fall_events  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent / "visual_file_test3"
OUT_DIR.mkdir(exist_ok=True)

VIDEOS = [
    AI_ROOT / "Datasets" / "File Test 3" / "gialap_vidnga_rtsp_1.mp4",
    AI_ROOT / "Datasets" / "File Test 3" / "gialap_vidnga_rtsp_2.mp4",
]

POSE_COLOR = {
    "lie": (0, 0, 255),      # do
    "sit": (0, 165, 255),    # cam
    "bend": (0, 255, 255),   # vang
    "stand": (0, 255, 0),    # xanh la
    "exercise": (255, 255, 0),
}
FALL_BANNER_MS = 1500  # hien banner canh bao trong 1.5s sau moi lan trigger
BOX_EXPIRE_FRAMES = 35  # so frame GOC (khong phai frame da sample) sau do 1 khung
# hinh khong con duoc ve nua neu khong thay lai. Khop voi GRACE_FRAMES=6 trong
# detect_classify_pipeline.py (6 lan sample * frame_skip=5 = 30 frame goc, +
# them chut du phong). Neu khong co gioi han nay, ID cu tu nhung canh truoc do
# (video ghep nhieu doan) se bi ve chong chat mai tren man hinh, gay roi hinh
# nhu anh chup man hinh da thay (day CHI la loi hien thi, khong phai loi cua
# rule that -- pipeline that da tu xoa dung qua GRACE_FRAMES).


def annotate_video(video_path, out_path, pipeline):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"  Dang chay pipeline tren {video_path.name} ...")
    records = pipeline.process_video(video_path, frame_skip=5)
    events = detect_fall_events(records, fps=fps)
    print(f"  {len(records)} record, {len(events)} su kien nga")

    by_frame = {}
    for r in records:
        by_frame.setdefault(r["frame_id"], []).append(r)

    event_windows = [(e["timestamp_ms"], e["timestamp_ms"] + FALL_BANNER_MS) for e in events]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    last_by_person = {}  # person_id -> (record, last_seen_frame_idx)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx in by_frame:
            for r in by_frame[frame_idx]:
                last_by_person[r["person_id"]] = (r, frame_idx)

        # Xoa cac ID lau khong thay lai -- neu khong, ID tu canh truoc (video
        # ghep nhieu doan, doi canh lien tuc) se ve chong chat mai tren man
        # hinh vi khong bao gio bi xoa (day la loi vua phat hien qua anh chup).
        stale_ids = [pid for pid, (_, seen) in last_by_person.items()
                     if frame_idx - seen > BOX_EXPIRE_FRAMES]
        for pid in stale_ids:
            del last_by_person[pid]

        t_ms = frame_idx / fps * 1000.0

        for pid, (r, _) in last_by_person.items():
            x1, y1, x2, y2 = int(r["bbox_x1"]), int(r["bbox_y1"]), int(r["bbox_x2"]), int(r["bbox_y2"])
            color = POSE_COLOR.get(r["pose"], (200, 200, 200))
            thickness = 3 if r.get("held") else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            label = f"ID{pid} {r['pose']} {r['pose_confidence']:.2f}" + (" (held)" if r.get("held") else "")
            cv2.putText(frame, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if any(lo <= t_ms <= hi for lo, hi in event_windows):
            cv2.putText(frame, "!!! FALL DETECTED !!!", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

        cv2.putText(frame, f"t={t_ms/1000:.1f}s", (30, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        writer.write(frame)

    cap.release()
    writer.release()
    return events


def main():
    pipeline = DetectClassifyPipeline()  # yolov8n mac dinh -- xem muc 4c trong
    # README.md ve ly do da chot giu detector nay sau khi so sanh voi yolov8x/
    # yolo11n/fine-tune rieng.
    for video_path in VIDEOS:
        print(f"=== {video_path.name} ===")
        out_path = OUT_DIR / f"{video_path.stem}_annotated.mp4"
        events = annotate_video(video_path, out_path, pipeline)
        print(f"  Da luu: {out_path}")
        for e in events:
            print(f"    -> Fall luc t={e['timestamp_ms']/1000:.1f}s "
                  f"({e['detection']['class_before']} -> lying)")
        print()


if __name__ == "__main__":
    main()
