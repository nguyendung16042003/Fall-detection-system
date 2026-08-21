"""
Xuat video co ve bbox + nhan tu the + banner "FALL DETECTED" luc rule trigger,
cho BAT KY video nao ban tu chon (MCFD, URFD, hoac duong dan bat ky) -- de tu
mat xem pipeline dang doan dung/sai o dau, khong chi tin vao con so metric.

Cach dung:
  python visualize_any_video.py chute05/cam3          -> MCFD dataset/chute05/cam3.avi
  python visualize_any_video.py chute12/cam7.avi      -> MCFD (duoi .avi tuy chon)
  python visualize_any_video.py adl-10                -> URFD ADL/adl-10-cam0-rgb.mp4 (tim gan dung ten)
  python visualize_any_video.py fall-05                -> URFD Fall/fall-05-cam0-rgb.mp4
  python visualize_any_video.py "D:\duong\dan\bat_ky.mp4"  -> duong dan tuyet doi/tuong doi bat ky

Output luu vao Test 4/visual_output/<ten_video>_annotated.mp4, cung lien
tuc in ra console cac su kien nga (thoi diem, tu the truoc, kieu trigger)
de doi chieu khi xem video.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Pipeline"))
from detect_classify_pipeline import DetectClassifyPipeline  # noqa: E402
from fall_rule import detect_fall_events  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
MCFD_DIR = AI_ROOT / "Datasets" / "File Test 2" / "MCFD" / "dataset"
URFD_DIR = AI_ROOT / "Datasets" / "File Test 2" / "URFD" / "Cam"
OUT_DIR = Path(__file__).resolve().parent / "visual_output"
OUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# NEU BAM NUT "RUN" TRONG IDE (khong go duoc tham so dong lenh): SUA DONG
# DUOI DAY thanh video ban muon xem, roi bam Run. Vi du:
#   VIDEO_TO_TEST = "chute05/cam3"        (MCFD)
#   VIDEO_TO_TEST = "adl-10"              (URFD ADL)
#   VIDEO_TO_TEST = "fall-05"             (URFD Fall)
#   VIDEO_TO_TEST = r"D:\video_cua_ban.mp4"  (video bat ky, nho co chu r truoc "")
# Neu chay bang terminal voi tham so (vd: python visualize_any_video.py chute05/cam3)
# thi tham so dong lenh se duoc uu tien, dong nay bi bo qua.
VIDEO_TO_TEST = "chute1/cam6"
# ============================================================================

POSE_COLOR = {
    "lie": (0, 0, 255),      # do
    "sit": (0, 165, 255),    # cam
    "bend": (0, 255, 255),   # vang
    "stand": (0, 255, 0),    # xanh la
    "exercise": (255, 255, 0),
}
FALL_BANNER_MS = 1500  # hien banner canh bao trong 1.5s sau moi lan trigger
BOX_EXPIRE_FRAMES = 35  # khop GRACE_FRAMES=6 * frame_skip=5 + du phong (xem
# detect_classify_pipeline.py) -- tranh bbox cu ve chong chat khi video ghep
# nhieu doan / doi canh (loi da tung gap va sua o visualize_file_test3.py).


def resolve_video(query: str) -> Path:
    """Cho phep go tat: 'chute05/cam3' -> MCFD, 'adl-10'/'fall-05' -> URFD,
    hoac duong dan bat ky (tuyet doi/tuong doi) neu khong khop mau nao o tren."""
    p = Path(query)
    if p.exists():
        return p

    q = query.strip().replace("\\", "/")

    # MCFD: "chuteNN/camM" hoac "chuteNN/camM.avi" -- cho phep go tat so
    # khong can dien du 2 chu so (vd "chute1" -> tu dong thanh "chute01").
    if q.lower().startswith("chute"):
        parts = q.split("/")
        chute_raw = parts[0]
        chute_num = "".join(ch for ch in chute_raw if ch.isdigit())
        chute = f"chute{int(chute_num):02d}" if chute_num else chute_raw
        cam = parts[1] if len(parts) > 1 else "cam1"
        if not cam.endswith(".avi"):
            cam += ".avi"
        candidate = MCFD_DIR / chute / cam
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Khong thay MCFD video: {candidate}")

    # URFD: "adl-10", "fall-05", co hoac khong duoi/duong dan day du
    ql = q.lower()
    if ql.startswith("adl") or ql.startswith("fall"):
        subdir = "ADL" if ql.startswith("adl") else "FALL"
        search_dir = URFD_DIR / subdir
        matches = sorted(search_dir.glob(f"*{q}*")) or sorted(search_dir.glob(f"*{ql}*"))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"Khong thay video URFD khop '{query}' trong {search_dir}")

    raise FileNotFoundError(
        f"Khong tim thay video cho '{query}'. Dung duong dan day du, hoac cu phap "
        f"'chuteNN/camM' (MCFD) / 'adl-NN' / 'fall-NN' (URFD)."
    )


def annotate_video(video_path: Path, out_path: Path, pipeline: DetectClassifyPipeline,
                    frame_skip: int = 5):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"Dang chay pipeline tren {video_path} ...")
    records = pipeline.process_video(video_path, frame_skip=frame_skip)
    events = detect_fall_events(records, fps=fps)
    print(f"{len(records)} record, {len(events)} su kien nga phat hien")

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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", nargs="?", default=None,
                         help="chuteNN/camM | adl-NN | fall-NN | duong dan video bat ky "
                              "(bo trong -> dung VIDEO_TO_TEST khai bao o dau file)")
    parser.add_argument("--frame-skip", type=int, default=5)
    args = parser.parse_args()

    video_query = args.video if args.video is not None else VIDEO_TO_TEST
    print(f"Video se chay: {video_query}")
    video_path = resolve_video(video_query)
    out_path = OUT_DIR / f"{video_path.stem}_annotated.mp4"

    pipeline = DetectClassifyPipeline()
    events = annotate_video(video_path, out_path, pipeline, frame_skip=args.frame_skip)

    print(f"\nDa luu video co annotate: {out_path}")
    if events:
        for e in events:
            print(f"  -> Fall luc t={e['timestamp_ms']/1000:.1f}s "
                  f"({e['detection']['class_before']} -> lying, trigger={e['rule']['trigger']})")
    else:
        print("  -> Khong phat hien su kien nga nao.")

    print("\nDang mo video len de xem (bang trinh phat mac dinh cua may)...")
    os.startfile(out_path)  # Windows: mo bang app video mac dinh, khong can tu di tim file


if __name__ == "__main__":
    main()
