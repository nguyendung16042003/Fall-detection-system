"""Video demo 3/3 -- Nga thuong (demo don gian cho hoi dong), dung DUNG
pipeline don-camera goc: Coding/Pipeline/detect_classify_pipeline.py +
fall_rule.py (KHONG sua), chuan hien thi giong
Coding/Test/Test 4/visualize_any_video.py (bbox mau theo TU THE, banner
"!!! FALL DETECTED !!!" 1.5s khi rule trigger). KHONG hien ID (khong can da
camera/Boundary Feature Fusion o day).

Du lieu: "Video demo/Ngã thường/" -- 2 scene x 2 cam, A di Cam1->Cam2 roi nga
o Cam2 (khong dinh Cam1). Ghep 2 cam song song moi scene, noi tiep Scene 1 ->
Scene 2 trong CUNG 1 file, moi scene co title card rieng.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "Coding" / "Pipeline"))
from detect_classify_pipeline import DetectClassifyPipeline  # noqa: E402
from fall_rule import detect_fall_events  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Video demo" / "Ngã thường"
OUT_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Video demo" / "Video cuối cùng"
OUT_NAME = "3_Nga_thuong.mp4"

SCENES = [
    # Nguoi dung chon giu lai DUY NHAT Scene 2 cho video cuoi cung (Scene 1 bo).
    ("Scene 2", DEMO_DIR / "CAM 1-SCENE 2.avi", DEMO_DIR / "CAM 2-SCENE 2.avi"),
]

TILE_H = 540
FALL_BANNER_MS = 1500
BOX_EXPIRE_FRAMES = 35
POSE_COLOR = {
    "lie": (0, 0, 255),
    "sit": (0, 165, 255),
    "bend": (0, 255, 255),
    "stand": (0, 255, 0),
    "exercise": (255, 255, 0),
}


def make_title_card(text, size, seconds, fps):
    w, h = size
    frame = np.full((h, w, 3), 30, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 1.1, 3
    lines = text.split("\n")
    heights = [cv2.getTextSize(line, font, scale, thick)[0][1] for line in lines]
    y0 = h // 2 - (len(lines) - 1) * (max(heights) + 20) // 2
    for i, line in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(line, font, scale, thick)
        cv2.putText(frame, line, ((w - tw) // 2, y0 + i * (th + 25)), font, scale, (255, 255, 255), thick)
    return [frame] * int(seconds * fps)


def process_cam(video_path, pipeline):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    records = pipeline.process_video(video_path, frame_skip=5)
    events = detect_fall_events(records, fps=fps)
    by_frame = {}
    for r in records:
        by_frame.setdefault(r["frame_id"], []).append(r)
    event_windows = [(e["timestamp_ms"], e["timestamp_ms"] + FALL_BANNER_MS) for e in events]
    return fps, by_frame, event_windows, events


def render_cam_frames(video_path, fps, by_frame, event_windows, cam_label):
    """Tra ve list frame da ve (bbox mau theo tu the + banner), full res.

    CHI ve 1 bbox/frame (moi nhat) -- kich ban demo nay LUON chi co 1 nguoi
    that. Khong theo doi rieng tung person_id: YOLO tracker (ByteTrack) hay
    tra ve box.id=None khi vua mat/bat lai dau (dac biet luc chuyen tu
    dung/ngoi sang nam -- kho detect, xem ghi chu trong
    detect_classify_pipeline.py), khien detect_classify_pipeline.py gan tam
    person_id=-1 rieng cho nhung lan do. Neu giu bbox theo TUNG person_id
    (ke ca -1) trong ca so BOX_EXPIRE_FRAMES nhu truoc, box "-1" ao va box id
    that se hien CHONG LEN NHAU cung luc (da do dac: 30-60/~100-140 frame moi
    clip, gan nua video) -- nhin loan dai bbox du chi 1 nguoi that. Ve 1 box
    moi nhat giai quyet tan goc, khong can quan tam id co bi doi hay khong."""
    cap = cv2.VideoCapture(str(video_path))
    frames_out = []
    frame_idx = 0
    last_seen = None  # (record, frame_idx) -- CHI 1, khong phai dict theo person_id
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx in by_frame:
            last_seen = (by_frame[frame_idx][-1], frame_idx)
        if last_seen is not None and frame_idx - last_seen[1] > BOX_EXPIRE_FRAMES:
            last_seen = None

        t_ms = frame_idx / fps * 1000.0
        if last_seen is not None:
            r, _ = last_seen
            x1, y1, x2, y2 = int(r["bbox_x1"]), int(r["bbox_y1"]), int(r["bbox_x2"]), int(r["bbox_y2"])
            color = POSE_COLOR.get(r["pose"], (200, 200, 200))
            thickness = 3 if r.get("held") else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            label = f"{r['pose']} {r['pose_confidence']:.2f}" + (" (held)" if r.get("held") else "")
            cv2.putText(frame, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if any(lo <= t_ms <= hi for lo, hi in event_windows):
            cv2.putText(frame, "!!! FALL DETECTED !!!", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
        cv2.putText(frame, cam_label, (30, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        frames_out.append(frame)
    cap.release()
    return frames_out


def make_scene_tiles(frames1, frames2, tile_h=TILE_H):
    n = max(len(frames1), len(frames2))
    h1, w1 = frames1[0].shape[:2]
    tile_w = int(tile_h * w1 / h1)
    black1 = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    black2 = black1.copy()
    tiles = []
    for i in range(n):
        f1 = frames1[i] if i < len(frames1) else None
        f2 = frames2[i] if i < len(frames2) else None
        t1 = cv2.resize(f1, (tile_w, tile_h)) if f1 is not None else black1
        t2 = cv2.resize(f2, (tile_w, tile_h)) if f2 is not None else black2
        sep = np.full((tile_h, 10, 3), 255, dtype=np.uint8)
        tiles.append(np.hstack([t1, sep, t2]))
    return tiles, tile_w * 2 + 10, tile_h


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Nap DetectClassifyPipeline (detector + classifier goc)...")
    pipeline = DetectClassifyPipeline()

    out_path = OUT_DIR / OUT_NAME
    writer = None
    out_fps = 15.0

    for scene_name, cam1_video, cam2_video in SCENES:
        print(f"\n=== {scene_name} ===")
        print(f"  Xu ly CAM 1 ({cam1_video.name})...")
        fps1, by_frame1, ev_win1, events1 = process_cam(cam1_video, pipeline)
        print(f"  Xu ly CAM 2 ({cam2_video.name})...")
        fps2, by_frame2, ev_win2, events2 = process_cam(cam2_video, pipeline)
        out_fps = fps1

        print(f"  CAM 1: {len(events1)} su kien nga")
        for e in events1:
            print(f"    -> t={e['timestamp_ms']/1000:.2f}s trigger={e['rule']['trigger']}")
        print(f"  CAM 2: {len(events2)} su kien nga")
        for e in events2:
            print(f"    -> t={e['timestamp_ms']/1000:.2f}s trigger={e['rule']['trigger']}")

        frames1 = render_cam_frames(cam1_video, fps1, by_frame1, ev_win1, "CAM 1")
        frames2 = render_cam_frames(cam2_video, fps2, by_frame2, ev_win2, "CAM 2")
        tiles, out_w, out_h = make_scene_tiles(frames1, frames2)

        if writer is None:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (out_w, out_h))
            assert writer.isOpened(), f"VideoWriter khong mo duoc: {out_path}"

        title = f"Nga thuong - {scene_name}\nA di tu Cam 1 sang Cam 2 roi nga o Cam 2"
        for f in make_title_card(title, (out_w, out_h), 2.5, out_fps):
            writer.write(f)
        for t in tiles:
            writer.write(t)

    writer.release()
    assert out_path.exists() and out_path.stat().st_size > 0, f"Video khong duoc ghi ra: {out_path}"
    print(f"\nDa luu: {out_path}")


if __name__ == "__main__":
    main()
