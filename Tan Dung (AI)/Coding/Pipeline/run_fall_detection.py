"""Chay tron 1 video qua toan bo pipeline MVP: detect -> crop -> classify -> rule."""
import argparse
import json

import cv2

from detect_classify_pipeline import DetectClassifyPipeline, DEFAULT_DETECTOR, DEFAULT_CLASSIFIER
from fall_rule import detect_fall_events, DEFAULT_WINDOW_MS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--detector", default=str(DEFAULT_DETECTOR))
    parser.add_argument("--classifier", default=str(DEFAULT_CLASSIFIER))
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--window-ms", type=int, default=DEFAULT_WINDOW_MS)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    pipeline = DetectClassifyPipeline(args.detector, args.classifier)
    records = pipeline.process_video(args.video, frame_skip=args.frame_skip)
    print(f"Da xu ly {len(records)} record, fps video = {fps:.1f}")

    events = detect_fall_events(records, fps=fps, window_ms=args.window_ms)

    if events:
        print(f"\n>>> PHAT HIEN {len(events)} SU KIEN NGA:")
        for e in events:
            print(json.dumps(e, indent=2, ensure_ascii=False))
    else:
        print("\n>>> Khong phat hien su kien nga nao.")


if __name__ == "__main__":
    main()
