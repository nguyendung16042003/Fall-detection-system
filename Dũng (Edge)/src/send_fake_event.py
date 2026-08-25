#!/usr/bin/env python3
"""Bắn 1 event ngã GIẢ lên broker để test trọn đường dây edge -> Khánh
(KHÔNG cần camera/GPU/pipeline).

Ví dụ:
    # broker giả local (đóng vai Khánh): amqtt &  +  tools/mqtt_sub.py
    MQTT_BROKER=127.0.0.1 python3 src/send_fake_event.py --cam cam_01 --trigger stand_to_lie

    # broker thật của Khánh: đặt MQTT_BROKER trong .env rồi:
    export $(grep -v '^#' .env | xargs) && python3 src/send_fake_event.py --cam cam_02 --trigger sit_to_lie
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from event_builder import build_fall_event, placeholder_frames
from fall_detector import FallCandidate
from mqtt_publisher import MqttPublisher


def build_fake(cam: str, trigger: str, person_id: int) -> dict:
    class_before = trigger.split("_to_")[0]  # stand | sit
    cand = FallCandidate(
        cam_id=cam,
        person_id=person_id,
        class_before=class_before,
        final_class="lie",
        confidence=0.92,
        trigger=trigger,
        transition_ms=1800,
        window_ms=2000,
        ts_ms=0,
        bbox_xyxy=(120, 80, 380, 420),
        frame_width=1280,
        frame_height=720,
    )
    return build_fall_event(cand, placeholder_frames())


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Bắn event ngã giả để test đường dây MQTT")
    ap.add_argument("--cam", default="cam_01", help="cam_id (vd cam_01)")
    ap.add_argument("--trigger", default="stand_to_lie",
                    choices=["stand_to_lie", "sit_to_lie"])
    ap.add_argument("--person-id", type=int, default=3)
    ap.add_argument("--broker", default=None, help="ghi đè MQTT_BROKER")
    ap.add_argument("--port", type=int, default=None, help="ghi đè MQTT_PORT")
    args = ap.parse_args()

    if args.broker:
        os.environ["MQTT_BROKER"] = args.broker
    if args.port:
        os.environ["MQTT_PORT"] = str(args.port)

    event = build_fake(args.cam, args.trigger, args.person_id)

    pub = MqttPublisher.from_env(client_id="fake-event-sender")
    print(f"[fake] kết nối {pub.host}:{pub.port} ...")
    pub.connect()
    ok = pub.publish_fall_event(args.cam, event)
    pub.disconnect()

    topic = f"events/{args.cam}/fall"
    print(f"[fake] {'ĐÃ GỬI' if ok else 'GỬI LỖI'} -> {topic} "
          f"(event_id={event['event_id']}, trigger={args.trigger}, 6 frame)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
