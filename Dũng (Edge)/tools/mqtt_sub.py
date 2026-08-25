#!/usr/bin/env python3
"""Subscriber MQTT — đóng vai consumer của Khánh để kiểm tra event/telemetry cục bộ.

In tóm tắt gọn (KHÔNG in cả base64 frame cho đỡ ngập terminal).

Ví dụ:
    MQTT_BROKER=127.0.0.1 python3 tools/mqtt_sub.py --topic "events/+/fall"
    MQTT_BROKER=127.0.0.1 python3 tools/mqtt_sub.py --topic "telemetry/+/status"
"""
import argparse
import json
import os
import signal
import sys

import paho.mqtt.client as mqtt

DEFAULT_PORT = 1883


def _summarize(topic: str, payload: dict) -> str:
    if "detection" in payload:  # fall_event
        d = payload.get("detection", {})
        n = len(payload.get("frames", []))
        return (f"[FALL] {topic} cam={payload.get('cam_id')} pid={payload.get('person_id')} "
                f"{d.get('class_before')}→{d.get('final_class')} "
                f"conf={d.get('confidence')} trigger={payload.get('rule', {}).get('trigger')} "
                f"frames={n} event_id={payload.get('event_id')}")
    if "pipeline" in payload:  # telemetry
        p = payload.get("pipeline", {})
        return (f"[TELE] {topic} cam={payload.get('cam_id')} state={p.get('state')} "
                f"fps_pgie={p.get('fps_pgie')} tracks={p.get('active_tracks')}")
    return f"[MSG ] {topic} keys={list(payload)}"


def on_message(_c, _u, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        print(_summarize(msg.topic, payload), flush=True)
    except (json.JSONDecodeError, UnicodeDecodeError):
        print(f"[RAW ] {msg.topic} ({len(msg.payload)} bytes, không phải JSON)", flush=True)


def on_connect(client, _u, _f, reason_code, *_p):
    # paho 1.x: reason_code là int (0=OK). paho 2.x: là ReasonCode, dùng .is_failure
    failed = reason_code != 0 if isinstance(reason_code, int) else reason_code.is_failure
    if not failed:
        topic = client._user_topic
        client.subscribe(topic, qos=1)
        print(f"[sub] kết nối OK, đang nghe topic: {topic}", flush=True)
    else:
        print(f"[sub] kết nối lỗi: {reason_code}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Subscriber MQTT (vai consumer của Khánh)")
    ap.add_argument("--topic", default="events/+/fall")
    ap.add_argument("--broker", default=os.environ.get("MQTT_BROKER", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("MQTT_PORT", DEFAULT_PORT)))
    args = ap.parse_args()

    if hasattr(mqtt, "CallbackAPIVersion"):
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="khanh-sub-sim")
    else:
        client = mqtt.Client(client_id="khanh-sub-sim")
    client._user_topic = args.topic
    user = os.environ.get("MQTT_USER")
    if user:
        client.username_pw_set(user, os.environ.get("MQTT_PASS", ""))
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"[sub] nối {args.broker}:{args.port} ... (Ctrl+C để dừng)", flush=True)
    client.connect(args.broker, args.port, 60)

    signal.signal(signal.SIGINT, lambda *_: (client.disconnect(), sys.exit(0)))
    client.loop_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
