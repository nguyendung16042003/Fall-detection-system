"""Giả lập Edge (Jetson) publish event/telemetry qua MQTT — mqtt_schema v2.

Publish 1 fall_event tới events/cam_{id}/fall và 1 telemetry tới
telemetry/cam_{id}/status để kiểm thử MQTT consumer của server.

Chạy: python scripts/mock_mqtt_publish.py [cam_id] [host] [port]
Yêu cầu: RabbitMQ bật MQTT plugin (port 1883) và server đang chạy.
"""

"""Giả lập Edge (Jetson) publish event/telemetry qua MQTT — mqtt_schema v2.

Publish 1 fall_event tới events/cam_{id}/fall và 1 telemetry tới
telemetry/cam_{id}/status để kiểm thử MQTT consumer của server.

Chạy: python scripts/mock_mqtt_publish.py [cam_id] [host] [port]
Yêu cầu: RabbitMQ bật MQTT plugin (port 1883) và server đang chạy.
"""

import base64
import json
import sys
import time
import uuid

import paho.mqtt.client as mqtt

CAM_ID = sys.argv[1] if len(sys.argv) > 1 else "cam_01"
HOST = sys.argv[2] if len(sys.argv) > 2 else "localhost"
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 1883
_JPEG = base64.b64encode(bytes.fromhex("ffd8ffd9")).decode()


def _client() -> mqtt.Client:
    if hasattr(mqtt, "CallbackAPIVersion"):
        c = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id="edge-mock",
        )
    else:
        c = mqtt.Client(client_id="edge-mock")
    c.username_pw_set("guest", "guest")
    return c


def main() -> None:
    fall_event = {
        "schema_version": "2.0",
        "event_id": str(uuid.uuid4()),
        "cam_id": CAM_ID,
        "person_id": 1,
        "timestamp": "2026-06-20T07:00:00Z",
        "event_type": "fall_candidate",
        "detection": {
            "class_before": "stand",
            "final_class": "lie",
            "confidence": 0.63,
            "bbox_xyxy": [100, 120, 320, 400],
            "frame_width": 1280,
            "frame_height": 720,
        },
        "rule": {"trigger": "stand_to_lie", "transition_ms": 800,
                 "window_ms": 2000},
        "frames": [
            {"offset_ms": off, "jpeg_b64": _JPEG}
            for off in (-2500, -2000, -1500, -1000, -500, 0)
        ],
    }
    telemetry = {
        "schema_version": "2.0",
        "cam_id": CAM_ID,
        "timestamp_utc": "2026-06-20T07:00:05Z",
        "pipeline": {
            "state": "running",
            "fps_pgie": 29.9,
            "fps_sgie": 15.0,
            "active_tracks": 1,
        },
        "system": {
            "ram_used_mb": 2048,
            "ram_total_mb": 4096,
            "cpu_temp_c": 54.0,
            "gpu_temp_c": 58.0,
            "cpu_usage_pct": 40.0,
            "disk_free_gb": 11.2,
        },
        "network": {
            "mqtt_connected": True,
            "last_event_sent_utc": "2026-06-20T07:00:00Z",
        },
    }

    c = _client()
    c.connect(HOST, PORT, 60)
    c.loop_start()
    c.publish(f"events/{CAM_ID}/fall", json.dumps(fall_event), qos=1)
    c.publish(f"telemetry/{CAM_ID}/status", json.dumps(telemetry), qos=1)
    print(f"Đã publish fall_event + telemetry cho {CAM_ID} -> {HOST}:{PORT}")
    time.sleep(2)
    c.loop_stop()
    c.disconnect()


if __name__ == "__main__":
    main()