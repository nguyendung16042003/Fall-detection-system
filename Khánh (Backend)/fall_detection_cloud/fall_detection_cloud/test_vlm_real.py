"""
Test VLM thật (Gemini) với 6 frame ảnh thật — không còn dry-run.

Script này TỰ CHỨA — không import gì từ package app.* của dự án, nên KHÔNG cần
đặt cùng thư mục với test_api.py. Chỉ cần 2 điều kiện:
  1. Thư mục test_frames/ (chứa 6 ảnh .jpg) nằm CÙNG CẤP với chính file này
     (ví dụ cả 2 đều trong vlm_test_package/ là được, không cần di chuyển ra ngoài).
  2. Chạy trên máy có thể kết nối tới localhost:8000 (backend) và localhost:1883
     (MQTT broker) — tức là chạy TRÊN máy Khánh, nơi docker compose đang chạy.

Cách chạy:
    cd vào đúng thư mục chứa file này, rồi: python test_vlm_real.py

Nếu backend báo lỗi format field "frames" không đúng (400/422, hoặc log báo lỗi
parse), đó là do định dạng base64 dưới đây (list chuỗi base64 thuần) không khớp
với những gì backend thật sự mong đợi — gửi lại lỗi đó để chỉnh lại cho đúng.
"""

import base64
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import urllib.request
import urllib.error

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("Cần cài paho-mqtt trước: pip install paho-mqtt")
    sys.exit(1)

API_BASE = "http://localhost:8000"
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_USERNAME = "guest"
MQTT_PASSWORD = "guest"

FRAMES_DIR = Path(__file__).parent / "test_frames"
NUM_RUNS = 3  # chạy 3 lần để lấy latency trung bình, giãn cách để tránh rate-limit Gemini

# 6 frame = 2 frame/giây trong 3 giây trước thời điểm ngã (khớp thiết kế gốc).
# offset_ms âm nghĩa là "trước" thời điểm ngã (0ms).
FRAME_OFFSETS_MS = [-3000, -2500, -2000, -1500, -1000, -500]


def load_frames_base64():
    frames = []
    for i in range(1, 7):
        path = FRAMES_DIR / f"frame_{i}.jpg"
        if not path.exists():
            print(f"LỖI: không tìm thấy {path}")
            sys.exit(1)
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
            frames.append(b64)
    return frames


def login():
    login_data = json.dumps({"username": "admin", "password": "secret"}).encode()
    req = urllib.request.Request(
        f"{API_BASE}/api/auth/login",
        data=login_data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req)
    result = json.loads(resp.read().decode())
    return result.get("access_token")


def get_event(token, event_id):
    req = urllib.request.Request(
        f"{API_BASE}/api/events/{event_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read().decode())


def publish_event(client, frames_b64, run_idx):
    event_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    payload = {
        "schema_version": "1.2",
        "event_id": event_id,
        "cam_id": "cam_01",
        "person_id": 1,
        "timestamp_utc": now.isoformat(),
        "event_type": "fall_candidate",
        "detection": {
            "class_before": "stand",
            "final_class": "lie",
            "confidence": 0.92,
            "bbox_xyxy": [120, 80, 380, 420],
            "frame_width": 1280,
            "frame_height": 720,
        },
        "rule": {
            "version": "1.0",
            "trigger": "stand_to_lie",
            "transition_ms": 1800,
            "window_ms": 2000,
        },
        # Định dạng ĐÚNG theo class FrameIn (app/schemas/event.py):
        #   FrameIn(offset_ms: int, jpeg_b64: str)
        # offset_ms: mốc thời gian (ms) của frame đó so với thời điểm ngã,
        # khớp với thiết kế "2 frame/giây trong 3 giây trước khi ngã".
        "frames": [
            {"offset_ms": offset, "jpeg_b64": b64}
            for offset, b64 in zip(FRAME_OFFSETS_MS, frames_b64)
        ],
        "clip_url": None,
    }
    topic = "events/cam_01/fall"
    print(f"\n[Run {run_idx}] Publish event_id={event_id} len(frames)={len(frames_b64)}")
    client.publish(topic, json.dumps(payload))
    return event_id


def main():
    print("=== Test VLM thật với 6 frame ảnh ===\n")

    print("1. Load 6 frame JPEG...")
    frames_b64 = load_frames_base64()
    total_kb = sum(len(f) for f in frames_b64) // 1024
    print(f"   Đã load {len(frames_b64)} frame, tổng ~{total_kb} KB (base64)")

    print("\n2. Login...")
    token = login()
    print("   Login OK")

    print("\n3. Kết nối MQTT...")
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="test_vlm_real_publisher")
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
    time.sleep(1)
    print("   MQTT connected")

    results = []
    for run_idx in range(1, NUM_RUNS + 1):
        event_id = publish_event(client, frames_b64, run_idx)
        print(f"   Đợi VLM xử lý (Gemini có thể mất 1-3s)...")
        time.sleep(8)

        try:
            event = get_event(token, event_id)
        except urllib.error.HTTPError as e:
            print(f"   LỖI khi GET event: {e.code} {e.read().decode()}")
            continue

        vlm_verdict = event.get("vlm_verdict")
        vlm_confidence = event.get("vlm_confidence")
        vlm_reason = event.get("vlm_reason")
        status = event.get("status")

        print(f"   status={status}")
        print(f"   vlm_verdict={vlm_verdict}  vlm_confidence={vlm_confidence}")
        print(f"   vlm_reason={vlm_reason}")

        results.append({
            "event_id": event_id,
            "status": status,
            "vlm_verdict": vlm_verdict,
            "vlm_confidence": vlm_confidence,
            "vlm_reason": vlm_reason,
        })

        if run_idx < NUM_RUNS:
            print("   Nghỉ 10s trước lần chạy tiếp theo (tránh rate-limit Gemini)...")
            time.sleep(10)

    client.loop_stop()
    client.disconnect()

    print("\n=== TỔNG KẾT ===")
    for i, r in enumerate(results, 1):
        print(f"Run {i}: status={r['status']} verdict={r['vlm_verdict']} "
              f"confidence={r['vlm_confidence']} reason={r['vlm_reason']}")

    print("\nQUAN TRỌNG: mở thêm cửa sổ khác chạy lệnh sau để xem latency thật")
    print("và xác nhận KHÔNG còn thấy dòng 'VLM dry-run' hay 'nhận được 0':")
    print("    docker logs fall_backend --tail 50")


if __name__ == "__main__":
    main()
