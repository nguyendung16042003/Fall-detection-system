"""Kiểm thử chuỗi kích hoạt (Trigger Chain) bằng sự kiện giả — contract v2.

Luồng: login -> POST /api/events (mqtt fall_event) -> đợi pipeline background
-> GET /api/alerts -> PATCH acknowledge. Xác minh:
    Event giả -> VLM xác minh -> tạo Alert (FCM + Telegram) -> cập nhật DB.

Chạy: python scripts/mock_trigger_chain.py [base_url]
"""

import base64
import sys
import time
import uuid

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
EMAIL, PASSWORD = "admin@example.com", "admin123"
_JPEG = base64.b64encode(bytes.fromhex("ffd8ffd9")).decode()


def main() -> None:
    with httpx.Client(base_url=BASE, timeout=15.0) as c:
        r = c.post(
            "/api/auth/login",
            json={"username": EMAIL, "password": PASSWORD},
        )
        r.raise_for_status()
        c.headers["Authorization"] = f"Bearer {r.json()['access_token']}"

        cameras = c.get("/api/cameras").json()
        if not cameras:
            print("Không có camera nào. Hãy chạy seed.py trước.")
            return
        cam_id = cameras[0]["cam_id"]
        print(f"Camera: {cameras[0]['name']} ({cam_id})")

        event = {
            "schema_version": "2.0",
            "event_id": str(uuid.uuid4()),
            "cam_id": cam_id,
            "person_id": 2,
            "timestamp": "2026-06-08T12:00:00Z",
            "event_type": "fall_candidate",
            "detection": {
                "class_before": "standing",
                "final_class": "lying",
                "confidence": 0.55,
                "bbox_xyxy": [10, 20, 110, 260],
                "frame_width": 1280,
                "frame_height": 720,
            },
            "rule": {"trigger": "transition", "transition_ms": 850,
                     "window_ms": 2000},
            "frames": [{"offset_ms": 0, "jpeg_b64": _JPEG}],
        }
        resp = c.post("/api/events", json=event)
        resp.raise_for_status()
        print("POST /api/events ->", resp.json())

        print("Đợi trigger chain (VLM -> alert -> FCM/Telegram)...")
        time.sleep(2)

        alerts = c.get("/api/alerts", params={"page_size": 1}).json()
        if not alerts["items"]:
            print("Chưa có alert (có thể VLM trả not_fall -> đã bỏ qua).")
            return
        alert = alerts["items"][0]
        print("Alert mới nhất:")
        print("  id        :", alert["id"])
        print("  cam_id    :", alert["cam_id"])
        print("  fcm_sent  :", alert["fcm_sent"])
        print("  vlm_reason:", alert["vlm_reason"])

        ack = c.patch(f"/api/alerts/{alert['id']}/acknowledge")
        ack.raise_for_status()
        print("PATCH acknowledge ->", ack.json())


if __name__ == "__main__":
    main()
