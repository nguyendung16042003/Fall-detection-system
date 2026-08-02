"""Nạp dữ liệu mẫu cho việc kiểm thử API và hiển thị giao diện App.

Chạy: python seed.py  (sau khi `alembic upgrade head`).
Script idempotent: chạy lại nhiều lần không tạo bản ghi trùng.
"""

import uuid
from datetime import datetime, timedelta, timezone

from app.core.database import SessionLocal
from app.core.security import hash_password
from app.models.camera import Camera
from app.models.camera_rule import CameraRule
from app.models.event import Event
from app.models.user import User

SAMPLE_USERS = [
    {
        "username": "admin",
        "email": "admin@example.com",
        "password": "admin123",
        "role": "admin",
    },
    {
        "username": "caregiver",
        "email": "caregiver@example.com",
        "password": "caregiver123",
        "role": "caregiver",
    },
]

SAMPLE_CAMERAS = [
    {
        "cam_id": "cam_01",
        "name": "Camera Phòng Khách",
        "rtsp_url": "rtsp://admin:pass@192.168.1.100:554/stream1",
        "edge_device_id": "jetson_nano_01",
        "location": "Phòng khách tầng 1",
        "status": "online",
        "is_active": True,
    },
    {
        "cam_id": "cam_02",
        "name": "Camera Phòng Ngủ",
        "rtsp_url": "rtsp://admin:pass@192.168.1.101:554/stream1",
        "edge_device_id": "jetson_nano_02",
        "location": "Phòng ngủ ông bà",
        "status": "offline",
        "is_active": False,
    },
]


def seed_data() -> None:
    db = SessionLocal()
    try:
        for data in SAMPLE_USERS:
            if (
                db.query(User).filter(User.username == data["username"]).first()
                is None
            ):
                db.add(
                    User(
                        username=data["username"],
                        email=data["email"],
                        hashed_password=hash_password(data["password"]),
                        role=data["role"],
                    )
                )
        db.commit()

        for data in SAMPLE_CAMERAS:
            camera = (
                db.query(Camera)
                .filter(Camera.cam_id == data["cam_id"])
                .first()
            )
            if camera is None:
                camera = Camera(**data)
                db.add(camera)
                db.flush()  # lấy camera.id

            # Quy tắc mặc định cho mỗi camera
            if (
                db.query(CameraRule)
                .filter(CameraRule.camera_id == camera.id)
                .first()
                is None
            ):
                db.add(CameraRule(camera_id=camera.id, enable_vlm_verify=True))

            # Một sự kiện mẫu cho mỗi camera
            if (
                db.query(Event)
                .filter(Event.camera_id == camera.id)
                .first()
                is None
            ):
                db.add(
                    Event(
                        event_id=uuid.uuid4(),
                        camera_id=camera.id,
                        event_type="fall_candidate",
                        person_id=1,
                        timestamp_utc=datetime.now(timezone.utc)
                        - timedelta(minutes=5),
                        class_before="standing",
                        final_class="lying",
                        detection_confidence=0.92,
                        bbox_json={"x1": 100, "y1": 120, "x2": 320, "y2": 400},
                        frame_width=1280,
                        frame_height=720,
                        rule_trigger="standing_to_lying",
                        transition_ms=850,
                        status="confirmed",
                    )
                )
        db.commit()
        print("Đã nạp dữ liệu mẫu thành công!")
    finally:
        db.close()


if __name__ == "__main__":
    seed_data()
