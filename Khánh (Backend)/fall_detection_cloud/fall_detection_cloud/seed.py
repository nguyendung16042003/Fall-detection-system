"""Nạp dữ liệu mẫu cho việc kiểm thử API và hiển thị giao diện App.

Chạy: python seed.py  (sau khi `alembic upgrade head`).
Script idempotent: chạy lại nhiều lần không tạo bản ghi trùng.
"""

from datetime import datetime, timedelta, timezone

from app.core.database import SessionLocal
from app.core.security import hash_password
from app.models.camera import Camera
from app.models.camera_rule import CameraRule
from app.models.event import Event
from app.models.user import User

SAMPLE_USERS = [
    {
        "email": "admin@example.com",
        "password": "admin123",
        "full_name": "Quản trị viên",
        "role": "admin",
    },
    {
        "email": "caregiver@example.com",
        "password": "caregiver123",
        "full_name": "Người chăm sóc",
        "role": "caregiver",
    },
]

SAMPLE_CAMERAS = [
    {
        "name": "Camera Phòng Khách",
        "rtsp_url": "rtsp://admin:pass@192.168.1.100:554/stream1",
        "edge_device_id": "jetson_nano_01",
        "location_note": "Phòng khách tầng 1",
        "status": "online",
    },
    {
        "name": "Camera Phòng Ngủ",
        "rtsp_url": "rtsp://admin:pass@192.168.1.101:554/stream1",
        "edge_device_id": "jetson_nano_02",
        "location_note": "Phòng ngủ ông bà",
        "status": "offline",
    },
]


def seed_data() -> None:
    db = SessionLocal()
    try:
        for data in SAMPLE_USERS:
            if (
                db.query(User).filter(User.email == data["email"]).first()
                is None
            ):
                db.add(
                    User(
                        email=data["email"],
                        hashed_password=hash_password(data["password"]),
                        full_name=data["full_name"],
                        role=data["role"],
                    )
                )
        db.commit()

        for data in SAMPLE_CAMERAS:
            camera = (
                db.query(Camera)
                .filter(Camera.name == data["name"])
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
                db.add(CameraRule(camera_id=camera.id))

            # Một sự kiện mẫu cho mỗi camera
            if (
                db.query(Event)
                .filter(Event.camera_id == camera.id)
                .first()
                is None
            ):
                db.add(
                    Event(
                        camera_id=camera.id,
                        event_type="fall_candidate",
                        severity="high",
                        person_id=1,
                        timestamp=datetime.now(timezone.utc)
                        - timedelta(minutes=5),
                        bbox_json={"x1": 100, "y1": 120, "x2": 320, "y2": 400},
                        state_before="standing",
                        state_after="lying",
                        transition_time_ms=850,
                        fall_confidence=0.92,
                        status="new",
                    )
                )
        db.commit()
        print("Đã nạp dữ liệu mẫu thành công!")
    finally:
        db.close()


if __name__ == "__main__":
    seed_data()
