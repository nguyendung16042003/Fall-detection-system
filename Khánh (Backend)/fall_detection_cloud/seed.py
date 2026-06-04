from app.core.database import SessionLocal
from app.models.user import User
from app.models.camera import Camera
# Dùng thư viện passlib để hash mật khẩu admin

def seed_data():
    db = SessionLocal()
    # 1. Tạo User mẫu (Admin)
    admin = User(email="admin@example.com", hashed_password="hashed_password_here", role="caregiver")
    db.add(admin)
    
    # 2. Tạo Camera mẫu
    cam1 = Camera(name="Phòng Khách", rtsp_url="rtsp://192.168.1.100:554/stream1")
    db.add(cam1)
    
    db.commit()
    print("Đã nạp dữ liệu mẫu thành công!")

if __name__ == "__main__":
    seed_data()