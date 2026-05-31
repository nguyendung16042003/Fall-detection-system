CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Bảng Người dùng (Phục vụ Auth API)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    role VARCHAR(50) DEFAULT 'caregiver', -- caregiver, admin
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Bảng Camera (Quản lý thiết bị biên)
CREATE TABLE cameras (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    rtsp_url VARCHAR(255) NOT NULL,
    location_note TEXT,
    status VARCHAR(20) DEFAULT 'offline', -- online, offline, error
    last_heartbeat TIMESTAMP WITH TIME ZONE, -- Cập nhật từ Telemetry REST
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Bảng Sự kiện (Lưu kết quả AI từ Edge)
CREATE TABLE events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    camera_id UUID REFERENCES cameras(id) ON DELETE CASCADE,
    person_id INTEGER, -- ID từ nvtracker (ByteTrack)
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    class_label VARCHAR(50), -- stand, sit, lie, fall
    confidence FLOAT,
    bbox_json JSONB, -- Lưu tọa độ [x, y, w, h]
    image_url VARCHAR(500), -- Đường dẫn ảnh crop trên MinIO/S3
    video_clip_url VARCHAR(500), -- Clip 10s lưu trên MinIO
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 4. Bảng Cảnh báo (Lưu trạng thái thông báo tới User)
CREATE TABLE alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_id UUID REFERENCES events(id) ON DELETE CASCADE,
    vlm_verdict VARCHAR(50), -- Kết quả xác minh từ Gemini/GPT-4V (True/False)
    notification_type VARCHAR(50), -- FCM, Telegram
    is_acknowledged BOOLEAN DEFAULT FALSE, -- App user đã xem/xác nhận chưa
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 5. Bảng MLOps Confidence Log (Phân tích hiệu năng model)
CREATE TABLE confidence_log (
    id SERIAL PRIMARY KEY,
    event_id UUID REFERENCES events(id),
    model_version VARCHAR(50), -- YOLOv8-cls v1/v2
    inference_latency_ms FLOAT,
    environment_metadata JSONB, -- Điều kiện ánh sáng, góc cam
    is_flagged_for_retrain BOOLEAN DEFAULT FALSE -- Đánh dấu cho MLOps collector
);

-- Index để tối ưu truy vấn lịch sử cho App
CREATE INDEX idx_events_camera_timestamp ON events(camera_id, timestamp DESC);
CREATE INDEX idx_alerts_is_acknowledged ON alerts(is_acknowledged);
