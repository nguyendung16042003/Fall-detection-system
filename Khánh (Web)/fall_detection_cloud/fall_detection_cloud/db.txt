-- Kích hoạt tiện ích tạo UUID
CREATE EXTENSION IF NOT EXISTS "uuid-ossp"; [2]

-- 1. BẢNG NGƯỜI DÙNG (USERS)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    role VARCHAR(50) DEFAULT 'caregiver', -- admin / caregiver [2, 7]
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. BẢNG THIẾT BỊ NGƯỜI DÙNG (USER_DEVICES) - Quản lý FCM Token
-- Hỗ trợ một người dùng đăng nhập trên nhiều thiết bị (Android/iOS) theo API 8.1 [8]
CREATE TABLE user_devices (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    device_id VARCHAR(255) NOT NULL, -- Định danh thiết bị (ví dụ: android_device_001) [8]
    platform VARCHAR(50), -- android / ios [8]
    fcm_token VARCHAR(500) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. BẢNG CAMERA
CREATE TABLE cameras (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    rtsp_url VARCHAR(255) NOT NULL,
    edge_device_id VARCHAR(100) NOT NULL, -- Định danh Jetson quản lý cam (ví dụ: jetson_nano_01) [3, 9]
    location_note TEXT,
    status VARCHAR(20) DEFAULT 'unknown', -- online / offline [3, 9]
    last_heartbeat TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 4. BẢNG CẤU HÌNH QUY TẮC (CAMERA_RULES)
-- Lưu trữ các thông số kỹ thuật cho Temporal State Transition Detector theo API 5.1 & 5.2 [10, 11]
CREATE TABLE camera_rules (
    id SERIAL PRIMARY KEY,
    camera_id UUID REFERENCES cameras(id) ON DELETE CASCADE,
    time_window_sec INTEGER DEFAULT 2, -- Cửa sổ thời gian (Temporal Window) [4, 10]
    min_lying_frames INTEGER DEFAULT 5, -- Số frame nằm tối thiểu để tính là ngã [10, 12]
    high_confidence_threshold FLOAT DEFAULT 0.85, -- Ngưỡng báo động ngay [10, 13]
    low_confidence_threshold FLOAT DEFAULT 0.40, -- Ngưỡng cần Cloud/VLM kiểm tra [10, 13]
    enable_vlm_verify BOOLEAN DEFAULT TRUE, -- Cho phép bật/tắt xác minh bằng AI [10]
    is_active BOOLEAN DEFAULT TRUE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 5. BẢNG SỰ KIỆN (EVENTS)
-- Lưu trữ dữ liệu chi tiết từ Edge gửi lên (MQTT/REST) theo API 3.1 & 3.3 [14, 15]
CREATE TABLE events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(), -- event_id [4, 14]
    camera_id UUID REFERENCES cameras(id) ON DELETE CASCADE,
    event_type VARCHAR(50) DEFAULT 'fall_candidate', [14, 15]
    severity VARCHAR(20) DEFAULT 'low', -- high / low [4, 14]
    person_id INTEGER, -- ID người từ nvtracker [4, 11, 14]
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    bbox_json JSONB, -- {x1, y1, x2, y2} [14-16]
    state_before VARCHAR(50), -- standing / sitting [11, 14, 15]
    state_after VARCHAR(50), -- lying [11, 14, 15]
    transition_time_ms INTEGER, -- Thời gian chuyển trạng thái [11, 14, 15]
    classification_confidence FLOAT, -- Độ tin cậy từ model YOLOv8-cls [14, 15]
    fall_confidence FLOAT, -- Độ tin cậy từ Temporal Detector [14, 15]
    vlm_verdict VARCHAR(50), -- fall_likely / false_alarm (kết quả Gemini/GPT-4V) [15, 17]
    vlm_confidence FLOAT, -- Độ tin cậy của AI xác minh [15]
    image_url VARCHAR(500), -- snapshot_url [14-16]
    video_clip_url VARCHAR(500), -- clip_url (rolling buffer 10s) [14-16]
    status VARCHAR(20) DEFAULT 'new', -- new / acknowledged / resolved [15, 16, 18]
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 6. BẢNG CẢNH BÁO (ALERTS)
-- Quản lý trạng thái thông báo và xác nhận từ người dùng theo API 4.1 & 4.2 [16, 19]
CREATE TABLE alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_id UUID REFERENCES events(id) ON DELETE CASCADE,
    title VARCHAR(255),
    message TEXT,
    is_acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by UUID REFERENCES users(id), -- Người thực hiện xác nhận [19]
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    note TEXT, -- Ghi chú khi xác nhận (ví dụ: "Đã kiểm tra qua live view") [19]
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 7. BẢNG NHẬT KÝ HỆ THỐNG (TELEMETRY_LOGS)
-- Theo dõi sức khỏe thiết bị Jetson theo API 6.1 & 6.2 [20, 21]
CREATE TABLE telemetry_logs (
    id BIGSERIAL PRIMARY KEY,
    edge_device_id VARCHAR(100) NOT NULL, [20]
    pipeline_status VARCHAR(50), -- running / error [20]
    fps FLOAT, [20, 21]
    ram_percent FLOAT, [20]
    cpu_usage FLOAT, [21]
    gpu_temp FLOAT, [20, 21]
    camera_status_json JSONB, -- {cam_01: online, cam_02: online} [20]
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 8. BẢNG MLOPS CONFIDENCE LOG
-- Phục vụ đánh giá mô hình và tái huấn luyện [13, 21]
CREATE TABLE confidence_log (
    id SERIAL PRIMARY KEY,
    event_id UUID REFERENCES events(id) ON DELETE CASCADE,
    model_version VARCHAR(50), [22]
    inference_latency_ms FLOAT, [22]
    environment_metadata JSONB, [22]
    is_flagged_for_retrain BOOLEAN DEFAULT FALSE [22]
);

-- CHỈ MỤC (INDEX) TỐI ƯU TRUY VẤN [22]
CREATE INDEX idx_events_camera_timestamp ON events(camera_id, timestamp DESC);
CREATE INDEX idx_events_status_severity ON events(status, severity); -- Tối ưu cho API 3.2 [22]
CREATE INDEX idx_alerts_is_acknowledged ON alerts(is_acknowledged); [22]
CREATE INDEX idx_telemetry_device_time ON telemetry_logs(edge_device_id, recorded_at DESC);