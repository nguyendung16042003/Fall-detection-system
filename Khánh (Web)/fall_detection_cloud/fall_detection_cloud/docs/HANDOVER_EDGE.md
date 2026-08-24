# Handover Documentation - Edge Integration (Dũng)

## Thông tin kết nối Cloud Infrastructure

### 1. MinIO Object Storage

**Endpoint:**
- API: `http://<CLOUD_IP>:9000`
- Console: `http://<CLOUD_IP>:9001`

**Credentials:**
- Access Key: `${MINIO_ROOT_USER}` (mặc định: `admin`)
- Secret Key: `${MINIO_ROOT_PASSWORD}` (mặc định: `password123`)

**Buckets:**
- `fall-events` - Lưu snapshots và clips sự kiện
- `false-positives` - Lưu clips cho MLOps (false positive cases)
- `clips` - Lưu video clips 10 giây khi có sự kiện

**Python SDK Example:**
```python
from minio import Minio
from minio.error import S3Error

client = Minio(
    "cloud-ip:9000",
    access_key="admin",
    secret_key="password123",
    secure=False,
)

# Upload clip
client.fput_object(
    "fall-events",
    "cam_01/event_123.zip",
    "/path/to/clip.zip",
    content_type="application/zip",
)
```

### 2. MediaMTX Live Streaming Server

**RTSP Push (Jetson → Cloud):**
- Protocol: RTSP over TCP
- Port: `8554`
- URL pattern: `rtsp://<CLOUD_IP>:8554/cam_<id>`
- Authentication:
  - Username: `jetson`
  - Password: `${RTSP_PASSWORD}` (mặc định: `jetsonpass`)

**HLS Playback (Cloud → App):**
- Protocol: HLS
- Port: `8888`
- URL pattern: `http://<CLOUD_IP>:8888/cam_<id>/index.m3u8`

**FFmpeg Push Example:**
```bash
ffmpeg -re -i /dev/video0 \
  -c:v libx264 -preset ultrafast -tune zerolatency \
  -f rtsp rtsp://<CLOUD_IP>:8554/cam_01
```

### 3. MQTT Broker (RabbitMQ)

**Connection:**
- Host: `<CLOUD_IP>`
- Port: `1883`
- Username: `${RABBITMQ_USER}` (mặc định: `guest`)
- Password: `${RABBITMQ_PASS}` (mặc định: `guest`)

**Topics:**
- Fall Events: `events/cam_<id>/fall`
- Telemetry: `telemetry/cam_<id>/status`

**MQTT Schema v2:** Xem file `mqtt_schema_v2.json`

## Payload Format

### Fall Event Payload

```json
{
  "schema_version": "1.1",
  "event_id": "<uuid-v4>",
  "cam_id": "cam_01",
  "person_id": 3,
  "timestamp_utc": "2026-06-15T10:30:00.123Z",
  "event_type": "fall_candidate",
  
  "detection": {
    "class_before": "standing",
    "final_class": "lying",
    "confidence": 0.92,
    "bbox_xyxy": [120, 80, 380, 420],
    "frame_width": 1280,
    "frame_height": 720
  },
  
  "rule": {
    "version": "1.0",
    "trigger": "standing_to_lying",
    "transition_ms": 1800,
    "window_ms": 2000
  },
  
  "frames": [
    { "offset_ms": -2500, "jpeg_b64": "<base64>" },
    { "offset_ms": -2000, "jpeg_b64": "<base64>" },
    { "offset_ms": -1500, "jpeg_b64": "<base64>" },
    { "offset_ms": -1000, "jpeg_b64": "<base64>" },
    { "offset_ms":  -500, "jpeg_b64": "<base64>" },
    { "offset_ms":     0, "jpeg_b64": "<base64>" }
  ]
}
```

**Lưu ý quan trọng:**
- Gửi chính xác 6 frame JPEG (base64)
- Frame offset_ms: -2500, -2000, -1500, -1000, -500, 0
- JPEG quality: 80, crop vùng person + padding 20%
- Tổng payload ước tính: 300-600 KB

### Telemetry Payload

```json
{
  "schema_version": "1.1",
  "cam_id": "cam_01",
  "timestamp_utc": "2026-06-15T10:30:00.000Z",
  
  "pipeline": {
    "state": "running",
    "fps_pgie": 15.2,
    "fps_sgie": 12.8,
    "active_tracks": 2
  },
  
  "system": {
    "ram_used_mb": 2048,
    "ram_total_mb": 4096,
    "cpu_temp_c": 65.3,
    "gpu_temp_c": 68.1,
    "cpu_usage_pct": 45.2,
    "disk_free_gb": 8.4
  },
  
  "network": {
    "mqtt_connected": true,
    "last_event_sent_utc": "2026-06-15T10:29:45.000Z"
  }
}
```

## Cloud Processing Pipeline

Khi Jetson gửi fall event lên Cloud, chuỗi xử lý:

1. **Validate** - Kiểm tra schema theo mqtt_schema_v2.json
2. **Dedup** - Gộp sự kiện từ 2 camera cách nhau < 2s
3. **VLM Verify** - Gọi Gemini Flash với 6 frame để xác thực
4. **Alert** - Nếu VLM xác nhận ngã → Tạo alert + Gửi FCM + Telegram
5. **MLOps** - Nếu VLM từ chối → Lưu clip với tag false_positive

## Testing Checklist

### 1. RTSP Stream Push
- [ ] Jetson có thể push RTSP stream đến MediaMTX
- [ ] HLS stream có thể phát được trên browser: `http://<CLOUD_IP>:8888/cam_01/index.m3u8`

### 2. MQTT Event Publish
- [ ] Jetson có thể connect đến MQTT broker
- [ ] Fall event được publish đúng topic: `events/cam_01/fall`
- [ ] Telemetry được publish đúng topic: `telemetry/cam_01/status`

### 3. MinIO Upload
- [ ] Jetson có thể upload clip đến MinIO bucket
- [ ] Presigned URL có thể truy cập được

### 4. End-to-End Test
- [ ] Tạo fall event giả lập
- [ ] Kiểm tra Cloud nhận và xử lý event
- [ ] Kiểm tra VLM verification chạy
- [ ] Kiểm tra alert được tạo

## Troubleshooting

### MQTT Connection Failed
- Kiểm tra RabbitMQ đang chạy: `docker ps | grep rabbitmq`
- Kiểm tra port 1883 có mở: `telnet <CLOUD_IP> 1883`
- Kiểm tra username/password trong .env

### RTSP Push Failed
- Kiểm tra MediaMTX đang chạy: `docker ps | grep mediamtx`
- Kiểm tra port 8554 có mở: `telnet <CLOUD_IP> 8554`
- Kiểm tra authentication trong mediamtx.yml

### MinIO Upload Failed
- Kiểm tra MinIO đang chạy: `docker ps | grep minio`
- Kiểm tra bucket đã tạo: `python scripts/setup_minio.py`
- Kiểm tra credentials trong .env

## Contact

- Cloud Developer: Khánh
- Cloud IP: `<CLOUD_IP>` (cập nhật khi có)
- Support: Slack channel #fall-detection-cloud
