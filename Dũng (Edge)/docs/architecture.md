# Architecture Diagram & System Flow — Hệ thống phát hiện ngã Edge–Cloud

**Owner:** Anh Dũng — Edge / Architecture  
**Version:** Week 1 draft  
**Scope:** Finalize kiến trúc tổng thể để nhóm Cloud, AI và App cùng code theo một chuẩn.

---

## 1. Mục tiêu hệ thống

Hệ thống nhận video từ 2 camera RTSP, xử lý realtime trên Jetson Nano bằng DeepStream, phát hiện sự kiện nghi ngờ té ngã bằng pipeline AI + logic thời gian, sau đó chỉ gửi dữ liệu đã lọc lên Cloud.

Nguyên tắc chính:

- Video realtime được xử lý ở Edge, không đẩy toàn bộ video lên Cloud.
- Cloud chỉ nhận event, snapshot/clip ngắn và telemetry.
- App Flutter nhận cảnh báo qua FCM push notification.
- Không dùng SMS trong main flow.
- Telegram không nằm trong main flow.

---

## 2. Luồng dữ liệu tổng thể

```text
Camera RTSP
  → nvstreammux
  → PGIE YOLOv8n person detection
  → nvtracker / ByteTrack
  → SGIE YOLOv8-cls posture classification
  → State Buffer
  → Custom Temporal Transition Detector
  → Event Builder
  → nvmsgconv + nvmsgbroker
  → MQTT Broker
  → MQTT Consumer
  → Fall Handler
  → VLM Verify nếu cần
  → Notification Service
  → FCM
  → Flutter App
```

---

## 3. Edge Layer — Jetson Nano + DeepStream

### 3.1 Camera RTSP

Input gồm 2 luồng RTSP:

- `cam_01`
- `cam_02`

Mỗi camera có `camera_id` riêng để Cloud và App biết event thuộc camera nào.

---

### 3.2 nvstreammux

`nvstreammux` gom frame từ 2 camera thành batch để DeepStream xử lý hiệu quả hơn.

Quan trọng: dù gom batch, mỗi frame vẫn phải giữ `source_id` hoặc `camera_id`. Khi xử lý nhiều camera, không được làm mất thông tin nguồn camera.

Ví dụ:

```text
source_id = 0 → cam_01
source_id = 1 → cam_02
```

---

### 3.3 PGIE — YOLOv8n Person Detection

PGIE là model inference đầu tiên.

Nhiệm vụ:

- nhận frame gốc;
- detect người;
- trả về bounding box, class person và confidence.

Output mẫu:

```json
{
  "camera_id": "cam_01",
  "source_id": 0,
  "bbox": [120, 80, 300, 420],
  "class": "person",
  "confidence": 0.92
}
```

---

### 3.4 nvtracker / ByteTrack

Tracker gán `person_id` ổn định cho từng người qua nhiều frame.

Nhiệm vụ:

- giữ ID của cùng một người theo thời gian;
- hỗ trợ State Buffer biết lịch sử trạng thái của đúng người;
- tránh nhầm người A đứng với người B nằm.

Key bắt buộc khi lưu state:

```text
(camera_id, person_id)
```

Không được chỉ dùng `person_id`, vì 2 camera có thể trùng ID.

---

### 3.5 SGIE — YOLOv8-cls Posture Classification

SGIE nhận crop người từ bbox của PGIE.

Nhiệm vụ:

- classify tư thế của từng crop người;
- output 4 class chính:
  - `standing`
  - `sitting`
  - `lying`
  - `other`

Output mẫu:

```json
{
  "camera_id": "cam_01",
  "person_id": 3,
  "state": "lying",
  "classification_confidence": 0.88
}
```

---

### 3.6 State Buffer

State Buffer lưu lịch sử trạng thái gần nhất của từng người.

Key lưu buffer:

```text
camera_id + person_id
```

Ví dụ:

```json
{
  "key": "cam_01:person_3",
  "states": [
    {"t": "10:30:10.000", "state": "standing"},
    {"t": "10:30:10.500", "state": "standing"},
    {"t": "10:30:11.000", "state": "sitting"},
    {"t": "10:30:11.500", "state": "lying"}
  ]
}
```

Mục tiêu: biết người đó vừa chuyển trạng thái hay đã nằm sẵn từ trước.

---

### 3.7 Custom Temporal Transition Detector

Đây là logic phát hiện ngã chính của đồ án.

Không kết luận ngã chỉ vì thấy người nằm. Hệ thống kiểm tra chuỗi trạng thái theo thời gian.

Rule version 1:

```text
Nếu cùng một (camera_id, person_id):
- trạng thái trước đó là standing hoặc sitting;
- trạng thái hiện tại là lying;
- thời gian chuyển sang lying <= 2 giây;
- lying xuất hiện liên tiếp ít nhất N frame;
→ tạo fall_candidate.
```

Bỏ qua nếu:

```text
Người đã lying liên tục > 2 giây trước khi tạo event
→ coi là nằm nghỉ/ngủ, không báo ngã.
```

Ghi chú kỹ thuật: phần này nên triển khai như custom logic trong DeepStream app / pad probe / user app, không nên gọi là `nvdsanalytics` vì rule này là logic riêng của hệ thống.

---

### 3.8 Event Builder

Event Builder đóng gói kết quả thành event có cấu trúc.

Nhiệm vụ:

- tạo `event_id`;
- gắn `camera_id`, `person_id`, `bbox`, `timestamp`;
- gắn trạng thái trước/sau;
- gắn confidence và rule version;
- liên kết snapshot/clip nếu có.

Output mẫu:

```json
{
  "event_id": "evt_20260531_000001",
  "event_type": "fall_candidate",
  "severity": "high",
  "edge_device_id": "jetson_nano_01",
  "camera_id": "cam_01",
  "source_id": 0,
  "person_id": 3,
  "timestamp": "2026-05-31T10:30:12Z",
  "bbox": {
    "x1": 120,
    "y1": 80,
    "x2": 300,
    "y2": 420
  },
  "state_before": "standing",
  "state_after": "lying",
  "transition_time_ms": 1200,
  "classification_confidence": 0.88,
  "fall_confidence": 0.86,
  "snapshot_path": "edge://cam_01/events/evt_20260531_000001.jpg",
  "clip_path": "edge://cam_01/events/evt_20260531_000001.mp4",
  "rule_version": "temporal_v1"
}
```

---

### 3.9 Local Rolling Buffer

Local Rolling Buffer lưu tạm snapshot hoặc clip ngắn quanh thời điểm event.

Mục đích:

- phục vụ Alert Detail trên app;
- phục vụ VLM verify;
- retry khi mất mạng;
- hỗ trợ debug và đánh giá false positive.

Gợi ý:

```text
clip length = 10 giây
pre-event = 5 giây
post-event = 5 giây
```

---

### 3.10 nvmsgconv + nvmsgbroker

`nvmsgconv` chuyển metadata/event thành JSON payload theo schema.

`nvmsgbroker` publish payload lên MQTT Broker.

Topic chính:

```text
event.cam.{camera_id}.fall.high
event.cam.{camera_id}.fall.low
telemetry.edge.{edge_device_id}
status.edge.{edge_device_id}
```

---

### 3.11 Edge Orchestrator

Edge Orchestrator theo dõi sức khỏe Jetson và pipeline.

Nhiệm vụ:

- camera online/offline;
- DeepStream pipeline status;
- FPS;
- RAM;
- nhiệt độ;
- auto-restart nếu pipeline crash;
- gửi telemetry lên Cloud.

---

## 4. Cloud Layer — API Gateway + Message Routing + Verification

### 4.1 MQTT Broker

Broker nhận event từ Jetson.

Edge publish event vào topic:

```text
event.cam.cam_01.fall.high
```

Cloud subscribe:

```text
event.cam.*.fall.#
```

---

### 4.2 MQTT Consumer Group

Consumer nhận message từ broker, parse JSON và chuyển sang Fall Handler.

Nhiệm vụ:

- validate JSON payload;
- kiểm tra schema version;
- retry nếu xử lý lỗi;
- tránh mất event.

---

### 4.3 Fall Handler

Fall Handler xử lý logic backend của event.

Nhiệm vụ:

- validate event;
- deduplicate event;
- lưu event vào database;
- route theo confidence;
- quyết định có cần VLM Verify không.

Routing gợi ý:

```text
fall_confidence >= 0.85 → tạo alert và gửi FCM ngay
0.40 <= fall_confidence < 0.85 → gửi qua VLM Verify trước
fall_confidence < 0.40 → lưu log, không cảnh báo
```

---

### 4.4 VLM Verify

VLM Verify là lớp xác minh bổ sung cho case chưa chắc chắn.

Input:

- snapshot;
- crop người;
- metadata event;
- prompt chuẩn hóa.

Output:

```json
{
  "verdict": "fall_likely",
  "confidence": 0.78,
  "reason": "person is lying on the floor after a sudden posture transition"
}
```

VLM là optional verification layer, không phải lõi phát hiện ngã chính.

---

### 4.5 Notification Service — FCM App Push

Notification Service gửi push notification đến app Flutter qua FCM.

Không dùng SMS. Telegram không nằm trong main flow.

Flow:

```text
Fall Handler / VLM Verify
  → Notification Service
  → Firebase Admin SDK
  → FCM
  → Flutter app
```

---

### 4.6 FastAPI Gateway

FastAPI Gateway cung cấp REST API cho Flutter app và admin dashboard.

Nhóm API chính:

- Auth API
- Camera API
- Event API
- Alert API
- Rule Config API
- Telemetry API
- Live View API

---

### 4.7 Storage

Storage gồm:

- PostgreSQL: users, cameras, events, alerts, rules, telemetry.
- MinIO/S3: snapshot, crop image, video clip.

Không nên lưu trực tiếp ảnh/video lớn vào PostgreSQL.

---

### 4.8 MLOps Collector

MLOps Collector lưu các case khó để phục vụ retrain.

Ví dụ:

- confidence thấp;
- VLM không chắc;
- người dùng đánh dấu false alarm;
- model nhầm sitting/lying.

---

### 4.9 Live Stream Service

Live Stream Service phục vụ app xem camera trực tiếp.

Có thể dùng:

- MediaMTX;
- WebRTC;
- HLS.

Lưu ý: Live view là luồng xem trực tiếp, không phải luồng inference chính.

---

### 4.10 Admin Rules

Admin Rules lưu cấu hình phát hiện ngã.

Ví dụ:

```json
{
  "camera_id": "cam_01",
  "time_window_sec": 2,
  "min_lying_frames": 5,
  "high_confidence_threshold": 0.85,
  "enable_vlm_verify": true
}
```

---

## 5. User Layer — Flutter Mobile / Desktop

### 5.1 Push cảnh báo realtime

App nhận push notification qua FCM khi có alert.

Ví dụ thông báo:

```text
Cảnh báo té ngã tại Camera 1
Thời gian: 10:30:12
Mức độ: High
```

---

### 5.2 Lịch sử và chi tiết event

App hiển thị:

- camera;
- thời gian;
- severity;
- snapshot/clip;
- trạng thái trước/sau;
- confidence;
- VLM verdict nếu có;
- trạng thái xử lý: new / acknowledged / resolved.

---

### 5.3 Live view camera

Người dùng mở app để xem camera trực tiếp.

---

### 5.4 Cấu hình camera/rule

Admin có thể chỉnh:

- tên camera;
- RTSP URL;
- bật/tắt camera;
- confidence threshold;
- time window;
- min lying frames.

---

## 6. Critical Design Decisions

| Quyết định | Lý do |
|---|---|
| Edge-first inference | Giảm tải Cloud, giảm băng thông, tăng realtime |
| PGIE + SGIE 2-stage | Tách detect người và classify tư thế, dễ giải thích và debug |
| Tracker + State Buffer | Cần lịch sử của đúng người để phát hiện chuyển trạng thái |
| Key `(camera_id, person_id)` | Tránh lẫn ID giữa nhiều camera |
| Temporal rule 2 giây | Phân biệt ngã thật với nằm nghỉ/ngủ |
| MQTT event | Tách Edge và Cloud, hợp với event-driven architecture |
| FCM push | Đúng yêu cầu app notification, không cần SMS |
| VLM optional | Tăng độ tin cậy cho case khó, không làm lõi phụ thuộc Cloud |

---

## 7. Deliverable tuần 1 của Anh Dũng

- `architecture.md`
- `mqtt_schema.json`
- `api_contract.md`
- `repo_convention.md`
- `docker-compose.skeleton.yml`

