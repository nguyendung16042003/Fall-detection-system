# Multi-Camera Fall Detection System on Edge Device with VLM Verification

Capstone project (đồ án tốt nghiệp): phát hiện té ngã cho người già theo thời gian
thực bằng **2 camera IP (RTSP)** chạy trên **Jetson Orin Nano 8GB**, xác minh lại
event bằng **VLM (Gemini)** trước khi báo động, gửi cảnh báo tới **app di động**
qua cloud backend. Đóng góp học thuật chính: kiến trúc xử lý đa camera giải 3 bài
toán con — Re-Identification, Identity Association, Boundary Feature Fusion — cho
phép hệ thống bám đúng danh tính và tư thế 1 người xuyên suốt 2 camera, kể cả khi
họ ngã ngay tại vùng giao giữa 2 góc quay.

## Team

| Thành viên | Vai trò | Folder trong repo | Branch |
|---|---|---|---|
| **Tấn Dũng** | AI / Model — nghiên cứu, huấn luyện, benchmark 3 module Re-ID/Identity Association/Boundary Feature Fusion | `Tan Dung (AI)/` | `feature/ai` |
| **Dũng** | Edge / Architecture (lead) — hạ tầng pipeline DeepStream trên Jetson, tích hợp toàn hệ thống, schema MQTT & API contract | `Dũng (Edge)/` | `feature/edge` |
| **Khánh** | Cloud / Backend — FastAPI, MQTT consumer, VLM verify, notification (FCM/Telegram), REST API cho mobile | `Khánh (Web)/` | `feature/backend` |
| **Duy** | Mobile App (Flutter) | *(chưa merge vào `main(to-nhất)`)* | `feature/mobile` |

> Lưu ý: "Dũng" (Edge, lead kiến trúc/hạ tầng) và "Tấn Dũng" (AI/model) là **2 thành
> viên khác nhau** trùng tên đầu — tài liệu trong `Dũng (Edge)/docs/` luôn phân biệt
> rõ 2 phạm vi việc này.

## Kiến trúc tổng thể (luồng dữ liệu)

```
2× camera IP (RTSP, Ezviz/Hikvision)
        │
        ▼
Jetson Orin Nano 8GB — DeepStream 7.0  (Dũng - Edge)
  PGIE  YOLOv8n (person, 640px)
  Tracker  NvSORT (thay ByteTrack — segfault trên DeepStream 7.0)
  SGIE  YOLOv8n-cls (6 lớp tư thế: bend/exercise/half_person/lie/sit/stand)
        │
        ├─ Re-Identification (OSNet_x0.25 KD, embedding 512-d)
        │    → nhận lại người sau khi mất dấu 1 camera (đi qua vùng chết)
        │
        ├─ Identity Association (homography sàn + Hungarian, THUẦN hình học)
        │    → ghép 2 camera cùng thấy 1 người thành 1 global ID, ngưỡng ~0.5m
        │
        └─ Boundary Feature Fusion (chỉ khi ghép được cặp)
             → gộp feature-grid 2 cam theo trọng số epipolar (không train),
               cho quyết định tư thế/ngã chính xác hơn khi người ở vùng biên
        │
        ▼
Temporal Fall Rule  (đứng/ngồi → nằm trong <2s, chạy SAU bước gộp danh tính)
        │
        ▼
MQTT (RabbitMQ + plugin MQTT, broker chạy trên máy Khánh)
  events/cam_{id}/fall        — event ngã + 6 ảnh JPEG quanh thời điểm trigger
  telemetry/cam_{id}/status   — heartbeat/trạng thái pipeline
        │
        ▼
Cloud Backend — FastAPI + PostgreSQL + MinIO + MediaMTX  (Khánh)
  MQTT consumer → lưu DB → VLM verify (Gemini, phân biệt "ngã" vs "nằm nghỉ")
  → tạo alert → push FCM + Telegram
  REST API (JWT) cho: auth, camera, event history, alert ack, telemetry, live HLS
        │
        ▼
Mobile App (Flutter)  (Duy)
  nhận push FCM, xem lịch sử event/alert, xem live HLS, quản lý camera
```

## Cấu trúc repo

```
Tan Dung (AI)/        Nghiên cứu + huấn luyện 3 module AI, report, slide, demo video
                       → xem Tan Dung (AI)/README.md
Dũng (Edge)/           Pipeline DeepStream chạy trên Jetson + tích hợp toàn hệ thống
  src/                   pipeline.py, fall_detector.py, event_builder.py,
                         mqtt_publisher.py, rolling_buffer.py, visual_dump.py,
                         reid.py / identity_association.py / boundary_feature_fusion.py
                         (port trực tiếp từ Tan Dung (AI)/Handoff_for_Edge/,
                          giữ nguyên công thức toán, chỉ đổi cách chạy model
                          sang TensRT/pyds cho tương thích Jetson)
  tests/                 unit test thuần Python (không cần GPU/camera)
  tools/                 mqtt_sub.py, compare_onnx_deepstream.py
  docs/                  api_contract_v2.md, mqtt_schema_v2.json
  cloud/                 Dockerfile/requirements cho phần backend chạy cạnh Edge
Khánh (Web)/            Cloud backend FastAPI
  fall_detection_cloud/fall_detection_cloud/
    app/                   api/, core/, models/, schemas/, services/ (VLM, FCM,
                            Telegram, MQTT consumer, alert pipeline)
    alembic/               DB migrations
    → xem Khánh (Web)/.../README.md để chạy local + đầy đủ API
```

## Hạ tầng & công nghệ

| Thành phần | Công nghệ |
|---|---|
| Edge inference | Jetson Orin Nano 8GB, JetPack 6.0, DeepStream 7.0, TensorRT 8.6.2, CUDA 12.2 |
| Camera | 2× IP camera RTSP (Ezviz CS-H6C/C6N hoặc Hikvision, tuỳ đợt lắp), qua NAT |
| Message broker | RabbitMQ + MQTT plugin (chạy trên máy Khánh, Docker Compose) |
| Backend | FastAPI, SQLAlchemy 2.0 + PostgreSQL, Alembic, JWT | 
| Lưu trữ media | MinIO (snapshot/clip), MediaMTX (HLS live view) |
| Xác minh event | Gemini (VLM) — phân loại "ngã thật" vs "nằm nghỉ" từ 6 ảnh quanh thời điểm trigger |
| Notification | Firebase Cloud Messaging + Telegram Bot |
| Mobile | Flutter |

## Trạng thái hiện tại (tóm tắt, xem chi tiết trong từng folder)

- **AI (Tấn Dũng)**: 3 module Re-ID/Identity Association/Boundary Feature Fusion đã
  huấn luyện, benchmark, và đóng gói bàn giao tại `Tan Dung (AI)/Handoff_for_Edge/`.
- **Edge (Dũng)**: pipeline DeepStream chạy 2 nguồn RTSP ổn định (~24 FPS), đã publish
  MQTT thật lên broker Khánh (verified end-to-end). `reid.py` đã viết/test độc lập
  nhưng **chưa nối vào `pipeline.py`**; calibration homography dùng cho Identity
  Association hiện là bản mẫu (phòng test của Tấn Dũng), **chưa phải calib phòng lắp
  camera thật**.
- **Backend (Khánh)**: API contract v2 đầy đủ (auth, camera, event, alert, telemetry,
  live), trigger chain VLM → alert → FCM/Telegram chạy được ở chế độ dry-run khi
  chưa có credential thật.
- **Mobile (Duy)**: đang phát triển trên `feature/mobile`, chưa merge vào
  `main(to-nhất)`.

## Đọc thêm

- `Tan Dung (AI)/README.md` — chi tiết 3 module AI, report, slide.
- `Khánh (Web)/fall_detection_cloud/fall_detection_cloud/README.md` — cài đặt +
  toàn bộ REST API backend.
- `Dũng (Edge)/docs/api_contract_v2.md`, `mqtt_schema_v2.json` — hợp đồng dữ liệu
  giữa Edge và Cloud.
- `(THESIS) Multi-Camera Fall Detection System on Edge Device with VLM
  Verification.pdf`, `(SLIDE) ...pdf` — báo cáo và slide đầy đủ.
