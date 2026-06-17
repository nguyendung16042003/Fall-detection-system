# API Contract v2.0 — Fall Detection System
**Owner:** Dũng (Edge/Architecture)  
**Version:** Week 3 — đã sync với timeline v3 và kiến trúc đã chốt  
**Base URL:** `http://<KHANH_PUBLIC_IP>:8000`  
**Auth:** Bearer JWT — header `Authorization: Bearer <token>` (trừ `/api/auth/*` và `/health`)  
**Content-Type:** `application/json`  
**Swagger UI:** `http://<KHANH_PUBLIC_IP>:8000/docs`

---

> ## 📋 GHI CHÚ CHO KHÁNH — THAY ĐỔI SO VỚI FILE CŨ
>
> File này thay thế hoàn toàn `api_contract.md` v1 (Week 1 draft). Các thay đổi Khánh cần chú ý:
>
> **[CHANGE-1] Xóa toàn bộ `severity: high/low`**  
> File cũ có `severity: high/low` và `high_confidence_threshold/low_confidence_threshold`.  
> Theo timeline v3 quyết định #2: KHÔNG có confidence routing. Tất cả fall event đều qua VLM verify.  
> Khánh xóa field `severity` khỏi DB schema và không cần tạo queue high/low.
>
> **[CHANGE-2] `event_id` đổi format**  
> File cũ: `evt_20260531_000001` (tự đặt)  
> File mới: UUID v4 dạng `a1b2c3d4-e5f6-...` — Jetson tự generate bằng `uuid.uuid4()`  
> Khánh dùng UUID làm primary key trong bảng events.
>
> **[CHANGE-3] `POST /telemetry` bị XÓA**  
> File cũ có REST endpoint nhận telemetry từ Edge.  
> File mới: Telemetry đi qua MQTT topic `telemetry/cam_{id}/status`, Khánh's consumer tự lưu vào DB.  
> Chỉ giữ lại `GET /api/telemetry/{cam_id}/latest` để Duy đọc.
>
> **[CHANGE-4] Live view đổi từ WebRTC → HLS**  
> File cũ: `stream_type: webrtc`  
> File mới: `stream_type: hls`, URL trả về là HLS endpoint của MediaMTX.  
> Khánh cần setup MediaMTX trong Docker Compose (tuần 5).
>
> **[CHANGE-5] Thêm `POST /api/devices/fcm-token`** ← GIỮ từ file cũ  
> Duy cần endpoint này để đăng ký FCM token khi app khởi động.
>
> **[CHANGE-6] Thêm standardized error format** ← GIỮ từ file cũ  
> Mọi API lỗi đều trả về format thống nhất, xem mục 9.
>
> **[CHANGE-7] `GET /api/alerts/{id}/acknowledge` → PATCH không cần body**  
> File cũ có `user_id` và `note` trong body.  
> File mới: body rỗng `{}`, server tự lấy user từ JWT token.

---

## 1. AUTH

### POST `/api/auth/login`
**Request:**
```json
{
  "username": "admin",
  "password": "secret"
}
```
**Response 200:**
```json
{
  "access_token": "<jwt>",
  "refresh_token": "<jwt>",
  "token_type": "bearer",
  "expires_in": 3600
}
```
**Errors:** `401` sai credentials

---

### POST `/api/auth/register`
**Request:**
```json
{
  "username": "user1",
  "password": "P@ssw0rd!",
  "email": "user1@example.com"
}
```
**Response 201:**
```json
{
  "id": 1,
  "username": "user1",
  "email": "user1@example.com",
  "created_at": "2026-06-15T10:00:00Z"
}
```

---

### POST `/api/auth/refresh`
**Request:**
```json
{ "refresh_token": "<jwt>" }
```
**Response 200:**
```json
{
  "access_token": "<jwt>",
  "expires_in": 3600
}
```

---

## 2. CAMERAS

**Camera object:**
```json
{
  "id": 1,
  "cam_id": "cam_01",
  "name": "Phòng khách",
  "rtsp_url": "rtsp://admin:pass@192.168.2.1:554/h264_stream",
  "location": "Tầng 1",
  "is_active": true,
  "created_at": "2026-06-01T00:00:00Z"
}
```

### GET `/api/cameras`
Danh sách tất cả camera.  
**Response 200:** `[...camera objects]`

### POST `/api/cameras`
**Request:**
```json
{
  "cam_id": "cam_01",
  "name": "Phòng khách",
  "rtsp_url": "rtsp://admin:pass@192.168.2.1:554/h264_stream",
  "location": "Tầng 1"
}
```
**Response 201:** camera object

### GET `/api/cameras/{cam_id}`
**Response 200:** camera object | `404` không tìm thấy

### PUT `/api/cameras/{cam_id}`
**Request:** subset của camera fields  
**Response 200:** updated camera object

### DELETE `/api/cameras/{cam_id}`
**Response 204:** no body

---

## 3. EVENTS

**Event object (đầy đủ):**
```json
{
  "id": 42,
  "event_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "cam_id": "cam_01",
  "person_id": 3,
  "timestamp_utc": "2026-06-15T10:30:00.123Z",
  "event_type": "fall_candidate",
  "status": "confirmed",
  "detection": {
    "final_class": "lying",
    "confidence": 0.92,
    "bbox_xyxy": [120, 80, 380, 420]
  },
  "vlm_result": {
    "fall": true,
    "confidence": 0.95,
    "reason": "Person transitioned from standing to lying in 1.8s"
  },
  "clip_url": "https://<minio-presigned>/clips/event_42.mp4",
  "created_at": "2026-06-15T10:30:03Z"
}
```

**Field `status`:** `pending` | `confirmed` | `false_positive`

> **[NOTE cho Khánh]:** Không có field `severity`. Không có `fall_confidence` riêng — dùng `detection.confidence` cho SGIE output, `vlm_result.confidence` cho VLM output.

### GET `/api/events`
**Query params:**

| Param | Type | Default | Mô tả |
|---|---|---|---|
| `cam_id` | string | — | Lọc theo camera |
| `status` | string | — | `confirmed` / `false_positive` / `pending` |
| `start_date` | ISO date | — | `2026-06-01` |
| `end_date` | ISO date | — | `2026-06-15` |
| `page` | int | 1 | |
| `page_size` | int | 20 | Max 100 |

**Response 200:**
```json
{
  "total": 150,
  "page": 1,
  "page_size": 20,
  "items": [ "...event objects" ]
}
```

### GET `/api/events/{event_id}`
**Response 200:** full event object | `404`

### POST `/api/events`
REST fallback — dùng khi MQTT lỗi hoặc Khánh bơm data test cho Duy.  
**Request:** full `mqtt_schema fall_event` object  
**Response 201:**
```json
{ "id": 42, "event_id": "a1b2c3d4-..." }
```

---

## 4. ALERTS

**Alert object:**
```json
{
  "id": 1,
  "event_id": "a1b2c3d4-...",
  "cam_id": "cam_01",
  "timestamp_utc": "2026-06-15T10:30:00.123Z",
  "acknowledged": false,
  "acknowledged_at": null,
  "acknowledged_by": null,
  "fcm_sent": true,
  "vlm_reason": "Person transitioned from standing to lying in 1.8s"
}
```

### GET `/api/alerts`
**Query params:** `cam_id`, `acknowledged` (bool), `page`, `page_size`  
**Response 200:** `{ "total": ..., "page": ..., "page_size": ..., "items": [...] }`

### GET `/api/alerts/{id}`
**Response 200:** alert object

### PATCH `/api/alerts/{id}/acknowledge`
> **[NOTE cho Khánh]:** Body rỗng — server lấy user từ JWT token, không cần client gửi `user_id`.

**Request:** `{}` (body rỗng)  
**Response 200:**
```json
{
  "id": 1,
  "acknowledged": true,
  "acknowledged_at": "2026-06-15T10:35:00Z",
  "acknowledged_by": "admin"
}
```

---

## 5. TELEMETRY

> **[NOTE cho Khánh]:** Không có `POST /telemetry` — Edge gửi qua MQTT topic `telemetry/cam_{id}/status`, consumer tự lưu DB. Chỉ expose GET cho Duy đọc.

### GET `/api/telemetry/{cam_id}/latest`
**Response 200:**
```json
{
  "cam_id": "cam_01",
  "timestamp_utc": "2026-06-15T10:30:00Z",
  "is_online": true,
  "pipeline_state": "running",
  "fps_pgie": 15.2,
  "fps_sgie": 12.8,
  "active_tracks": 2,
  "ram_used_mb": 2048,
  "ram_total_mb": 4096,
  "cpu_temp_c": 65.3,
  "gpu_temp_c": 68.1
}
```
`is_online = true` nếu `timestamp_utc` trong vòng 60s qua.

---

## 6. CONFIG / RULES

Edge đọc config này khi khởi động pipeline.

### GET `/api/config/rules`
**Response 200:**
```json
{
  "fall_detection": {
    "transition_window_ms": 2000,
    "lying_ignore_after_ms": 2000,
    "min_confidence_sgie": 0.6,
    "confirm_frames": 5
  },
  "dedup": {
    "window_ms": 2000
  },
  "vlm": {
    "enabled": true,
    "model": "gemini-1.5-flash",
    "timeout_s": 10
  },
  "notifications": {
    "fcm_enabled": true,
    "cameras": {
      "cam_01": { "notify": true },
      "cam_02": { "notify": true }
    }
  }
}
```

### PUT `/api/config/rules`
**Request:** subset của object trên (partial update OK)  
**Response 200:** full updated config

---

## 7. LIVE VIEW

> **[NOTE cho Khánh]:** Đổi từ WebRTC → HLS qua MediaMTX (timeline v3 quyết định #7).  
> Khánh setup MediaMTX trong Docker Compose tuần 5, expose port 8888.  
> Endpoint này chỉ trả URL — Duy dùng `video_player` Flutter để phát HLS.

### GET `/api/live/{cam_id}`
**Response 200:**
```json
{
  "cam_id": "cam_01",
  "stream_type": "hls",
  "stream_url": "http://<KHANH_PUBLIC_IP>:8888/cam_01/index.m3u8"
}
```

---

## 8. FCM DEVICE TOKEN

> **[NOTE cho Khánh]:** Giữ từ file cũ. Duy gọi endpoint này mỗi khi app khởi động để đăng ký/cập nhật FCM token. Khánh lưu vào bảng `device_tokens`, dùng khi gửi push notification.

### POST `/api/devices/fcm-token`
**Request:**
```json
{
  "device_id": "android_device_001",
  "platform": "android",
  "fcm_token": "<fcm_registration_token>"
}
```
> **[NOTE]:** Bỏ `user_id` trong request — lấy từ JWT token.  
> `platform`: `android` | `ios`

**Response 201:**
```json
{ "status": "registered" }
```

---

## 9. HEALTH

### GET `/health`
Không cần auth.  
**Response 200:**
```json
{
  "status": "ok",
  "timestamp": "2026-06-15T10:30:00Z",
  "services": {
    "database": "ok",
    "rabbitmq": "ok",
    "minio": "ok"
  }
}
```

---

## 10. ERROR FORMAT (chuẩn cho mọi API)

> **[NOTE cho Khánh]:** Giữ từ file cũ. Mọi response lỗi đều dùng format này — Duy parse dựa vào `error.code`.

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "cam_id is required",
    "details": {}
  }
}
```

**Các error code thường dùng:**

| Code | HTTP Status | Ý nghĩa |
|---|---|---|
| `VALIDATION_ERROR` | 400 | Request sai format |
| `UNAUTHORIZED` | 401 | Chưa đăng nhập / token hết hạn |
| `FORBIDDEN` | 403 | Không có quyền |
| `NOT_FOUND` | 404 | Resource không tồn tại |
| `INTERNAL_ERROR` | 500 | Lỗi server |

---

## 11. HANDOFF CHECKLIST

| Ai cần | Cần gì | Từ ai | Deadline |
|---|---|---|---|
| Dũng (Jetson) | IP public + port 1883 + user/pass broker | Khánh | T3 ngày 4 |
| Duy (Flutter) | Postman collection + swagger URL | Khánh | T3 ngày 4 |
| Khánh (Consumer) | `mqtt_schema.json` | Dũng | T3 ngày 3 ✅ |
| Duy (FCM) | FCM payload format + Alert API docs | Khánh | T4 giữa tuần |