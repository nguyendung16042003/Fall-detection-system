# API Contract — Edge–Cloud Fall Detection System

**Owner:** Anh Dũng — Edge / Architecture  
**Version:** Week 1 draft  
**Base URL:** `/api/v1`  
**Main notification channel:** FCM push notification to Flutter app. No SMS in main flow.

---

## 1. Auth API

### 1.1 Login

`POST /auth/login`

Request:

```json
{
  "email": "admin@example.com",
  "password": "123456"
}
```

Response:

```json
{
  "access_token": "jwt_access_token",
  "refresh_token": "jwt_refresh_token",
  "user": {
    "id": "user_01",
    "name": "Admin",
    "email": "admin@example.com",
    "role": "admin"
  }
}
```

---

### 1.2 Refresh token

`POST /auth/refresh`

Request:

```json
{
  "refresh_token": "jwt_refresh_token"
}
```

Response:

```json
{
  "access_token": "new_jwt_access_token"
}
```

---

## 2. Camera API

### 2.1 Get camera list

`GET /cameras`

Response:

```json
[
  {
    "id": "cam_01",
    "name": "Living Room",
    "rtsp_url": "rtsp://username:password@192.168.1.10:554/stream1",
    "status": "online",
    "edge_device_id": "jetson_nano_01",
    "created_at": "2026-05-31T10:00:00Z"
  }
]
```

---

### 2.2 Create camera

`POST /cameras`

Request:

```json
{
  "name": "Living Room",
  "rtsp_url": "rtsp://username:password@192.168.1.10:554/stream1",
  "edge_device_id": "jetson_nano_01"
}
```

Response:

```json
{
  "id": "cam_01",
  "name": "Living Room",
  "rtsp_url": "rtsp://username:password@192.168.1.10:554/stream1",
  "status": "unknown",
  "edge_device_id": "jetson_nano_01"
}
```

---

### 2.3 Update camera

`PUT /cameras/{camera_id}`

Request:

```json
{
  "name": "Living Room Camera",
  "rtsp_url": "rtsp://username:password@192.168.1.10:554/stream1",
  "enabled": true
}
```

Response:

```json
{
  "id": "cam_01",
  "name": "Living Room Camera",
  "enabled": true
}
```

---

### 2.4 Delete camera

`DELETE /cameras/{camera_id}`

Response:

```json
{
  "message": "camera deleted successfully"
}
```

---

## 3. Event API

### 3.1 Create event from Edge fallback REST

Normally Edge sends events through MQTT. This endpoint is for fallback/debug.

`POST /events`

Request:

```json
{
  "schema_version": "1.0.0",
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

Response:

```json
{
  "event_id": "evt_20260531_000001",
  "status": "received"
}
```

---

### 3.2 Get event history

`GET /events`

Query parameters:

| Name | Type | Required | Description |
|---|---|---:|---|
| `camera_id` | string | no | Filter by camera |
| `from` | string | no | ISO-8601 start time |
| `to` | string | no | ISO-8601 end time |
| `severity` | string | no | high / low |
| `status` | string | no | new / acknowledged / resolved |
| `page` | integer | no | default 1 |
| `limit` | integer | no | default 20 |

Response:

```json
{
  "items": [
    {
      "event_id": "evt_20260531_000001",
      "camera_id": "cam_01",
      "camera_name": "Living Room",
      "timestamp": "2026-05-31T10:30:12Z",
      "severity": "high",
      "status": "new",
      "fall_confidence": 0.86,
      "snapshot_url": "https://storage.example.com/events/evt_20260531_000001.jpg"
    }
  ],
  "page": 1,
  "limit": 20,
  "total": 1
}
```

---

### 3.3 Get event detail

`GET /events/{event_id}`

Response:

```json
{
  "event_id": "evt_20260531_000001",
  "event_type": "fall_candidate",
  "severity": "high",
  "camera_id": "cam_01",
  "camera_name": "Living Room",
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
  "vlm_verdict": "fall_likely",
  "vlm_confidence": 0.78,
  "snapshot_url": "https://storage.example.com/events/evt_20260531_000001.jpg",
  "clip_url": "https://storage.example.com/events/evt_20260531_000001.mp4",
  "status": "new"
}
```

---

## 4. Alert API

### 4.1 Get alerts

`GET /alerts`

Response:

```json
{
  "items": [
    {
      "alert_id": "alert_001",
      "event_id": "evt_20260531_000001",
      "title": "Fall detected at Living Room",
      "message": "A fall candidate was detected at Camera 1",
      "severity": "high",
      "status": "new",
      "created_at": "2026-05-31T10:30:13Z"
    }
  ]
}
```

---

### 4.2 Acknowledge alert

`PATCH /alerts/{alert_id}/ack`

Request:

```json
{
  "user_id": "user_01",
  "note": "Đã kiểm tra qua live view"
}
```

Response:

```json
{
  "alert_id": "alert_001",
  "status": "acknowledged",
  "acknowledged_by": "user_01",
  "acknowledged_at": "2026-05-31T10:35:00Z"
}
```

---

## 5. Rule Config API

### 5.1 Get camera rule

`GET /rules/{camera_id}`

Response:

```json
{
  "camera_id": "cam_01",
  "time_window_sec": 2,
  "min_lying_frames": 5,
  "high_confidence_threshold": 0.85,
  "low_confidence_threshold": 0.40,
  "enable_vlm_verify": true,
  "enabled": true
}
```

---

### 5.2 Update camera rule

`PUT /rules/{camera_id}`

Request:

```json
{
  "time_window_sec": 2,
  "min_lying_frames": 5,
  "high_confidence_threshold": 0.85,
  "low_confidence_threshold": 0.40,
  "enable_vlm_verify": true,
  "enabled": true
}
```

Response:

```json
{
  "camera_id": "cam_01",
  "message": "rule updated successfully"
}
```

---

## 6. Telemetry API

### 6.1 Send telemetry from Edge

`POST /telemetry`

Request:

```json
{
  "schema_version": "1.0.0",
  "edge_device_id": "jetson_nano_01",
  "timestamp": "2026-05-31T10:30:12Z",
  "pipeline_status": "running",
  "fps": 12.5,
  "ram_percent": 72,
  "temperature_celsius": 68,
  "camera_status": {
    "cam_01": "online",
    "cam_02": "online"
  }
}
```

Response:

```json
{
  "status": "received"
}
```

---

### 6.2 Get edge status

`GET /telemetry/{edge_device_id}`

Response:

```json
{
  "edge_device_id": "jetson_nano_01",
  "last_seen": "2026-05-31T10:30:12Z",
  "pipeline_status": "running",
  "fps": 12.5,
  "ram_percent": 72,
  "temperature_celsius": 68,
  "camera_status": {
    "cam_01": "online",
    "cam_02": "online"
  }
}
```

---

## 7. Live View API

### 7.1 Get live stream URL

`GET /live/{camera_id}`

Response:

```json
{
  "camera_id": "cam_01",
  "stream_type": "webrtc",
  "stream_url": "https://stream.example.com/webrtc/cam_01"
}
```

---

## 8. FCM Device Token API

### 8.1 Register app device token

`POST /devices/fcm-token`

Request:

```json
{
  "user_id": "user_01",
  "device_id": "android_device_001",
  "platform": "android",
  "fcm_token": "fcm_registration_token"
}
```

Response:

```json
{
  "status": "registered"
}
```

---

## 9. Error Response Format

All APIs should use this error format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "camera_id is required",
    "details": {}
  }
}
```

