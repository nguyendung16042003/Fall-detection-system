# Handover Documentation - Mobile App Integration (Duy)

## API Base URL

**Development:** `http://<CLOUD_IP>:8000/api`
**Production:** `https://<DOMAIN>/api`

## Authentication

### JWT Token Authentication

**Login Endpoint:** `POST /auth/login`

**Request:**
```json
{
  "username": "caregiver",
  "password": "password"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

**Usage:** Include `Authorization: Bearer <token>` header in all API requests.

## Alert API

### List Alerts

**Endpoint:** `GET /alerts`

**Query Parameters:**
- `cam_id` (optional): Filter by camera ID (e.g., `cam_01`)
- `acknowledged` (optional): Filter by acknowledgment status (`true`/`false`)
- `page` (optional): Page number (default: 1)
- `page_size` (optional): Items per page (default: 20, max: 100)

**Example Request:**
```
GET /alerts?acknowledged=false&page=1&page_size=20
```

**Response:**
```json
{
  "total": 45,
  "page": 1,
  "page_size": 20,
  "items": [
    {
      "id": 123,
      "event_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "cam_id": "cam_01",
      "timestamp_utc": "2026-06-15T10:30:00.123Z",
      "acknowledged": false,
      "acknowledged_at": null,
      "acknowledged_by": null,
      "fcm_sent": true,
      "vlm_reason": "Person transitioned from standing to lying on floor with sudden motion"
    }
  ]
}
```

### Get Single Alert

**Endpoint:** `GET /alerts/{alert_id}`

**Response:**
```json
{
  "id": 123,
  "event_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "cam_id": "cam_01",
  "timestamp_utc": "2026-06-15T10:30:00.123Z",
  "acknowledged": false,
  "acknowledged_at": null,
  "acknowledged_by": null,
  "fcm_sent": true,
  "vlm_reason": "Person transitioned from standing to lying on floor with sudden motion"
}
```

### Acknowledge Alert

**Endpoint:** `PATCH /alerts/{alert_id}/acknowledge`

**Request Body:** Empty

**Response:**
```json
{
  "id": 123,
  "acknowledged": true,
  "acknowledged_at": "2026-06-15T10:35:00.000Z",
  "acknowledged_by": "caregiver"
}
```

## Event API

### Get Event Details

**Endpoint:** `GET /events/{event_id}`

**Response:**
```json
{
  "id": 456,
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
    "confidence": 0.85,
    "reason": "Person transitioned from standing to lying on floor with sudden motion"
  },
  "clip_url": "http://<CLOUD_IP>:9000/fall-events/abc123.zip",
  "snapshot_url": "http://<CLOUD_IP>:9000/fall-events/def456.jpg",
  "created_at": "2026-06-15T10:30:01.000Z"
}
```

**Important Fields:**
- `clip_url`: URL to download 6-frame ZIP clip (10 seconds before fall)
- `snapshot_url`: URL to download trigger frame JPEG
- `vlm_result.fall`: VLM verification result (true = confirmed fall)
- `vlm_result.confidence`: VLM confidence score (0.0 - 1.0)

## Live Stream API

### Get Live Stream URL

**Endpoint:** `GET /cameras/{cam_id}/live`

**Response:**
```json
{
  "cam_id": "cam_01",
  "name": "Living Room Camera",
  "hls_url": "http://<CLOUD_IP>:8888/cam_01/index.m3u8",
  "rtsp_url": "rtsp://<CLOUD_IP>:8554/cam_01",
  "status": "online"
}
```

**HLS URL Format:**
- Pattern: `http://<CLOUD_IP>:8888/cam_<id>/index.m3u8`
- Protocol: HLS (HTTP Live Streaming)
- Supported by: iOS, Android (ExoPlayer), Web (hls.js)

**Example URLs:**
- Camera 01: `http://192.168.1.100:8888/cam_01/index.m3u8`
- Camera 02: `http://192.168.1.100:8888/cam_02/index.m3u8`
- Camera 03: `http://192.168.1.100:8888/cam_03/index.m3u8`

## Camera API

### List Cameras

**Endpoint:** `GET /cameras`

**Response:**
```json
{
  "items": [
    {
      "id": 1,
      "cam_id": "cam_01",
      "name": "Living Room Camera",
      "location": "Living Room",
      "rtsp_url": "rtsp://jetson-cam01:8554/cam_01",
      "is_active": true,
      "status": "online"
    }
  ]
}
```

## FCM Push Notification

### Notification Payload

When a fall is detected and confirmed by VLM, the app will receive an FCM push notification with the following payload:

**Data Payload:**
```json
{
  "event_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "alert_id": "123",
  "cam_id": "cam_01",
  "type": "fall_alert"
}
```

**Notification Display:**
- Title: `Phát hiện ngã tại {camera_name}`
- Body: Camera ID, timestamp, confidence, VLM reason

### FCM Token Registration

**Endpoint:** `POST /users/me/device-tokens`

**Request:**
```json
{
  "device_id": "unique_device_id",
  "platform": "ios",
  "fcm_token": "firebase_device_token"
}
```

## Error Response Format

All API errors follow this format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": {}
  }
}
```

**Common Error Codes:**
- `UNAUTHORIZED` (401): Invalid or missing authentication token
- `FORBIDDEN` (403): User does not have permission
- `NOT_FOUND` (404): Resource not found
- `VALIDATION_ERROR` (422): Invalid request data
- `INTERNAL_ERROR` (500): Server error

## Clip URL Structure

### Presigned URL Format

Clips are stored in MinIO and accessible via presigned URLs:

**Pattern:** `http://<CLOUD_IP>:9000/<bucket>/<object_key>`

**Buckets:**
- `fall-events`: Regular event clips
- `false-positives`: MLOps clips (VLM rejected)

**Object Key Format:**
- Snapshots: `snapshots/{uuid}.jpg`
- Clips: `clips/{uuid}.zip` (6-frame ZIP)
- False Positives: `false_positives/{uuid}.zip`

**Example URLs:**
```
Snapshot: http://192.168.1.100:9000/fall-events/snapshots/abc123.jpg
Clip: http://192.168.1.100:9000/fall-events/clips/def456.zip
False Positive: http://192.168.1.100:9000/false-positives/ghi789.zip
```

**Downloading Clips:**
- Clips are ZIP files containing 6 JPEG frames
- Frame naming: `frame_000.jpg`, `frame_001.jpg`, ..., `frame_005.jpg`
- Frame order: chronological (offset_ms: -2500 to 0)

## Testing Checklist

### 1. Authentication
- [ ] Login with valid credentials returns JWT token
- [ ] Protected endpoints require valid token
- [ ] Invalid token returns 401 error

### 2. Alert List
- [ ] GET /alerts returns list of alerts
- [ ] Pagination works correctly
- [ ] Filter by `acknowledged` works
- [ ] Filter by `cam_id` works

### 3. Alert Acknowledge
- [ ] PATCH /alerts/{id}/acknowledge marks alert as acknowledged
- [ ] Response includes acknowledgment timestamp
- [ ] Cannot acknowledge already acknowledged alert

### 4. Live Stream
- [ ] GET /cameras/{id}/live returns HLS URL
- [ ] HLS URL is playable in video player
- [ ] Stream latency is acceptable (< 5 seconds)

### 5. Event Details
- [ ] GET /events/{id} returns full event details
- [ ] `clip_url` is accessible and downloadable
- [ ] `snapshot_url` is accessible and downloadable
- [ ] VLM result fields are populated

### 6. FCM Notifications
- [ ] App receives push notification when fall is detected
- [ ] Notification payload contains correct data
- [ ] Tapping notification opens alert details

## Video Player Recommendations

### iOS
- Use `AVPlayer` with `AVPlayerViewController`
- Supports HLS natively
- Example:
```swift
let url = URL(string: "http://<CLOUD_IP>:8888/cam_01/index.m3u8")
let player = AVPlayer(url: url!)
let playerViewController = AVPlayerViewController()
playerViewController.player = player
present(playerViewController, animated: true)
```

### Android
- Use ExoPlayer
- Supports HLS with HlsMediaSource
- Example:
```kotlin
val url = "http://<CLOUD_IP>:8888/cam_01/index.m3u8"
val mediaItem = MediaItem.fromUri(url)
player.setMediaItem(mediaItem)
player.prepare()
player.play()
```

### React Native
- Use `react-native-video`
- Supports HLS on both iOS and Android
- Example:
```jsx
<Video
  source={{ uri: 'http://<CLOUD_IP>:8888/cam_01/index.m3u8' }}
  resizeMode="contain"
  style={{ width: '100%', height: 300 }}
/>
```

## Rate Limiting

API rate limits (if implemented):
- 100 requests per minute per user
- 1000 requests per hour per user

Rate limit headers:
- `X-RateLimit-Limit`: Request limit
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Unix timestamp when limit resets

## WebSocket (Optional)

For real-time updates, WebSocket endpoint may be available:

**Endpoint:** `ws://<CLOUD_IP>:8000/ws/alerts`

**Message Format:**
```json
{
  "type": "new_alert",
  "data": {
    "alert_id": 123,
    "cam_id": "cam_01",
    "timestamp_utc": "2026-06-15T10:30:00.123Z"
  }
}
```

## Contact

- Backend Developer: Khánh
- Cloud IP: `<CLOUD_IP>` (cập nhật khi có)
- API Documentation: `http://<CLOUD_IP>:8000/docs` (Swagger UI)
- Support: Slack channel #fall-detection-app
