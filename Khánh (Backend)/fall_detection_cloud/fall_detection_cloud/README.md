# Fall Detection Cloud API (Backend)

Backend FastAPI cho hệ thống phát hiện té ngã theo **api_contract_v2** +
**mqtt_schema_v2**: Auth (JWT), quản lý Camera (định danh `cam_id`), quản lý
User và FCM token, Event API (ingest qua MQTT + REST fallback, lịch sử cho
Mobile), Telemetry (qua MQTT), VLM verify (Gemini), Notification (FCM +
Telegram), Alert API, Live HLS, Alembic migrations và dữ liệu seed.

## Công nghệ
- FastAPI + Uvicorn (prefix `/api`)
- SQLAlchemy 2.0 + PostgreSQL (psycopg2)
- Alembic (migrations)
- JWT (python-jose) + bcrypt (hash mật khẩu)
- MQTT (paho-mqtt) — ingest event/telemetry từ Edge qua RabbitMQ MQTT plugin (port 1883)
- MinIO (lưu snapshot) + MediaMTX (HLS live view)

## Cài đặt & chạy

```bash
cd "Khánh (Backend)/fall_detection_cloud"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Cấu hình biến môi trường
cp .env.example .env   # chỉnh DATABASE_URL, SECRET_KEY

# Khởi tạo schema
alembic upgrade head

# Nạp dữ liệu mẫu (idempotent)
python seed.py

# Chạy server
uvicorn app.main:app --reload
```

Tài liệu API tương tác: `http://localhost:8000/docs`

## Tài khoản seed
| Username | Mật khẩu | Email | Role |
|----------|----------|-------|------|
| admin | admin123 | admin@example.com | admin |
| caregiver | caregiver123 | caregiver@example.com | caregiver |

## Cấu hình (.env)
| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `DATABASE_URL` | postgresql://postgres:postgres@localhost:5432/falldetection | Chuỗi kết nối Postgres |
| `SECRET_KEY` | change-me-in-production | Khóa ký JWT |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | 60 | Hạn access token |
| `REFRESH_TOKEN_EXPIRE_DAYS` | 7 | Hạn refresh token |
| `MQTT_HOST` / `MQTT_PORT` | localhost / 1883 | Broker MQTT (RabbitMQ MQTT plugin) |
| `MQTT_USERNAME` / `MQTT_PASSWORD` | guest / guest | Thông tin đăng nhập MQTT |
| `MQTT_ENABLED` | true | Bật consumer MQTT (đặt `false` khi không có broker) |
| `MQTT_TOPIC_FALL` | events/+/fall | Topic subscribe sự kiện ngã |
| `MQTT_TOPIC_TELEMETRY` | telemetry/+/status | Topic subscribe telemetry |
| `MINIO_ENDPOINT` / `MINIO_*` | (trống) | Lưu snapshot; trống = bỏ qua upload |
| `MEDIAMTX_BASE_URL` | http://localhost:8888 | Base URL HLS cho live view |
| `TELEMETRY_ONLINE_WINDOW_SEC` | 60 | Cửa sổ coi camera là online |

## API contract (prefix `/api`)

> Tuân theo `Dũng (Edge)/docs/api_contract_v2.md` + `mqtt_schema_v2.json`. Điểm chính của v2:
> - **Bỏ `severity` high/low** — mọi event đều qua VLM verify (không phân luồng theo confidence).
> - Camera định danh bằng **`cam_id`** chuỗi (`cam_01`); khóa chính nội bộ là `id` số nguyên.
> - Ingest event/telemetry chủ yếu **qua MQTT**; REST `POST /events` chỉ là fallback.
> - Error format chuẩn: `{"error": {"code", "message", "details"}}`.
> - `acknowledged_by` trả **username** người dùng.
> - Đăng nhập/đăng ký bằng **`username`** (cột riêng trong bảng `users`, `email` vẫn unique).
> - DB v2: khóa chính số nguyên cho `users/cameras/events/alerts`; `events.event_id` là UUID v4 do Edge tạo; bảng FCM tên `device_tokens`.


### Auth
| Method | Path | Auth | Mô tả |
|--------|------|------|-------|
| POST | `/auth/register` | - | Đăng ký tài khoản |
| POST | `/auth/login` | - | Đăng nhập, trả access + refresh token |
| POST | `/auth/refresh` | - | Làm mới access token từ refresh token |

### Users
| Method | Path | Auth | Mô tả |
|--------|------|------|-------|
| GET | `/users/me` | Bearer | Thông tin người dùng hiện tại |
| PUT | `/users/me` | Bearer | Cập nhật thông tin |
| PUT | `/users/me/fcm-token` | Bearer | Đăng ký/cập nhật FCM token theo thiết bị |
| GET | `/users/me/devices` | Bearer | Danh sách thiết bị (FCM token) |
| DELETE | `/users/me/devices/{device_id}` | Bearer | Xóa thiết bị |

### Devices
| Method | Path | Auth | Mô tả |
|--------|------|------|-------|
| POST | `/devices/fcm-token` | Bearer | Đăng ký FCM token theo thiết bị (user lấy từ JWT) |

### Cameras
| Method | Path | Auth | Mô tả |
|--------|------|------|-------|
| GET | `/cameras` | Bearer | Danh sách camera (gồm `id` UUID + `cam_id`) |
| POST | `/cameras` | Bearer | Tạo camera (cần `cam_id`, `name`, `rtsp_url`) |
| GET | `/cameras/{cam_id}` | Bearer | Chi tiết camera theo `cam_id` |
| PUT | `/cameras/{cam_id}` | Bearer | Cập nhật (`name`, `rtsp_url`, `is_active`...) |
| DELETE | `/cameras/{cam_id}` | Bearer | Xóa camera |

### Config (rules toàn hệ thống)
| Method | Path | Auth | Mô tả |
|--------|------|------|-------|
| GET | `/config/rules` | Bearer | Cấu hình quy tắc (fall_detection, dedup, vlm, notifications) |
| PUT | `/config/rules` | Bearer | Cập nhật cấu hình quy tắc |

### Events
| Method | Path | Auth | Mô tả |
|--------|------|------|-------|
| POST | `/events` | Bearer | REST fallback nhận fall_event (mqtt_schema_v2) |
| GET | `/events` | Bearer | Lịch sử sự kiện (phân trang + filter) cho Mobile |
| GET | `/events/{event_id}` | Bearer | Chi tiết 1 sự kiện (nested `detection`/`vlm_result`) |

**`GET /events` query params:** `cam_id`, `status` (pending/confirmed/false_positive),
`start_date`/`end_date` (date), `page` (mặc định 1), `page_size` (mặc định 20).
Trả về `{items, page, page_size, total}`.

> Event nhận đúng payload `mqtt_schema_v2`: nested `detection{}` (class_before,
> final_class, confidence, bbox_xyxy), `rule{}`, `frames[]` (6 ảnh base64 JPEG,
> offset_ms -2500..0). Server decode frame trigger (offset_ms=0), upload MinIO,
> rồi chạy trigger chain. `event_id` do Edge tạo (UUID v4).

### Alerts
| Method | Path | Auth | Mô tả |
|--------|------|------|-------|
| GET | `/alerts` | Bearer | Lịch sử cảnh báo (filter `cam_id`, `acknowledged`, phân trang) |
| GET | `/alerts/{alert_id}` | Bearer | Chi tiết 1 cảnh báo |
| PATCH | `/alerts/{alert_id}/acknowledge` | Bearer | Xác nhận (body rỗng, user lấy từ JWT) |

`PATCH /alerts/{id}/acknowledge` **body rỗng** — set `is_acknowledged=TRUE`,
`acknowledged_by` = người dùng hiện tại (từ JWT), `acknowledged_at` = thời điểm.
AlertOut gồm `fcm_sent` (bool) và `vlm_reason` (giải thích của VLM).

### Telemetry
| Method | Path | Auth | Mô tả |
|--------|------|------|-------|
| GET | `/telemetry/{cam_id}/latest` | Bearer | Telemetry mới nhất của camera (pipeline/system/network + `is_online`) |

> v2 **bỏ `POST /telemetry`** — telemetry chỉ vào qua MQTT consumer.

### Live view
| Method | Path | Auth | Mô tả |
|--------|------|------|-------|
| GET | `/live/{cam_id}` | Bearer | Trả URL HLS (MediaMTX): `{MEDIAMTX_BASE_URL}/{cam_id}/index.m3u8` |

## VLM Adapter & Notification Service (Trigger Chain)

Khi nhận `POST /events`, server chạy **trigger chain ở background**
(`BackgroundTasks`) — không chặn phản hồi cho Edge:

```
POST /events
  -> lưu event + publish RabbitMQ
  -> [background] camera_rules.enable_vlm_verify == TRUE ?
        -> VLM (Gemini Flash) xác minh ảnh -> ghi events.vlm_verdict / vlm_confidence
        -> verdict == not_fall ? -> bỏ qua (giảm báo động giả), event.status=false_positive
  -> tạo alert
  -> gửi FCM (multicast tới device_tokens.fcm_token) + Telegram (kèm ảnh)
```

**Fail-safe / dry-run:** mỗi kênh tự chạy dry-run (chỉ log) nếu chưa cấu hình,
nên luồng test được ngay cả khi chưa có credential. Cấu hình `.env`:

| Biến | Mô tả |
|------|-------|
| `GEMINI_API_KEY` | Key Google AI Studio; trống = VLM dry-run |
| `GEMINI_MODEL` | Mặc định `gemini-2.0-flash` |
| `FCM_CREDENTIALS_FILE` | Đường dẫn service account JSON (Firebase); trống = FCM dry-run |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Bot token + chat đích; trống = Telegram dry-run |
| `VLM_ENABLED` / `FCM_ENABLED` / `TELEGRAM_ENABLED` | Bật/tắt từng kênh |

- **VLM** (`app/services/vlm.py`): gọi Gemini Flash REST, prompt phân biệt
  "nằm do ngã" vs "nằm nghỉ", trả `{verdict, confidence}` + đo latency
  (cảnh báo nếu >2s).
- **FCM** (`app/services/notifications/fcm.py`): `firebase-admin` multicast.
- **Telegram** (`app/services/notifications/telegram.py`): `sendPhoto` kèm
  caption (camera, thời gian, độ tin cậy, link ảnh).
- **Pipeline** (`app/services/alert_pipeline.py`): điều phối toàn bộ chuỗi.

## MQTT (ingest từ Edge)

Edge (Jetson) publish lên RabbitMQ MQTT plugin (port 1883). Server chạy
consumer ở background (lifespan) subscribe 2 topic wildcard và lưu DB +
chạy trigger chain cho event:

| Topic | Schema | Xử lý |
|-------|--------|-------|
| `events/cam_{id}/fall` | fall_event (mqtt_schema_v2) | lưu event → VLM → alert → FCM/Telegram |
| `telemetry/cam_{id}/status` | telemetry (mqtt_schema_v2) | lưu telemetry + cập nhật heartbeat camera |

`cam_id` được trích từ topic. Consumer fail-safe: thiếu `paho-mqtt` hoặc
broker không tới được thì log cảnh báo, app vẫn chạy. Đặt `MQTT_ENABLED=false`
để tắt hẳn.

Bật MQTT plugin trên RabbitMQ:
```bash
docker exec fall_mq rabbitmq-plugins enable rabbitmq_mqtt
# hoặc dùng docker-compose.cloud.yml (đã mount enabled_plugins + mở port 1883)
```

## Kiểm thử (mock)
```bash
# 1) Trigger chain qua REST: login -> POST event -> đợi pipeline -> alert -> ack
python scripts/mock_trigger_chain.py [base_url]

# 2) Ingest qua MQTT: publish fall_event + telemetry như Edge thật
python scripts/mock_mqtt_publish.py [cam_id] [host] [port]
```

## Postman
Import `postman/fall_detection_cloud.postman_collection.json` (đã cập nhật v2,
prefix `/api`). Chạy **Auth → Login** trước để tự động lưu
`access_token`/`refresh_token`; các request được bảo vệ sẽ tự dùng Bearer token.
Biến `cam_id` mặc định `cam_01` (chạy `seed.py`).

## Cấu trúc thư mục
```
app/
  api/deps.py            # JWT guard (get_current_user / active user)
  api/v1/api.py          # gom router (prefix /api)
  api/v1/endpoints/      # auth, users, devices, cameras, config, events, alerts, telemetry, live
  core/config.py         # cấu hình (pydantic-settings)
  core/database.py       # engine, SessionLocal, Base, get_db
  core/storage.py        # MinIO upload snapshot (graceful fallback)
  core/security.py       # hash mật khẩu + JWT (access/refresh/decode)
  models/                # user, user_device, camera, camera_rule, event, alert, telemetry
  schemas/               # Pydantic models (nested detection/vlm/pipeline...)
  services/vlm.py        # VLM Adapter (Gemini Flash)
  services/event_ingest.py     # parse fall_event, decode frame, upload snapshot
  services/telemetry_ingest.py # parse telemetry, cập nhật heartbeat
  services/mqtt_consumer.py    # MQTT subscriber (paho) chạy nền
  services/alert_pipeline.py        # trigger chain: VLM -> alert -> notify
  services/notifications/fcm.py      # push FCM (firebase-admin)
  services/notifications/telegram.py # Telegram Bot (sendPhoto)
alembic/                 # migrations (gồm v2_contract_align)
scripts/                 # mock_trigger_chain.py, mock_mqtt_publish.py
db_schema_v2.sql         # schema đầy đủ cho cài mới
seed.py                  # dữ liệu mẫu (cam_01, cam_02)
```
