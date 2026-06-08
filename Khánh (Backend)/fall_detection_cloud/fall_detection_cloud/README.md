# Fall Detection Cloud API (Backend)

Backend FastAPI cho hệ thống phát hiện té ngã: Auth (JWT), quản lý Camera,
quản lý User và FCM token, Alembic migrations và dữ liệu seed.

## Công nghệ
- FastAPI + Uvicorn
- SQLAlchemy 2.0 + PostgreSQL (psycopg2)
- Alembic (migrations)
- JWT (python-jose) + bcrypt (hash mật khẩu)

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
| Email | Mật khẩu | Role |
|-------|----------|------|
| admin@example.com | admin123 | admin |
| caregiver@example.com | caregiver123 | caregiver |

## Cấu hình (.env)
| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `DATABASE_URL` | postgresql://postgres:postgres@localhost:5432/falldetection | Chuỗi kết nối Postgres |
| `SECRET_KEY` | change-me-in-production | Khóa ký JWT |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | 60 | Hạn access token |
| `REFRESH_TOKEN_EXPIRE_DAYS` | 7 | Hạn refresh token |

## API contract (prefix `/api/v1`)

> Tuân theo `Dũng (Edge)/docs/api_contract.md`. Một số điểm khớp contract:
> - `POST /auth/login` trả thêm object `user: {id, name, email, role}`.
> - `/rules/{camera_id}` dùng field `enabled` (lưu ở cột DB `is_active`).
> - `PUT /cameras/{id}` chấp nhận `enabled` (true→`status=online`, false→`offline`); `GET /cameras` vẫn trả `status`.


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

### Cameras
| Method | Path | Auth | Mô tả |
|--------|------|------|-------|
| GET | `/cameras` | Bearer | Danh sách camera |
| POST | `/cameras` | Bearer | Tạo camera |
| GET | `/cameras/{id}` | Bearer | Chi tiết camera |
| PUT | `/cameras/{id}` | Bearer | Cập nhật camera (gồm `status`, `rtsp_url`) |
| DELETE | `/cameras/{id}` | Bearer | Xóa camera |

### Rules
| Method | Path | Auth | Mô tả |
|--------|------|------|-------|
| GET | `/rules/{camera_id}` | Bearer | Cấu hình quy tắc của camera |
| PUT | `/rules/{camera_id}` | Bearer | Tạo/cập nhật quy tắc |

## Postman
Import `postman/fall_detection_cloud.postman_collection.json`. Chạy
**Auth → Login** trước để tự động lưu `access_token`/`refresh_token`; các request
được bảo vệ sẽ tự dùng Bearer token. **Create camera** lưu `camera_id` cho các
request liên quan.

## Cấu trúc thư mục
```
app/
  api/deps.py            # JWT guard (get_current_user / active user)
  api/v1/api.py          # gom router v1
  api/v1/endpoints/      # auth.py, users.py, cameras.py, rules.py
  core/config.py         # cấu hình (pydantic-settings)
  core/database.py       # engine, SessionLocal, Base, get_db
  core/security.py       # hash mật khẩu + JWT (access/refresh/decode)
  models/                # user, user_device, camera, camera_rule, event, alert, telemetry
  schemas/               # Pydantic models
alembic/                 # migrations
seed.py                  # dữ liệu mẫu
```
