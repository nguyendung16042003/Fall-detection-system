from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Cấu hình ứng dụng, đọc từ biến môi trường hoặc file .env."""

    PROJECT_NAME: str = "Fall Detection Cloud API"
    # Contract v2: base path là /api (bỏ /v1)
    API_PREFIX: str = "/api"
    # Giữ tên cũ làm bí danh để không vỡ chỗ tham chiếu
    API_V1_PREFIX: str = "/api"

    # Database
    DATABASE_URL: str = (
        "postgresql://postgres:postgres@localhost:5432/falldetection"
    )

    # JWT
    SECRET_KEY: str = "change-me-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # RabbitMQ (AMQP) — vẫn dùng cho publish nội bộ tùy chọn
    RABBITMQ_URL: str = "amqp://guest:guest@localhost:5672/"
    RABBITMQ_EXCHANGE: str = "fall_detection"
    # Bật/tắt publish (tắt khi chạy test không có broker)
    RABBITMQ_ENABLED: bool = False

    # MQTT (RabbitMQ MQTT plugin) — Edge publish event/telemetry qua đây
    MQTT_HOST: str = "localhost"
    MQTT_PORT: int = 1883
    MQTT_USERNAME: str = "guest"
    MQTT_PASSWORD: str = "guest"
    MQTT_ENABLED: bool = True
    # Topic (contract v2): events/cam_{id}/fall, telemetry/cam_{id}/status
    MQTT_TOPIC_FALL: str = "events/+/fall"
    MQTT_TOPIC_TELEMETRY: str = "telemetry/+/status"

    # MinIO (lưu snapshot/clip) — để trống endpoint sẽ bỏ qua upload
    MINIO_ENDPOINT: str = ""
    MINIO_ACCESS_KEY: str = ""
    MINIO_SECRET_KEY: str = ""
    MINIO_BUCKET: str = "fall-events"
    MINIO_SECURE: bool = False
    MINIO_PUBLIC_URL: str = ""

    # Live view (HLS qua MediaMTX) — contract v2 mục 7
    MEDIAMTX_BASE_URL: str = "http://localhost:8888"

    # Telemetry: coi camera online nếu nhận tin trong vòng N giây
    TELEMETRY_ONLINE_WINDOW_SEC: int = 60

    # VLM (Gemini Flash) — để trống GEMINI_API_KEY sẽ chạy dry-run
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta"
    VLM_TIMEOUT_SEC: float = 8.0

    # FCM (Firebase Admin SDK) — đường dẫn tới service account JSON
    FCM_CREDENTIALS_FILE: str = ""

    # Telegram Bot — để trống token sẽ chạy dry-run
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    # Bật/tắt từng kênh (mặc định bật; tự fallback dry-run nếu thiếu cấu hình)
    VLM_ENABLED: bool = True
    FCM_ENABLED: bool = True
    TELEGRAM_ENABLED: bool = True

    # URL gốc để dựng link snapshot công khai (MinIO/S3) nếu cần
    PUBLIC_MEDIA_BASE_URL: str = ""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
