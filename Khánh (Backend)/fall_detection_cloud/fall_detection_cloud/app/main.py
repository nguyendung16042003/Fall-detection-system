import logging

from fastapi import FastAPI

from app.api.v1.api import api_router
from app.core.config import settings
from app.core.messaging import declare_topology

logger = logging.getLogger(__name__)

app = FastAPI(title=settings.PROJECT_NAME, version="1.0.0")

app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.on_event("startup")
def _setup_messaging() -> None:
    """Khai báo exchange/queue RabbitMQ khi khởi động (không chặn nếu lỗi)."""
    if not settings.RABBITMQ_ENABLED:
        return
    try:
        declare_topology()
        logger.info("RabbitMQ topology đã sẵn sàng")
    except Exception as exc:  # noqa: BLE001 - không để chặn app khởi động
        logger.warning("Bỏ qua khai báo RabbitMQ topology: %s", exc)


@app.get("/", tags=["health"])
def root() -> dict[str, str]:
    return {"message": "Fall Detection System API is running"}


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    return {"status": "healthy"}
