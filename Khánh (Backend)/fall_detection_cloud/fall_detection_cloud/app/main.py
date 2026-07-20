import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1.api import api_router
from app.core.config import settings
from app.core.database import SessionLocal
from app.services.mqtt_consumer import consumer
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start MQTT consumer in background thread (consumer.start() is blocking)
    def start_consumer():
        try:
            consumer.start()
            logger.info("MQTT consumer started successfully")
        except Exception as e:
            logger.error(f"MQTT consumer failed to start: {e}")

    thread = threading.Thread(target=start_consumer, daemon=True)
    thread.start()
    logger.info("MQTT consumer thread started")
    
    # Wait a bit for consumer to initialize
    await asyncio.sleep(1)
    
    yield
    
    consumer.stop()
    logger.info("MQTT consumer stopped")


app = FastAPI(title=settings.PROJECT_NAME, version="2.0.0", lifespan=lifespan)

app.include_router(api_router, prefix=settings.API_PREFIX)

# Mount static files for video clips
app.mount("/clips", StaticFiles(directory="/app/clips"), name="clips")


# ---- Error format chuẩn hóa (contract v2 mục 10) ----
_STATUS_CODE_NAMES = {
    400: "VALIDATION_ERROR",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    409: "CONFLICT",
    422: "VALIDATION_ERROR",
    500: "INTERNAL_ERROR",
}
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _error_body(code: str, message: object, details: object = None) -> dict:
    body: dict[str, object] = {"code": code, "message": message}
    if details is not None:
        body["details"] = details
    return {"error": body}


@app.exception_handler(StarletteHTTPException)
async def _http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    code = _STATUS_CODE_NAMES.get(exc.status_code, "INTERNAL_ERROR")
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_body(code, exc.detail),
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=_error_body(
            "VALIDATION_ERROR", "Dữ liệu không hợp lệ", exc.errors()
        ),
    )


@app.get("/", tags=["health"])
def root() -> dict[str, str]:
    return {"message": "Fall Detection System API is running"}


def _check_db() -> bool:
    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        return False
    finally:
        db.close()


def _check_rabbitmq() -> bool:
    try:
        import socket
        from urllib.parse import urlparse

        parsed = urlparse(settings.RABBITMQ_URL)
        host = parsed.hostname or "localhost"
        port = parsed.port or 5672
        with socket.create_connection((host, port), timeout=2):
            return True
    except Exception:  # noqa: BLE001
        return False


def _check_minio() -> bool:
    if not settings.MINIO_ENDPOINT:
        return False
    try:
        import socket

        host, _, port = settings.MINIO_ENDPOINT.partition(":")
        with socket.create_connection(
            (host, int(port or 9000)), timeout=2
        ):
            return True
    except Exception:  # noqa: BLE001
        return False


@app.get("/health", tags=["health"])
def health_check() -> dict:
    """Health check (contract v2 mục 9)."""
    db_ok = _check_db()
    services = {
        "database": "ok" if db_ok else "down",
        "rabbitmq": "ok" if _check_rabbitmq() else "down",
        "minio": "ok" if _check_minio() else "down",
    }
    return {
        "status": "ok" if db_ok else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": services,
    }
