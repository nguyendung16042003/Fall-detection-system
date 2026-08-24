"""Health check endpoints for monitoring and orchestration.

Provides /health/live (liveness) and /health/ready (readiness) endpoints
for Kubernetes/Docker health checks and orchestration systems.
"""

from fastapi import APIRouter, Response
from sqlalchemy import text

from app.core.database import DB_MAX_OVERFLOW, engine
from app.core.logging_config import get_logger
from app.services.mqtt_consumer import consumer

router = APIRouter()
logger = get_logger(__name__)


@router.get("/live")
async def live() -> dict:
    """
    Liveness probe - checks if the process is running.
    
    This endpoint always returns 200 if the service is running.
    Used by orchestration systems to detect if the container needs restart.
    """
    return {"status": "ok"}


@router.get("/ready")
async def ready(response: Response) -> dict:
    """
    Readiness probe - checks if the service is ready to handle requests.
    
    Checks:
    - Database connection
    - MQTT broker connection
    - Database connection pool status
    
    Returns 503 if any critical dependency is unavailable.
    """
    checks = {}
    all_ok = True
    
    # Check database connection
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"fail: {str(e)}"
        all_ok = False
        logger.warning("health_check_postgres_failed", error=str(e))
    
    # Check MQTT connection
    try:
        mqtt_connected = consumer._client is not None
        checks["rabbitmq"] = "ok" if mqtt_connected else "disconnected"
        if not mqtt_connected:
            all_ok = False
            logger.warning("health_check_mqtt_disconnected")
    except Exception as e:
        checks["rabbitmq"] = f"fail: {str(e)}"
        all_ok = False
        logger.warning("health_check_mqtt_failed", error=str(e))
    
    # Check database pool status
    try:
        pool = engine.pool
        pool_size = pool.size()
        pool_overflow = pool.overflow()
        checks["pool_usage"] = f"{pool_size}+{pool_overflow}/{pool_size + DB_MAX_OVERFLOW}"

        # Warn if pool is saturated
        if pool_overflow >= DB_MAX_OVERFLOW:
            logger.warning("health_check_pool_saturated", pool_size=pool_size, pool_overflow=pool_overflow)
    except Exception as e:
        checks["pool_usage"] = f"fail: {str(e)}"
        logger.warning("health_check_pool_failed", error=str(e))
    
    # Set response status code
    response.status_code = 200 if all_ok else 503
    
    logger.info("health_check", status="ok" if all_ok else "not_ready", checks=checks)
    
    return checks
