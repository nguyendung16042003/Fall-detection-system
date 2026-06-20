from fastapi import APIRouter

from app.api.v1.endpoints import (
    alerts,
    auth,
    cameras,
    config,
    devices,
    events,
    live,
    telemetry,
    users,
)

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(devices.router, prefix="/devices", tags=["devices"])
api_router.include_router(cameras.router, prefix="/cameras", tags=["cameras"])
api_router.include_router(config.router, prefix="/config", tags=["config"])
api_router.include_router(events.router, prefix="/events", tags=["events"])
api_router.include_router(alerts.router, prefix="/alerts", tags=["alerts"])
api_router.include_router(live.router, prefix="/live", tags=["live"])
api_router.include_router(
    telemetry.router, prefix="/telemetry", tags=["telemetry"]
)
