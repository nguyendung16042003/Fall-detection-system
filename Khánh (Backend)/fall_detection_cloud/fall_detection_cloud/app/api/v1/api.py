from fastapi import APIRouter

from app.api.v1.endpoints import (
    auth,
    cameras,
    events,
    rules,
    telemetry,
    users,
)

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(cameras.router, prefix="/cameras", tags=["cameras"])
api_router.include_router(rules.router, prefix="/rules", tags=["rules"])
api_router.include_router(events.router, prefix="/events", tags=["events"])
api_router.include_router(
    telemetry.router, prefix="/telemetry", tags=["telemetry"]
)
