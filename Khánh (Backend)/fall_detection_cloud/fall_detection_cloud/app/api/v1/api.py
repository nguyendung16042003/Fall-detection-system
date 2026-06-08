from fastapi import APIRouter

from app.api.v1.endpoints import auth, cameras, rules, users

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(cameras.router, prefix="/cameras", tags=["cameras"])
api_router.include_router(rules.router, prefix="/rules", tags=["rules"])
