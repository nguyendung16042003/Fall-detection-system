from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class CameraRuleBase(BaseModel):
    time_window_sec: int = Field(2, ge=1, le=30)
    min_lying_frames: int = Field(5, ge=1, le=300)
    min_confidence_sgie: float = Field(0.6, ge=0.0, le=1.0)
    enable_vlm_verify: bool = True
    is_active: bool = True


class CameraRuleUpdate(BaseModel):
    time_window_sec: int | None = Field(default=None, ge=1, le=30)
    min_lying_frames: int | None = Field(default=None, ge=1, le=300)
    min_confidence_sgie: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    enable_vlm_verify: bool | None = None
    is_active: bool | None = None


class CameraRuleRead(CameraRuleBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    camera_id: int
    updated_at: datetime
