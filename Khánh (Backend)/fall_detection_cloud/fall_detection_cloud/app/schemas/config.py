from pydantic import BaseModel


class FallDetectionCfg(BaseModel):
    transition_window_ms: int | None = None
    lying_ignore_after_ms: int | None = None
    min_confidence_sgie: float | None = None
    confirm_frames: int | None = None


class DedupCfg(BaseModel):
    window_ms: int | None = None


class VLMCfg(BaseModel):
    enabled: bool | None = None
    model: str | None = None
    timeout_s: float | None = None


class NotificationsCfg(BaseModel):
    fcm_enabled: bool | None = None
    cameras: dict[str, dict] | None = None


class RulesConfig(BaseModel):
    """Cấu hình pipeline toàn cục (contract v2 mục 6)."""

    fall_detection: FallDetectionCfg = FallDetectionCfg()
    dedup: DedupCfg = DedupCfg()
    vlm: VLMCfg = VLMCfg()
    notifications: NotificationsCfg = NotificationsCfg()


class RulesConfigUpdate(BaseModel):
    fall_detection: FallDetectionCfg | None = None
    dedup: DedupCfg | None = None
    vlm: VLMCfg | None = None
    notifications: NotificationsCfg | None = None
