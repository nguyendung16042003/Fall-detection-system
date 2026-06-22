"""VLM Adapter — xác minh ngã bằng Gemini Flash với phân tích chuỗi ảnh.

Nhận chuỗi 6 frame JPEG (tương đương 3 giây trước thời điểm ngã) do Edge gửi,
hỏi Gemini phân tích yếu tố thời gian (temporal) để xác định người có đang
nằm do *ngã* hay chỉ *nằm nghỉ*, trả về verdict + confidence và đo latency.
Nếu chưa cấu hình `GEMINI_API_KEY` (hoặc tắt VLM), chạy **dry-run**:
trả verdict mô phỏng để luồng trigger chain vẫn test được.
"""

import base64
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Sequence

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# Verdict chuẩn hóa ghi vào events.vlm_verdict
VERDICT_FALL = "fall"
VERDICT_NOT_FALL = "not_fall"
VERDICT_UNCERTAIN = "uncertain"

PROMPT_TEMPLATE = (
    "Bạn là trợ lý an toàn y tế phân tích chuỗi 6 ảnh từ camera giám sát người cao "
    "tuổi. Mỗi ảnh là vùng crop của MỘT người được chụp cách nhau 500ms, "
    "tổng thời gian 2.5 giây từ frame đầu đến frame cuối. Frame cuối (offset_ms=0) "
    "là thời điểm phát hiện ngã.\n\n"
    "Nhiệm vụ: phân tích yếu tố thời gian (temporal) để xác định người đó có "
    "đang nằm trên sàn DO BỊ NGÃ hay không, để phân biệt với tư thế nằm nghỉ "
    "bình thường (giường/sofa) hoặc ngồi/đứng.\n\n"
    "Quy tắc phán đoán:\n"
    "- 'fall': người chuyển động từ đứng/ngồi sang nằm trên sàn một cách đột ngột, "
    "  tư thế bất thường, có dấu hiệu té ngã trong chuỗi ảnh.\n"
    "- 'not_fall': đứng, ngồi, hoặc nằm nghỉ bình thường trên giường/sofa, "
    "  không có chuyển động đột ngột sang tư thế nằm.\n"
    "- 'uncertain': ảnh mờ, bị che khuất, hoặc không đủ thông tin để xác định.\n\n"
    "CHỈ trả về JSON đúng định dạng:\n"
    '{"verdict": "fall|not_fall|uncertain", "confidence": <số 0..1>, '
    '"reason": "<giải thích ngắn về phân tích temporal>"}'
)


@dataclass
class VLMResult:
    verdict: str
    confidence: float
    latency_ms: float
    reason: str = ""
    dry_run: bool = False
    error: str | None = None
    raw: dict = field(default_factory=dict)


def _is_configured() -> bool:
    return bool(settings.VLM_ENABLED and settings.GEMINI_API_KEY)


def _resolve_image_bytes(
    image_bytes: bytes | None, image_ref: str | None
) -> bytes | None:
    """Lấy bytes ảnh: ưu tiên bytes truyền vào, sau đó tải từ URL http(s)."""
    if image_bytes:
        return image_bytes
    if image_ref and image_ref.startswith(("http://", "https://")):
        try:
            resp = httpx.get(image_ref, timeout=settings.VLM_TIMEOUT_SEC)
            resp.raise_for_status()
            return resp.content
        except httpx.HTTPError as exc:
            logger.warning("Không tải được ảnh VLM từ %s: %s", image_ref, exc)
    return None


def _resolve_frame_sequence(
    frames_bytes: Sequence[bytes] | None,
    frames_refs: Sequence[str] | None,
) -> list[bytes] | None:
    """Lấy bytes chuỗi 6 frame: ưu tiên bytes truyền vào, sau đó tải từ URL."""
    if frames_bytes and len(frames_bytes) == 6:
        return list(frames_bytes)
    if frames_refs and len(frames_refs) == 6:
        resolved = []
        for ref in frames_refs:
            if ref.startswith(("http://", "https://")):
                try:
                    resp = httpx.get(ref, timeout=settings.VLM_TIMEOUT_SEC)
                    resp.raise_for_status()
                    resolved.append(resp.content)
                except httpx.HTTPError as exc:
                    logger.warning("Không tải được frame VLM từ %s: %s", ref, exc)
                    return None
            else:
                logger.warning("Frame ref không phải URL: %s", ref)
                return None
        return resolved if len(resolved) == 6 else None
    return None


def _parse_gemini_json(data: dict) -> tuple[str, float, str]:
    """Bóc verdict/confidence/reason từ response Gemini."""
    text = (
        data["candidates"][0]["content"]["parts"][0]["text"]
        if data.get("candidates")
        else "{}"
    )
    parsed = json.loads(text)
    verdict = str(parsed.get("verdict", VERDICT_UNCERTAIN)).lower()
    if verdict not in (VERDICT_FALL, VERDICT_NOT_FALL, VERDICT_UNCERTAIN):
        verdict = VERDICT_UNCERTAIN
    confidence = float(parsed.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))
    return verdict, confidence, str(parsed.get("reason", ""))


def verify_fall(
    image_bytes: bytes | None = None,
    image_ref: str | None = None,
    frames_bytes: Sequence[bytes] | None = None,
    frames_refs: Sequence[str] | None = None,
    mime_type: str = "image/jpeg",
) -> VLMResult:
    """Gọi Gemini Flash để xác minh ngã với chuỗi 6 frame. Luôn trả VLMResult (không raise)."""
    start = time.perf_counter()

    if not _is_configured():
        # Dry-run: mô phỏng verdict để test trigger chain khi chưa có key.
        latency = (time.perf_counter() - start) * 1000
        logger.info("VLM dry-run (chưa cấu hình GEMINI_API_KEY)")
        return VLMResult(
            verdict=VERDICT_FALL,
            confidence=0.75,
            latency_ms=latency,
            reason="dry-run: giả lập xác minh ngã",
            dry_run=True,
        )

    # Ưu tiên xử lý chuỗi 6 frame (temporal analysis)
    frame_sequence = _resolve_frame_sequence(frames_bytes, frames_refs)
    if frame_sequence:
        return _verify_with_frames(frame_sequence, start, mime_type)

    # Fallback: xử lý single image (backward compatibility)
    raw_bytes = _resolve_image_bytes(image_bytes, image_ref)
    if not raw_bytes:
        latency = (time.perf_counter() - start) * 1000
        return VLMResult(
            verdict=VERDICT_UNCERTAIN,
            confidence=0.0,
            latency_ms=latency,
            error="không có ảnh/chuỗi frame để phân tích",
        )

    return _verify_with_single_image(raw_bytes, start, mime_type)


def _verify_with_frames(
    frames: list[bytes], start: float, mime_type: str
) -> VLMResult:
    """Gọi Gemini Flash với chuỗi 6 frame để phân tích temporal."""
    url = (
        f"{settings.GEMINI_BASE_URL}/models/"
        f"{settings.GEMINI_MODEL}:generateContent"
    )
    
    # Build parts with prompt + 6 frames
    parts = [{"text": PROMPT_TEMPLATE}]
    for frame in frames:
        parts.append({
            "inline_data": {
                "mime_type": mime_type,
                "data": base64.b64encode(frame).decode("ascii"),
            }
        })
    
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.0,
        },
    }

    try:
        resp = httpx.post(
            url,
            params={"key": settings.GEMINI_API_KEY},
            json=payload,
            timeout=settings.VLM_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        data = resp.json()
        verdict, confidence, reason = _parse_gemini_json(data)
        latency = (time.perf_counter() - start) * 1000
        if latency > 3000:
            logger.warning("VLM latency %.0fms vượt mục tiêu 3s", latency)
        logger.info("VLM verify với 6 frame: verdict=%s, confidence=%.2f, latency=%.0fms", 
                    verdict, confidence, latency)
        return VLMResult(
            verdict=verdict,
            confidence=confidence,
            latency_ms=latency,
            reason=reason,
            raw=data,
        )
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        latency = (time.perf_counter() - start) * 1000
        logger.warning("Gọi Gemini với chuỗi frame thất bại: %s", exc)
        return VLMResult(
            verdict=VERDICT_UNCERTAIN,
            confidence=0.0,
            latency_ms=latency,
            error=str(exc),
        )


def _verify_with_single_image(
    image_bytes: bytes, start: float, mime_type: str
) -> VLMResult:
    """Gọi Gemini Flash với single image (fallback)."""
    url = (
        f"{settings.GEMINI_BASE_URL}/models/"
        f"{settings.GEMINI_MODEL}:generateContent"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": PROMPT_TEMPLATE},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(image_bytes).decode("ascii"),
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.0,
        },
    }

    try:
        resp = httpx.post(
            url,
            params={"key": settings.GEMINI_API_KEY},
            json=payload,
            timeout=settings.VLM_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        data = resp.json()
        verdict, confidence, reason = _parse_gemini_json(data)
        latency = (time.perf_counter() - start) * 1000
        if latency > 3000:
            logger.warning("VLM latency %.0fms vượt mục tiêu 3s", latency)
        logger.info("VLM verify với single image: verdict=%s, confidence=%.2f, latency=%.0fms", 
                    verdict, confidence, latency)
        return VLMResult(
            verdict=verdict,
            confidence=confidence,
            latency_ms=latency,
            reason=reason,
            raw=data,
        )
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        latency = (time.perf_counter() - start) * 1000
        logger.warning("Gọi Gemini thất bại: %s", exc)
        return VLMResult(
            verdict=VERDICT_UNCERTAIN,
            confidence=0.0,
            latency_ms=latency,
            error=str(exc),
        )
