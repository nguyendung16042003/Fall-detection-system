"""Đệm ảnh gần đây theo camera — cấp 6 ảnh THẬT quanh thời điểm ngã, thay cho
PLACEHOLDER_JPEG_B64 (Tuần 5).

Thuần Python, không phụ thuộc pyds/GStreamer — add_frame() nhận jpeg bytes bất
kỳ nên test được không cần GPU/camera. Ghi từ probe thread capture (liên tục),
đọc từ probe thread SGIE lúc xác nhận ngã -> cần khóa.
"""

import base64
import threading
from collections import deque
from typing import Deque, Dict, List, Tuple

from event_builder import REQUIRED_OFFSETS

# Giữ dài hơn offset xa nhất (2500ms) một chút để luôn có frame khớp quanh mép.
MAX_AGE_MS = 4000


class RollingBuffer:
    """Đệm theo cam_id: mỗi cam giữ danh sách (ts_ms, jpeg_bytes) tăng dần theo
    thời gian, tự dọn frame cũ hơn max_age_ms mỗi lần add_frame()."""

    def __init__(self, max_age_ms: int = MAX_AGE_MS):
        self._max_age_ms = max_age_ms
        self._buffers: Dict[str, Deque[Tuple[int, bytes]]] = {}
        self._lock = threading.Lock()

    def add_frame(self, cam_id: str, ts_ms: int, jpeg_bytes: bytes) -> None:
        with self._lock:
            buf = self._buffers.setdefault(cam_id, deque())
            buf.append((ts_ms, jpeg_bytes))
            cutoff = ts_ms - self._max_age_ms
            while buf and buf[0][0] < cutoff:
                buf.popleft()

    def has_frames(self, cam_id: str) -> bool:
        with self._lock:
            return bool(self._buffers.get(cam_id))

    def get_frames_before(self, cam_id: str, event_ts_ms: int) -> List[Dict[str, object]]:
        """Đúng 6 frame theo REQUIRED_OFFSETS của event_builder (-2500..0ms).
        Mỗi phần tử chọn frame GẦN NHẤT thời điểm target = event_ts_ms + offset_ms.
        Buffer rỗng hoàn toàn (cam chưa có frame nào, vd mới khởi động) -> trả []
        để nơi gọi tự quyết định fallback (KHÔNG tự chế placeholder ở đây)."""
        with self._lock:
            buf = list(self._buffers.get(cam_id, ()))
        if not buf:
            return []
        result: List[Dict[str, object]] = []
        for offset_ms in REQUIRED_OFFSETS:
            target = event_ts_ms + offset_ms
            nearest_ts, nearest_jpeg = min(buf, key=lambda item: abs(item[0] - target))
            result.append({
                "offset_ms": offset_ms,
                "jpeg_b64": base64.b64encode(nearest_jpeg).decode("ascii"),
            })
        return result
