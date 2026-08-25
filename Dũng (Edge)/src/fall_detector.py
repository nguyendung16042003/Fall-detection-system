#!/usr/bin/env python3
"""Logic phát hiện ngã — THUẦN Python (không import DeepStream/GPU) nên test được
không cần camera.

Gồm 2 phần theo thiết kế Tuần 4:
  - State Buffer          : với mỗi (cam_id, person_id) giữ lịch sử N frame gần nhất
                            (class + confidence + timestamp).
  - Temporal Transition   : phát hiện chuyển tư thế stand/sit -> lie trong <= window_ms
    Detector                thì coi là "fall_candidate".

Mọi ngưỡng lấy từ configs/fall_rules.yaml (không hardcode).

Cách dùng trong pipeline (probe sau SGIE):
    det = FallDetector(load_rules())
    cand = det.update(cam_id, person_id, cls, conf, ts_ms, bbox, w, h)
    if cand:
        # dựng payload bằng event_builder + publish qua mqtt_publisher
"""

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import yaml

_CONFIG_DEFAULT = Path(__file__).resolve().parent.parent / "configs" / "fall_rules.yaml"


# ============================================================
# Cấu hình luật (immutable)
# ============================================================
@dataclass(frozen=True)
class FallRules:
    version: str
    window_ms: int
    confirm_frames: int
    min_confidence: float
    cooldown_ms: int
    person_ttl_ms: int
    class_before_enum: Tuple[str, ...]
    fall_class: str
    # Port từ fall_rule.py (Handoff Tấn Dũng 2026-08-08) — xem CLAUDE.md "SGIE v2".
    min_lying_aspect_ratio: float
    min_sustained_lying_ms: int


def load_rules(path: Optional[str] = None) -> FallRules:
    """Đọc configs/fall_rules.yaml -> FallRules. Fail nhanh nếu thiếu khoá."""
    p = Path(path) if path else _CONFIG_DEFAULT
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data or "fall_rule" not in data:
        raise ValueError(f"fall_rules.yaml thiếu khoá 'fall_rule': {p}")
    r = data["fall_rule"]
    return FallRules(
        version=str(r["version"]),
        window_ms=int(r["window_ms"]),
        confirm_frames=int(r["confirm_frames"]),
        min_confidence=float(r["min_confidence"]),
        cooldown_ms=int(r["cooldown_ms"]),
        person_ttl_ms=int(r["person_ttl_ms"]),
        class_before_enum=tuple(r["class_before_enum"]),
        fall_class=str(r["fall_class"]),
        min_lying_aspect_ratio=float(r["min_lying_aspect_ratio"]),
        min_sustained_lying_ms=int(r["min_sustained_lying_ms"]),
    )


# ============================================================
# Kết quả phát hiện ngã (immutable)
# ============================================================
@dataclass(frozen=True)
class FallCandidate:
    cam_id: str
    person_id: int
    class_before: str          # tư thế trước ngã: stand | sit | unknown (sustained_lying)
    final_class: str           # tư thế lúc trigger: lie
    confidence: float          # confidence SGIE của frame lie xác nhận
    trigger: str               # stand_to_lie | sit_to_lie | sustained_lying
    transition_ms: int         # stand/sit->lie: thời gian chuyển. sustained_lying: số ms đã nằm liên tục
    window_ms: int             # cửa sổ luật (để đưa vào payload)
    ts_ms: int                 # thời điểm trigger (ms)
    bbox_xyxy: Tuple[int, int, int, int]
    frame_width: int
    frame_height: int


# ============================================================
# State per (cam_id, person_id)
# ============================================================
@dataclass
class _PersonState:
    samples: Deque[Tuple[int, str, float]] = field(default_factory=deque)  # (ts, cls, conf)
    last_alert_ts: Optional[int] = None
    alerted_latch: bool = False   # True trong khi vẫn nằm ở episode đã cảnh báo
    last_seen_ts: int = 0


# ============================================================
# Detector
# ============================================================
class FallDetector:
    def __init__(self, rules: FallRules):
        self.rules = rules
        # Giữ đủ lịch sử để tìm class_before trong window + margin an toàn.
        self._history_ms = rules.window_ms + 5000
        self._states: Dict[Tuple[str, int], _PersonState] = {}

    def update(
        self,
        cam_id: str,
        person_id: int,
        cls: str,
        conf: float,
        ts_ms: int,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        frame_width: int = 0,
        frame_height: int = 0,
    ) -> Optional[FallCandidate]:
        """Nạp 1 quan sát (1 person tại 1 frame). Trả FallCandidate nếu vừa phát hiện
        ngã, ngược lại None."""
        key = (cam_id, person_id)
        st = self._states.get(key)

        # TTL: mất dấu quá lâu -> coi như người mới, reset buffer.
        if st is None or (ts_ms - st.last_seen_ts) > self.rules.person_ttl_ms:
            st = _PersonState()
            self._states[key] = st

        st.last_seen_ts = ts_ms

        # Lọc aspect-ratio TRƯỚC KHI ghi nhận sample 'lie' (port từ fall_rule.py,
        # Handoff Tấn Dũng 2026-08-08): bbox chưa đủ "dẹt ngang" (width/height <
        # ngưỡng) nhiều khả năng là "ngồi ngả lưng" bị SGIE nhầm thành 'lie', không
        # phải nằm thật. Bỏ HẲN sample này (không append, không đổi latch/streak) —
        # khác các nhãn khác luôn được ghi nhận bình thường.
        if cls == self.rules.fall_class and bbox is not None:
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            aspect = (w / h) if h > 0 else 0.0
            if aspect < self.rules.min_lying_aspect_ratio:
                return None

        st.samples.append((ts_ms, cls, conf))

        # Loại bỏ sample quá cũ.
        cutoff = ts_ms - self._history_ms
        while st.samples and st.samples[0][0] < cutoff:
            st.samples.popleft()

        # Không còn nằm -> mở lại latch để lần ngã sau được phép trigger.
        if cls != self.rules.fall_class:
            st.alerted_latch = False
            return None

        # cls == fall_class ('lie'): kiểm tra đã xác nhận nằm chưa.
        if not self._is_confirmed_lie(st.samples):
            return None
        if st.alerted_latch:
            return None  # đã cảnh báo cho episode nằm này rồi
        if st.last_alert_ts is not None and (ts_ms - st.last_alert_ts) < self.rules.cooldown_ms:
            return None  # đang trong cooldown

        lie_start_ts = self._lie_run_start(st.samples)
        before = self._latest_before(st.samples, lie_start_ts)

        # Tiêu chí 1 (OR, ưu tiên): có mỏ neo stand/sit hợp lệ, chuyển đủ nhanh.
        transition_ms: Optional[int] = None
        before_cls: Optional[str] = None
        if before is not None:
            before_ts, before_cls, _before_conf = before
            candidate_transition = lie_start_ts - before_ts
            if candidate_transition <= self.rules.window_ms:
                transition_ms = int(candidate_transition)

        if transition_ms is not None:
            trigger = f"{before_cls}_to_{self.rules.fall_class}"
        else:
            # Tiêu chí 2 (dự phòng, port từ fall_rule.py "sustained_lying"): không
            # có mỏ neo hợp lệ (nằm sẵn từ đầu, track mới xuất hiện đã nằm, hoặc
            # chuyển quá chậm) -> vẫn báo nếu nằm LIÊN TỤC đủ lâu, bất kể trước đó
            # là gì. class_before="unknown" khi không xác định được.
            sustained_ms = ts_ms - lie_start_ts
            if sustained_ms < self.rules.min_sustained_lying_ms:
                return None
            trigger = "sustained_lying"
            transition_ms = int(sustained_ms)

        # --- TRIGGER ---
        st.last_alert_ts = ts_ms
        st.alerted_latch = True
        return FallCandidate(
            cam_id=cam_id,
            person_id=person_id,
            class_before=before_cls or "unknown",
            final_class=self.rules.fall_class,
            confidence=conf,
            trigger=trigger,
            transition_ms=transition_ms,
            window_ms=self.rules.window_ms,
            ts_ms=int(ts_ms),
            bbox_xyxy=tuple(bbox) if bbox else (0, 0, 0, 0),
            frame_width=int(frame_width),
            frame_height=int(frame_height),
        )

    def evict_stale(self, now_ms: int) -> None:
        """Dọn buffer các person đã mất dấu quá TTL (gọi định kỳ, tuỳ chọn)."""
        dead = [k for k, s in self._states.items()
                if (now_ms - s.last_seen_ts) > self.rules.person_ttl_ms]
        for k in dead:
            del self._states[k]

    def active_track_count(self, cam_id: str) -> int:
        """Số person đang được theo dõi cho 1 cam (dùng cho telemetry)."""
        return sum(1 for (c, _pid) in self._states if c == cam_id)

    # ---------- helpers ----------
    def _is_confirmed_lie(self, samples: Deque[Tuple[int, str, float]]) -> bool:
        n = self.rules.confirm_frames
        if len(samples) < n:
            return False
        tail = list(samples)[-n:]
        return all(
            c == self.rules.fall_class and conf >= self.rules.min_confidence
            for (_ts, c, conf) in tail
        )

    def _lie_run_start(self, samples: Deque[Tuple[int, str, float]]) -> int:
        """ts của frame 'lie' đầu tiên trong chuỗi nằm liên tục ở đuôi buffer."""
        start_ts = samples[-1][0]
        for ts, c, _conf in reversed(samples):
            if c == self.rules.fall_class:
                start_ts = ts
            else:
                break
        return start_ts

    def _latest_before(
        self, samples: Deque[Tuple[int, str, float]], lie_start_ts: int
    ) -> Optional[Tuple[int, str, float]]:
        """Sample stand/sit gần nhất TRƯỚC khi bắt đầu nằm."""
        best: Optional[Tuple[int, str, float]] = None
        for ts, c, conf in samples:
            if ts < lie_start_ts and c in self.rules.class_before_enum:
                if best is None or ts > best[0]:
                    best = (ts, c, conf)
        return best
