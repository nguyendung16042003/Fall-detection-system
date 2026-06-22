"""Dedup Multi-camera Logic — gộp sự kiện từ nhiều camera.

Nếu nhận được 2 sự kiện ngã từ 2 camera khác nhau nhưng có timestamp cách nhau
dưới 2 giây, hệ thống phải gộp lại thành 1 sự kiện duy nhất. Tối ưu chi phí:
chỉ gọi VLM 1 lần và gửi 1 thông báo duy nhất cho người dùng đối với các sự
kiện đã gộp.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.event import Event
from app.models.camera import Camera

logger = logging.getLogger(__name__)

# Thời gian window để gộp sự kiện (2 giây)
DEDUP_WINDOW_SEC = 2.0


class EventDeduplicator:
    """Quản lý deduplication cho sự kiện multi-camera."""

    def __init__(self):
        # Cache các event đang chờ gộp: {timestamp_key: [event_ids]}
        self._pending_events: Dict[str, List[int]] = defaultdict(list)
        # Map event_id -> timestamp_key để lookup nhanh
        self._event_to_key: Dict[int, str] = {}
        # Set các event đã được gộp để tránh xử lý lại
        self._merged_events: Set[int] = set()
        # Map merged event_id -> list of original event_ids
        self._merge_groups: Dict[int, List[int]] = {}

    def _get_timestamp_key(self, timestamp: datetime) -> str:
        """Tạo key cho timestamp theo window 2 giây."""
        # Làm tròn timestamp xuống bội số của 2 giây
        epoch_seconds = timestamp.timestamp()
        windowed = int(epoch_seconds // DEDUP_WINDOW_SEC) * DEDUP_WINDOW_SEC
        return str(int(windowed))

    def add_event(self, event: Event) -> Optional[List[int]]:
        """
        Thêm event vào dedup system.
        
        Trả về:
        - None nếu event không được gộp (event duy nhất trong window)
        - List[int] các event_id được gộp nếu tìm thấy event khác trong window
        """
        if event.id in self._merged_events:
            logger.debug("Event %d đã được gộp trước đó", event.id)
            return None

        timestamp_key = self._get_timestamp_key(event.timestamp_utc)
        
        # Thêm vào pending
        self._pending_events[timestamp_key].append(event.id)
        self._event_to_key[event.id] = timestamp_key

        # Kiểm tra xem có event khác trong cùng window không
        events_in_window = self._pending_events[timestamp_key]
        
        if len(events_in_window) > 1:
            # Có nhiều event trong window -> gộp lại
            logger.info(
                "Phát hiện %d sự kiện trong window 2s: %s",
                len(events_in_window),
                events_in_window,
            )
            
            # Đánh dấu tất cả là đã gộp
            for eid in events_in_window:
                self._merged_events.add(eid)
            
            # Lưu group merge (dùng event đầu tiên làm representative)
            representative_id = events_in_window[0]
            self._merge_groups[representative_id] = events_in_window
            
            # Cleanup pending sau khi gộp
            del self._pending_events[timestamp_key]
            for eid in events_in_window:
                self._event_to_key.pop(eid, None)
            
            return events_in_window
        
        return None

    def get_merge_group(self, event_id: int) -> Optional[List[int]]:
        """
        Lấy danh sách event_id được gộp với event này.
        
        Trả về None nếu event không được gộp.
        """
        # Kiểm tra xem event có phải là representative không
        if event_id in self._merge_groups:
            return self._merge_groups[event_id]
        
        # Kiểm tra xem event có trong group nào không
        for rep_id, group in self._merge_groups.items():
            if event_id in group:
                return group
        
        return None

    def is_merged(self, event_id: int) -> bool:
        """Kiểm tra event đã được gộp chưa."""
        return event_id in self._merged_events

    def cleanup_old_events(self, before_timestamp: datetime) -> None:
        """
        Cleanup các event cũ trong pending để tránh memory leak.
        
        Gọi định kỳ (ví dụ mỗi 5 phút) để cleanup các event quá cũ.
        """
        cutoff_key = self._get_timestamp_key(before_timestamp)
        cutoff_key_int = int(cutoff_key)
        
        keys_to_remove = []
        for key_str in self._pending_events:
            key_int = int(key_str)
            if key_int < cutoff_key_int:
                keys_to_remove.append(key_str)
        
        for key_str in keys_to_remove:
            event_ids = self._pending_events.pop(key_str, [])
            for eid in event_ids:
                self._event_to_key.pop(eid, None)
            logger.debug("Cleanup %d events từ window %s", len(event_ids), key_str)


def find_nearby_events(
    db: Session, 
    event: Event, 
    window_sec: float = DEDUP_WINDOW_SEC
) -> List[Event]:
    """
    Tìm các sự kiện từ camera khác trong khoảng thời gian window_sec.
    
    Sử dụng database query để tìm các event từ camera khác có timestamp
    trong khoảng ±window_sec từ timestamp của event hiện tại.
    """
    start_time = event.timestamp_utc - timedelta(seconds=window_sec)
    end_time = event.timestamp_utc + timedelta(seconds=window_sec)
    
    # Query các event từ camera khác trong cùng time window
    nearby_events = (
        db.query(Event)
        .join(Camera)
        .filter(
            Event.camera_id != event.camera_id,
            Event.timestamp_utc >= start_time,
            Event.timestamp_utc <= end_time,
            Event.event_type == "fall_candidate",
            Event.status == "pending",
        )
        .all()
    )
    
    return nearby_events


def should_merge_events(event1: Event, event2: Event) -> bool:
    """
    Quyết định xem 2 event có nên được gộp không.
    
    Điều kiện gộp:
    - Từ 2 camera khác nhau
    - Timestamp cách nhau dưới 2 giây
    - Cùng type (fall_candidate)
    """
    if event1.camera_id == event2.camera_id:
        return False
    
    if event1.event_type != event2.event_type:
        return False
    
    time_diff = abs((event1.timestamp_utc - event2.timestamp_utc).total_seconds())
    if time_diff > DEDUP_WINDOW_SEC:
        return False
    
    return True


def merge_events_metadata(db: Session, primary_event: Event, merged_events: List[Event]) -> None:
    """
    Cập nhật metadata cho các event đã gộp.
    
    - Đánh dấu các event phụ là merged
    - Thêm reference đến primary event
    """
    for merged_event in merged_events:
        if merged_event.id != primary_event.id:
            # Lưu thông tin về event primary vào note hoặc metadata
            note = f"Merged with primary event {primary_event.event_id}"
            if merged_event.note:
                merged_event.note = f"{merged_event.note}; {note}"
            else:
                merged_event.note = note
            
            # Đánh dấu status để tránh xử lý lại
            merged_event.status = "merged"
            db.add(merged_event)
    
    # Cập nhật primary event với thông tin về các event đã gộp
    merged_ids = [str(e.event_id) for e in merged_events if e.id != primary_event.id]
    if merged_ids:
        note = f"Merged events: {', '.join(merged_ids)}"
        if primary_event.note:
            primary_event.note = f"{primary_event.note}; {note}"
        else:
            primary_event.note = note
        db.add(primary_event)
    
    db.commit()


# Global instance
_deduplicator: Optional[EventDeduplicator] = None


def get_deduplicator() -> EventDeduplicator:
    """Trả về global EventDeduplicator instance."""
    global _deduplicator
    if _deduplicator is None:
        _deduplicator = EventDeduplicator()
    return _deduplicator
