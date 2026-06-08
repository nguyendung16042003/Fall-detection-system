from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_active_user
from app.core.database import get_db
from app.core.messaging import RoutingKey, publish
from app.models.camera import Camera
from app.models.event import Event
from app.models.user import User
from app.schemas.event import (
    EventCreate,
    EventCreateResponse,
    EventDetail,
    EventListItem,
    EventListResponse,
)

router = APIRouter()


def _routing_key_for(severity: str) -> str:
    """high -> fall_events.high; còn lại -> fall_events.low."""
    if (severity or "").lower() == "high":
        return RoutingKey.FALL_HIGH.value
    return RoutingKey.FALL_LOW.value


def _to_detail(event: Event) -> EventDetail:
    return EventDetail(
        event_id=event.id,
        event_type=event.event_type,
        severity=event.severity,
        camera_id=event.camera_id,
        camera_name=event.camera.name if event.camera else None,
        person_id=event.person_id,
        timestamp=event.timestamp,
        bbox=event.bbox_json,
        state_before=event.state_before,
        state_after=event.state_after,
        transition_time_ms=event.transition_time_ms,
        classification_confidence=event.classification_confidence,
        fall_confidence=event.fall_confidence,
        vlm_verdict=event.vlm_verdict,
        vlm_confidence=event.vlm_confidence,
        snapshot_url=event.image_url,
        clip_url=event.video_clip_url,
        status=event.status,
    )


@router.post(
    "",
    response_model=EventCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def ingest_event(
    payload: EventCreate,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),
) -> EventCreateResponse:
    """Nhận sự kiện ngã từ Edge: lưu DB + publish RabbitMQ theo severity."""
    camera = db.query(Camera).filter(Camera.id == payload.camera_id).first()
    if camera is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera không tồn tại",
        )

    event = Event(
        camera_id=payload.camera_id,
        event_type=payload.class_label or payload.event_type,
        severity=payload.severity,
        person_id=payload.person_id,
        timestamp=payload.timestamp,
        bbox_json=payload.bbox.model_dump() if payload.bbox else None,
        state_before=payload.state_before,
        state_after=payload.state_after,
        transition_time_ms=payload.transition_time_ms,
        classification_confidence=payload.classification_confidence,
        fall_confidence=(
            payload.fall_confidence
            if payload.fall_confidence is not None
            else payload.confidence
        ),
        vlm_verdict=payload.vlm_verdict,
        vlm_confidence=payload.vlm_confidence,
        image_url=payload.snapshot_path,
        video_clip_url=payload.clip_path,
        status=payload.status,
    )
    db.add(event)
    db.commit()
    db.refresh(event)

    # Publish sang RabbitMQ (không chặn nếu broker lỗi)
    publish(
        _routing_key_for(event.severity),
        {
            "event_id": str(event.id),
            "edge_event_id": payload.event_id,
            "camera_id": str(event.camera_id),
            "edge_device_id": payload.edge_device_id,
            "event_type": event.event_type,
            "severity": event.severity,
            "person_id": event.person_id,
            "timestamp": event.timestamp,
            "fall_confidence": event.fall_confidence,
            "classification_confidence": event.classification_confidence,
            "bbox": event.bbox_json,
            "status": event.status,
        },
    )

    # Theo contract: echo lại event_id của Edge nếu có, không thì trả UUID server
    return EventCreateResponse(
        event_id=payload.event_id or str(event.id), status="received"
    )


@router.get("", response_model=EventListResponse)
def list_events(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),
    camera_id: UUID | None = Query(default=None),
    from_: datetime | None = Query(default=None, alias="from"),
    to: datetime | None = Query(default=None),
    severity: str | None = Query(default=None),
    status_: str | None = Query(default=None, alias="status"),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
) -> EventListResponse:
    """Lịch sử sự kiện cho Mobile: phân trang + filter."""
    query = db.query(Event)
    if camera_id is not None:
        query = query.filter(Event.camera_id == camera_id)
    if from_ is not None:
        query = query.filter(Event.timestamp >= from_)
    if to is not None:
        query = query.filter(Event.timestamp <= to)
    if severity is not None:
        query = query.filter(Event.severity == severity)
    if status_ is not None:
        query = query.filter(Event.status == status_)

    total = query.count()
    events = (
        query.order_by(Event.timestamp.desc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )

    items = [
        EventListItem(
            event_id=e.id,
            camera_id=e.camera_id,
            camera_name=e.camera.name if e.camera else None,
            timestamp=e.timestamp,
            severity=e.severity,
            status=e.status,
            fall_confidence=e.fall_confidence,
            snapshot_url=e.image_url,
        )
        for e in events
    ]
    return EventListResponse(
        items=items, page=page, limit=limit, total=total
    )


@router.get("/{event_id}", response_model=EventDetail)
def get_event(
    event_id: UUID,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),
) -> EventDetail:
    event = db.query(Event).filter(Event.id == event_id).first()
    if event is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sự kiện không tồn tại",
        )
    return _to_detail(event)
