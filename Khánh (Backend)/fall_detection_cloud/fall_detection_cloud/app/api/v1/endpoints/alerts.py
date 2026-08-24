from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_active_user
from app.models.alert import Alert
from app.models.camera import Camera
from app.models.event import Event
from app.models.user import User
from app.core.database import get_db
from app.schemas.alert import AlertAckResponse, AlertListResponse, AlertOut

router = APIRouter()


def _to_out(alert: Alert, db: Session) -> AlertOut:
    event = alert.event
    cam_id = None
    if event is not None and event.camera is not None:
        cam_id = event.camera.cam_id
    ack_by = None
    if alert.acknowledged_by is not None:
        user = db.query(User).filter(User.id == alert.acknowledged_by).first()
        if user is not None:
            ack_by = user.username
    return AlertOut(
        id=alert.id,
        event_id=alert.event_id,
        cam_id=cam_id,
        timestamp_utc=event.timestamp_utc if event else None,
        acknowledged=alert.is_acknowledged,
        acknowledged_at=alert.acknowledged_at,
        acknowledged_by=ack_by,
        fcm_sent=alert.fcm_sent,
        vlm_reason=event.vlm_reason if event else None,
    )


@router.get("", response_model=AlertListResponse)
def list_alerts(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),
    cam_id: str | None = Query(default=None),
    acknowledged: bool | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> AlertListResponse:
    """Lịch sử cảnh báo cho Mobile (phân trang + filter)."""
    query = db.query(Alert)
    if acknowledged is not None:
        query = query.filter(Alert.is_acknowledged == acknowledged)
    if cam_id is not None:
        query = (
            query.join(Event, Alert.event_id == Event.event_id)
            .join(Camera, Event.camera_id == Camera.id)
            .filter(Camera.cam_id == cam_id)
        )

    total = query.count()
    alerts = (
        query.order_by(Alert.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return AlertListResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=[_to_out(a, db) for a in alerts],
    )


def _get_alert_or_404(alert_id: int, db: Session) -> Alert:
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if alert is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Cảnh báo không tồn tại",
        )
    return alert


@router.get("/{alert_id}", response_model=AlertOut)
def get_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),
) -> AlertOut:
    return _to_out(_get_alert_or_404(alert_id, db), db)


@router.patch("/{alert_id}/acknowledge", response_model=AlertAckResponse)
def acknowledge_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> AlertAckResponse:
    """Xác nhận đã xử lý cảnh báo (contract v2: body rỗng, user từ JWT)."""
    alert = _get_alert_or_404(alert_id, db)

    alert.is_acknowledged = True
    alert.acknowledged_by = current_user.id
    alert.acknowledged_at = datetime.now(timezone.utc)
    db.add(alert)
    db.commit()
    db.refresh(alert)

    return AlertAckResponse(
        id=alert.id,
        acknowledged=True,
        acknowledged_at=alert.acknowledged_at,
        acknowledged_by=current_user.username,
    )
