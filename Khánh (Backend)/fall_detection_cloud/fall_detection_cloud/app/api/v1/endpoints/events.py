import logging
from datetime import date, datetime
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    Response,
    status,
)
from sqlalchemy.orm import Session

from app.api.deps import get_current_active_user
from app.core.database import SessionLocal, get_db
from app.core.storage import get_jpeg
from app.models.camera import Camera
from app.models.event import Event
from app.models.user import User
from app.schemas.event import (
    DetectionOut,
    EventCreate,
    EventCreateResponse,
    EventListResponse,
    EventOut,
    VLMResultOut,
)
from app.services.alert_pipeline import run_pipeline
from app.services.event_ingest import CameraNotFoundError, ingest_event

logger = logging.getLogger(__name__)

router = APIRouter()


def _run_pipeline_bg(event_pk: int) -> None:
    """Chạy trigger chain (VLM -> alert -> FCM/Telegram) ở background."""
    db = SessionLocal()
    try:
        event = db.query(Event).filter(Event.id == event_pk).first()
        if event is None:
            return
        result = run_pipeline(db, event)
        logger.info("Pipeline %s: %s", event_pk, result)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Pipeline lỗi cho event %s: %s", event_pk, exc)
    finally:
        db.close()


def to_event_out(event: Event) -> EventOut:
    bbox = event.bbox_json or {}
    bbox_xyxy = None
    if all(k in bbox for k in ("x1", "y1", "x2", "y2")):
        bbox_xyxy = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]

    vlm_result = None
    if event.vlm_verdict is not None:
        vlm_result = VLMResultOut(
            fall=event.vlm_verdict == "fall",
            confidence=event.vlm_confidence,
            reason=event.vlm_reason,
        )

    return EventOut(
        id=event.id,
        event_id=event.event_id,
        cam_id=event.camera.cam_id if event.camera else None,
        person_id=event.person_id,
        timestamp_utc=event.timestamp_utc,
        event_type=event.event_type,
        status=event.status,
        detection=DetectionOut(
            final_class=event.final_class,
            confidence=event.detection_confidence,
            bbox_xyxy=bbox_xyxy,
        ),
        vlm_result=vlm_result,
        clip_url=event.clip_url,
        snapshot_url=event.image_url,
        created_at=event.created_at,
    )


@router.post(
    "",
    response_model=EventCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_event(
    payload: EventCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),
) -> EventCreateResponse:
    """REST fallback nhận sự kiện ngã từ Edge (khi MQTT lỗi / bơm data test)."""
    if payload.timestamp_utc is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Thiếu timestamp_utc",
        )
    try:
        event, _frame, _frames = ingest_event(db, payload)
    except CameraNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera không tồn tại: {payload.cam_id}",
        ) from None

    # Trigger chain chạy nền: VLM verify -> alert -> FCM/Telegram
    background_tasks.add_task(_run_pipeline_bg, event.id)
    return EventCreateResponse(id=event.id, event_id=event.event_id)


@router.get("", response_model=EventListResponse)
def list_events(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),
    cam_id: str | None = Query(default=None),
    status_: str | None = Query(default=None, alias="status"),
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> EventListResponse:
    """Lịch sử sự kiện cho Mobile: phân trang + filter (contract v2)."""
    query = db.query(Event).join(Camera, Event.camera_id == Camera.id)
    if cam_id is not None:
        query = query.filter(Camera.cam_id == cam_id)
    if status_ is not None:
        query = query.filter(Event.status == status_)
    if start_date is not None:
        query = query.filter(
            Event.timestamp_utc
            >= datetime.combine(start_date, datetime.min.time())
        )
    if end_date is not None:
        query = query.filter(
            Event.timestamp_utc
            <= datetime.combine(end_date, datetime.max.time())
        )

    total = query.count()
    events = (
        query.order_by(Event.timestamp_utc.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return EventListResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=[to_event_out(e) for e in events],
    )


@router.get("/{event_id}", response_model=EventOut)
def get_event(
    event_id: UUID,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),
) -> EventOut:
    event = db.query(Event).filter(Event.event_id == event_id).first()
    if event is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sự kiện không tồn tại",
        )
    return to_event_out(event)


@router.get("/{event_id}/snapshot")
def get_event_snapshot(
    event_id: UUID,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),
) -> Response:
    """Serve ảnh snapshot từ MinIO qua backend API (tránh vấn đề localhost qua Ngrok)."""
    event = db.query(Event).filter(Event.event_id == event_id).first()
    if event is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sự kiện không tồn tại",
        )

    if not event.image_url:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sự kiện không có snapshot",
        )

    image_bytes = get_jpeg(event.image_url)
    if image_bytes is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Không thể tải ảnh từ MinIO",
        )

    return Response(content=image_bytes, media_type="image/jpeg")
