from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_current_active_user
from app.core.config import settings
from app.core.database import get_db
from app.models.camera import Camera
from app.models.user import User

router = APIRouter()


class LiveStream(BaseModel):
    cam_id: str
    stream_type: str = "hls"
    stream_url: str


@router.get("/{cam_id}", response_model=LiveStream)
def get_live_stream(
    cam_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),
) -> LiveStream:
    """URL HLS (MediaMTX) cho live view (contract v2 mục 7).

    MediaMTX dự kiến setup ở tuần 5; endpoint chỉ dựng URL theo cam_id.
    """
    camera = db.query(Camera).filter(Camera.cam_id == cam_id).first()
    if camera is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy Camera",
        )
    base = settings.MEDIAMTX_BASE_URL.rstrip("/")
    return LiveStream(
        cam_id=cam_id,
        stream_type="hls",
        stream_url=f"{base}/{cam_id}/index.m3u8",
    )
