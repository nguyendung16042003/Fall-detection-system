from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.dialects.postgresql import UUID
from app.core.database import Base
import uuid
import datetime

class Camera(Base):
    __tablename__ = "cameras"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    rtsp_url = Column(String(255), nullable=False)  # Quan trọng cho Lớp 1 (Edge) [1, 3]
    location_note = Column(Text)
    status = Column(String(20), default='offline') # Trạng thái camera [2]
    last_heartbeat = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=datetime.datetime.utcnow)