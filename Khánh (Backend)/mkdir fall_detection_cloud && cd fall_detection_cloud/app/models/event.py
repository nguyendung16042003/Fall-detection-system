from sqlalchemy import Column, String, Float, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from app.core.database import Base
import uuid, datetime

class Event(Base):
    __tablename__ = "events"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    camera_id = Column(UUID(as_uuid=True), ForeignKey("cameras.id"))
    timestamp = Column(DateTime, nullable=False)
    severity = Column(String(20), default="low") # Theo API Contract 3.2
    status = Column(String(20), default="new")   # Theo API Contract 3.2
    bbox_json = Column(JSONB)
    image_url = Column(String(500))