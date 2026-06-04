from sqlalchemy import Column, String, Boolean, DateTime
from sqlalchemy.dialects.postgresql import UUID
from app.core.database import Base
import uuid, datetime

class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    fcm_token = Column(String(500))  # Theo API Contract 8.1
    role = Column(String, default="caregiver")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)