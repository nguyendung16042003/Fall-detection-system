from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings

DB_POOL_SIZE = 5  # Minimum connections in pool
DB_MAX_OVERFLOW = 15  # Additional connections beyond pool_size (total max = 20)

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=DB_POOL_SIZE,
    max_overflow=DB_MAX_OVERFLOW,
    pool_recycle=3600,  # Recycle connections after 1 hour
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Lớp Base chung cho toàn bộ SQLAlchemy models."""


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
