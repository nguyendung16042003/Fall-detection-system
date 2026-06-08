"""Tập hợp Base metadata cho Alembic.

Import Base cùng toàn bộ models để ``Base.metadata`` chứa đầy đủ bảng khi
Alembic autogenerate / tạo migration.
"""

from app.core.database import Base  # noqa: F401
from app.models import (  # noqa: F401
    Alert,
    Camera,
    CameraRule,
    ConfidenceLog,
    Event,
    TelemetryLog,
    User,
    UserDevice,
)
