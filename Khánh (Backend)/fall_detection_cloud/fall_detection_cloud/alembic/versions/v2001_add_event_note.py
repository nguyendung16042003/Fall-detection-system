"""add events.note column

Revision ID: v2001addnote
Revises: v2000initschema
Create Date: 2026-08-04

`services/dedup.py` và `services/alert_pipeline.py` gán `event.note` để ghi
dấu vết gộp sự kiện (dedup multi-camera) nhưng bảng `events` chưa có cột này
-> giá trị bị mất âm thầm khi commit. Thêm cột để khớp với code thật.
"""

import sqlalchemy as sa

from alembic import op

revision = "v2001addnote"
down_revision = "v2000initschema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("events", sa.Column("note", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("events", "note")
