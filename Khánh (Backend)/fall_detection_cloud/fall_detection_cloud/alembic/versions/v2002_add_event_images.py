"""add events.image_urls column

Revision ID: v2002addimgs
Revises: v2001addnote
Create Date: 2026-08-18

Trước đây chỉ frame trigger (offset_ms=0) được upload lên MinIO; 5 frame còn
lại chỉ tồn tại tạm trong bộ nhớ để VLM phân tích rồi mất. Yêu cầu mới: lưu
và hiển thị đủ 6 ảnh bằng chứng. Thêm cột JSONB lưu danh sách
{index, offset_ms, url} theo thứ tự offset_ms tăng dần — cùng phong cách với
`bbox_json`. Cột `image_url` (ảnh trigger, dùng cho VLM/Telegram) và
`clip_url` (zip MLOps false_positive) giữ nguyên, không đụng tới.
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "v2002addimgs"
down_revision = "v2001addnote"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "events",
        sa.Column(
            "image_urls", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
    )


def downgrade() -> None:
    op.drop_column("events", "image_urls")
