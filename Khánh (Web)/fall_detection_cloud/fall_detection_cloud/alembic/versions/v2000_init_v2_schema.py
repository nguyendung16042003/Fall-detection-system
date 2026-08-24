"""init v2 schema (contract v2 + mqtt_schema v2, integer PKs)

Revision ID: v2000initschema
Revises:
Create Date: 2026-06-08

Schema fully aligned with api_contract_v2 + mqtt_schema_v2 (bỏ db.txt cũ):
- Khóa chính số nguyên auto-increment cho users/cameras/events/alerts.
- `users.username` (đăng nhập bằng username), bỏ `full_name`.
- `events.event_id` UUID v4 do Edge tạo (unique); bỏ `severity`.
- Bảng FCM đổi tên thành `device_tokens`.
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision = "v2000initschema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("username", sa.String(150), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column(
            "role", sa.String(50), nullable=False, server_default="caregiver"
        ),
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_users_username", "users", ["username"], unique=True)
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "cameras",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("cam_id", sa.String(50), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("rtsp_url", sa.String(255), nullable=False),
        sa.Column("location", sa.String(255)),
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
        sa.Column(
            "edge_device_id",
            sa.String(100),
            nullable=False,
            server_default="",
        ),
        sa.Column(
            "status", sa.String(20), nullable=False, server_default="unknown"
        ),
        sa.Column("last_heartbeat", sa.DateTime(timezone=True)),
        sa.Column("note", sa.Text()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_cameras_cam_id", "cameras", ["cam_id"], unique=True)

    op.create_table(
        "camera_rules",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "camera_id",
            sa.Integer(),
            sa.ForeignKey("cameras.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "time_window_sec", sa.Integer(), nullable=False, server_default="2"
        ),
        sa.Column(
            "min_lying_frames",
            sa.Integer(),
            nullable=False,
            server_default="5",
        ),
        sa.Column(
            "min_confidence_sgie",
            sa.Float(),
            nullable=False,
            server_default="0.6",
        ),
        sa.Column(
            "enable_vlm_verify",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_camera_rules_camera_id", "camera_rules", ["camera_id"]
    )

    op.create_table(
        "events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("event_id", UUID(as_uuid=True), nullable=False),
        sa.Column(
            "camera_id",
            sa.Integer(),
            sa.ForeignKey("cameras.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("person_id", sa.Integer()),
        sa.Column(
            "timestamp_utc", sa.DateTime(timezone=True), nullable=False
        ),
        sa.Column(
            "event_type",
            sa.String(50),
            nullable=False,
            server_default="fall_candidate",
        ),
        sa.Column("class_before", sa.String(50)),
        sa.Column("final_class", sa.String(50)),
        sa.Column("detection_confidence", sa.Float()),
        sa.Column("bbox_json", JSONB()),
        sa.Column("frame_width", sa.Integer()),
        sa.Column("frame_height", sa.Integer()),
        sa.Column("rule_trigger", sa.String(50)),
        sa.Column("transition_ms", sa.Integer()),
        sa.Column("vlm_verdict", sa.String(50)),
        sa.Column("vlm_confidence", sa.Float()),
        sa.Column("vlm_reason", sa.String(1000)),
        sa.Column("image_url", sa.String(500)),
        sa.Column("clip_url", sa.String(500)),
        sa.Column(
            "status", sa.String(20), nullable=False, server_default="pending"
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_events_event_id", "events", ["event_id"], unique=True)
    op.create_index("ix_events_camera_id", "events", ["camera_id"])

    op.create_table(
        "alerts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "event_id",
            UUID(as_uuid=True),
            sa.ForeignKey("events.event_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.String(255)),
        sa.Column("message", sa.Text()),
        sa.Column(
            "fcm_sent",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column(
            "is_acknowledged",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column(
            "acknowledged_by",
            sa.Integer(),
            sa.ForeignKey("users.id"),
        ),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_alerts_event_id", "alerts", ["event_id"])
    op.create_index(
        "ix_alerts_is_acknowledged", "alerts", ["is_acknowledged"]
    )

    op.create_table(
        "device_tokens",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("device_id", sa.String(255), nullable=False),
        sa.Column("platform", sa.String(50)),
        sa.Column("fcm_token", sa.String(500), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_device_tokens_user_id", "device_tokens", ["user_id"]
    )

    op.create_table(
        "telemetry_logs",
        sa.Column(
            "id", sa.BigInteger(), primary_key=True, autoincrement=True
        ),
        sa.Column("cam_id", sa.String(50), nullable=False),
        sa.Column("timestamp_utc", sa.DateTime(timezone=True)),
        sa.Column("pipeline_status", sa.String(50)),
        sa.Column("fps_pgie", sa.Float()),
        sa.Column("fps_sgie", sa.Float()),
        sa.Column("active_tracks", sa.Integer()),
        sa.Column("ram_used_mb", sa.Float()),
        sa.Column("ram_total_mb", sa.Float()),
        sa.Column("cpu_temp_c", sa.Float()),
        sa.Column("gpu_temp_c", sa.Float()),
        sa.Column("cpu_usage_pct", sa.Float()),
        sa.Column("disk_free_gb", sa.Float()),
        sa.Column("mqtt_connected", sa.Boolean()),
        sa.Column("last_event_sent_utc", sa.DateTime(timezone=True)),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_telemetry_logs_cam_id", "telemetry_logs", ["cam_id"]
    )

    op.create_table(
        "confidence_log",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "event_id",
            sa.Integer(),
            sa.ForeignKey("events.id", ondelete="CASCADE"),
        ),
        sa.Column("model_version", sa.String(50)),
        sa.Column("inference_latency_ms", sa.Float()),
        sa.Column("environment_metadata", JSONB()),
        sa.Column(
            "is_flagged_for_retrain",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    op.drop_table("confidence_log")
    op.drop_index("ix_telemetry_logs_cam_id", table_name="telemetry_logs")
    op.drop_table("telemetry_logs")
    op.drop_index("ix_device_tokens_user_id", table_name="device_tokens")
    op.drop_table("device_tokens")
    op.drop_index("ix_alerts_is_acknowledged", table_name="alerts")
    op.drop_index("ix_alerts_event_id", table_name="alerts")
    op.drop_table("alerts")
    op.drop_index("ix_events_camera_id", table_name="events")
    op.drop_index("ix_events_event_id", table_name="events")
    op.drop_table("events")
    op.drop_index("ix_camera_rules_camera_id", table_name="camera_rules")
    op.drop_table("camera_rules")
    op.drop_index("ix_cameras_cam_id", table_name="cameras")
    op.drop_table("cameras")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_index("ix_users_username", table_name="users")
    op.drop_table("users")
