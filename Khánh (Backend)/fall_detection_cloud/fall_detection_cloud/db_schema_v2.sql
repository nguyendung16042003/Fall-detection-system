-- ============================================================
-- Fall Detection System — DB schema v2
-- Khớp api_contract_v2.md + mqtt_schema_v2.json (bỏ db.txt cũ)
-- Khóa chính số nguyên; users.username; events.event_id UUID; device_tokens.
-- Dùng cho docker-entrypoint-initdb.d (khởi tạo DB sạch).
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- USERS -------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id              SERIAL PRIMARY KEY,
    username        VARCHAR(150) NOT NULL UNIQUE,
    email           VARCHAR(255) NOT NULL UNIQUE,
    hashed_password VARCHAR(255) NOT NULL,
    role            VARCHAR(50)  NOT NULL DEFAULT 'caregiver',
    is_active       BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- CAMERAS -----------------------------------------------------
CREATE TABLE IF NOT EXISTS cameras (
    id             SERIAL PRIMARY KEY,
    cam_id         VARCHAR(50)  NOT NULL UNIQUE,
    name           VARCHAR(255) NOT NULL,
    rtsp_url       VARCHAR(255) NOT NULL,
    location       VARCHAR(255),
    is_active      BOOLEAN      NOT NULL DEFAULT TRUE,
    edge_device_id VARCHAR(100) NOT NULL DEFAULT '',
    status         VARCHAR(20)  NOT NULL DEFAULT 'unknown',
    last_heartbeat TIMESTAMPTZ,
    note           TEXT,
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- CAMERA RULES ------------------------------------------------
CREATE TABLE IF NOT EXISTS camera_rules (
    id                  SERIAL PRIMARY KEY,
    camera_id           INTEGER NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    time_window_sec     INTEGER NOT NULL DEFAULT 2,
    min_lying_frames    INTEGER NOT NULL DEFAULT 5,
    min_confidence_sgie DOUBLE PRECISION NOT NULL DEFAULT 0.6,
    enable_vlm_verify   BOOLEAN NOT NULL DEFAULT TRUE,
    is_active           BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_camera_rules_camera_id ON camera_rules(camera_id);

-- EVENTS ------------------------------------------------------
CREATE TABLE IF NOT EXISTS events (
    id                   SERIAL PRIMARY KEY,
    event_id             UUID NOT NULL UNIQUE,
    camera_id            INTEGER NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    person_id            INTEGER,
    timestamp_utc        TIMESTAMPTZ NOT NULL,
    event_type           VARCHAR(50) NOT NULL DEFAULT 'fall_candidate',
    class_before         VARCHAR(50),
    final_class          VARCHAR(50),
    detection_confidence DOUBLE PRECISION,
    bbox_json            JSONB,
    frame_width          INTEGER,
    frame_height         INTEGER,
    rule_trigger         VARCHAR(50),
    transition_ms        INTEGER,
    vlm_verdict          VARCHAR(50),
    vlm_confidence       DOUBLE PRECISION,
    vlm_reason           VARCHAR(1000),
    image_url            VARCHAR(500),
    image_urls           JSONB,
    clip_url             VARCHAR(500),
    status               VARCHAR(20) NOT NULL DEFAULT 'pending',
    note                 TEXT,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_events_camera_id ON events(camera_id);

-- ALERTS ------------------------------------------------------
CREATE TABLE IF NOT EXISTS alerts (
    id              SERIAL PRIMARY KEY,
    event_id        UUID NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
    title           VARCHAR(255),
    message         TEXT,
    fcm_sent        BOOLEAN NOT NULL DEFAULT FALSE,
    is_acknowledged BOOLEAN NOT NULL DEFAULT FALSE,
    acknowledged_by INTEGER REFERENCES users(id),
    acknowledged_at TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_alerts_event_id ON alerts(event_id);
CREATE INDEX IF NOT EXISTS ix_alerts_is_acknowledged ON alerts(is_acknowledged);

-- DEVICE TOKENS (FCM) -----------------------------------------
CREATE TABLE IF NOT EXISTS device_tokens (
    id         SERIAL PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    device_id  VARCHAR(255) NOT NULL,
    platform   VARCHAR(50),
    fcm_token  VARCHAR(500) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_device_tokens_user_id ON device_tokens(user_id);

-- TELEMETRY LOGS ----------------------------------------------
CREATE TABLE IF NOT EXISTS telemetry_logs (
    id                  BIGSERIAL PRIMARY KEY,
    cam_id              VARCHAR(50) NOT NULL,
    timestamp_utc       TIMESTAMPTZ,
    pipeline_status     VARCHAR(50),
    fps_pgie            DOUBLE PRECISION,
    fps_sgie            DOUBLE PRECISION,
    active_tracks       INTEGER,
    ram_used_mb         DOUBLE PRECISION,
    ram_total_mb        DOUBLE PRECISION,
    cpu_temp_c          DOUBLE PRECISION,
    gpu_temp_c          DOUBLE PRECISION,
    cpu_usage_pct       DOUBLE PRECISION,
    disk_free_gb        DOUBLE PRECISION,
    mqtt_connected      BOOLEAN,
    last_event_sent_utc TIMESTAMPTZ,
    recorded_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_telemetry_logs_cam_id ON telemetry_logs(cam_id);

-- CONFIDENCE LOG (MLOps) --------------------------------------
CREATE TABLE IF NOT EXISTS confidence_log (
    id                     SERIAL PRIMARY KEY,
    event_id               INTEGER REFERENCES events(id) ON DELETE CASCADE,
    model_version          VARCHAR(50),
    inference_latency_ms   DOUBLE PRECISION,
    environment_metadata   JSONB,
    is_flagged_for_retrain BOOLEAN NOT NULL DEFAULT FALSE
);
