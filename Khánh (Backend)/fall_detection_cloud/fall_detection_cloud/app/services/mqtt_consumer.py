"""MQTT consumer — nhận sự kiện & telemetry từ Edge (Jetson).

Subscribe (mqtt_schema v2):
  - events/+/fall        -> lưu Event + chạy trigger chain (VLM -> alert)
  - telemetry/+/status   -> lưu TelemetryLog

Broker: RabbitMQ + MQTT plugin (port 1883). Chạy nền trong một thread riêng,
fail-safe: nếu thiếu `paho-mqtt`, tắt MQTT, hoặc broker lỗi -> chỉ log, không
làm sập app (REST POST /events vẫn là đường dự phòng).
"""

import json
import logging
import threading

from app.core.config import settings
from app.core.database import SessionLocal
from app.schemas.event import EventCreate
from app.schemas.telemetry import TelemetryIn
from app.services.alert_pipeline import run_pipeline
from app.services.event_ingest import CameraNotFoundError, ingest_event
from app.services.telemetry_ingest import ingest_telemetry

logger = logging.getLogger(__name__)


def _cam_id_from_topic(topic: str) -> str | None:
    # events/cam_01/fall hoặc telemetry/cam_01/status
    parts = topic.split("/")
    return parts[1] if len(parts) >= 2 else None


def _handle_fall(payload: dict, cam_id: str | None) -> None:
    if cam_id and not payload.get("cam_id"):
        payload["cam_id"] = cam_id
    data = EventCreate.model_validate(payload)
    db = SessionLocal()
    try:
        event, frame, all_frames = ingest_event(db, data)
        result = run_pipeline(db, event, image_bytes=frame, frames_bytes=all_frames)
        logger.info("MQTT fall %s -> %s", event.id, result)
    except CameraNotFoundError as exc:
        logger.warning("MQTT fall: camera không tồn tại cam_id=%s", exc)
    finally:
        db.close()


def _handle_telemetry(payload: dict, cam_id: str | None) -> None:
    if cam_id and not payload.get("cam_id"):
        payload["cam_id"] = cam_id
    data = TelemetryIn.model_validate(payload)
    db = SessionLocal()
    try:
        ingest_telemetry(db, data)
    finally:
        db.close()


def _on_connect(client, userdata, flags, rc, properties=None) -> None:  # noqa: ANN001
    if rc == 0:
        client.subscribe(
            [(settings.MQTT_TOPIC_FALL, 1), (settings.MQTT_TOPIC_TELEMETRY, 1)]
        )
        logger.info(
            "MQTT đã kết nối, subscribe %s + %s",
            settings.MQTT_TOPIC_FALL,
            settings.MQTT_TOPIC_TELEMETRY,
        )
    else:
        logger.warning("MQTT kết nối thất bại, rc=%s", rc)


def _on_message(client, userdata, msg) -> None:  # noqa: ANN001
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("MQTT payload không phải JSON (%s): %s", msg.topic, exc)
        return
    cam_id = _cam_id_from_topic(msg.topic)
    try:
        if msg.topic.endswith("/fall"):
            _handle_fall(payload, cam_id)
        elif msg.topic.endswith("/status"):
            _handle_telemetry(payload, cam_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Lỗi xử lý MQTT message %s: %s", msg.topic, exc)


class MQTTConsumer:
    def __init__(self) -> None:
        self._client = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        logger.info("MQTT consumer.start() called")
        logger.info(f"MQTT_ENABLED={settings.MQTT_ENABLED}")
        logger.info(f"MQTT_HOST={settings.MQTT_HOST}")
        logger.info(f"MQTT_PORT={settings.MQTT_PORT}")
        
        if not settings.MQTT_ENABLED:
            logger.info("MQTT tắt (MQTT_ENABLED=false), bỏ qua consumer")
            return
        try:
            import paho.mqtt.client as mqtt
            logger.info("paho-mqtt imported successfully")
        except ImportError:
            logger.warning("Chưa cài 'paho-mqtt'; bỏ qua MQTT consumer")
            return

        if hasattr(mqtt, "CallbackAPIVersion"):
            # paho-mqtt >= 2.x
            client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id="fall-cloud-consumer",
            )
        else:
            # paho-mqtt 1.x
            client = mqtt.Client(client_id="fall-cloud-consumer")

        if settings.MQTT_USERNAME:
            client.username_pw_set(
                settings.MQTT_USERNAME, settings.MQTT_PASSWORD
            )
        client.on_connect = _on_connect
        client.on_message = _on_message

        try:
            logger.info(f"Attempting to connect to {settings.MQTT_HOST}:{settings.MQTT_PORT}")
            client.connect_async(settings.MQTT_HOST, settings.MQTT_PORT, 60)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Không kết nối được MQTT broker: %s", exc)
            return

        client.loop_start()
        self._client = client
        logger.info(
            "MQTT consumer khởi động (%s:%s)",
            settings.MQTT_HOST,
            settings.MQTT_PORT,
        )

    def stop(self) -> None:
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None


consumer = MQTTConsumer()
