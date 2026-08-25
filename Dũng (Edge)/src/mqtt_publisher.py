#!/usr/bin/env python3
"""Publisher MQTT — gửi fall_event / telemetry lên broker Khánh (RabbitMQ + plugin MQTT).

- Tương thích cả paho-mqtt 1.x (Python 3.6, vd Jetson Nano 4GB — paho 2.x đã bỏ
  hỗ trợ 3.6) lẫn 2.x (CallbackAPIVersion.VERSION2) — tự dò qua hasattr, xem
  _HAS_CALLBACK_API_VERSION bên dưới.
- Cấu hình đọc từ biến môi trường (KHÔNG hardcode broker/credential):
    MQTT_BROKER (bắt buộc)   IP broker Khánh (IP public động — hỏi lại trước demo)
    MQTT_PORT   (mặc định 1883)
    MQTT_USER / MQTT_PASS    (tuỳ chọn; broker Khánh dùng guest/guest)
    MQTT_CLIENT_ID (tuỳ chọn)

Dùng:
    pub = MqttPublisher.from_env()
    pub.connect()
    pub.publish_fall_event("cam_01", event_dict)
    pub.publish_json("debug/cam_01/preview", payload)   # topic tuỳ ý
    pub.disconnect()
"""

import json
import logging
import os
import time
from typing import Optional

import paho.mqtt.client as mqtt

from event_builder import fall_topic, telemetry_topic

log = logging.getLogger("mqtt_publisher")

DEFAULT_PORT = 1883
CONNECT_TIMEOUT_S = 10
PUBLISH_TIMEOUT_S = 10

_HAS_CALLBACK_API_VERSION = hasattr(mqtt, "CallbackAPIVersion")


class MqttPublisher:
    def __init__(
        self,
        host: str,
        port: int = DEFAULT_PORT,
        username: Optional[str] = None,
        password: Optional[str] = None,
        client_id: str = "jetson-edge",
        keepalive: int = 60,
    ):
        if not host:
            raise ValueError("Thiếu MQTT broker host (đặt MQTT_BROKER trong .env)")
        self.host = host
        self.port = int(port)
        self.keepalive = keepalive
        self._connected = False

        if _HAS_CALLBACK_API_VERSION:
            self._client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2, client_id=client_id, clean_session=True
            )
        else:
            self._client = mqtt.Client(client_id=client_id, clean_session=True)
        if username:
            self._client.username_pw_set(username, password or "")
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

    @classmethod
    def from_env(cls, client_id: str = "jetson-edge") -> "MqttPublisher":
        host = os.environ.get("MQTT_BROKER", "")
        return cls(
            host=host,
            port=int(os.environ.get("MQTT_PORT", DEFAULT_PORT)),
            username=os.environ.get("MQTT_USER") or None,
            password=os.environ.get("MQTT_PASS") or None,
            client_id=client_id,
        )

    # ---------- lifecycle ----------
    def _on_connect(self, _client, _userdata, _flags, reason_code, *_props):
        # paho 1.x: reason_code là int (0=OK). paho 2.x: là ReasonCode, dùng .is_failure
        # (KHÔNG ép int() được ReasonCode -> phải rẽ nhánh theo type).
        if isinstance(reason_code, int):
            failed = reason_code != 0
        else:
            failed = reason_code.is_failure
        self._connected = not failed
        if self._connected:
            log.info("MQTT kết nối OK -> %s:%s", self.host, self.port)
        else:
            log.error("MQTT kết nối thất bại: %s", reason_code)

    def _on_disconnect(self, _client, _userdata, *_args):
        self._connected = False
        log.info("MQTT đã ngắt kết nối")

    def connect(self, timeout_s: int = CONNECT_TIMEOUT_S) -> bool:
        self._client.connect(self.host, self.port, self.keepalive)
        self._client.loop_start()
        deadline = time.time() + timeout_s
        while not self._connected and time.time() < deadline:
            time.sleep(0.05)
        if not self._connected:
            raise TimeoutError(
                f"Không CONNACK trong {timeout_s}s từ {self.host}:{self.port} "
                f"(broker tắt? plugin rabbitmq_mqtt chưa bật? IP đổi?)"
            )
        return True

    def disconnect(self) -> None:
        try:
            self._client.loop_stop()
            self._client.disconnect()
        finally:
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ---------- publish ----------
    def publish_json(self, topic: str, payload: dict, qos: int = 1) -> bool:
        """Publish 1 dict dưới dạng JSON. Chờ xác nhận gửi (QoS1) tối đa PUBLISH_TIMEOUT_S."""
        data = json.dumps(payload, ensure_ascii=False)
        info = self._client.publish(topic, data, qos=qos)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            log.error("publish rc=%s topic=%s", info.rc, topic)
            return False
        try:
            info.wait_for_publish(PUBLISH_TIMEOUT_S)
        except (ValueError, RuntimeError) as e:
            log.error("wait_for_publish lỗi topic=%s: %s", topic, e)
            return False
        ok = info.is_published()
        log.info("publish topic=%s qos=%s bytes=%d ok=%s", topic, qos, len(data), ok)
        return ok

    def publish_fall_event(self, cam_id: str, event: dict, qos: int = 1) -> bool:
        return self.publish_json(fall_topic(cam_id), event, qos=qos)

    def publish_telemetry(self, cam_id: str, telemetry: dict, qos: int = 0) -> bool:
        return self.publish_json(telemetry_topic(cam_id), telemetry, qos=qos)
