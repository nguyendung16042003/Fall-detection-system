"""Tích hợp RabbitMQ: khai báo topology và publish message sự kiện.

Topology (topic exchange ``fall_detection``):
- queue ``fall_events.high``      <- routing key ``fall_events.high``
- queue ``fall_events.low``       <- routing key ``fall_events.low``
- queue ``telemetry_events``      <- routing key ``telemetry_events``

Publish được bọc try/except: nếu broker không sẵn sàng, API vẫn hoạt động
(chỉ ghi log cảnh báo) để không chặn luồng lưu DB / phản hồi cho client.
"""

import json
import logging
from enum import Enum

import pika
from pika.exceptions import AMQPError

from app.core.config import settings

logger = logging.getLogger(__name__)


class RoutingKey(str, Enum):
    FALL_HIGH = "fall_events.high"
    FALL_LOW = "fall_events.low"
    TELEMETRY = "telemetry_events"


# Các queue cần khai báo; routing key trùng tên queue cho đơn giản.
QUEUES: tuple[str, ...] = (
    RoutingKey.FALL_HIGH.value,
    RoutingKey.FALL_LOW.value,
    RoutingKey.TELEMETRY.value,
)


def _connect() -> pika.BlockingConnection:
    return pika.BlockingConnection(
        pika.URLParameters(settings.RABBITMQ_URL)
    )


def declare_topology() -> None:
    """Khai báo exchange + queue + binding (idempotent)."""
    connection = _connect()
    try:
        channel = connection.channel()
        channel.exchange_declare(
            exchange=settings.RABBITMQ_EXCHANGE,
            exchange_type="topic",
            durable=True,
        )
        for queue in QUEUES:
            channel.queue_declare(queue=queue, durable=True)
            channel.queue_bind(
                queue=queue,
                exchange=settings.RABBITMQ_EXCHANGE,
                routing_key=queue,
            )
    finally:
        connection.close()


def publish(routing_key: str, payload: dict) -> bool:
    """Publish message JSON tới exchange theo routing key.

    Trả về True nếu publish thành công, False nếu lỗi/đã tắt broker.
    """
    if not settings.RABBITMQ_ENABLED:
        logger.info("RabbitMQ disabled; bỏ qua publish %s", routing_key)
        return False

    try:
        connection = _connect()
        try:
            channel = connection.channel()
            channel.exchange_declare(
                exchange=settings.RABBITMQ_EXCHANGE,
                exchange_type="topic",
                durable=True,
            )
            channel.queue_declare(routing_key, durable=True)
            channel.queue_bind(
                queue=routing_key,
                exchange=settings.RABBITMQ_EXCHANGE,
                routing_key=routing_key,
            )
            channel.basic_publish(
                exchange=settings.RABBITMQ_EXCHANGE,
                routing_key=routing_key,
                body=json.dumps(payload, default=str).encode("utf-8"),
                properties=pika.BasicProperties(
                    content_type="application/json",
                    delivery_mode=2,  # persistent
                ),
            )
            return True
        finally:
            connection.close()
    except (AMQPError, OSError) as exc:
        logger.warning("Publish RabbitMQ thất bại (%s): %s", routing_key, exc)
        return False
