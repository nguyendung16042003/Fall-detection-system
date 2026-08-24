"""Tiện ích kiểm thử RabbitMQ: xem message trong 1 queue.

Dùng để xác minh API publish đúng (không thay thế consumer thật).

Ví dụ:
    python scripts/consume_queue.py fall_events.high          # peek (không xóa)
    python scripts/consume_queue.py telemetry_events --ack    # lấy & xóa
"""

import argparse
import json
import sys

import pika

sys.path.insert(0, ".")
from app.core.config import settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queue", help="Tên queue cần đọc")
    parser.add_argument(
        "--ack",
        action="store_true",
        help="Ack (xóa) message thay vì requeue (peek)",
    )
    parser.add_argument(
        "--max", type=int, default=10, help="Số message tối đa đọc"
    )
    args = parser.parse_args()

    conn = pika.BlockingConnection(pika.URLParameters(settings.RABBITMQ_URL))
    channel = conn.channel()
    count = 0
    try:
        while count < args.max:
            method, _props, body = channel.basic_get(
                args.queue, auto_ack=False
            )
            if method is None:
                break
            count += 1
            try:
                payload = json.loads(body)
            except ValueError:
                payload = body.decode("utf-8", "replace")
            print(f"--- message {count} ---")
            print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
            if args.ack:
                channel.basic_ack(method.delivery_tag)
            else:
                channel.basic_nack(method.delivery_tag, requeue=True)
    finally:
        conn.close()
    print(f"\nĐã đọc {count} message từ '{args.queue}'"
          f" ({'đã xóa' if args.ack else 'giữ nguyên'}).")


if __name__ == "__main__":
    main()
