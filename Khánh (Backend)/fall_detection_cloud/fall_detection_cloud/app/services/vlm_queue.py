"""VLM Queue Worker — xử lý VLM verification tuần tự với asyncio.Queue.

Sử dụng asyncio.Queue và worker để xử lý các sự kiện VLM một cách tuần tự,
đảm bảo không làm mất dữ liệu khi có nhiều sự kiện dồn dập. Mục tiêu độ trễ
dưới 3 giây cho mỗi lượt verify.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.models.event import Event
from app.services import vlm

logger = logging.getLogger(__name__)


@dataclass
class VLMTask:
    """Task cho VLM verification."""
    event_id: int
    frames_bytes: list[bytes] | None = None
    frames_refs: list[str] | None = None
    callback: Callable[[Session, Event, vlm.VLMResult], None] | None = None


class VLMQueueWorker:
    """Worker xử lý VLM verification từ queue."""

    def __init__(self, max_queue_size: int = 100, num_workers: int = 2):
        self._queue: asyncio.Queue[VLMTask] = asyncio.Queue(maxsize=max_queue_size)
        self._num_workers = num_workers
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """Khởi động worker tasks."""
        if self._running:
            logger.warning("VLM Queue Worker đã chạy")
            return

        self._running = True
        self._tasks = [
            asyncio.create_task(self._worker(f"worker-{i}"))
            for i in range(self._num_workers)
        ]
        logger.info(
            "VLM Queue Worker khởi động với %d workers, queue size=%d",
            self._num_workers,
            self._queue.maxsize,
        )

    async def stop(self) -> None:
        """Dừng worker tasks."""
        if not self._running:
            return

        self._running = False
        # Hủy tất cả tasks
        for task in self._tasks:
            task.cancel()
        
        # Chờ tasks hoàn thành
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("VLM Queue Worker đã dừng")

    async def enqueue(self, task: VLMTask) -> bool:
        """Thêm task vào queue. Trả True nếu thành công, False nếu queue đầy."""
        try:
            await asyncio.wait_for(self._queue.put(task), timeout=1.0)
            logger.debug("VLM task enqueued: event_id=%d", task.event_id)
            return True
        except asyncio.TimeoutError:
            logger.warning("VLM queue đầy, không thể thêm task: event_id=%d", task.event_id)
            return False

    async def _worker(self, name: str) -> None:
        """Worker loop xử lý tasks."""
        logger.info("VLM worker %s started", name)
        while self._running:
            try:
                # Lấy task từ queue với timeout để kiểm tra _running
                task = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._process_task(task, name)
                self._queue.task_done()
            except asyncio.TimeoutError:
                # Timeout là bình thường, tiếp tục loop
                continue
            except Exception as exc:  # noqa: BLE001
                logger.error("VLM worker %s error: %s", name, exc)
        logger.info("VLM worker %s stopped", name)

    async def _process_task(self, task: VLMTask, worker_name: str) -> None:
        """Xử lý một VLM task."""
        start_time = datetime.now(timezone.utc)
        logger.info(
            "VLM worker %s processing event_id=%d", worker_name, task.event_id
        )

        db = SessionLocal()
        try:
            event = db.query(Event).filter(Event.id == task.event_id).first()
            if not event:
                logger.warning("Event không tồn tại: event_id=%d", task.event_id)
                return

            # Gọi VLM verification với chuỗi 6 frame
            vlm_res = vlm.verify_fall(
                frames_bytes=task.frames_bytes,
                frames_refs=task.frames_refs,
            )

            # Cập nhật event với kết quả VLM
            event.vlm_verdict = vlm_res.verdict
            event.vlm_confidence = vlm_res.confidence
            event.vlm_reason = vlm_res.reason or None
            db.add(event)
            db.commit()

            latency_ms = vlm_res.latency_ms
            logger.info(
                "VLM worker %s completed event_id=%d: verdict=%s, confidence=%.2f, latency=%.0fms",
                worker_name,
                task.event_id,
                vlm_res.verdict,
                vlm_res.confidence,
                latency_ms,
            )

            if latency_ms > 3000:
                logger.warning(
                    "VLM latency %.0fms vượt mục tiêu 3s cho event_id=%d",
                    latency_ms,
                    task.event_id,
                )

            # Gọi callback nếu có (để tiếp tục pipeline)
            if task.callback:
                task.callback(db, event, vlm_res)

        except Exception as exc:  # noqa: BLE001
            logger.error("Lỗi xử lý VLM task event_id=%d: %s", task.event_id, exc)
            db.rollback()
        finally:
            db.close()

    def get_queue_size(self) -> int:
        """Trả về số task đang chờ trong queue."""
        return self._queue.qsize()

    def is_running(self) -> bool:
        """Trả về True nếu worker đang chạy."""
        return self._running


# Global instance
_vlm_queue_worker: VLMQueueWorker | None = None


def get_vlm_queue_worker() -> VLMQueueWorker:
    """Trả về global VLM Queue Worker instance."""
    global _vlm_queue_worker
    if _vlm_queue_worker is None:
        _vlm_queue_worker = VLMQueueWorker(max_queue_size=100, num_workers=2)
    return _vlm_queue_worker
