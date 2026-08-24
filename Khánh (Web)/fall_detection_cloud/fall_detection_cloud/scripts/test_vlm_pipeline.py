"""Mock Event Test — kiểm tra toàn bộ chuỗi: Event → VLM verdict → FCM → App.

Script này giả lập sự kiện có 6 frame ảnh để kiểm tra toàn bộ pipeline:
1. Tạo mock event với 6 frame JPEG
2. Gọi event_ingest để lưu vào database
3. Gọi alert_pipeline để chạy VLM verification
4. Kiểm tra kết quả VLM, alert creation, và notifications
"""

import base64
import io
import logging
import uuid
from datetime import datetime, timezone

from PIL import Image

from app.core.database import SessionLocal, engine, Base
from app.models.camera import Camera
from app.models.camera_rule import CameraRule
from app.schemas.event import EventCreate, DetectionIn, RuleIn, FrameIn
from app.services.event_ingest import ingest_event
from app.services.alert_pipeline import run_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_jpeg(width: int = 640, height: int = 480, color: str = "white") -> bytes:
    """Tạo mock JPEG image cho testing."""
    img = Image.new("RGB", (width, height), color)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=80)
    return buffer.getvalue()


def setup_test_data(db: Session):
    """Setup test camera và camera rule."""
    # Tạo test camera
    camera = db.query(Camera).filter(Camera.cam_id == "test_cam_01").first()
    if not camera:
        camera = Camera(
            cam_id="test_cam_01",
            name="Test Camera 01",
            rtsp_url="rtsp://test.local/stream",
            location="Test Location",
            is_active=True,
        )
        db.add(camera)
        db.commit()
        db.refresh(camera)
    
    # Tạo camera rule với VLM enabled
    rule = db.query(CameraRule).filter(CameraRule.camera_id == camera.id).first()
    if not rule:
        rule = CameraRule(
            camera_id=camera.id,
            time_window_sec=2,
            min_lying_frames=5,
            min_confidence_sgie=0.6,
            enable_vlm_verify=True,
            is_active=True,
        )
        db.add(rule)
        db.commit()
        db.refresh(rule)
    
    return camera, rule


def test_single_event_pipeline():
    """Test pipeline với single event có 6 frames."""
    logger.info("=" * 60)
    logger.info("TEST: Single Event Pipeline với 6 Frames")
    logger.info("=" * 60)
    
    db = SessionLocal()
    try:
        # Setup test data
        camera, rule = setup_test_data(db)
        logger.info(f"Setup camera: {camera.cam_id}, rule VLM enabled: {rule.enable_vlm_verify}")
        
        # Tạo 6 mock frames (mỗi frame cách nhau 500ms)
        frames = []
        for i in range(6):
            offset_ms = -2500 + (i * 500)  # -2500, -2000, -1500, -1000, -500, 0
            frame_bytes = create_mock_jpeg(640, 480, color="red" if i == 5 else "white")
            frame_b64 = base64.b64encode(frame_bytes).decode("utf-8")
            frames.append(FrameIn(offset_ms=offset_ms, jpeg_b64=frame_b64))
        
        # Tạo mock event payload
        event_data = EventCreate(
            schema_version="1.1",
            event_id=str(uuid.uuid4()),
            cam_id="test_cam_01",
            person_id=1,
            timestamp_utc=datetime.now(timezone.utc),
            event_type="fall_candidate",
            detection=DetectionIn(
                class_before="standing",
                final_class="lying",
                confidence=0.92,
                bbox_xyxy=[120, 80, 380, 420],
                frame_width=1280,
                frame_height=720,
            ),
            rule=RuleIn(
                version="1.0",
                trigger="standing_to_lying",
                transition_ms=1800,
                window_ms=2000,
            ),
            frames=frames,
            status="pending",
        )
        
        logger.info(f"Created mock event with {len(frames)} frames")
        
        # Ingest event
        event, frame_bytes, all_frames = ingest_event(db, event_data)
        logger.info(f"Ingested event: id={event.id}, event_id={event.event_id}")
        logger.info(f"Decoded {len(all_frames) if all_frames else 0} frames for VLM")
        
        # Run pipeline
        result = run_pipeline(db, event, image_bytes=frame_bytes, frames_bytes=all_frames)
        
        logger.info("=" * 60)
        logger.info("PIPELINE RESULT:")
        logger.info(f"  Event ID: {result.event_id}")
        logger.info(f"  VLM Ran: {result.vlm_ran}")
        logger.info(f"  VLM Verdict: {result.vlm_verdict}")
        logger.info(f"  VLM Confidence: {result.vlm_confidence}")
        logger.info(f"  VLM Latency: {result.vlm_latency_ms}ms")
        logger.info(f"  Alert Created: {result.alert_created}")
        logger.info(f"  Alert ID: {result.alert_id}")
        logger.info(f"  FCM Sent: {result.fcm_sent}")
        logger.info(f"  Telegram OK: {result.telegram_ok}")
        logger.info(f"  Suppressed: {result.suppressed}")
        logger.info(f"  Notes: {result.notes}")
        logger.info("=" * 60)
        
        # Verify event status
        db.refresh(event)
        logger.info(f"Final event status: {event.status}")
        logger.info(f"VLM verdict in DB: {event.vlm_verdict}")
        logger.info(f"VLM confidence in DB: {event.vlm_confidence}")
        logger.info(f"VLM reason in DB: {event.vlm_reason}")
        logger.info(f"Clip URL (MLOps): {event.clip_url}")
        
        return result
        
    finally:
        db.close()


def test_multi_camera_dedup():
    """Test dedup logic với 2 events từ 2 camera khác nhau."""
    logger.info("=" * 60)
    logger.info("TEST: Multi-Camera Dedup")
    logger.info("=" * 60)
    
    db = SessionLocal()
    try:
        # Setup 2 test cameras
        camera1, _ = setup_test_data(db)
        
        camera2 = db.query(Camera).filter(Camera.cam_id == "test_cam_02").first()
        if not camera2:
            camera2 = Camera(
                cam_id="test_cam_02",
                name="Test Camera 02",
                rtsp_url="rtsp://test.local/stream2",
                location="Test Location 2",
                is_active=True,
            )
            db.add(camera2)
            db.commit()
            db.refresh(camera2)
        
        # Tạo rule cho camera2
        rule2 = db.query(CameraRule).filter(CameraRule.camera_id == camera2.id).first()
        if not rule2:
            rule2 = CameraRule(
                camera_id=camera2.id,
                time_window_sec=2,
                min_lying_frames=5,
                min_confidence_sgie=0.6,
                enable_vlm_verify=True,
                is_active=True,
            )
            db.add(rule2)
            db.commit()
        
        # Tạo 2 events với timestamp gần nhau (cách nhau < 2s)
        base_time = datetime.now(timezone.utc)
        
        # Event 1 từ camera1
        frames1 = []
        for i in range(6):
            offset_ms = -2500 + (i * 500)
            frame_bytes = create_mock_jpeg(640, 480, color="blue")
            frame_b64 = base64.b64encode(frame_bytes).decode("utf-8")
            frames1.append(FrameIn(offset_ms=offset_ms, jpeg_b64=frame_b64))
        
        event_data1 = EventCreate(
            schema_version="1.1",
            event_id=str(uuid.uuid4()),
            cam_id="test_cam_01",
            person_id=1,
            timestamp_utc=base_time,
            event_type="fall_candidate",
            detection=DetectionIn(
                class_before="standing",
                final_class="lying",
                confidence=0.92,
                bbox_xyxy=[120, 80, 380, 420],
                frame_width=1280,
                frame_height=720,
            ),
            rule=RuleIn(
                version="1.0",
                trigger="standing_to_lying",
                transition_ms=1800,
                window_ms=2000,
            ),
            frames=frames1,
            status="pending",
        )
        
        event1, _, frames1_bytes = ingest_event(db, event_data1)
        logger.info(f"Created event1: id={event1.id}, timestamp={event1.timestamp_utc}")
        
        # Event 2 từ camera2 (cách nhau 1 giây)
        frames2 = []
        for i in range(6):
            offset_ms = -2500 + (i * 500)
            frame_bytes = create_mock_jpeg(640, 480, color="green")
            frame_b64 = base64.b64encode(frame_bytes).decode("utf-8")
            frames2.append(FrameIn(offset_ms=offset_ms, jpeg_b64=frame_b64))
        
        event_data2 = EventCreate(
            schema_version="1.1",
            event_id=str(uuid.uuid4()),
            cam_id="test_cam_02",
            person_id=1,
            timestamp_utc=base_time.replace(microsecond=base_time.microsecond + 1000000),  # +1s
            event_type="fall_candidate",
            detection=DetectionIn(
                class_before="standing",
                final_class="lying",
                confidence=0.88,
                bbox_xyxy=[120, 80, 380, 420],
                frame_width=1280,
                frame_height=720,
            ),
            rule=RuleIn(
                version="1.0",
                trigger="standing_to_lying",
                transition_ms=1600,
                window_ms=2000,
            ),
            frames=frames2,
            status="pending",
        )
        
        event2, _, frames2_bytes = ingest_event(db, event_data2)
        logger.info(f"Created event2: id={event2.id}, timestamp={event2.timestamp_utc}")
        
        # Run pipeline cho cả 2 events
        result1 = run_pipeline(db, event1, frames_bytes=frames1_bytes)
        logger.info(f"Event1 result: suppressed={result1.suppressed}, notes={result1.notes}")
        
        result2 = run_pipeline(db, event2, frames_bytes=frames2_bytes)
        logger.info(f"Event2 result: suppressed={result2.suppressed}, notes={result2.notes}")
        
        # Check dedup
        from app.services.dedup import get_deduplicator
        deduplicator = get_deduplicator()
        
        merge_group1 = deduplicator.get_merge_group(event1.id)
        merge_group2 = deduplicator.get_merge_group(event2.id)
        
        logger.info(f"Event1 merge group: {merge_group1}")
        logger.info(f"Event2 merge group: {merge_group2}")
        
        db.refresh(event1)
        db.refresh(event2)
        logger.info(f"Event1 status: {event1.status}")
        logger.info(f"Event2 status: {event2.status}")
        
        logger.info("=" * 60)
        
    finally:
        db.close()


if __name__ == "__main__":
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Test single event pipeline
    test_single_event_pipeline()
    
    # Test multi-camera dedup
    test_multi_camera_dedup()
    
    logger.info("All tests completed!")
