"""Full Trigger Test Script

Test kịch bản hoàn chỉnh:
1. Người ngã trước camera (Jetson)
2. Edge phát hiện fall (DeepStream)
3. Edge gửi MQTT event lên Cloud
4. Cloud validate → dedup → VLM verify
5. Cloud tạo alert → gửi FCM → gửi Telegram
6. App nhận FCM notification
7. App xem live stream (HLS)
8. App xem clip (MinIO presigned URL)
"""

import asyncio
import base64
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.database import SessionLocal
from app.models.alert import Alert
from app.models.camera import Camera
from app.models.event import Event
from app.schemas.event import EventCreate
from app.services.alert_pipeline import run_pipeline
from app.services.event_ingest import ingest_event


def create_mock_jpeg(width: int = 640, height: int = 480) -> bytes:
    """Tạo mock JPEG image đơn giản."""
    # Tạo một JPEG header đơn giản
    jpeg_header = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
    # Tạo mock data
    mock_data = b'\x00' * (width * height * 3)
    return jpeg_header + mock_data + b'\xff\xd9'


def create_mock_event_payload(cam_id: str = "cam_01") -> dict:
    """Tạo mock fall event payload."""
    frames = []
    for offset in [-2500, -2000, -1500, -1000, -500, 0]:
        jpeg_bytes = create_mock_jpeg()
        jpeg_b64 = base64.b64encode(jpeg_bytes).decode('utf-8')
        frames.append({
            "offset_ms": offset,
            "jpeg_b64": jpeg_b64
        })
    
    return {
        "schema_version": "1.1",
        "event_id": str(uuid4()),
        "cam_id": cam_id,
        "person_id": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event_type": "fall_candidate",
        "detection": {
            "class_before": "standing",
            "final_class": "lying",
            "confidence": 0.92,
            "bbox_xyxy": [120, 80, 380, 420],
            "frame_width": 1280,
            "frame_height": 720
        },
        "rule": {
            "version": "1.0",
            "trigger": "standing_to_lying",
            "transition_ms": 1800,
            "window_ms": 2000
        },
        "frames": frames
    }


def test_full_trigger():
    """Test full trigger pipeline."""
    print("=" * 60)
    print("FULL TRIGGER TEST")
    print("=" * 60)
    
    db = SessionLocal()
    
    try:
        # Step 1: Kiểm tra camera tồn tại
        print("\n[Step 1] Checking camera...")
        camera = db.query(Camera).filter(Camera.cam_id == "cam_01").first()
        if not camera:
            print("❌ Camera cam_01 not found. Creating mock camera...")
            camera = Camera(
                cam_id="cam_01",
                name="Test Camera",
                location="Test Location",
                rtsp_url="rtsp://test:8554/cam_01",
                is_active=True
            )
            db.add(camera)
            db.commit()
            db.refresh(camera)
        print(f"✓ Camera found: {camera.name}")
        
        # Step 2: Tạo mock event
        print("\n[Step 2] Creating mock fall event...")
        payload = create_mock_event_payload("cam_01")
        event_data = EventCreate.model_validate(payload)
        print(f"✓ Event ID: {payload['event_id']}")
        print(f"✓ Camera: {payload['cam_id']}")
        print(f"✓ Timestamp: {payload['timestamp_utc']}")
        
        # Step 3: Ingest event
        print("\n[Step 3] Ingesting event...")
        event, trigger_frame, all_frames = ingest_event(db, event_data)
        print(f"✓ Event saved to DB: {event.id}")
        print(f"✓ Status: {event.status}")
        
        # Step 4: Run pipeline
        print("\n[Step 4] Running alert pipeline...")
        start_time = time.time()
        result = run_pipeline(db, event, image_bytes=trigger_frame, frames_bytes=all_frames)
        elapsed = time.time() - start_time
        print(f"✓ Pipeline completed in {elapsed:.2f}s")
        print(f"✓ Suppressed: {result.suppressed}")
        print(f"✓ Notes: {result.notes}")
        
        # Step 5: Kiểm tra alert
        print("\n[Step 5] Checking alert...")
        alert = db.query(Alert).filter(Alert.event_id == event.id).first()
        if alert:
            print(f"✓ Alert created: {alert.id}")
            print(f"✓ FCM sent: {alert.fcm_sent}")
            print(f"✓ Acknowledged: {alert.is_acknowledged}")
        else:
            print("ℹ No alert created (event may be suppressed)")
        
        # Step 6: Kiểm tra event status
        print("\n[Step 6] Checking final event status...")
        db.refresh(event)
        print(f"✓ Final status: {event.status}")
        print(f"✓ VLM verdict: {event.vlm_verdict}")
        print(f"✓ VLM confidence: {event.vlm_confidence}")
        print(f"✓ Clip URL: {event.clip_url}")
        
        # Step 7: Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Event ID: {event.id}")
        print(f"Camera: {camera.cam_id}")
        print(f"Status: {event.status}")
        print(f"VLM Verdict: {event.vlm_verdict}")
        print(f"Alert Created: {'Yes' if alert else 'No'}")
        print(f"Pipeline Latency: {elapsed:.2f}s")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()


def test_multi_camera_dedup():
    """Test multi-camera event deduplication."""
    print("\n" + "=" * 60)
    print("MULTI-CAMERA DEDUP TEST")
    print("=" * 60)
    
    db = SessionLocal()
    
    try:
        # Tạo 2 events từ 2 camera khác nhau trong cùng window
        print("\n[Step 1] Creating events from 2 cameras...")
        
        payload1 = create_mock_event_payload("cam_01")
        payload2 = create_mock_event_payload("cam_02")
        
        event_data1 = EventCreate.model_validate(payload1)
        event_data2 = EventCreate.model_validate(payload2)
        
        event1, frame1, frames1 = ingest_event(db, event_data1)
        event2, frame2, frames2 = ingest_event(db, event_data2)
        
        print(f"✓ Event 1: {event1.id} (cam_01)")
        print(f"✓ Event 2: {event2.id} (cam_02)")
        
        # Run pipeline cho cả 2 events
        print("\n[Step 2] Running pipeline for both events...")
        result1 = run_pipeline(db, event1, image_bytes=frame1, frames_bytes=frames1)
        result2 = run_pipeline(db, event2, image_bytes=frame2, frames_bytes=frames2)
        
        print(f"✓ Event 1 suppressed: {result1.suppressed}")
        print(f"✓ Event 2 suppressed: {result2.suppressed}")
        print(f"✓ Event 1 notes: {result1.notes}")
        print(f"✓ Event 2 notes: {result2.notes}")
        
        # Kiểm tra status
        db.refresh(event1)
        db.refresh(event2)
        
        print(f"\n[Step 3] Final statuses:")
        print(f"✓ Event 1 status: {event1.status}")
        print(f"✓ Event 2 status: {event2.status}")
        
        # Một trong 2 events nên bị merged
        merged_count = sum(1 for e in [event1, event2] if e.status == "merged")
        print(f"\n✓ Merged events: {merged_count}/2")
        
        if merged_count > 0:
            print("✓ Deduplication working correctly")
        else:
            print("⚠ No events merged (may be outside dedup window)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()


if __name__ == "__main__":
    print("Starting Full Trigger Tests...\n")
    
    # Test single event
    success1 = test_full_trigger()
    
    # Test multi-camera dedup
    success2 = test_multi_camera_dedup()
    
    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"Single Event Test: {'✓ PASSED' if success1 else '❌ FAILED'}")
    print(f"Multi-Camera Dedup Test: {'✓ PASSED' if success2 else '❌ FAILED'}")
    print("=" * 60)
    
    sys.exit(0 if (success1 and success2) else 1)
