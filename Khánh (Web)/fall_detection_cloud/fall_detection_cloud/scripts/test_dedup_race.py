"""Test to reproduce and verify race condition fix in dedup.

This test sends 2 events simultaneously from different cameras in the same area
with timestamps <2s apart to trigger the race condition scenario.
"""

import asyncio
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.database import SessionLocal
from app.models.event import Event
from app.models.camera import Camera
from app.schemas.event import EventCreate
from app.services.alert_pipeline import run_pipeline
from app.services.event_ingest import CameraNotFoundError, ingest_event
from app.services.dedup import compute_lock_key, acquire_dedup_lock


def create_mock_event_payload(cam_id: str, timestamp_offset: float = 0.0) -> dict:
    """Create mock fall event payload."""
    return {
        "schema_version": "1.2",
        "event_id": str(uuid.uuid4()),
        "cam_id": cam_id,
        "person_id": 1,
        "timestamp_utc": (datetime.now(timezone.utc).timestamp() + timestamp_offset),
        "event_type": "fall_candidate",
        "detection": {
            "class_before": "stand",
            "final_class": "lie",
            "confidence": 0.92,
            "bbox_xyxy": [120, 80, 380, 420],
            "frame_width": 1280,
            "frame_height": 720
        },
        "rule": {
            "version": "1.0",
            "trigger": "stand_to_lie",
            "transition_ms": 1800,
            "window_ms": 2000
        },
        "frames": []
    }


def test_no_double_notification():
    """
    Test that 2 concurrent events from different cameras in same area
    with timestamps <2s apart result in only 1 VLM call and 1 alert.
    """
    print("=" * 60)
    print("TEST DEDUP RACE CONDITION")
    print("=" * 60)
    
    db = SessionLocal()
    
    try:
        # Ensure cameras exist
        cam1 = db.query(Camera).filter(Camera.cam_id == "cam_01").first()
        cam2 = db.query(Camera).filter(Camera.cam_id == "cam_02").first()
        
        if not cam1:
            print("Creating cam_01...")
            cam1 = Camera(
                cam_id="cam_01",
                name="Test Camera 1",
                location="Test Location",
                rtsp_url="rtsp://test:8554/cam_01",
                is_active=True
            )
            db.add(cam1)
            db.commit()
            db.refresh(cam1)
        
        if not cam2:
            print("Creating cam_02...")
            cam2 = Camera(
                cam_id="cam_02",
                name="Test Camera 2",
                location="Test Location",
                rtsp_url="rtsp://test:8554/cam_02",
                is_active=True
            )
            db.add(cam2)
            db.commit()
            db.refresh(cam2)
        
        # Clear existing events
        db.query(Event).filter(Event.camera_id.in_([cam1.id, cam2.id])).delete()
        db.commit()
        
        # Create 2 events with timestamps <2s apart
        print("\n[Step 1] Creating 2 concurrent events...")
        event1_data = EventCreate.model_validate(create_mock_event_payload("cam_01", 0.0))
        event2_data = EventCreate.model_validate(create_mock_event_payload("cam_02", 0.5))
        
        # Ingest events
        print("[Step 2] Ingesting events...")
        event1, _, _ = ingest_event(db, event1_data)
        event2, _, _ = ingest_event(db, event2_data)
        
        print(f"Event 1: id={event1.id}, camera_id={event1.camera_id}, timestamp={event1.timestamp_utc}")
        print(f"Event 2: id={event2.id}, camera_id={event2.camera_id}, timestamp={event2.timestamp_utc}")
        
        # Process events concurrently (simulating race condition)
        print("[Step 3] Processing events concurrently...")
        
        def process_event(event_id):
            try:
                db_session = SessionLocal()
                event = db_session.query(Event).filter(Event.id == event_id).first()
                if event:
                    result = run_pipeline(db_session, event)
                db_session.close()
                return result
            except Exception as e:
                print(f"Error processing event {event_id}: {e}")
                return None
        
        # Run in threads to simulate concurrent processing
        import threading
        t1 = threading.Thread(target=process_event, args=(event1.id,))
        t2 = threading.Thread(target=process_event, args=(event2.id,))
        
        t1.start()
        t2.start()
        
        t1.join()
        t2.join()
        
        # Check results
        print("[Step 4] Checking results...")
        db.refresh(event1)
        db.refresh(event2)
        
        confirmed_count = db.query(Event).filter(
            Event.camera_id.in_([cam1.id, cam2.id]),
            Event.status == "confirmed"
        ).count()
        
        merged_count = db.query(Event).filter(
            Event.camera_id.in_([cam1.id, cam2.id]),
            Event.status == "merged"
        ).count()
        
        total_count = db.query(Event).filter(
            Event.camera_id.in_([cam1.id, cam2.id])
        ).count()
        
        print(f"\nResults:")
        print(f"  Total events: {total_count}")
        print(f"  Confirmed events: {confirmed_count}")
        print(f"  Merged events: {merged_count}")
        
        # Verify dedup worked correctly
        if confirmed_count == 1 and merged_count == 1:
            print("\n✅ TEST PASSED: Dedup working correctly - only 1 VLM call")
            return True
        elif confirmed_count == 2:
            print("\n❌ TEST FAILED: Race condition detected - 2 VLM calls made")
            print("   This means both events were processed independently")
            return False
        else:
            print(f"\n⚠️  TEST UNCERTAIN: Unexpected state - confirmed={confirmed_count}, merged={merged_count}")
            return False
            
    finally:
        db.close()


if __name__ == "__main__":
    success = test_no_double_notification()
    exit(0 if success else 1)
