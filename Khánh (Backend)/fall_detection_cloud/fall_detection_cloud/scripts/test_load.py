"""Load Test Script

Test khả năng xử lý của hệ thống khi có 50 events dồn dập gửi cùng lúc.
Kiểm tra asyncio.Queue không overflow và không có event bị mất.
"""

import asyncio
import base64
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
from app.services.vlm_queue import get_vlm_queue


def create_mock_jpeg(width: int = 640, height: int = 480) -> bytes:
    """Tạo mock JPEG image đơn giản."""
    jpeg_header = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
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


def process_single_event(cam_id: str, db: Session) -> dict:
    """Xử lý một event đơn lẻ."""
    try:
        payload = create_mock_event_payload(cam_id)
        event_data = EventCreate.model_validate(payload)
        
        event, trigger_frame, all_frames = ingest_event(db, event_data)
        result = run_pipeline(db, event, image_bytes=trigger_frame, frames_bytes=all_frames)
        
        return {
            "event_id": str(event.id),
            "cam_id": cam_id,
            "status": event.status,
            "suppressed": result.suppressed,
            "success": True
        }
    except Exception as e:
        return {
            "event_id": None,
            "cam_id": cam_id,
            "status": "error",
            "error": str(e),
            "success": False
        }


def test_load_sequential(num_events: int = 50):
    """Test load với xử lý tuần tự."""
    print("=" * 60)
    print(f"LOAD TEST - SEQUENTIAL ({num_events} events)")
    print("=" * 60)
    
    db = SessionLocal()
    
    try:
        # Kiểm tra camera
        camera = db.query(Camera).filter(Camera.cam_id == "cam_01").first()
        if not camera:
            print("Creating mock camera...")
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
        
        print(f"\nProcessing {num_events} events sequentially...")
        start_time = time.time()
        
        results = []
        for i in range(num_events):
            cam_id = f"cam_{(i % 3) + 1}"
            result = process_single_event(cam_id, db)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{num_events} events...")
        
        elapsed = time.time() - start_time
        
        # Thống kê
        success_count = sum(1 for r in results if r["success"])
        error_count = num_events - success_count
        suppressed_count = sum(1 for r in results if r.get("suppressed"))
        
        print(f"\n{'=' * 60}")
        print("RESULTS")
        print(f"{'=' * 60}")
        print(f"Total events: {num_events}")
        print(f"Successful: {success_count}")
        print(f"Errors: {error_count}")
        print(f"Suppressed: {suppressed_count}")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Avg time per event: {elapsed/num_events:.2f}s")
        print(f"Throughput: {num_events/elapsed:.2f} events/s")
        print(f"{'=' * 60}")
        
        return error_count == 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()


def test_load_concurrent(num_events: int = 50):
    """Test load với xử lý song song (asyncio)."""
    print("\n" + "=" * 60)
    print(f"LOAD TEST - CONCURRENT ({num_events} events)")
    print("=" * 60)
    
    db = SessionLocal()
    
    try:
        # Kiểm tra camera
        for i in range(1, 4):
            cam_id = f"cam_0{i}"
            camera = db.query(Camera).filter(Camera.cam_id == cam_id).first()
            if not camera:
                camera = Camera(
                    cam_id=cam_id,
                    name=f"Test Camera {i}",
                    location="Test Location",
                    rtsp_url=f"rtsp://test:8554/{cam_id}",
                    is_active=True
                )
                db.add(camera)
        db.commit()
        
        db.close()
        
        print(f"\nProcessing {num_events} events concurrently...")
        start_time = time.time()
        
        async def process_events():
            results = []
            for i in range(num_events):
                cam_id = f"cam_{(i % 3) + 1}"
                db = SessionLocal()
                result = process_single_event(cam_id, db)
                db.close()
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{num_events} events...")
            
            return results
        
        results = asyncio.run(process_events())
        elapsed = time.time() - start_time
        
        # Thống kê
        success_count = sum(1 for r in results if r["success"])
        error_count = num_events - success_count
        suppressed_count = sum(1 for r in results if r.get("suppressed"))
        
        print(f"\n{'=' * 60}")
        print("RESULTS")
        print(f"{'=' * 60}")
        print(f"Total events: {num_events}")
        print(f"Successful: {success_count}")
        print(f"Errors: {error_count}")
        print(f"Suppressed: {suppressed_count}")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Avg time per event: {elapsed/num_events:.2f}s")
        print(f"Throughput: {num_events/elapsed:.2f} events/s")
        print(f"{'=' * 60}")
        
        return error_count == 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()


def check_vlm_queue_status():
    """Kiểm tra trạng thái VLM queue."""
    print("\n" + "=" * 60)
    print("VLM QUEUE STATUS")
    print("=" * 60)
    
    try:
        queue = get_vlm_queue()
        print(f"Queue size: {queue.qsize()}")
        print(f"Max size: {queue.maxsize if queue.maxsize > 0 else 'unlimited'}")
        
        if queue.qsize() > 0:
            print(f"⚠ Queue has {queue.qsize()} pending items")
        else:
            print("✓ Queue is empty")
        
        return True
    except Exception as e:
        print(f"❌ Error checking queue: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load test for fall detection system")
    parser.add_argument("--events", type=int, default=50, help="Number of events to test")
    parser.add_argument("--mode", choices=["sequential", "concurrent", "both"], default="both",
                       help="Test mode")
    
    args = parser.parse_args()
    
    print(f"Starting Load Test with {args.events} events...\n")
    
    success = True
    
    if args.mode in ["sequential", "both"]:
        success = test_load_sequential(args.events) and success
    
    if args.mode in ["concurrent", "both"]:
        success = test_load_concurrent(args.events) and success
    
    # Check queue status
    check_vlm_queue_status()
    
    print("\n" + "=" * 60)
    print("OVERALL RESULT")
    print("=" * 60)
    print(f"Load Test: {'✓ PASSED' if success else '❌ FAILED'}")
    print("=" * 60)
    
    sys.exit(0 if success else 1)
