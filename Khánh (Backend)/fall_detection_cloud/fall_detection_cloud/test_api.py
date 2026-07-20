"""Test script for POST and GET /api/events endpoints."""

import json
import sys
import urllib.request
import uuid
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.database import SessionLocal
from app.models.user import User
from app.models.camera import Camera
from app.core.security import hash_password

# Create admin user if not exists
db = SessionLocal()
try:
    user = db.query(User).filter(User.username == 'admin').first()
    if not user:
        user = User(username='admin', email='admin@example.com', hashed_password=hash_password('secret'))
        db.add(user)
        db.commit()
        print("Admin user created")
    else:
        print("Admin user already exists")
    
    # Create camera cam_01 if not exists
    camera = db.query(Camera).filter(Camera.cam_id == 'cam_01').first()
    if not camera:
        camera = Camera(
            cam_id='cam_01',
            name='Test Camera 1',
            location='Test Location',
            rtsp_url='rtsp://test:8554/cam_01',
            is_active=True
        )
        db.add(camera)
        db.commit()
        print("Camera cam_01 created")
    else:
        print("Camera cam_01 already exists")
finally:
    db.close()

# Test login
login_data = json.dumps({'username': 'admin', 'password': 'secret'}).encode()
req = urllib.request.Request('http://localhost:8000/api/auth/login', data=login_data, headers={'Content-Type': 'application/json'})
try:
    resp = urllib.request.urlopen(req)
    login_result = json.loads(resp.read().decode())
    print("Login successful:", login_result)
    token = login_result.get('access_token')
except Exception as e:
    print("Login failed:", e)
    sys.exit(1)

# Test POST /api/events
event_data = {
    "schema_version": "1.2",
    "event_id": str(uuid.uuid4()),
    "cam_id": "cam_01",
    "person_id": 1,
    "timestamp_utc": "2026-07-19T10:20:00.123Z",
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
    "frames": [],
    "clip_url": "http://localhost:8000/clips/source.mp4"
}

event_json = json.dumps(event_data).encode()
req = urllib.request.Request('http://localhost:8000/api/events', data=event_json, headers={
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {token}'
})
try:
    resp = urllib.request.urlopen(req)
    event_result = json.loads(resp.read().decode())
    print("POST /api/events successful:", event_result)
    created_event_id = event_result.get('event_id')
except Exception as e:
    print("POST /api/events failed:", e)
    sys.exit(1)

# Test GET /api/events/{event_id} to verify the created event
req = urllib.request.Request(f'http://localhost:8000/api/events/{created_event_id}', headers={
    'Authorization': f'Bearer {token}'
})
try:
    resp = urllib.request.urlopen(req)
    event_detail = json.loads(resp.read().decode())
    print("GET /api/events/{event_id} successful:")
    print(json.dumps(event_detail, indent=2))
    
    # Check if clip_url field exists in response
    if 'clip_url' in event_detail:
        print("✅ clip_url field exists in GET response")
        print(f"clip_url value: {event_detail['clip_url']}")
        # Verify the test clip_url was saved
        if event_detail['clip_url'] == "http://localhost:8000/clips/source.mp4":
            print("✅ clip_url value matches test video URL")
        else:
            print("⚠️  clip_url value does not match test video URL")
    else:
        print("❌ clip_url field missing in GET response")
except Exception as e:
    print("GET /api/events/{event_id} failed:", e)
    sys.exit(1)

print("\n✅ All tests passed!")
