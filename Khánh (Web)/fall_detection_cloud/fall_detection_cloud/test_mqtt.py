"""Test MQTT path: publish fall event via MQTT, verify it appears in DB."""

import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import paho.mqtt.client as mqtt

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
    print("Login successful")
    token = login_result.get('access_token')
except Exception as e:
    print("Login failed:", e)
    sys.exit(1)

# MQTT setup
MQTT_BROKER = "localhost"  # Local testing
MQTT_PORT = 1883
MQTT_TOPIC = "events/cam_01/fall"
MQTT_USERNAME = "guest"
MQTT_PASSWORD = "guest"

# Event payload (same schema as test_api.py)
event_id = str(uuid.uuid4())
event_data = {
    "schema_version": "1.2",
    "event_id": event_id,
    "cam_id": "cam_01",
    "person_id": 1,
    "timestamp_utc": "2026-07-25T10:20:00.123Z",
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

# Publish via MQTT
print(f"\nPublishing event via MQTT to topic: {MQTT_TOPIC}")
print(f"Event ID: {event_id}")

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("✅ MQTT connected successfully")
    else:
        print(f"❌ MQTT connection failed with code {rc}")
        sys.exit(1)

def on_publish(client, userdata, mid):
    print(f"✅ MQTT message published (mid={mid})")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="test_mqtt_publisher")
client.on_connect = on_connect
client.on_publish = on_publish

try:
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
    
    # Wait for connection
    time.sleep(1)
    
    # Publish
    client.publish(MQTT_TOPIC, json.dumps(event_data))
    time.sleep(1)
    client.loop_stop()
    client.disconnect()
except Exception as e:
    print(f"❌ MQTT publish failed: {e}")
    sys.exit(1)

# Wait for consumer to process
print("Waiting 5 seconds for consumer to process...")
time.sleep(5)

# Verify via GET /api/events/{event_id}
print(f"\nVerifying event via GET /api/events/{event_id}")
req = urllib.request.Request(f'http://localhost:8000/api/events/{event_id}', headers={
    'Authorization': f'Bearer {token}'
})
try:
    resp = urllib.request.urlopen(req)
    event_detail = json.loads(resp.read().decode())
    print("✅ Event found in database via GET /api/events/{event_id}")
    print(f"Event ID: {event_detail.get('event_id')}")
    print(f"Cam ID: {event_detail.get('cam_id')}")
    print(f"Status: {event_detail.get('status')}")
    print(f"Clip URL: {event_detail.get('clip_url')}")
    
    # Verify it's the correct event
    if event_detail.get('event_id') == event_id:
        print("\n✅✅✅ MQTT TEST PASSED ✅✅✅")
        print("Event successfully published via MQTT and saved to database")
    else:
        print("\n❌ MQTT TEST FAILED - Event ID mismatch")
        sys.exit(1)
except urllib.error.HTTPError as e:
    if e.code == 404:
        print(f"❌ MQTT TEST FAILED - Event not found in database (404)")
        print("Event was published via MQTT but consumer did not save it")
        sys.exit(1)
    else:
        print(f"❌ HTTP error {e.code}: {e}")
        sys.exit(1)
except Exception as e:
    print(f"❌ Verification failed: {e}")
    sys.exit(1)
