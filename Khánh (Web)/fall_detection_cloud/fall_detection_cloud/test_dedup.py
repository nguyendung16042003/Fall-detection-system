"""Test dedup multi-camera: 2 events từ 2 camera khác nhau trong 2s window."""

import json
import sys
import time
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import paho.mqtt.client as mqtt

from app.core.database import SessionLocal
from app.models.user import User
from app.models.camera import Camera
from app.core.security import hash_password

# Login
login_data = json.dumps({'username': 'admin', 'password': 'secret'}).encode()
req = urllib.request.Request('http://localhost:8000/api/auth/login', data=login_data, headers={'Content-Type': 'application/json'})
resp = urllib.request.urlopen(req)
login_result = json.loads(resp.read().decode())
token = login_result.get('access_token')
print("Login successful")

# MQTT setup
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_USERNAME = "guest"
MQTT_PASSWORD = "guest"

# Base timestamp
base_time = datetime.now(timezone.utc)

# Event 1 - cam_01
event_id_1 = str(uuid.uuid4())
timestamp_1 = base_time
event_data_1 = {
    "schema_version": "1.2",
    "event_id": event_id_1,
    "cam_id": "cam_01",
    "person_id": 1,
    "timestamp_utc": timestamp_1.isoformat(),
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

# Event 2 - cam_02 (0.5s sau)
event_id_2 = str(uuid.uuid4())
timestamp_2 = base_time + timedelta(seconds=0.5)
event_data_2 = {
    "schema_version": "1.2",
    "event_id": event_id_2,
    "cam_id": "cam_02",
    "person_id": 1,
    "timestamp_utc": timestamp_2.isoformat(),
    "event_type": "fall_candidate",
    "detection": {
        "class_before": "stand",
        "final_class": "lie",
        "confidence": 0.88,
        "bbox_xyxy": [100, 90, 350, 400],
        "frame_width": 1280,
        "frame_height": 720
    },
    "rule": {
        "version": "1.0",
        "trigger": "stand_to_lie",
        "transition_ms": 1700,
        "window_ms": 2000
    },
    "frames": [],
    "clip_url": "http://localhost:8000/clips/source.mp4"
}

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("✅ MQTT connected successfully")
    else:
        print(f"❌ MQTT connection failed with code {rc}")
        sys.exit(1)

def on_publish(client, userdata, mid, reason_code=None, properties=None):
    print(f"✅ MQTT message published (mid={mid})")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="test_dedup_publisher")
client.on_connect = on_connect
client.on_publish = on_publish

try:
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
    
    time.sleep(1)
    
    # Publish event 1
    print(f"\nPublishing event 1 to topic: events/cam_01/fall")
    print(f"Event ID: {event_id_1}")
    print(f"Timestamp: {timestamp_1.isoformat()}")
    client.publish("events/cam_01/fall", json.dumps(event_data_1))
    time.sleep(0.5)
    
    # Publish event 2
    print(f"\nPublishing event 2 to topic: events/cam_02/fall")
    print(f"Event ID: {event_id_2}")
    print(f"Timestamp: {timestamp_2.isoformat()}")
    client.publish("events/cam_02/fall", json.dumps(event_data_2))
    time.sleep(1)
    
    client.loop_stop()
    client.disconnect()
except Exception as e:
    print(f"❌ MQTT publish failed: {e}")
    sys.exit(1)

# Wait for consumer to process
print("\nWaiting 5 seconds for consumer to process both events...")
time.sleep(5)

# Verify both events
print("\n" + "="*60)
print("VERIFYING EVENTS")
print("="*60)

for event_id, cam_id in [(event_id_1, "cam_01"), (event_id_2, "cam_02")]:
    print(f"\nChecking {cam_id} event: {event_id}")
    req = urllib.request.Request(f'http://localhost:8000/api/events/{event_id}', headers={
        'Authorization': f'Bearer {token}'
    })
    try:
        resp = urllib.request.urlopen(req)
        event_detail = json.loads(resp.read().decode())
        print(f"Status: {event_detail.get('status')}")
        print(f"Cam ID: {event_detail.get('cam_id')}")
        if event_detail.get('note'):
            print(f"Note: {event_detail.get('note')}")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"❌ Event not found (404)")
        else:
            print(f"❌ HTTP error {e.code}: {e}")

print("\n" + "="*60)
print("DEDUP TEST COMPLETE")
print("="*60)
