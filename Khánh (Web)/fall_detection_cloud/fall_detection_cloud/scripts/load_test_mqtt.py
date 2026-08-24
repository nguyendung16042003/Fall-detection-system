"""Load test script for MQTT events - 100 concurrent events.

Tests dedup race condition by sending events from cam_01/cam_02 in same area
with timestamps <2s apart.
"""

import asyncio
import json
import random
import time
import uuid
from datetime import datetime, timezone

try:
    from gmqtt import Client as MQTTClient
except ImportError:
    print("gmqtt not installed, falling back to paho-mqtt")
    import paho.mqtt.client as mqtt

BROKER = "localhost"  # hoặc IP public nhà Khánh
PORT = 1883
N_EVENTS = 100


async def publish_event(client, cam_id, area_id, ts_offset):
    """Publish a fall event to MQTT broker."""
    event = {
        "schema_version": "1.2",
        "event_id": str(uuid.uuid4()),
        "cam_id": cam_id,
        "person_id": random.randint(1, 20),
        "timestamp_utc": (datetime.now(timezone.utc).timestamp() + ts_offset),
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
        "frames": []  # Empty for load test to reduce payload size
    }
    topic = f"events/{cam_id}/fall"
    
    if hasattr(client, 'publish'):
        # gmqtt
        client.publish(topic, json.dumps(event), qos=1)
    else:
        # paho-mqtt
        client.publish(topic, json.dumps(event), qos=1)


async def main_gmqtt():
    """Main function using gmqtt."""
    client = MQTTClient("load-tester")
    await client.connect(BROKER, PORT)
    
    tasks = []
    for i in range(N_EVENTS):
        # Cố tình cho 1 cặp cam_01/cam_02 cùng area, lệch nhau <2s để test dedup
        cam_id = "cam_01" if i % 2 == 0 else "cam_02"
        tasks.append(publish_event(client, cam_id, area_id="zone_A", ts_offset=random.uniform(0, 1.8)))
    
    start = time.time()
    await asyncio.gather(*tasks)
    elapsed = time.time() - start
    print(f"Bắn xong {N_EVENTS} event trong {elapsed:.2f}s")
    print(f"Rate: {N_EVENTS/elapsed:.2f} events/s")
    
    await client.disconnect()


def main_paho():
    """Main function using paho-mqtt (synchronous)."""
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="load-tester")
    client.connect(BROKER, PORT, 60)
    client.loop_start()
    
    # Wait for connection
    time.sleep(1)
    
    tasks = []
    for i in range(N_EVENTS):
        cam_id = "cam_01" if i % 2 == 0 else "cam_02"
        tasks.append(publish_event(client, cam_id, area_id="zone_A", ts_offset=random.uniform(0, 1.8)))
    
    start = time.time()
    # Run tasks synchronously for paho-mqtt
    for task in tasks:
        asyncio.run(task)
    elapsed = time.time() - start
    print(f"Bắn xong {N_EVENTS} event trong {elapsed:.2f}s")
    print(f"Rate: {N_EVENTS/elapsed:.2f} events/s")
    
    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main_gmqtt())
    except:
        print("Falling back to paho-mqtt")
        main_paho()
