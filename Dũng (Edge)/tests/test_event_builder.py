"""Test event_builder — payload khớp mqtt_schema v3 (1.2), không cần broker."""
import uuid

import pytest

from event_builder import (
    build_fall_event,
    build_telemetry,
    fall_topic,
    telemetry_topic,
    to_iso_z,
    REQUIRED_OFFSETS,
)
from fall_detector import FallCandidate


def make_candidate(**over) -> FallCandidate:
    base = dict(
        cam_id="cam_01", person_id=3, class_before="stand", final_class="lie",
        confidence=0.92, trigger="stand_to_lie", transition_ms=1800, window_ms=2000,
        ts_ms=123456, bbox_xyxy=(120, 80, 380, 420), frame_width=1280, frame_height=720,
    )
    base.update(over)
    return FallCandidate(**base)


def make_frames():
    return [{"offset_ms": o, "jpeg_b64": "AAAA"} for o in REQUIRED_OFFSETS]


def test_build_fall_event_has_schema_keys():
    ev = build_fall_event(make_candidate(), make_frames())
    assert ev["schema_version"] == "1.2"
    assert ev["event_type"] == "fall_candidate"
    assert ev["cam_id"] == "cam_01"
    assert set(ev["detection"]) == {
        "class_before", "final_class", "confidence", "bbox_xyxy",
        "frame_width", "frame_height",
    }
    assert set(ev["rule"]) == {"version", "trigger", "transition_ms", "window_ms"}
    assert len(ev["frames"]) == 6
    assert ev["detection"]["bbox_xyxy"] == [120, 80, 380, 420]


def test_event_id_is_valid_uuid_when_not_provided():
    ev = build_fall_event(make_candidate(), make_frames())
    uuid.UUID(ev["event_id"])  # raises nếu không hợp lệ


def test_event_id_injectable():
    ev = build_fall_event(make_candidate(), make_frames(), event_id="fixed-id")
    assert ev["event_id"] == "fixed-id"


def test_timestamp_ends_with_z():
    ev = build_fall_event(make_candidate(), make_frames())
    assert ev["timestamp_utc"].endswith("Z")
    assert "T" in ev["timestamp_utc"]


def test_wrong_frame_count_raises():
    with pytest.raises(ValueError):
        build_fall_event(make_candidate(), make_frames()[:5])


def test_wrong_offsets_raise():
    bad = [{"offset_ms": o, "jpeg_b64": "AAAA"} for o in (0, 500, 1000, 1500, 2000, 2500)]
    with pytest.raises(ValueError):
        build_fall_event(make_candidate(), bad)


def test_invalid_trigger_raises():
    with pytest.raises(ValueError):
        build_fall_event(make_candidate(trigger="jump_to_lie"), make_frames())


def test_sustained_lying_trigger_with_unknown_class_before_accepted():
    ev = build_fall_event(
        make_candidate(trigger="sustained_lying", class_before="unknown"),
        make_frames(),
    )
    assert ev["rule"]["trigger"] == "sustained_lying"
    assert ev["detection"]["class_before"] == "unknown"


def test_confidence_out_of_range_raises():
    with pytest.raises(ValueError):
        build_fall_event(make_candidate(confidence=1.5), make_frames())


def test_bad_cam_id_raises():
    with pytest.raises(ValueError):
        build_fall_event(make_candidate(cam_id="camera1"), make_frames())


def test_topics():
    assert fall_topic("cam_01") == "events/cam_01/fall"
    assert telemetry_topic("cam_02") == "telemetry/cam_02/status"


def test_build_telemetry_keys():
    t = build_telemetry(
        "cam_01",
        pipeline={"state": "running", "fps_pgie": 15.2, "fps_sgie": 12.8, "active_tracks": 2},
        system={"ram_used_mb": 2048},
        network={"mqtt_connected": True},
    )
    assert t["cam_id"] == "cam_01"
    assert t["pipeline"]["state"] == "running"
    assert t["timestamp_utc"].endswith("Z")
