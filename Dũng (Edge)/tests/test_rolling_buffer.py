import base64
from event_builder import REQUIRED_OFFSETS
from rolling_buffer import RollingBuffer

def jpeg(tag: str) -> bytes:
    return f"jpeg-{tag}".encode("ascii")

def test_empty_buffer_returns_empty_list():
    rb = RollingBuffer()
    assert rb.get_frames_before("cam_1", 10_000) == []
    assert rb.has_frames("cam_1") is False

def test_returns_exactly_6_frames_matching_required_offsets():
    rb = RollingBuffer()
    for ts in range(7_000, 10_001, 250):
        rb.add_frame("cam_1", ts, jpeg(str(ts)))

    frames = rb.get_frames_before("cam_1", 10_000)

    assert [f["offset_ms"] for f in frames] == list(REQUIRED_OFFSETS)
    assert len(frames) == 6


def test_get_frames_before_picks_nearest_for_each_offset():
    rb = RollingBuffer()
    # Frame duy nhất tại mỗi mốc 500ms từ 7500 tới 10000.
    for ts in (7500, 8000, 8500, 9000, 9500, 10000):
        rb.add_frame("cam_1", ts, jpeg(str(ts)))

    frames = rb.get_frames_before("cam_1", 10_000)

    expected_ts = [7500, 8000, 8500, 9000, 9500, 10000]
    got_tags = [base64.b64decode(f["jpeg_b64"]).decode("ascii") for f in frames]
    assert got_tags == [f"jpeg-{ts}" for ts in expected_ts]


def test_old_frames_pruned_beyond_max_age():
    rb = RollingBuffer(max_age_ms=2000)
    rb.add_frame("cam_1", 0, jpeg("too-old"))
    rb.add_frame("cam_1", 2500, jpeg("kept"))

    frames = rb.get_frames_before("cam_1", 2500)

    for f in frames:
        assert base64.b64decode(f["jpeg_b64"]) == jpeg("kept")


def test_separate_cameras_independent():
    rb = RollingBuffer()
    rb.add_frame("cam_1", 1000, jpeg("cam1-frame"))
    rb.add_frame("cam_2", 1000, jpeg("cam2-frame"))

    assert rb.has_frames("cam_1") is True
    assert rb.has_frames("cam_2") is True
    assert rb.get_frames_before("cam_3", 1000) == []
