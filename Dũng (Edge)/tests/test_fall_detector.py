"""15 test cho fall_detector — chạy KHÔNG cần camera/GPU: python3 -m pytest tests/ -v

Phủ: 3 kịch bản bắt buộc (ngã / nằm sẵn / ngồi xuống) + chống nhiễu + confidence
thấp + chuyển chậm + cooldown + TTL + đa person/đa camera + tính đúng transition_ms.
"""
import pytest

from fall_detector import FallDetector, FallRules, load_rules

STAND, SIT, LIE = "stand", "sit", "lie"


def make_detector(**overrides) -> FallDetector:
    base = dict(
        version="1.0", window_ms=2000, confirm_frames=3, min_confidence=0.60,
        cooldown_ms=10000, person_ttl_ms=5000,
        class_before_enum=("stand", "sit"), fall_class="lie",
        min_lying_aspect_ratio=0.7, min_sustained_lying_ms=1500,
    )
    base.update(overrides)
    return FallDetector(FallRules(**base))


# bbox mặc định cho test: w=140,h=150 -> aspect=0.933 (>= min_lying_aspect_ratio
# mặc định 0.7) để các test cũ (không nhắm riêng vào aspect filter) không bị chặn.
DEFAULT_BBOX = (10, 20, 150, 170)


def feed(det, frames, cam="cam_01", pid=3, bbox=DEFAULT_BBOX):
    """frames: list (ts_ms, cls, conf). Trả list các FallCandidate được bắn ra."""
    out = []
    for ts, cls, conf in frames:
        cand = det.update(cam, pid, cls, conf, ts,
                          bbox=bbox, frame_width=1280, frame_height=720)
        if cand is not None:
            out.append(cand)
    return out


# 1
def test_stand_to_lie_triggers_fall():
    det = make_detector()
    out = feed(det, [(0, STAND, .9), (200, STAND, .9),
                     (400, LIE, .9), (600, LIE, .9), (800, LIE, .9)])
    assert len(out) == 1
    assert out[0].class_before == STAND
    assert out[0].final_class == LIE
    assert out[0].trigger == "stand_to_lie"


# 2
def test_sit_to_lie_triggers_fall():
    det = make_detector()
    out = feed(det, [(0, SIT, .9), (200, SIT, .9),
                     (400, LIE, .9), (600, LIE, .9), (800, LIE, .9)])
    assert len(out) == 1
    assert out[0].trigger == "sit_to_lie"
    assert out[0].class_before == SIT


# 3 — nằm sẵn: không có stand/sit trước đó
def test_already_lying_no_trigger():
    det = make_detector()
    out = feed(det, [(0, LIE, .9), (200, LIE, .9), (400, LIE, .9), (600, LIE, .9)])
    assert out == []


# 4 — ngồi xuống: stand -> sit, không có lie
def test_sitting_down_no_trigger():
    det = make_detector()
    out = feed(det, [(0, STAND, .9), (200, STAND, .9),
                     (400, SIT, .9), (600, SIT, .9), (800, SIT, .9)])
    assert out == []


# 5 — nhiễu 1 frame lie giữa các frame stand
def test_single_lie_frame_noise_no_trigger():
    det = make_detector()
    out = feed(det, [(0, STAND, .9), (200, LIE, .9),
                     (400, STAND, .9), (600, STAND, .9)])
    assert out == []


# 6 — lie nhưng confidence dưới ngưỡng
def test_low_confidence_lie_no_trigger():
    det = make_detector()
    out = feed(det, [(0, STAND, .9), (200, STAND, .9),
                     (400, LIE, .5), (600, LIE, .5), (800, LIE, .5)])
    assert out == []


# 7 — chuyển quá chậm (vượt window_ms)
def test_slow_transition_no_trigger():
    det = make_detector()
    out = feed(det, [(0, STAND, .9),
                     (2500, LIE, .9), (2700, LIE, .9), (2900, LIE, .9)])
    assert out == []


# 8 — cooldown chặn lần ngã thứ hai ngay sau đó
def test_cooldown_suppresses_second_fall():
    det = make_detector(cooldown_ms=10000)
    out = feed(det, [(0, STAND, .9), (200, STAND, .9),
                     (400, LIE, .9), (600, LIE, .9), (800, LIE, .9),   # trigger @800
                     (1000, STAND, .9),                                # mở latch
                     (1200, LIE, .9), (1400, LIE, .9), (1600, LIE, .9)])  # bị cooldown
    assert len(out) == 1
    assert out[0].ts_ms == 800


# 9 — hết cooldown thì được bắn lại
def test_after_cooldown_triggers_again():
    det = make_detector(cooldown_ms=1000)
    out = feed(det, [(0, STAND, .9), (200, STAND, .9),
                     (400, LIE, .9), (600, LIE, .9), (800, LIE, .9),   # trigger @800
                     (1000, STAND, .9),
                     (1900, LIE, .9), (2100, LIE, .9), (2300, LIE, .9)])  # trigger @2300
    assert len(out) == 2
    assert out[1].ts_ms == 2300


# 10 — mất dấu quá TTL rồi thấy đang nằm: không báo giả
def test_ttl_reset_no_false_trigger():
    det = make_detector(person_ttl_ms=5000)
    out = feed(det, [(0, STAND, .9), (200, STAND, .9),
                     (400, LIE, .9), (600, LIE, .9), (800, LIE, .9),   # trigger @800
                     (6000, LIE, .9), (6200, LIE, .9), (6400, LIE, .9)])  # sau reset, nằm sẵn
    assert len(out) == 1


# 11 — transition_ms đo đúng
def test_transition_ms_measured_correctly():
    det = make_detector()
    out = feed(det, [(100, STAND, .9), (200, STAND, .9),
                     (1000, LIE, .9), (1200, LIE, .9), (1400, LIE, .9)])
    assert len(out) == 1
    assert out[0].transition_ms == 800  # 1000 - 200


# 12 — latch chặn báo lặp trong cùng một episode nằm
def test_latch_single_trigger_per_episode():
    det = make_detector()
    out = feed(det, [(0, STAND, .9), (200, STAND, .9),
                     (400, LIE, .9), (600, LIE, .9), (800, LIE, .9),
                     (1000, LIE, .9), (1200, LIE, .9)])  # vẫn nằm, KHÔNG báo lại
    assert len(out) == 1


# 13 — hai người trong cùng cam độc lập
def test_two_persons_independent():
    det = make_detector()
    # pid=3 ngã
    feed(det, [(0, STAND, .9), (200, STAND, .9),
               (400, LIE, .9), (600, LIE, .9), (800, LIE, .9)], pid=3)
    # pid=7 nằm sẵn -> không báo
    out7 = feed(det, [(0, LIE, .9), (200, LIE, .9), (400, LIE, .9)], pid=7)
    assert out7 == []


# 14 — hai camera độc lập
def test_two_cameras_independent():
    det = make_detector()
    out1 = feed(det, [(0, STAND, .9), (200, STAND, .9),
                      (400, LIE, .9), (600, LIE, .9), (800, LIE, .9)], cam="cam_01")
    out2 = feed(det, [(0, LIE, .9), (200, LIE, .9), (400, LIE, .9)], cam="cam_02")
    assert len(out1) == 1
    assert out2 == []


# 15 — bbox + kích thước khung được mang theo trong candidate
def test_candidate_carries_bbox_and_frame_dims():
    det = make_detector()
    out = feed(det, [(0, STAND, .9), (200, STAND, .9),
                     (400, LIE, .95), (600, LIE, .95), (800, LIE, .95)])
    assert len(out) == 1
    c = out[0]
    assert c.bbox_xyxy == DEFAULT_BBOX
    assert c.frame_width == 1280 and c.frame_height == 720
    assert c.confidence == 0.95


# 16 — bbox chưa đủ "dẹt ngang" (aspect < min_lying_aspect_ratio) -> 'lie' bị bỏ
# qua hẳn, không trigger dù confidence cao + lặp nhiều frame (port fall_rule.py,
# chống nhầm "ngồi ngả lưng" thành nằm).
def test_low_aspect_ratio_lie_no_trigger():
    det = make_detector()
    tall_bbox = (10, 20, 90, 220)  # w=80,h=200 -> aspect=0.4 < 0.7
    out = feed(det, [(0, STAND, .9), (200, STAND, .9),
                     (400, LIE, .95), (600, LIE, .95), (800, LIE, .95)],
               bbox=tall_bbox)
    assert out == []


# 17 — sustained_lying: nằm sẵn từ đầu (không có mỏ neo), nhưng đủ lâu vẫn báo
def test_sustained_lying_triggers_without_anchor():
    det = make_detector()
    out = feed(det, [(0, LIE, .9), (200, LIE, .9), (400, LIE, .9),
                     (600, LIE, .9), (1600, LIE, .9)])
    assert len(out) == 1
    assert out[0].trigger == "sustained_lying"
    assert out[0].class_before == "unknown"
    assert out[0].transition_ms == 1600  # ts_ms - lie_start_ts (0)


# 18 — nằm sẵn nhưng CHƯA đủ min_sustained_lying_ms -> vẫn không báo (test #3
# cũ vẫn còn xanh, đây thêm biên ngay dưới ngưỡng)
def test_already_lying_below_sustained_threshold_no_trigger():
    det = make_detector()
    out = feed(det, [(0, LIE, .9), (200, LIE, .9), (400, LIE, .9), (600, LIE, .9)])
    assert out == []  # 600ms < min_sustained_lying_ms=1500ms


# bonus — file cấu hình thật parse được và đúng giá trị đã chốt
def test_load_rules_from_yaml():
    rules = load_rules()
    assert rules.window_ms == 3000
    assert rules.confirm_frames == 4
    assert rules.fall_class == "lie"
    assert rules.class_before_enum == ("stand", "sit")
    assert rules.min_lying_aspect_ratio == 0.7
    assert rules.min_sustained_lying_ms == 1500
