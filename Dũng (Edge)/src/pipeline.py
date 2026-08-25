#!/usr/bin/env python3
"""
Fall Detection Pipeline - Week 3 Baseline
=========================================
2 RTSP sources → nvstreammux → PGIE (YOLOv8n) → ByteTrack → SGIE (YOLOv8n-cls)
→ probe (in metadata + FPS ra console) → fakesink

Chạy:
    cd ~/fall_detection-dungha25/src
    export $(grep -v '^#' ../.env | xargs)
    python3 pipeline.py
"""

import sys
import os
import time
import signal
import logging
import threading
from collections import defaultdict
from typing import Optional

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

import pyds

# cv2/numpy PHẢI import ở main thread, TRƯỚC khi pipeline chạy (không phải lazy-import
# trong probe callback) — bug static-TLS aarch64: dlopen libgomp (OpenCV dùng OpenMP) từ
# thread phụ (GStreamer streaming thread) sau khi TLS block của main thread đã bị pyds/
# GStreamer chiếm hết chỗ -> "cannot allocate memory in static TLS block". Import sớm ở
# main thread thì loader còn dư chỗ, không lỗi.
import cv2
import numpy as np

# Module logic ngã — thuần python, test được không cần GPU (xem tests/)
from fall_detector import FallDetector, load_rules
from event_builder import build_fall_event, build_telemetry, placeholder_frames
from mqtt_publisher import MqttPublisher
from rolling_buffer import RollingBuffer
from identity_association import IdentityAssociator
from boundary_feature_fusion import BoundaryFusionEngine
from reid import ReIDExtractorTrt, PersonGallery, TrackEmbeddingBuffer

log = logging.getLogger("pipeline")

# ============================================================
# CONFIG
# ============================================================
# Tự suy gốc project từ vị trí file này (src/pipeline.py) -> KHÔNG hardcode path.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PGIE_CONFIG = os.environ.get("PGIE_CONFIG", f"{PROJECT_ROOT}/configs/pgie_yolov8n.txt")
# SGIE v2 (2026-08-19): pose_classifier.onnx của Tấn Dũng (Handoff_for_Edge
# 2026-08-08), 5 lớp (bỏ half_person), cần bbox phồng +20% trước khi crop — xem
# PAD_RATIO/pad_bbox_for_sgie_probe bên dưới + CLAUDE.md mục "SGIE v2". Bản 6 lớp
# cũ (sgie_yolov8n_cls.txt) VẪN CÒN, dùng lại qua SGIE_CONFIG=... nếu cần rollback.
SGIE_CONFIG = os.environ.get("SGIE_CONFIG", f"{PROJECT_ROOT}/configs/sgie_yolov8n_cls_v2.txt")
# Máy Nano 4GB chạy DS6.0 -> dùng tracker_nvdcf_ds6.yml (bản NVIDIA khuyến nghị
# cho DS6.0/Jetson Nano). tracker_nvsort.yml (bản cũ, cho Orin/DS7.0) chứa param
# riêng của NvMultiObjectTracker DS7.0 mà thư viện tracker DS6.0 không nhận diện
# được -> tracker âm thầm loại bỏ 100% object mỗi frame (PGIE detect đúng nhưng
# track/SGIE luôn rỗng, không throw lỗi gì). Xem chú thích đầy đủ trong
# configs/tracker_nvdcf_ds6.yml. Override qua env để thử tracker khác/quay lại
# Orin.
TRACKER_CONFIG = os.environ.get("TRACKER_CONFIG_OVERRIDE",
    f"{PROJECT_ROOT}/configs/tracker_nvdcf_ds6.yml")

# Identity Association đa camera (2026-08-20) — TẮT MẶC ĐỊNH (kiến trúc P2, chạy
# SONG SONG với rule single-cam đã verify, không thay thế). Bật thử nghiệm bằng
# ENABLE_MULTICAM=1. ⚠️ Dùng calib MẪU (phòng test Tấn Dũng, không phải phòng
# thật) — xem identity_association.py đầu file, không dùng số liệu ra quyết định
# thật. IA_HOMOGRAPHY_PATH trỏ sang file calib thật khi có.
ENABLE_MULTICAM = os.environ.get("ENABLE_MULTICAM") == "1"
IA_HOMOGRAPHY_PATH = os.environ.get("IA_HOMOGRAPHY_PATH")  # None -> dùng mặc định (mẫu)
# CHỈ để TEST luồng end-to-end khi calib sai phòng khiến gần như không bao giờ
# ghép cặp được (xem lịch sử chat 2026-08-21) — nới ngưỡng ép ra ít nhất 1 cặp,
# xác nhận dây chuyền Module 2 -> Module 3 sống thật trên camera thật. KHÔNG
# dùng giá trị này cho production (mặc định None -> IdentityAssociator tự dùng
# DEFAULT_DIST_THRESHOLD_M=0.5m đúng như thiết kế).
IA_DIST_THRESHOLD_M = os.environ.get("IA_DIST_THRESHOLD_M")
IA_DIST_THRESHOLD_M = float(IA_DIST_THRESHOLD_M) if IA_DIST_THRESHOLD_M else None

# Boundary Feature Fusion đa camera (2026-08-21) — CHỈ có ý nghĩa khi ENABLE_MULTICAM=1
# (cần Identity Association ghép cặp trước). TẮT MẶC ĐỊNH RIÊNG (ENABLE_FUSION=1) vì
# nặng hơn hẳn (chạy thêm 1 engine TensorRT riêng qua pycuda, ngoài các engine
# DeepStream đã có) — tách cờ riêng để bật multicam (rẻ) mà không kéo theo fusion
# (đắt) nếu chưa cần. Dùng calib MẪU giống Identity Association — xem CLAUDE.md.
ENABLE_FUSION = os.environ.get("ENABLE_FUSION") == "1"

# Re-Identification (2026-08-21) — ĐỘC LẬP với ENABLE_MULTICAM/ENABLE_FUSION
# (không cần Identity Association, chỉ cần track_id + crop từng cam riêng lẻ).
# Model OSNet_x0.25 (Distill/KD) — xem src/reid.py đầu file để biết vì sao
# KHÁC bản OSNet_x0.5 nhắc trong CLAUDE.md (đối chiếu README Handoff mới hơn).
ENABLE_REID = os.environ.get("ENABLE_REID") == "1"

RTSP_SOURCES = [
    os.environ.get("RTSP_CAM1"),
    os.environ.get("RTSP_CAM2"),
]

MUXER_WIDTH = 1280
MUXER_HEIGHT = 720
MUXER_BATCH_TIMEOUT_USEC = 40000  # 40ms

SGIE_CLASS_NAMES = ["bend", "exercise", "lie", "sit", "stand"]  # SGIE v2, 5 lớp

# ============================================================
# PADDING 20% quanh bbox trước khi SGIE crop — khớp crop_with_padding()/PAD_RATIO
# trong pose_classifier.onnx của Tấn Dũng (Handoff_for_Edge 2026-08-08,
# detect_classify_pipeline.py). nvinfer secondary tự crop theo obj_meta.rect_params
# hiện tại, không có property "padding %" riêng -> phồng bbox TRƯỚC khi vào SGIE
# (trên src pad tracker), rồi PHỤC HỒI lại bbox gốc trên src pad SGIE (aspect-ratio
# filter + payload event PHẢI dùng bbox THẬT, không phải bbox đã phồng để crop).
# ============================================================
PAD_RATIO = 0.20
_pending_orig_bbox = {}  # (cam_idx, object_id) -> (left, top, width, height) gốc


def pad_bbox_for_sgie_probe(pad, info, u_data):
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    if not batch_meta:
        return Gst.PadProbeReturn.OK
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        cam_idx = frame_meta.pad_index
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            rect = obj_meta.rect_params
            l0, t0, w0, h0 = rect.left, rect.top, rect.width, rect.height
            _pending_orig_bbox[(cam_idx, obj_meta.object_id)] = (l0, t0, w0, h0)
            pad_x, pad_y = w0 * PAD_RATIO, h0 * PAD_RATIO
            nl = max(0.0, l0 - pad_x)
            nt = max(0.0, t0 - pad_y)
            nr = min(float(MUXER_WIDTH), l0 + w0 + pad_x)
            nb = min(float(MUXER_HEIGHT), t0 + h0 + pad_y)
            rect.left = nl
            rect.top = nt
            rect.width = max(1.0, nr - nl)
            rect.height = max(1.0, nb - nt)
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK


def restore_orig_bbox(cam_idx, obj_meta) -> None:
    """Gọi ở ĐẦU vòng lặp object trong sgie_src_pad_probe — trả obj_meta.rect_params
    về bbox THẬT (trước khi bị pad_bbox_for_sgie_probe phồng lên để SGIE crop)."""
    orig = _pending_orig_bbox.pop((cam_idx, obj_meta.object_id), None)
    if orig is not None:
        l0, t0, w0, h0 = orig
        obj_meta.rect_params.left = l0
        obj_meta.rect_params.top = t0
        obj_meta.rect_params.width = w0
        obj_meta.rect_params.height = h0

# ============================================================
# GLOBALS - FPS counter per source
# ============================================================
fps_counter = defaultdict(lambda: {"frames": 0, "last_time": time.time(), "fps": 0.0})
PRINT_EVERY_N_FRAMES = 30

# ============================================================
# ROLLING BUFFER — ảnh THẬT quanh thời điểm ngã, thay placeholder_frames()
# (Tuần 5). LUÔN BẬT (không qua EDGE_DEBUG) vì publish_fall_async cần nó bất
# cứ lúc nào có ngã. Nhánh capture (NV12->RGBA->JPEG) chạy song song với nhánh
# debug crop cũ nếu EDGE_DEBUG=1 — không đụng nhau, chỉ gắn thêm 1 probe khác
# trên CÙNG pad.
# ============================================================
ROLLING_BUFFER = RollingBuffer()
CAPTURE_INTERVAL_MS = 200  # ~5 ảnh/giây/pipeline -> đủ dày hơn khoảng cách 500ms
                           # giữa các offset yêu cầu trong schema (REQUIRED_OFFSETS).
_last_capture_ts_ms = 0

# Boundary Feature Fusion (module 3) — cầu nối giữa sgie_src_pad_probe (biết
# CẶP nào vừa ghép ở buffer NV12, xem identity_associator.last_pairs) và
# capture_frame_probe (buffer RGBA cùng frame, chạy SAU trên pipeline nhưng
# CÙNG 1 buffer vật lý) — vì pyds.get_nvds_buf_surface chỉ đọc được RGBA nên
# không crop ảnh được ngay tại sgie_src_pad_probe. Ghi đè mỗi batch, KHÔNG
# hàng đợi (xem chú thích trong sgie_src_pad_probe).
_pending_fusion_pairs = []


def _crop_bgr(frame_bgr, bbox_xyxy):
    x1, y1, x2, y2 = bbox_xyxy
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_bgr[y1:y2, x1:x2]


def run_fusion_for_batch(ctx, frames_by_cam: dict) -> None:
    """frames_by_cam: {cam_id_str: (frame_bgr, ts_ms)} của batch RGBA hiện tại.
    Với mỗi cặp đang chờ (_pending_fusion_pairs) mà CẢ 2 cam đều có frame trong
    batch này -> crop + chạy BoundaryFusionEngine -> update detector_multicam
    bằng nhãn ĐÃ GỘP (thay nhãn thô từng cam, xem sgie_src_pad_probe)."""
    global _pending_fusion_pairs
    pairs, _pending_fusion_pairs = _pending_fusion_pairs, []
    if not pairs or ctx.fusion_engine is None:
        return
    for pair in pairs:
        c1, c2 = pair["cam_1"], pair["cam_2"]
        f1 = frames_by_cam.get(c1["cam_id"])
        f2 = frames_by_cam.get(c2["cam_id"])
        if f1 is None or f2 is None:
            continue  # cam đó không nằm trong batch RGBA này (throttle bỏ) -> bỏ qua
        frame1_bgr, ts_ms1 = f1
        frame2_bgr, _ = f2
        crop1 = _crop_bgr(frame1_bgr, c1["bbox_xyxy"])
        crop2 = _crop_bgr(frame2_bgr, c2["bbox_xyxy"])
        if crop1 is None or crop2 is None:
            continue
        try:
            label, conf = ctx.fusion_engine.classify_pair(
                crop1, crop2, c1["ground_xy"], c2["ground_xy"],
                c1["bbox_xyxy"], c2["bbox_xyxy"],
            )
        except Exception as e:  # noqa: BLE001 - fusion là nhánh thử nghiệm, không được làm chết pipeline
            log.error("Lỗi Boundary Feature Fusion (bỏ qua cặp global_id=%s): %s",
                      pair["global_id"], e)
            continue
        log.info("FUSION global_id=%s cam_1+cam_2 -> %s (conf=%.2f)",
                  pair["global_id"], label, conf)
        candidate = ctx.detector_multicam.update(
            "global", pair["global_id"], label, conf, ts_ms1,
            bbox=c1["bbox_xyxy"], frame_width=MUXER_WIDTH, frame_height=MUXER_HEIGHT,
        )
        if candidate is not None:
            publish_fall_async(ctx, candidate)


# Re-Identification (module Re-ID) — cùng cơ chế cầu nối RGBA như
# _pending_fusion_pairs ở trên: sgie_src_pad_probe (biết mọi object trong batch,
# ở buffer NV12) ghi vào đây, capture_frame_probe (buffer RGBA cùng frame) đọc
# ra để crop + embed. Ghi đè mỗi batch.
_pending_reid_objs = []


def run_reid_for_batch(ctx, frames_by_cam: dict) -> None:
    """frames_by_cam: {cam_id_str: (frame_bgr, ts_ms)}. Với mỗi object trong
    _pending_reid_objs: crop -> embed -> gom vào reid_buffer theo (cam_id,
    track_id). Khi buffer đủ frame (MIN_BUFFER_FRAMES) VÀ track đó CHƯA tra
    Gallery lần nào (reid_known_tracks) -> match Gallery: khớp người cũ thì
    chỉ log (chưa map lại state ngã — xem docstring EdgeContext); không khớp
    ("NEW") thì tự đăng ký làm người mới để lần sau nhận ra."""
    global _pending_reid_objs
    objs, _pending_reid_objs = _pending_reid_objs, []
    if not objs or ctx.reid_extractor is None:
        return
    for o in objs:
        key = (o["cam_id_str"], o["track_id"])
        if key in ctx.reid_known_tracks:
            continue  # track này đã tra Gallery rồi, khỏi lặp lại mỗi frame
        f = frames_by_cam.get(o["cam_id_str"])
        if f is None:
            continue
        frame_bgr, _ = f
        crop = _crop_bgr(frame_bgr, o["bbox_xyxy"])
        if crop is None:
            continue
        try:
            emb = ctx.reid_extractor.embed(crop)
        except Exception as e:  # noqa: BLE001 - reid là nhánh thử nghiệm, không được làm chết pipeline
            log.error("Lỗi ReID embed (cam=%s track=%s): %s",
                      o["cam_id_str"], o["track_id"], e)
            continue
        ctx.reid_buffer.add(key, emb)
        avg = ctx.reid_buffer.get_average_if_ready(key)
        if avg is None:
            continue  # chưa đủ frame, đợi batch sau
        ctx.reid_known_tracks.add(key)
        ctx.reid_buffer.forget(key)
        name, score = ctx.reid_gallery.match(avg)
        if name == "NEW":
            new_name = f"person_{ctx._reid_next_person_id}"
            ctx._reid_next_person_id += 1
            ctx.reid_gallery.register(new_name, [avg])
            log.info("REID cam=%s track=%s -> NGƯỜI MỚI, đăng ký %s (best_score=%.2f)",
                      o["cam_id_str"], o["track_id"], new_name, score)
        else:
            log.info("REID cam=%s track=%s -> KHỚP %s (score=%.2f)",
                      o["cam_id_str"], o["track_id"], name, score)


def capture_throttle_probe(pad, info, u_data):
    """Chặn sớm TRƯỚC khi vào nvvideoconvert (tốn GPU) — chỉ cho buffer đi qua
    mỗi CAPTURE_INTERVAL_MS. Áp dụng chung cho cả batch (2 cam đi cùng 1 buffer
    sau streammux nên không tách throttle riêng từng cam được)."""
    global _last_capture_ts_ms
    now_ms = int(time.monotonic() * 1000)
    if now_ms - _last_capture_ts_ms < CAPTURE_INTERVAL_MS:
        return Gst.PadProbeReturn.DROP
    _last_capture_ts_ms = now_ms
    return Gst.PadProbeReturn.OK


def capture_frame_probe(pad, info, u_data):
    """Encode NGUYÊN FRAME (không crop theo người) mỗi cam trong buffer RGBA
    này, đẩy vào ROLLING_BUFFER. Chạy sau throttle nên tần suất đã thấp.

    u_data = ctx (EdgeContext) — TIỆN THỂ gom frame_bgr theo cam vào
    frames_by_cam để chạy run_fusion_for_batch() (module 3, cần crop THẬT từ
    CẢ 2 cam của CÙNG 1 buffer — chỉ đọc được RGBA ở đây, không phải ở
    sgie_src_pad_probe, xem chú thích _pending_fusion_pairs)."""
    ctx: Optional[EdgeContext] = u_data if isinstance(u_data, EdgeContext) else None
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    if not batch_meta:
        return Gst.PadProbeReturn.OK
    frames_by_cam = {}
    try:
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            cam_id_str = cam_id_from_pad(frame_meta.pad_index)
            ts_ms = ts_ms_from_frame(frame_meta)
            n_frame = pyds.get_nvds_buf_surface(hash(buf), frame_meta.batch_id)
            frame_rgba = np.array(n_frame, copy=True, order="C")
            frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
            frames_by_cam[cam_id_str] = (frame_bgr, ts_ms)
            ok, jpeg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                ROLLING_BUFFER.add_frame(cam_id_str, ts_ms, jpeg.tobytes())
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
    except Exception as e:  # noqa: BLE001 - capture phụ, không được làm chết pipeline
        log.warning("Không capture được frame cho rolling buffer: %s", e)

    if ctx is not None and ctx.fusion_engine is not None and _pending_fusion_pairs:
        try:
            run_fusion_for_batch(ctx, frames_by_cam)
        except Exception as e:  # noqa: BLE001 - fusion là nhánh thử nghiệm, không được làm chết pipeline
            log.error("Lỗi chạy Boundary Feature Fusion cho batch này: %s", e)

    if ctx is not None and ctx.reid_extractor is not None and _pending_reid_objs:
        try:
            run_reid_for_batch(ctx, frames_by_cam)
        except Exception as e:  # noqa: BLE001 - reid là nhánh thử nghiệm, không được làm chết pipeline
            log.error("Lỗi chạy Re-Identification cho batch này: %s", e)

    return Gst.PadProbeReturn.OK


# ============================================================
# DEBUG chẩn đoán model (bật bằng EDGE_DEBUG=1). Tắt mặc định -> không tốn gì
# lúc chạy thường. Đếm: số người PGIE bắt + phân bố nhãn SGIE + conf trung bình.
# Mục đích: tách "PGIE bỏ sót người" vs "SGIE phân loại sai tư thế".
# ============================================================
DEBUG_LABELS = os.environ.get("EDGE_DEBUG") == "1"

# Lưu crop THẬT (đúng bbox DeepStream dùng) để so sánh độc lập với ONNX offline.
# Đây là "bài test quyết định" trong CLAUDE.md: chạy ONNX trên CÙNG crop này rồi
# so với sgie_label DeepStream đã ra -> khớp+sai = lỗi model, lệch = lỗi tiền xử lý.
CROP_SAVE_DIR = os.path.join(PROJECT_ROOT, "output", "debug_crops")
CROP_SAVE_MAX = 16
_crop_saved_count = 0


def maybe_save_debug_crop(buf, frame_meta, obj_meta, cam_id_str, sgie_label, sgie_conf):
    """Lưu crop bbox từ buffer THẬT mà SGIE vừa suy luận trên đó (probe ở src pad
    SGIE nên classifier_meta đã gắn sẵn vào buffer này — bbox và nhãn khớp nhau)."""
    global _crop_saved_count
    if _crop_saved_count >= CROP_SAVE_MAX:
        return
    # Rải đều theo thời gian (không chỉ lấy frame đầu) để bắt nhiều tư thế khác nhau.
    if _diag["pgie_persons"] % 60 != 1:
        return
    try:
        n_frame = pyds.get_nvds_buf_surface(hash(buf), frame_meta.batch_id)
        frame_rgba = np.array(n_frame, copy=True, order="C")
        frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
        bbox = obj_meta.rect_params
        x1 = max(0, int(bbox.left))
        y1 = max(0, int(bbox.top))
        x2 = min(frame_bgr.shape[1], int(bbox.left + bbox.width))
        y2 = min(frame_bgr.shape[0], int(bbox.top + bbox.height))
        if x2 <= x1 or y2 <= y1:
            return
        crop = frame_bgr[y1:y2, x1:x2]
        os.makedirs(CROP_SAVE_DIR, exist_ok=True)
        label = sgie_label or "none"
        fname = (f"{cam_id_str}_f{frame_meta.frame_num}_pid{obj_meta.object_id}_"
                 f"{label}_{sgie_conf:.2f}.jpg")
        cv2.imwrite(os.path.join(CROP_SAVE_DIR, fname), crop)
        _crop_saved_count += 1
        log.info("Đã lưu crop debug #%d/%d: %s", _crop_saved_count, CROP_SAVE_MAX, fname)
    except Exception as e:  # noqa: BLE001 - debug phụ, không được làm chết pipeline
        log.warning("Không lưu được crop debug: %s", e)
_diag = {
    "pgie_persons": 0,   # tổng số box người PGIE ra
    "with_label": 0,     # số người có nhãn SGIE (conf >= classifier-threshold)
    "no_label": 0,       # số người KHÔNG có nhãn SGIE (dưới ngưỡng -> '(none)')
    "hist": defaultdict(int),      # phân bố nhãn: {stand: n, lie: n, ...}
    "conf_sum": defaultdict(float),  # tổng conf theo nhãn -> tính trung bình
}
DEBUG_REPORT_EVERY = 120  # cứ mỗi 120 frame (mỗi cam) in 1 bảng tổng kết


def _diag_report(cam_id: int, frame_num: int) -> None:
    total = _diag["pgie_persons"]
    if total == 0:
        print(f"[DIAG] cam={cam_id} frame={frame_num} — PGIE CHƯA bắt được người nào (0)")
        return
    parts = []
    for lbl, n in sorted(_diag["hist"].items(), key=lambda kv: -kv[1]):
        avg = _diag["conf_sum"][lbl] / max(1, n)
        parts.append(f"{lbl}={n}({avg:.2f})")
    none_pct = 100.0 * _diag["no_label"] / total
    print(
        f"[DIAG] cam={cam_id} frame={frame_num} | người PGIE={total} "
        f"| có nhãn={_diag['with_label']} không nhãn={_diag['no_label']} "
        f"({none_pct:.0f}% dưới ngưỡng) | nhãn(avg conf): {' '.join(parts) or '(chưa có)'}"
    )


# Lưu 6 ảnh thật của 1 event ngã ra đĩa để XEM BẰNG MẮT (bật qua EDGE_SAVE_FALL_FRAMES=1).
# Tắt mặc định -> không tốn gì lúc chạy thường/production.
SAVE_FALL_FRAMES = os.environ.get("EDGE_SAVE_FALL_FRAMES") == "1"
FALL_FRAMES_DIR = os.path.join(PROJECT_ROOT, "output", "fall_events")


def maybe_save_fall_frames(event: dict) -> None:
    if not SAVE_FALL_FRAMES:
        return
    try:
        import base64
        event_dir = os.path.join(FALL_FRAMES_DIR, event["event_id"])
        os.makedirs(event_dir, exist_ok=True)
        for f in event["frames"]:
            fname = f"offset_{f['offset_ms']}ms.jpg"
            with open(os.path.join(event_dir, fname), "wb") as fh:
                fh.write(base64.b64decode(f["jpeg_b64"]))
        log.warning("Đã lưu 6 ảnh ngã ra %s", event_dir)
    except Exception as e:  # noqa: BLE001 - debug phụ, không được làm chết pipeline
        log.warning("Không lưu được ảnh ngã: %s", e)


# ============================================================
# FALL DETECTION context + helpers
# ============================================================
class EdgeContext:
    """Giữ detector (logic ngã) + publisher (MQTT). Truyền vào probe qua u_data.

    publisher=None nghĩa là chạy ở chế độ 'câm MQTT' (broker Khánh tắt) —
    detector VẪN chạy và log ra console, chỉ không gửi lên broker.

    detector_multicam/identity_associator: đường CHẠY SONG SONG, KHÔNG đụng
    `detector` (single-cam, đã verify thật trên camera thật) — Identity
    Association (2026-08-20, xem identity_association.py) gộp track 2 cam
    thành global_id, feed vào 1 FallDetector RIÊNG (state buffer key="global"
    -> merge xuyên 2 cam). None nếu tắt (mặc định — xem ENABLE_MULTICAM).

    fusion_engine: Boundary Feature Fusion (module 3, 2026-08-21) — CHỈ dùng
    khi Identity Association ghép được cặp (người ở vùng giao 2 cam). Nhãn ra
    từ fusion_engine.classify_pair() được feed vào CÙNG detector_multicam,
    THAY vì nhãn SGIE riêng từng cam, cho những object đã ghép cặp trong batch
    đó (xem sgie_src_pad_probe). None nếu tắt (mặc định — xem ENABLE_FUSION).

    reid_extractor/reid_gallery/reid_buffer: Re-Identification (2026-08-21,
    xem reid.py) — ĐỘC LẬP với 2 module trên (không cần Identity Association).
    Track MỚI (chưa thấy track_id đó bao giờ — xem reid_known_tracks) được
    gom embedding qua vài frame (reid_buffer), rồi tra reid_gallery: khớp
    người đã biết -> chỉ log (chưa map lại state ngã, xem docstring reid.py);
    "NEW" -> tự đăng ký làm người mới vào gallery. None nếu tắt (ENABLE_REID).
    """
    def __init__(self, detector: FallDetector, publisher,
                 detector_multicam: Optional[FallDetector] = None,
                 identity_associator: Optional[IdentityAssociator] = None,
                 fusion_engine: Optional[BoundaryFusionEngine] = None,
                 reid_extractor: Optional[ReIDExtractorTrt] = None,
                 reid_gallery: Optional[PersonGallery] = None,
                 reid_buffer: Optional[TrackEmbeddingBuffer] = None):
        self.detector = detector
        self.publisher = publisher
        self.detector_multicam = detector_multicam
        self.identity_associator = identity_associator
        self.fusion_engine = fusion_engine
        self.reid_extractor = reid_extractor
        self.reid_gallery = reid_gallery
        self.reid_buffer = reid_buffer
        self.reid_known_tracks = set()  # (cam_id_str, track_id) đã tra Gallery rồi
        self._reid_next_person_id = 1


def cam_id_from_pad(pad_index: int) -> str:
    """pad_index 0/1 -> 'cam_1'/'cam_2' (khớp topic events/cam_1/fall trong CLAUDE.md)."""
    return f"cam_{pad_index + 1}"


def ts_ms_from_frame(frame_meta) -> int:
    """buf_pts (ns) -> ms; nếu 0 (một số nguồn live) thì fallback đồng hồ monotonic."""
    pts = frame_meta.buf_pts
    if pts and pts > 0:
        return int(pts // 1_000_000)
    return int(time.monotonic() * 1000)


def publish_fall_async(ctx: EdgeContext, candidate) -> None:
    """Dựng payload schema v3 từ FallCandidate rồi gửi MQTT trong THREAD NỀN.

    Gửi ở thread riêng để KHÔNG chặn thread streaming của GStreamer (publish QoS1
    có chờ ack, chặn probe sẽ rớt frame). Ngã hiếm + có cooldown nên thread/1 event
    là đủ đơn giản.

    Ảnh lấy từ ROLLING_BUFFER (frame thật quanh thời điểm ngã, Tuần 5). Chỉ rơi
    về placeholder GIẢ khi buffer THẬT SỰ rỗng (vd ngã xảy ra ngay lúc mới khởi
    động, chưa kịp có frame nào) — vẫn luôn log rõ khi phải fallback.
    """
    frames = ROLLING_BUFFER.get_frames_before(candidate.cam_id, candidate.ts_ms)
    if frames:
        log.warning(
            "NGÃ %s pid=%s %s->%s conf=%.2f transition=%dms — 6 ảnh THẬT từ rolling buffer",
            candidate.cam_id, candidate.person_id, candidate.class_before,
            candidate.final_class, candidate.confidence, candidate.transition_ms,
        )
    else:
        log.warning(
            "NGÃ %s pid=%s %s->%s conf=%.2f transition=%dms — rolling buffer RỖNG "
            "(mới khởi động?) -> GỬI KÈM 6 ẢNH GIẢ",
            candidate.cam_id, candidate.person_id, candidate.class_before,
            candidate.final_class, candidate.confidence, candidate.transition_ms,
        )
        frames = placeholder_frames()
    try:
        event = build_fall_event(candidate, frames)
    except ValueError as e:
        log.error("Bỏ event ngã vì payload sai schema: %s", e)
        return

    maybe_save_fall_frames(event)

    if ctx.publisher is None:
        log.warning("MQTT đang TẮT -> KHÔNG gửi event_id=%s (chỉ log)", event["event_id"])
        return

    def _send():
        try:
            ok = ctx.publisher.publish_fall_event(candidate.cam_id, event)
            log.info("Gửi event ngã %s -> %s (ok=%s)",
                     event["event_id"], candidate.cam_id, ok)
        except Exception as e:  # noqa: BLE001 - không để lỗi mạng làm chết pipeline
            log.error("Lỗi gửi event ngã %s: %s", event["event_id"], e)

    threading.Thread(target=_send, name="fall-publish", daemon=True).start()


TELEMETRY_INTERVAL_SEC = 30


def telemetry_tick(ctx: EdgeContext) -> bool:
    """Gửi telemetry mỗi TELEMETRY_INTERVAL_SEC giây cho từng cam (Tuần 5).
    build_telemetry() đã có sẵn + có test — chỗ này chỉ gom số liệu THẬT rồi gọi.
    Trả True để GLib.timeout_add_seconds() lặp lại (trả False sẽ dừng hẳn)."""
    try:
        import psutil
        ram_used_mb = round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1)
    except Exception:  # noqa: BLE001 - telemetry phụ, không được làm chết pipeline
        ram_used_mb = None

    mqtt_connected = ctx.publisher.is_connected if ctx.publisher is not None else False

    for cam_idx in range(len(RTSP_SOURCES)):
        cam_id_str = cam_id_from_pad(cam_idx)
        fps = fps_counter[cam_idx]["fps"]
        active_tracks = ctx.detector.active_track_count(cam_id_str)
        payload = build_telemetry(
            cam_id_str,
            pipeline={
                "state": "running",
                "fps_pgie": round(fps, 1),
                "fps_sgie": round(fps, 1),  # PGIE/SGIE chạy đồng bộ cùng 1 probe -> cùng FPS
                "active_tracks": active_tracks,
            },
            system={"ram_used_mb": ram_used_mb},
            network={"mqtt_connected": mqtt_connected},
        )
        if ctx.publisher is not None:
            ctx.publisher.publish_telemetry(cam_id_str, payload)
        log.info("Telemetry %s: fps=%.1f tracks=%d ram=%sMB mqtt=%s",
                  cam_id_str, fps, active_tracks, ram_used_mb, mqtt_connected)
    return True


TENSOR_DUMP_MAX = 16
_tensor_dumped_count = 0


def maybe_dump_raw_tensor(obj_meta, cam_id_str, frame_num, sgie_label=None, sgie_conf=0.0):
    """Đọc RAW output 6 lớp nvinfer THỰC SỰ tính cho object này (cần
    output-tensor-meta=1 trong config SGIE) - bỏ qua threshold/argmax, để so
    trực tiếp với ONNX ngoại tuyến chạy trên crop đã lưu của CÙNG object.
    Chạy ngay ở src pad SGIE (không qua nhánh debug NV12->RGBA riêng)."""
    global _tensor_dumped_count
    if _tensor_dumped_count >= TENSOR_DUMP_MAX:
        return
    if _diag["pgie_persons"] % 60 != 1:
        return
    try:
        import ctypes
        l_user = obj_meta.obj_user_meta_list
        while l_user is not None:
            user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            if user_meta.base_meta.meta_type == pyds.NVDSINFER_TENSOR_OUTPUT_META:
                tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                vals = [round(ptr[i], 4) for i in range(len(SGIE_CLASS_NAMES))]
                pairs = list(zip(SGIE_CLASS_NAMES, vals))
                # Liệt kê TOÀN BỘ classifier_meta_list (mọi component_id, mọi
                # label_info) - không lọc như read_sgie_label - để xem có bị
                # sót entry nào không.
                all_meta = []
                l_class = obj_meta.classifier_meta_list
                while l_class is not None:
                    cm = pyds.NvDsClassifierMeta.cast(l_class.data)
                    labels_here = []
                    l_lbl = cm.label_info_list
                    while l_lbl is not None:
                        li = pyds.NvDsLabelInfo.cast(l_lbl.data)
                        labels_here.append((li.result_label, round(li.result_prob, 3)))
                        try:
                            l_lbl = l_lbl.next
                        except StopIteration:
                            break
                    all_meta.append((cm.unique_component_id, labels_here))
                    try:
                        l_class = l_class.next
                    except StopIteration:
                        break
                log.warning(
                    "TENSOR_RAW cam=%s frame=%s pid=%s vals=%s | classifier_meta_label=%s(%.3f) | ALL_META=%s",
                    cam_id_str, frame_num, obj_meta.object_id, pairs,
                    sgie_label, sgie_conf, all_meta,
                )
                _tensor_dumped_count += 1
            try:
                l_user = l_user.next
            except StopIteration:
                break
    except Exception as e:  # noqa: BLE001 - debug phụ, không được làm chết pipeline
        log.warning("Không đọc được raw tensor: %s", e)


def read_sgie_label(obj_meta):
    """Đọc nhãn+conf SGIE (gie-unique-id=2) gắn vào 1 object. None nếu dưới
    classifier-threshold (DeepStream không gắn classifier_meta khi đó)."""
    l_class = obj_meta.classifier_meta_list
    while l_class is not None:
        try:
            class_meta = pyds.NvDsClassifierMeta.cast(l_class.data)
        except StopIteration:
            break
        if class_meta.unique_component_id == 2:  # SGIE gie-unique-id
            l_label = class_meta.label_info_list
            while l_label is not None:
                try:
                    label_info = pyds.NvDsLabelInfo.cast(l_label.data)
                except StopIteration:
                    break
                return label_info.result_label, label_info.result_prob
        try:
            l_class = l_class.next
        except StopIteration:
            break
    return None, 0.0


# ============================================================
# PROBE TẠM: đếm raw object count NGAY SAU PGIE (trước tracker) — debug
# 2026-08-18 để xác định detection biến mất ở PGIE hay ở downstream.
# Bật qua env PGIE_RAW_DEBUG=1, tắt mặc định.
# ============================================================
def pgie_raw_debug_probe(pad, info, u_data):
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    if not batch_meta:
        print("[PGIE_RAW] no batch_meta")
        return Gst.PadProbeReturn.OK
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        n = 0
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            n += 1
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        print(f"[PGIE_RAW] pad={frame_meta.pad_index} frame={frame_meta.frame_num} n_obj={n}")
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK


# ============================================================
# PROBE: đọc metadata từ buffer sau SGIE
# ============================================================
def sgie_src_pad_probe(pad, info, u_data):
    ctx: EdgeContext = u_data
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    # Gom object của CẢ batch (mọi cam) cho Identity Association (2026-08-20) —
    # cần thấy cam_1 VÀ cam_2 CÙNG LÚC mới ghép cặp Hungarian được, nên không xử
    # lý ngay trong vòng lặp per-object bên dưới (đang duyệt tuần tự từng frame_meta
    # 1 cam 1 lúc) mà gom vào đây, xử lý 1 lần sau khi hết batch. batch_objs:
    # {cam_id_str: [{"track_id":.., "bbox_xyxy":.., "label":.., "conf":.., "ts_ms":..}]}.
    # Re-Identification (2026-08-21) DÙNG LẠI CHUNG danh sách này (không cần
    # ghép cặp, chỉ cần track_id + bbox từng cam riêng) nên cũng bật gom khi
    # ENABLE_REID, dù identity_associator có None hay không.
    need_batch_objs = ctx.identity_associator is not None or ctx.reid_extractor is not None
    batch_objs = defaultdict(list) if need_batch_objs else None

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        cam_id = frame_meta.pad_index
        cam_id_str = cam_id_from_pad(cam_id)   # 'cam_1' / 'cam_2' cho topic + detector
        ts_ms = ts_ms_from_frame(frame_meta)
        frame_num = frame_meta.frame_num

        # Đếm FPS theo từng source
        ctr = fps_counter[cam_id]
        ctr["frames"] += 1
        if ctr["frames"] % PRINT_EVERY_N_FRAMES == 0:
            now = time.time()
            fps = PRINT_EVERY_N_FRAMES / (now - ctr["last_time"])
            ctr["last_time"] = now
            ctr["fps"] = fps  # đọc lại ở telemetry_tick(), khỏi tính lại
            print(f"[FPS] cam={cam_id} frame={frame_num} fps={fps:.1f}")

        if DEBUG_LABELS and ctr["frames"] % DEBUG_REPORT_EVERY == 0:
            _diag_report(cam_id, frame_num)

        # Duyệt từng object (người) trong frame
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            # Trả bbox về đúng bbox THẬT (pad_bbox_for_sgie_probe đã phồng +20% để
            # SGIE crop) TRƯỚC KHI đọc rect_params — aspect-ratio filter + payload
            # event phải dùng bbox thật, không phải bbox đã phồng.
            restore_orig_bbox(cam_id, obj_meta)

            person_id = obj_meta.object_id
            bbox = obj_meta.rect_params
            det_conf = obj_meta.confidence
            sgie_label, sgie_conf = read_sgie_label(obj_meta)

            if os.environ.get("TRACK_DEBUG") == "1":
                print(f"[TRK] cam={cam_id_str} pid={person_id} frame={frame_num} ts_ms={ts_ms} label={sgie_label} conf={sgie_conf:.2f}")

            # DEBUG: gom thống kê PGIE/SGIE để chẩn đoán "model bắt sai"
            if DEBUG_LABELS:
                _diag["pgie_persons"] += 1
                maybe_dump_raw_tensor(obj_meta, cam_id_str, frame_num, sgie_label, sgie_conf)
                if sgie_label is not None:
                    _diag["with_label"] += 1
                    _diag["hist"][sgie_label] += 1
                    _diag["conf_sum"][sgie_label] += sgie_conf
                else:
                    _diag["no_label"] += 1

            # Đưa tư thế frame này vào detector. Detector tự lo temporal/chống nhiễu/
            # cooldown; chỉ trả về FallCandidate ĐÚNG lúc xác nhận ngã (stand/sit->lie).
            if sgie_label is not None:
                # bbox DeepStream là (left, top, width, height) -> đổi sang xyxy
                x1 = int(bbox.left)
                y1 = int(bbox.top)
                x2 = int(bbox.left + bbox.width)
                y2 = int(bbox.top + bbox.height)
                candidate = ctx.detector.update(
                    cam_id_str, person_id, sgie_label, sgie_conf, ts_ms,
                    bbox=(x1, y1, x2, y2),
                    frame_width=MUXER_WIDTH, frame_height=MUXER_HEIGHT,
                )
                if candidate is not None:
                    publish_fall_async(ctx, candidate)

                # Gom cho Identity Association — CHỈ thêm vào danh sách, chưa
                # gọi detector_multicam ở đây (phải đợi duyệt hết batch/cả 2 cam
                # trước mới ghép Hungarian đúng được).
                if batch_objs is not None:
                    batch_objs[cam_id_str].append({
                        "track_id": person_id, "bbox_xyxy": (x1, y1, x2, y2),
                        "label": sgie_label, "conf": sgie_conf, "ts_ms": ts_ms,
                    })

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    # Re-Identification (2026-08-21) — đẩy TOÀN BỘ object trong batch cho
    # reid_probe (chạy sau, trên buffer RGBA cùng frame) — ĐỘC LẬP với Identity
    # Association (không cần ghép cặp, chỉ cần track_id + crop từng cam riêng).
    if batch_objs and ctx.reid_extractor is not None:
        global _pending_reid_objs
        _pending_reid_objs = [
            {"cam_id_str": cam_id_str, **o}
            for cam_id_str, objs in batch_objs.items() for o in objs
        ]

    # Identity Association: ghép cặp 2 cam (Hungarian trên toạ độ sàn) -> global_id,
    # feed vào detector_multicam RIÊNG (state buffer key="global", merge xuyên 2 cam).
    # CHẠY SONG SONG với ctx.detector single-cam ở trên, KHÔNG thay thế.
    if batch_objs and ctx.identity_associator is not None:
        try:
            global_ids = ctx.identity_associator.assign_batch(batch_objs)
            # Object nào đã GHÉP CẶP (thấy ở cả 2 cam, xem last_pairs) thì bỏ qua
            # update "thô" theo nhãn SGIE riêng từng cam ở đây — Boundary Feature
            # Fusion (nếu bật, xem fusion_probe) sẽ tự update detector_multicam
            # bằng nhãn ĐÃ GỘP epipolar cho global_id đó. Không bật fusion thì vẫn
            # update thô như cũ (giữ hành vi gốc của Identity Association).
            paired_keys = set()
            if ctx.fusion_engine is not None:
                for pair in ctx.identity_associator.last_pairs:
                    paired_keys.add((pair["cam_1"]["cam_id"], pair["cam_1"]["track_id"]))
                    paired_keys.add((pair["cam_2"]["cam_id"], pair["cam_2"]["track_id"]))
                # Đẩy pending pairs cho fusion_probe (chạy sau, trên buffer RGBA
                # cùng frame) — ghi đè mỗi batch, KHÔNG cộng dồn (tần suất ghép cặp
                # thấp + FPS máy này vốn thấp, đủ dùng cho luồng "chạy thông").
                global _pending_fusion_pairs
                _pending_fusion_pairs = list(ctx.identity_associator.last_pairs)

            for cam_id_str, objs in batch_objs.items():
                for o in objs:
                    if (cam_id_str, o["track_id"]) in paired_keys:
                        continue
                    gid = global_ids.get((cam_id_str, o["track_id"]))
                    if gid is None:
                        continue
                    candidate = ctx.detector_multicam.update(
                        "global", gid, o["label"], o["conf"], o["ts_ms"],
                        bbox=o["bbox_xyxy"],
                        frame_width=MUXER_WIDTH, frame_height=MUXER_HEIGHT,
                    )
                    if candidate is not None:
                        publish_fall_async(ctx, candidate)
        except Exception as e:  # noqa: BLE001 - multicam là nhánh thử nghiệm, không được làm chết pipeline single-cam
            log.error("Lỗi Identity Association (bỏ qua batch này): %s", e)

    return Gst.PadProbeReturn.OK


def debug_crop_probe(pad, info, u_data):
    """Probe RIÊNG cho debug — chạy sau nvvideoconvert(RGBA) chỉ bật khi EDGE_DEBUG=1
    (xem main()). pyds.get_nvds_buf_surface CHỈ đọc được buffer RGBA, nên phải tách
    khỏi sgie_src_pad_probe (buffer ở đó vẫn là NV12 nội bộ)."""
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        cam_id_str = cam_id_from_pad(frame_meta.pad_index)

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            sgie_label, sgie_conf = read_sgie_label(obj_meta)
            maybe_save_debug_crop(buf, frame_meta, obj_meta, cam_id_str,
                                  sgie_label, sgie_conf)
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK


# ============================================================
# Tạo element + check
# ============================================================
def make_elm(factory, name):
    elm = Gst.ElementFactory.make(factory, name)
    if not elm:
        sys.stderr.write(f"FATAL: không tạo được element {factory} ({name})\n")
        sys.exit(1)
    return elm


def cb_newpad(decodebin, pad, source_bin):
    """Callback khi nvurisrcbin có new pad — link sang ghost pad của source_bin."""
    caps = pad.get_current_caps()
    if not caps:
        caps = pad.query_caps()
    name = caps.to_string()
    if "video" in name:
        ghost = source_bin.get_static_pad("src")
        if not ghost.is_linked():
            if pad.link(ghost.get_peer() or ghost) != Gst.PadLinkReturn.OK:
                # link qua ghost target
                ghost.set_target(pad)


def create_source_bin(index, uri):
    """Source bin cho 1 RTSP — chuỗi giải mã thủ công.

        rtspsrc -> [rtph264depay+h264parse HOẶC rtph265depay+h265parse] -> nvv4l2decoder

    Lý do KHÔNG dùng nvurisrcbin: decoder nvv4l2 của Jetson cần SPS/PPS (thông
    tin mô tả khung hình) nhúng LẶP LẠI trong luồng. RTSP thường chỉ để SPS/PPS
    ngoài luồng (trong SDP) nên decoder báo "Stream format not found" và drop
    sạch frame. h264parse/h265parse với config-interval=-1 chèn SPS/PPS trước
    MỖI keyframe -> decoder luôn có đủ thông tin. (Đã kiểm chứng bằng gst-launch.)

    depay/parser được tạo ĐỘNG theo codec thật của RTP payload (đọc từ caps SDP
    lúc rtspsrc tạo pad) — sim dùng H.264, camera thật (Hikvision) dùng HEVC/H.265,
    cùng 1 pipeline phải chạy được cả 2 chứ không hardcode 1 codec.
    """
    bin_name = f"source-bin-{index:02d}"
    nbin = Gst.Bin.new(bin_name)

    src = make_elm("rtspsrc", f"rtsp-src-{index}")
    src.set_property("location", uri)
    src.set_property("latency", 200)
    nbin.add(src)

    decoder = make_elm("nvv4l2decoder", f"decoder-{index}")
    nbin.add(decoder)
    ghost = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
    nbin.add_pad(ghost)

    # rtspsrc tạo pad động (video/audio/rtcp) -> chỉ nối pad VIDEO, chọn
    # depay/parse theo encoding-name thật (H264 hay H265) trong caps.
    def _on_rtsp_pad(_rtspsrc, new_pad):
        caps = new_pad.get_current_caps() or new_pad.query_caps()
        desc = caps.to_string() if caps else ""
        if "x-rtp" not in desc or "media=(string)video" not in desc:
            return

        if "encoding-name=(string)H265" in desc:
            depay = make_elm("rtph265depay", f"depay-{index}")
            parser = make_elm("h265parse", f"parse-{index}")
        else:
            depay = make_elm("rtph264depay", f"depay-{index}")
            parser = make_elm("h264parse", f"parse-{index}")
        parser.set_property("config-interval", -1)

        nbin.add(depay)
        nbin.add(parser)
        depay.sync_state_with_parent()
        parser.sync_state_with_parent()
        depay.link(parser)
        parser.link(decoder)

        sinkpad = depay.get_static_pad("sink")
        new_pad.link(sinkpad)
        ghost.set_target(decoder.get_static_pad("src"))

    src.connect("pad-added", _on_rtsp_pad)
    return nbin


def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("[BUS] End-of-stream")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, dbg = message.parse_warning()
        print(f"[WARN] {err}: {dbg}")
    elif t == Gst.MessageType.ERROR:
        err, dbg = message.parse_error()
        print(f"[ERROR] {err}: {dbg}")
        loop.quit()
    return True


def build_edge_context() -> EdgeContext:
    """Nạp luật ngã + detector, và kết nối MQTT (CÓ PHÒNG HỜ).

    Nếu broker Khánh tắt / chưa đặt MQTT_BROKER / đặt EDGE_DISABLE_MQTT=1 -> chạy
    ở chế độ 'câm MQTT': detector vẫn phát hiện ngã và log, chỉ không gửi lên broker.
    Pipeline KHÔNG chết chỉ vì broker offline (broker Khánh hay tắt — xem CLAUDE.md).
    """
    rules = load_rules()  # đọc configs/fall_rules.yaml (đường dẫn tự suy từ src/)
    detector = FallDetector(rules)
    log.info("Đã nạp luật ngã: window=%dms confirm=%d cooldown=%dms",
             rules.window_ms, rules.confirm_frames, rules.cooldown_ms)

    publisher = None
    if os.environ.get("EDGE_DISABLE_MQTT") == "1":
        log.warning("EDGE_DISABLE_MQTT=1 -> chạy KHÔNG gửi MQTT (chỉ log ngã)")
    elif not os.environ.get("MQTT_BROKER"):
        log.warning("Chưa đặt MQTT_BROKER -> chạy KHÔNG gửi MQTT (chỉ log ngã)")
    else:
        try:
            pub = MqttPublisher.from_env(client_id="jetson-edge-pipeline")
            pub.connect()
            publisher = pub
            log.info("MQTT sẵn sàng -> %s:%s", pub.host, pub.port)
        except Exception as e:  # noqa: BLE001 - broker tắt không được làm chết pipeline
            log.warning("Không kết nối được MQTT (%s) -> chạy chế độ câm, chỉ log ngã", e)

    detector_multicam = None
    identity_associator = None
    if ENABLE_MULTICAM:
        try:
            ia_kwargs = {"homography_path": IA_HOMOGRAPHY_PATH}
            if IA_DIST_THRESHOLD_M is not None:
                ia_kwargs["distance_threshold"] = IA_DIST_THRESHOLD_M
                log.warning(
                    "IA_DIST_THRESHOLD_M=%.1f -- NGƯỠNG TEST, không phải giá trị "
                    "production (mặc định 0.5m)", IA_DIST_THRESHOLD_M)
            identity_associator = IdentityAssociator(**ia_kwargs)
            detector_multicam = FallDetector(rules)  # cùng luật, buffer RIÊNG (key="global")
            log.warning(
                "ENABLE_MULTICAM=1 -> Identity Association BẬT (dùng calib %s — "
                "xem cảnh báo trong identity_association.py nếu là file mẫu)",
                IA_HOMOGRAPHY_PATH or "mặc định (mẫu)",
            )
        except Exception as e:  # noqa: BLE001 - lỗi nạp calib không được làm chết pipeline single-cam
            log.error("Không bật được Identity Association (%s) -> chạy KHÔNG có multicam", e)

    fusion_engine = None
    if ENABLE_MULTICAM and ENABLE_FUSION:
        try:
            fusion_engine = BoundaryFusionEngine()
            log.warning(
                "ENABLE_FUSION=1 -> Boundary Feature Fusion BẬT (engine TensorRT + "
                "calib h_by_height MẪU — xem cảnh báo trong boundary_feature_fusion.py)"
            )
        except Exception as e:  # noqa: BLE001 - thiếu file/RAM không được làm chết pipeline
            log.error("Không bật được Boundary Feature Fusion (%s) -> chạy KHÔNG có module 3", e)
    elif ENABLE_FUSION and not ENABLE_MULTICAM:
        log.warning("ENABLE_FUSION=1 nhưng ENABLE_MULTICAM=0 -> bỏ qua (fusion cần Identity Association)")

    reid_extractor = reid_gallery = reid_buffer = None
    if ENABLE_REID:
        try:
            reid_extractor = ReIDExtractorTrt()
            reid_gallery = PersonGallery()
            reid_buffer = TrackEmbeddingBuffer()
            log.warning(
                "ENABLE_REID=1 -> Re-Identification BẬT (OSNet_x0.25 Distill/KD, "
                "gallery bắt đầu RỖNG -> track mới đầu tiên nào cũng tự đăng ký)"
            )
        except Exception as e:  # noqa: BLE001 - thiếu file/RAM không được làm chết pipeline
            log.error("Không bật được Re-Identification (%s) -> chạy KHÔNG có reid", e)

    return EdgeContext(detector, publisher, detector_multicam, identity_associator,
                        fusion_engine, reid_extractor, reid_gallery, reid_buffer)


# ============================================================
# MAIN
# ============================================================
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    # Kiểm tra RTSP URL
    for i, uri in enumerate(RTSP_SOURCES):
        if not uri:
            sys.stderr.write(f"FATAL: thiếu RTSP_CAM{i+1} trong .env\n")
            sys.exit(1)

    ctx = build_edge_context()

    Gst.init(None)
    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write("FATAL: không tạo được Pipeline\n")
        sys.exit(1)

    # streammux
    streammux = make_elm("nvstreammux", "stream-muxer")
    streammux.set_property("width", MUXER_WIDTH)
    streammux.set_property("height", MUXER_HEIGHT)
    streammux.set_property("batch-size", len(RTSP_SOURCES))
    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)
    streammux.set_property("live-source", 1)
    pipeline.add(streammux)

    # source bins
    for i, uri in enumerate(RTSP_SOURCES):
        sb = create_source_bin(i, uri)
        pipeline.add(sb)
        # request_pad_simple chỉ có từ GStreamer >=1.20; DS6.0/Nano (Ubuntu 18.04)
        # dùng GStreamer 1.14 -> fallback get_request_pad (API cũ, tương đương).
        if hasattr(streammux, "request_pad_simple"):
            sinkpad = streammux.request_pad_simple(f"sink_{i}")
        else:
            sinkpad = streammux.get_request_pad(f"sink_{i}")
        srcpad = sb.get_static_pad("src")
        srcpad.link(sinkpad)
        print(f"[INIT] Đã add source {i}: {uri.split('@')[-1]}")

    # PGIE
    pgie = make_elm("nvinfer", "primary-inference")
    pgie.set_property("config-file-path", PGIE_CONFIG)
    pgie.set_property("batch-size", len(RTSP_SOURCES))
    pipeline.add(pgie)

    # tracker
    tracker = make_elm("nvtracker", "tracker")
    tracker.set_property("ll-lib-file",
                        "/opt/nvidia/deepstream/deepstream-6.0/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property("ll-config-file", TRACKER_CONFIG)
    tracker.set_property("tracker-width", 640)
    tracker.set_property("tracker-height", 384)
    tracker.set_property("display-tracking-id", 1)
    pipeline.add(tracker)

    # SGIE
    sgie = make_elm("nvinfer", "secondary-inference")
    sgie.set_property("config-file-path", SGIE_CONFIG)
    pipeline.add(sgie)

    # sink (fakesink — không output video, đúng option 3)
    sink = make_elm("fakesink", "fakesink")
    sink.set_property("sync", 0)
    sink.set_property("async", 0)
    pipeline.add(sink)

    # Link: streammux → pgie → tracker → sgie → sink
    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(sgie)

    # Nhánh capture LUÔN BẬT (không qua EDGE_DEBUG): queue(leaky, tách thread để
    # không nghẽn ngược lên GStreamer chính) → throttle (chặn sớm TRƯỚC khi tốn
    # GPU convert) → nvvideoconvert NV12->RGBA (pyds.get_nvds_buf_surface chỉ đọc
    # được RGBA) → capture_frame_probe đẩy vào ROLLING_BUFFER cho publish_fall_async.
    # EDGE_DEBUG=1 gắn thêm debug_crop_probe trên CÙNG pad (không đụng nhau).
    cap_queue = make_elm("queue", "capture-queue")
    cap_queue.set_property("leaky", 2)
    cap_queue.set_property("max-size-buffers", 5)
    cap_conv = make_elm("nvvideoconvert", "capture-convert")
    cap_caps = make_elm("capsfilter", "capture-caps")
    cap_caps.set_property(
        "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"))
    pipeline.add(cap_queue)
    pipeline.add(cap_conv)
    pipeline.add(cap_caps)
    sgie.link(cap_queue)
    cap_queue.link(cap_conv)
    cap_conv.link(cap_caps)
    cap_caps.link(sink)

    cap_queue_src = cap_queue.get_static_pad("src")
    cap_queue_src.add_probe(Gst.PadProbeType.BUFFER, capture_throttle_probe, 0)
    cap_caps_src = cap_caps.get_static_pad("src")
    cap_caps_src.add_probe(Gst.PadProbeType.BUFFER, capture_frame_probe, ctx)
    if DEBUG_LABELS:
        cap_caps_src.add_probe(Gst.PadProbeType.BUFFER, debug_crop_probe, 0)

    # Phồng bbox +20% (PAD_RATIO) TRƯỚC khi vào SGIE — khớp crop_with_padding()
    # bên train pose_classifier.onnx (Handoff Tấn Dũng). Gắn trên src pad tracker
    # (đầu vào SGIE), phục hồi lại ở restore_orig_bbox() trong sgie_src_pad_probe.
    tracker_src = tracker.get_static_pad("src")
    tracker_src.add_probe(Gst.PadProbeType.BUFFER, pad_bbox_for_sgie_probe, 0)

    # Probe trên src pad của SGIE để đọc metadata + chạy logic ngã (ctx đi kèm)
    sgie_src = sgie.get_static_pad("src")
    sgie_src.add_probe(Gst.PadProbeType.BUFFER, sgie_src_pad_probe, ctx)

    if os.environ.get("PGIE_RAW_DEBUG") == "1":
        pgie_src = pgie.get_static_pad("src")
        pgie_src.add_probe(Gst.PadProbeType.BUFFER, pgie_raw_debug_probe, 0)

    # Bus
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Telemetry heartbeat mỗi 30s (Tuần 5) — chạy dù MQTT tắt (lúc đó chỉ log).
    GLib.timeout_add_seconds(TELEMETRY_INTERVAL_SEC, telemetry_tick, ctx)

    # Ctrl+C handler
    def sigint(_sig, _frame):
        print("\n[INIT] SIGINT → tắt pipeline")
        loop.quit()
    signal.signal(signal.SIGINT, sigint)

    print("[INIT] Bắt đầu pipeline...")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        if ctx.publisher is not None:
            ctx.publisher.disconnect()
        print("[INIT] Pipeline đã tắt")


if __name__ == "__main__":
    main()
