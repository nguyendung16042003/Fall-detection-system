#!/usr/bin/env python3
"""Bài test QUYẾT ĐỊNH: chạy ONNX (SGIE) trực tiếp trên CÙNG crop mà DeepStream
đã dùng, rồi so kết quả.

- Khớp nhau (kể cả khớp SAI, vd cả 2 đều ra 'bend' cho người đang đứng)
  -> LỖI MODEL (yếu trên domain video nhà) -> việc của Tấn Dũng.
- LỆCH nhau (ONNX ra nhãn khác/hợp lý hơn DeepStream)
  -> LỖI TIỀN XỬ LÝ trong pipeline (net-scale-factor / color-format / resize)
  -> việc của Dũng, sửa configs/sgie_yolov8n_cls.txt.

Input: các crop .jpg lưu ở output/debug_crops/ (tên file mã hoá sẵn nhãn +
conf mà DeepStream đã ra, xem pipeline.py::maybe_save_debug_crop).

Chạy:
    python3 tools/compare_onnx_deepstream.py
"""
from __future__ import annotations

import glob
import os
import re

import cv2
import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_PATH = os.path.join(ROOT, "models", "yolov8n-cls-datav4.onnx")
LABELS_PATH = os.path.join(ROOT, "labels", "sgie_labels.txt")
CROP_DIR = os.path.join(ROOT, "output", "debug_crops")

# Phải khớp configs/sgie_yolov8n_cls.txt
INFER_SIZE = 224
NET_SCALE_FACTOR = 0.0039215697906911373  # = 1/255
MODEL_COLOR_FORMAT_RGB = True              # model-color-format=0 -> RGB
MAINTAIN_ASPECT_RATIO = True

FNAME_RE = re.compile(r"^(?P<cam>cam_\d+)_f(?P<frame>\d+)_pid(?P<pid>\d+)_"
                       r"(?P<label>[a-z_]+)_(?P<conf>[0-9.]+)\.jpg$")


def letterbox_resize(img: np.ndarray, size: int) -> np.ndarray:
    """Resize giữ tỉ lệ + pad về size x size (khớp maintain-aspect-ratio=1)."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def preprocess(bgr: np.ndarray) -> np.ndarray:
    img = letterbox_resize(bgr, INFER_SIZE) if MAINTAIN_ASPECT_RATIO else \
        cv2.resize(bgr, (INFER_SIZE, INFER_SIZE))
    if MODEL_COLOR_FORMAT_RGB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) * NET_SCALE_FACTOR
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return np.expand_dims(img, axis=0)  # NCHW


def main() -> int:
    labels = [l.strip() for l in open(LABELS_PATH) if l.strip()]
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    files = sorted(glob.glob(os.path.join(CROP_DIR, "*.jpg")))
    if not files:
        print(f"KHÔNG có crop nào trong {CROP_DIR} — chạy pipeline với EDGE_DEBUG=1 trước.")
        return 1

    print(f"{'file':45s} {'DS_label':12s} {'DS_conf':8s} | {'ONNX_label':12s} {'ONNX_conf':9s} | khớp?")
    print("-" * 100)
    n_match, n_total = 0, 0
    for f in files:
        base = os.path.basename(f)
        m = FNAME_RE.match(base)
        if not m:
            continue
        ds_label, ds_conf = m.group("label"), float(m.group("conf"))

        bgr = cv2.imread(f)
        if bgr is None:
            continue
        x = preprocess(bgr)
        out = sess.run(None, {in_name: x})[0][0]  # (6,)
        idx = int(np.argmax(out))
        onnx_label = labels[idx]
        onnx_conf = float(out[idx])

        match = "✅ KHỚP" if onnx_label == ds_label else "❌ LỆCH"
        n_total += 1
        n_match += (onnx_label == ds_label)
        print(f"{base:45s} {ds_label:12s} {ds_conf:<8.2f} | {onnx_label:12s} {onnx_conf:<9.3f} | {match}")

    print("-" * 100)
    print(f"Tổng: {n_total} crop | khớp {n_match} ({100*n_match/max(1,n_total):.0f}%)")
    if n_total == 0:
        return 1
    if n_match / n_total > 0.8:
        print("\n=> KẾT LUẬN: ONNX khớp DeepStream -> KHÔNG PHẢI lỗi tiền xử lý pipeline.")
        print("   Model tự nó phân loại sai/yếu trên domain video này -> việc của Tấn Dũng.")
    else:
        print("\n=> KẾT LUẬN: ONNX LỆCH nhiều so với DeepStream -> NGHI lỗi tiền xử lý")
        print("   trong configs/sgie_yolov8n_cls.txt (net-scale-factor/color-format/resize).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
