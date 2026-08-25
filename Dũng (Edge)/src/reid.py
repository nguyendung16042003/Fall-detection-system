#!/usr/bin/env python3
"""Re-Identification — nhận lại người khi mất dấu 1 track (đi qua "vùng chết":
mất dấu camera này, xuất hiện lại camera khác/cùng camera sau 1 lúc) bằng cách
so khớp đặc trưng ngoại hình (embedding 512-d) với Gallery đã đăng ký.

Model: OSNet_x0.25 (bản Distill/KD từ OSNet_x1.0) — `reid_osnet_x0_25_kd.onnx`,
theo README `tan_dung_gui/Handoff_for_Edge/2_Multi_Camera_Modules/README.md`
mục 4 ghi "ĐANG DÙNG THẬT" (KHÁC với tóm tắt cũ trong CLAUDE.md nói OSNet_x0.5 —
xem lịch sử chat 2026-08-21, đã đối chiếu source thật và dùng bản mới hơn).
TPR thật trên gallery nhỏ (2 người, đúng bài toán triển khai): 75-76%.

Port từ `tan_dung_gui/Handoff_for_Edge/2_Multi_Camera_Modules/pipeline_code/
reid/reid_matching.py` — CHỈ đổi cách chạy model: bản gốc dùng `torchreid`
(cần Python ≥3.8, không cài được trên Jetson Nano Python 3.6.9 — GIỐNG hệt lý
do đổi cách chạy pose_classifier ở boundary_feature_fusion.py), bản này chạy
thẳng qua TensorRT Python API + pycuda (đã verify: input (1,3,256,128) float32,
output (1,512) float32 — engine build OK trên máy này, xem lịch sử chat).
`PersonGallery`/`TrackEmbeddingBuffer` giữ NGUYÊN logic, không đổi gì.

⚠️ CHƯA nối vào pipeline.py (chỉ mới viết + test module độc lập) — xem
CLAUDE.md/lịch sử chat để biết bước tiếp theo (cần quyết định lúc nào 1 track
được coi là "track MỚI cần tra Gallery" trong kiến trúc DeepStream, khác hẳn
với `.track()` của Ultralytics mà code gốc viết cho).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("reid")

MATCH_THRESHOLD = 0.6   # hiệu chỉnh cho bài toán Gallery (so track mới với vài
                         # người đã đăng ký) — KHÔNG dùng để so trực tiếp 2 crop
                         # cùng lúc từ 2 cam khác nhau (xem README mục 6).
MIN_BUFFER_FRAMES = 5
MAX_BUFFER_FRAMES = 10

_DEFAULT_ENGINE_PATH = (
    Path(__file__).resolve().parent.parent
    / "models" / "reid_osnet_x0_25_kd.onnx_b1_gpu0_fp16.engine"
)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_crop_for_reid(crop_bgr: np.ndarray) -> np.ndarray:
    """Crop OpenCV BGR (H,W,3) -> (1,3,256,128) float32, resize THẲNG (không
    giữ tỉ lệ — khác pose_classifier) + chuẩn hoá ImageNet mean/std, đúng như
    README ghi ("crop người, resize 256×128, chuẩn hoá ImageNet")."""
    import cv2
    resized = cv2.resize(crop_bgr, (128, 256), interpolation=cv2.INTER_LINEAR)
    rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
    rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
    return chw[np.newaxis, ...]


class ReIDExtractorTrt:
    """Chạy engine `reid_osnet_x0_25_kd.onnx_*.engine` qua TensorRT Python API
    + pycuda trực tiếp — KHÔNG qua torchreid (không cài được trên Jetson Nano
    Python 3.6.9). Cùng pattern với TrtGridBackbone trong
    boundary_feature_fusion.py."""

    def __init__(self, engine_path: Optional[str] = None):
        import pycuda.driver as cuda
        import tensorrt as trt

        # KHÔNG dùng `import pycuda.autoinit` — context CUDA của nó gắn vào
        # thread hiện tại (main thread lúc build_edge_context() chạy), nhưng
        # embed() thực ra được gọi từ THREAD KHÁC (GStreamer streaming thread,
        # qua capture_frame_probe) -> lỗi CUDA ngẫu nhiên ("invalid resource
        # handle") đã bắt được thật khi chạy live cùng lúc với
        # BoundaryFusionEngine 2026-08-21 (xem lịch sử chat + chú thích tương
        # tự trong boundary_feature_fusion.py:TrtGridBackbone). Tự tạo + tự
        # push/pop context THỦ CÔNG quanh mỗi lần suy luận.
        cuda.init()
        self.cuda_ctx = cuda.Device(0).make_context()
        self._cuda = cuda
        try:
            path = Path(engine_path) if engine_path else _DEFAULT_ENGINE_PATH
            if not path.exists():
                raise FileNotFoundError(f"Chưa có engine: {path} (build bằng trtexec trước)")

            trt_logger = trt.Logger(trt.Logger.WARNING)
            with open(path, "rb") as f, trt.Runtime(trt_logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()

            self.input_shape = tuple(self.engine.get_binding_shape(0))   # (1,3,256,128)
            self.output_shape = tuple(self.engine.get_binding_shape(1))  # (1,512)

            self.d_input = cuda.mem_alloc(int(np.prod(self.input_shape)) * 4)
            self.d_output = cuda.mem_alloc(int(np.prod(self.output_shape)) * 4)
            self.stream = cuda.Stream()
        finally:
            self.cuda_ctx.pop()  # nhả context khỏi main thread -- _forward() tự push lại khi cần
        log.info("ReIDExtractorTrt nạp OK: %s (input=%s output=%s)",
                 path.name, self.input_shape, self.output_shape)

    def _forward(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        assert x.shape == self.input_shape, f"input shape {x.shape} != {self.input_shape}"
        cuda = self._cuda
        self.cuda_ctx.push()
        try:
            cuda.memcpy_htod_async(self.d_input, x, self.stream)
            self.context.execute_async_v2(
                bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)
            out = np.empty(self.output_shape, dtype=np.float32)
            cuda.memcpy_dtoh_async(out, self.d_output, self.stream)
            self.stream.synchronize()
        finally:
            self.cuda_ctx.pop()
        return out

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        """crop_bgr: 1 ảnh crop người (numpy BGR, từ OpenCV). Trả về vector
        512-d đã L2-normalize (numpy)."""
        x = preprocess_crop_for_reid(crop_bgr)
        emb = self._forward(x).reshape(-1)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 1e-9 else emb


class PersonGallery:
    """Gallery các người đã đăng ký, mỗi người 1 vector embedding trung bình."""

    def __init__(self):
        self._gallery: Dict[str, np.ndarray] = {}

    def register(self, person_name: str, embeddings: List[np.ndarray]) -> None:
        avg = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg)
        self._gallery[person_name] = avg / norm if norm > 1e-9 else avg

    def match(self, query_embedding: np.ndarray,
              threshold: float = MATCH_THRESHOLD) -> Tuple[str, float]:
        """Trả về (person_name, score) nếu khớp (score > threshold), ngược
        lại ("NEW", best_score)."""
        best_name, best_score = None, -1.0
        for name, gal_emb in self._gallery.items():
            score = float(np.dot(query_embedding, gal_emb))
            if score > best_score:
                best_name, best_score = name, score
        if best_score > threshold:
            return best_name, best_score
        return "NEW", best_score


class TrackEmbeddingBuffer:
    """Gom embedding cho 1 track_id qua nhiều frame, trả về embedding trung
    bình khi đã đủ (>= MIN_BUFFER_FRAMES)."""

    def __init__(self):
        self._buffers: Dict[object, List[np.ndarray]] = {}

    def add(self, track_id, embedding: np.ndarray) -> None:
        buf = self._buffers.setdefault(track_id, [])
        if len(buf) < MAX_BUFFER_FRAMES:
            buf.append(embedding)

    def get_average_if_ready(self, track_id) -> Optional[np.ndarray]:
        buf = self._buffers.get(track_id, [])
        if len(buf) < MIN_BUFFER_FRAMES:
            return None
        avg = np.mean(buf, axis=0)
        norm = np.linalg.norm(avg)
        return avg / norm if norm > 1e-9 else avg

    def forget(self, track_id) -> None:
        self._buffers.pop(track_id, None)
