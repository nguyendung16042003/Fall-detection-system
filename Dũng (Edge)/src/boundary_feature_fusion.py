#!/usr/bin/env python3
"""Boundary Feature Fusion — gộp feature map 2 camera theo trọng số epipolar
(THUẦN toán, không train) khi Identity Association (module 2) ghép được 1
người xuất hiện ở CẢ 2 cam CÙNG LÚC. Port từ `tan_dung_gui/Handoff_for_Edge/
2_Multi_Camera_Modules/pipeline_code/boundary_feature_fusion/
cross_camera_fuse.py` — giữ NGUYÊN công thức toán, chỉ đổi cách chạy backbone:
bản gốc dùng PyTorch (`shared_backbone.py`, cần `ultralytics` — không cài
được trên Jetson Nano Python 3.6.9), bản này dùng thẳng TensorRT (đã có sẵn
trên máy, không cần cài gì thêm) qua engine `pose_classifier_backbone_grid.onnx`
(export 2026-08-20 — xem CLAUDE.md/lịch sử chat: cùng trọng số với SGIE sản
xuất `pose_classifier_v2.onnx`, chỉ dừng SỚM HƠN, trước bước pool/dropout/
linear, để lộ ra lưới đặc trưng 7×7×1280 thay vì 5 lớp cuối).

⚠️ CHỈ dùng khi Identity Association ghép được cặp (hiếm, gần như không xảy
ra với calib MẪU hiện tại — xem identity_association.py) — không chạy trên
MỌI object, tránh tốn GPU vô ích.

⚠️ CÒN THIẾU bước cuối: sau khi gộp xong (`v_fused`, vector 1280-d), cần đi
qua đúng lớp `Linear(1280,5)` GỐC của `pose_classifier.pt` (phần "pose head",
dropout ở lúc suy luận không có tác dụng nên có thể bỏ qua) để ra 5 lớp cuối
— trọng số lớp đó CHƯA có trên máy này (engine hiện tại đã cắt bỏ nó). Cần
xin thêm 1 file `.npy` nhỏ (weight+bias của Linear(1280,5)) export từ máy có
`ultralytics` (xem `classify_from_vector()` trong `shared_backbone.py` của
Handoff). `PoseHead.classify()` bên dưới raise NotImplementedError cho tới
lúc có file đó — mọi phần KHÁC (TensorRT inference, distance matrix, gộp
epipolar) đã chạy được, test được ngay bây giờ.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

log = logging.getLogger("boundary_feature_fusion")

ASSUMED_HEIGHTS_M = np.linspace(0.0, 2.0, num=8)

_DEFAULT_H_BY_HEIGHT_PATH = (
    Path(__file__).resolve().parent.parent
    / "configs" / "calibration" / "h_by_height_SAMPLE.json"
)
_DEFAULT_ENGINE_PATH = (
    Path(__file__).resolve().parent.parent
    / "models" / "pose_classifier_backbone_grid.onnx_b1_gpu0_fp16.engine"
)
_DEFAULT_POSE_HEAD_WEIGHTS_PATH = (
    Path(__file__).resolve().parent.parent
    / "models" / "pose_head_linear.npz"  # weight(1280,5) + bias(5) — CHƯA CÓ, xem docstring
)


# ============================================================
# 1. Backbone qua TensorRT (thay PyTorch/ultralytics của bản gốc)
# ============================================================
class TrtGridBackbone:
    """Chạy engine `pose_classifier_backbone_grid.onnx_*.engine` qua TensorRT
    Python API + pycuda trực tiếp (KHÔNG qua nvinfer/DeepStream — engine này
    chỉ cần chạy khi có cặp ghép, không phải mọi object mọi frame, nên tách
    riêng khỏi pipeline GStreamer chính cho đơn giản).

    input: (1,3,224,224) float32 đã tiền xử lý giống hệt SGIE (resize+crop
    giữ tỉ lệ, /255, KHÔNG chuẩn hoá mean/std — xem `preprocess_crop` trong
    `shared_backbone.py` gốc + `net-scale-factor` trong config SGIE hiện tại).
    output: (1,1280,7,7) float32.
    """

    def __init__(self, engine_path: Optional[str] = None):
        import pycuda.driver as cuda
        import tensorrt as trt

        # KHÔNG dùng `import pycuda.autoinit` — nó gắn CUDA context vào thread
        # HIỆN TẠI (main thread lúc build_edge_context() chạy), nhưng forward()
        # thực ra được gọi từ THREAD KHÁC (GStreamer streaming thread, qua
        # capture_frame_probe) — context CUDA là thread-local, gọi từ thread
        # không "sở hữu" context gây lỗi CUDA ngẫu nhiên ("invalid resource
        # handle", "Cask Convolution execution" — đã bắt được thật khi chạy
        # live 2026-08-21, xem lịch sử chat). Tự tạo + tự push/pop context
        # THỦ CÔNG quanh mỗi forward() để đúng bất kể gọi từ thread nào.
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

            self.input_name = self.engine.get_binding_name(0)
            self.output_name = self.engine.get_binding_name(1)
            self.input_shape = tuple(self.engine.get_binding_shape(0))   # (1,3,224,224)
            self.output_shape = tuple(self.engine.get_binding_shape(1))  # (1,1280,7,7)

            self.d_input = cuda.mem_alloc(int(np.prod(self.input_shape)) * 4)
            self.d_output = cuda.mem_alloc(int(np.prod(self.output_shape)) * 4)
            self.stream = cuda.Stream()
        finally:
            self.cuda_ctx.pop()  # nhả context khỏi main thread -- forward() tự push lại khi cần
        log.info("TrtGridBackbone nạp OK: %s (input=%s output=%s)",
                 path.name, self.input_shape, self.output_shape)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (1,3,224,224) float32 contiguous. Trả về (1,1280,7,7) float32."""
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


def preprocess_crop_for_grid(crop_bgr: np.ndarray, imgsz: int = 224) -> np.ndarray:
    """Crop OpenCV BGR (H,W,3) -> (1,3,imgsz,imgsz) float32, khớp tiền xử lý
    SGIE hiện tại (resize giữ tỉ lệ qua padding, KHÔNG mean/std normalize —
    net-scale-factor=1/255 trong config nvinfer đã lo việc /255)."""
    import cv2
    h, w = crop_bgr.shape[:2]
    scale = imgsz / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(crop_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    return chw[np.newaxis, ...]


# ============================================================
# 2. H_by_height calib (nội suy theo chiều cao — port homography_by_height.py)
# ============================================================
def load_h_by_height(path: Optional[str] = None) -> Dict[str, Dict[float, np.ndarray]]:
    """{"cam_1": {height_m: H 3x3, ...}, "cam_2": {...}}."""
    import json
    p = Path(path) if path else _DEFAULT_H_BY_HEIGHT_PATH
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for key, by_h in raw.items():
        cam_id = "cam_" + key.strip().split()[-1]
        out[cam_id] = {float(h): np.array(mat, dtype=np.float64) for h, mat in by_h.items()}
    log.info("Đã nạp h_by_height calib (%s): %s", p.name, list(out.keys()))
    return out


# ============================================================
# 3. Công thức epipolar + gộp feature (port cross_camera_fuse.py — GIỮ NGUYÊN
#    toán học, chỉ đổi torch -> numpy vì chạy tay trên grid nhỏ 7x7, không
#    cần GPU cho bước gộp này (chỉ backbone mới cần GPU/TensorRT ở trên).
# ============================================================
def _fit_line_least_squares(points) -> Tuple[float, float, float]:
    pts = np.array(points, dtype=np.float64)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, vt = np.linalg.svd(centered)
    direction = vt[0]
    a, b = -direction[1], direction[0]
    norm = np.hypot(a, b)
    if norm < 1e-9:
        a, b = 0.0, 1.0
        norm = 1.0
    a, b = a / norm, b / norm
    c = -(a * centroid[0] + b * centroid[1])
    return a, b, c


def _apply_homography_point(H, point_xy):
    x, y = point_xy
    p = np.array([x, y, 1.0])
    p_proj = H @ p
    p_proj = p_proj / p_proj[2]
    return (float(p_proj[0]), float(p_proj[1]))


def approximate_epipolar_line(p1_floor_point, H_by_height_target_cam,
                               assumed_heights_m=ASSUMED_HEIGHTS_M):
    candidate_points = []
    for h in assumed_heights_m:
        H_h = H_by_height_target_cam[float(h)]
        H_h_inv = np.linalg.inv(H_h)
        candidate_points.append(_apply_homography_point(H_h_inv, p1_floor_point))
    return _fit_line_least_squares(candidate_points)


def _point_to_line_distance(p2_xy, line_abc):
    a, b, c = line_abc
    x2, y2 = p2_xy
    return abs(a * x2 + b * y2 + c) / np.hypot(a, b)


def build_distance_matrices(grid_hw, foot_point_cam1, foot_point_cam2,
                             H_by_height_cam1, H_by_height_cam2,
                             bbox_cam1, bbox_cam2):
    h, w = grid_hw
    n = h * w
    line_1to2 = approximate_epipolar_line(foot_point_cam1, H_by_height_cam2)
    line_2to1 = approximate_epipolar_line(foot_point_cam2, H_by_height_cam1)

    def grid_to_full_frame_px(bbox):
        x1, y1, x2, y2 = bbox
        bbox_w, bbox_h = x2 - x1, y2 - y1
        return [
            (x1 + (gx + 0.5) / w * bbox_w, y1 + (gy + 0.5) / h * bbox_h)
            for gy in range(h) for gx in range(w)
        ]

    grid_px_cam2 = grid_to_full_frame_px(bbox_cam2)
    grid_px_cam1 = grid_to_full_frame_px(bbox_cam1)

    dist_1to2 = np.zeros((n, n), dtype=np.float32)
    dist_2to1 = np.zeros((n, n), dtype=np.float32)
    for j, (px, py) in enumerate(grid_px_cam2):
        dist_1to2[:, j] = _point_to_line_distance((px, py), line_1to2)
    for j, (px, py) in enumerate(grid_px_cam1):
        dist_2to1[:, j] = _point_to_line_distance((px, py), line_2to1)
    return dist_1to2, dist_2to1


def _epipolar_weight(dist_matrix: np.ndarray, sigma: float) -> np.ndarray:
    raw = np.exp(-(dist_matrix ** 2) / (2 * sigma ** 2))
    return raw / (raw.sum(axis=-1, keepdims=True) + 1e-8)


def cross_camera_fuse(F1_grid, F2_grid, dist_1to2, dist_2to1, sigma=2.0):
    """F1_grid, F2_grid: (hw, d) numpy, đã flatten từ (d,h,w). sigma=2.0 —
    giá trị đã sweep thực nghiệm trong Handoff, KHÔNG phải học được."""
    w_1to2 = _epipolar_weight(dist_1to2, sigma)
    w_2to1 = _epipolar_weight(dist_2to1, sigma)
    F1_enriched = F1_grid + w_1to2 @ F2_grid
    F2_enriched = F2_grid + w_2to1 @ F1_grid
    return F1_enriched, F2_enriched


def fuse_to_single_vector(F1_enriched: np.ndarray, F2_enriched: np.ndarray) -> np.ndarray:
    v1 = F1_enriched.mean(axis=0)
    v2 = F2_enriched.mean(axis=0)
    return (v1 + v2) / 2.0


def grid_to_flat(grid: np.ndarray):
    """(1,d,h,w) -> (hw,d)."""
    _, d, h, w = grid.shape
    return grid.reshape(d, h * w).T, (h, w)


# ============================================================
# 4. Pose head cuối (Linear 1280->5) — CHƯA CÓ TRỌNG SỐ, xem docstring đầu file.
# ============================================================
class PoseHead:
    CLASS_NAMES = ["bend", "exercise", "lie", "sit", "stand"]  # khớp sgie_labels_v2.txt

    def __init__(self, weights_path: Optional[str] = None):
        path = Path(weights_path) if weights_path else _DEFAULT_POSE_HEAD_WEIGHTS_PATH
        self.ready = path.exists()
        if self.ready:
            data = np.load(path)
            # nn.Linear.weight của PyTorch lưu (out_features, in_features) =
            # (5,1280) -- transpose ngay lúc nạp để dùng "v @ W" trực tiếp.
            self.W = data["weight"].T  # (1280,5)
            self.b = data["bias"]      # (5,)
        else:
            log.warning(
                "Chưa có %s (weight Linear(1280,5) cuối) -> PoseHead.classify() sẽ "
                "raise NotImplementedError. Xem docstring đầu boundary_feature_fusion.py.",
                path.name,
            )

    def classify(self, v: np.ndarray) -> Tuple[str, float]:
        if not self.ready:
            raise NotImplementedError(
                "Thiếu models/pose_head_linear.npz (weight lớp Linear(1280,5) cuối "
                "của pose_classifier.pt) -- xin thêm từ máy có ultralytics.")
        v = np.asarray(v).reshape(-1)  # (1280,) -- chấp nhận cả (1280,) lẫn (1,1280)
        logits = v @ self.W + self.b
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        idx = int(probs.argmax())
        return self.CLASS_NAMES[idx], float(probs[idx])


# ============================================================
# 5. BoundaryFusionEngine — gói gọn cả pipeline (backbone TensorRT + epipolar +
#    pose head) thành 1 hàm duy nhất `classify_pair()` cho pipeline.py gọi.
#    Chỉ khởi tạo (nạp engine + calib) MỘT LẦN lúc build_edge_context(); nếu
#    thiếu file nào (engine/calib/npz) thì raise ngay ở __init__ — pipeline.py
#    bắt exception đó, log lỗi, chạy KHÔNG có module 3 (không chết pipeline).
# ============================================================
class BoundaryFusionEngine:
    def __init__(self, engine_path: Optional[str] = None,
                 h_by_height_path: Optional[str] = None,
                 pose_head_path: Optional[str] = None,
                 sigma: float = 2.0):
        self.backbone = TrtGridBackbone(engine_path)
        self.H_by_height = load_h_by_height(h_by_height_path)
        self.pose_head = PoseHead(pose_head_path)
        if not self.pose_head.ready:
            raise FileNotFoundError(
                "Thiếu models/pose_head_linear.npz -- BoundaryFusionEngine cần đủ "
                "3 file (engine + h_by_height calib + pose_head npz) mới chạy được.")
        self.sigma = sigma

    def classify_pair(self, crop1_bgr: np.ndarray, crop2_bgr: np.ndarray,
                       ground_xy1: Tuple[float, float], ground_xy2: Tuple[float, float],
                       bbox1_xyxy, bbox2_xyxy) -> Tuple[str, float]:
        """1 lệnh gọi duy nhất: 2 crop RGB thô (đã cắt theo bbox THẬT, chưa
        resize) của CÙNG 1 người thấy ở CẢ 2 cam -> (label, conf) đã gộp
        epipolar. Chạy backbone TensorRT 2 lần (crop1, crop2), fusion+pose head
        thuần numpy (rẻ, không cần GPU thêm)."""
        x1 = preprocess_crop_for_grid(crop1_bgr)
        x2 = preprocess_crop_for_grid(crop2_bgr)
        grid1 = self.backbone.forward(x1)
        grid2 = self.backbone.forward(x2)
        F1, hw = grid_to_flat(grid1)
        F2, _ = grid_to_flat(grid2)

        dist_1to2, dist_2to1 = build_distance_matrices(
            hw, ground_xy1, ground_xy2,
            self.H_by_height["cam_1"], self.H_by_height["cam_2"],
            bbox1_xyxy, bbox2_xyxy,
        )
        F1_enriched, F2_enriched = cross_camera_fuse(
            F1, F2, dist_1to2, dist_2to1, sigma=self.sigma)
        v_fused = fuse_to_single_vector(F1_enriched, F2_enriched)
        return self.pose_head.classify(v_fused)
