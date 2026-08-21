"""Xuat ONNX MOI cho Boundary Feature Fusion (P3) -- khac pose_classifier.onnx
(Part 1, xuat DU ca pool+drop+linear -> 5 lop cuoi, LUOI DAC TRUNG (B,1280,h,w)
TRUOC do chi la node an trong do thi, KHONG duoc khai bao output nen
TensorRT/nvinfer (ke ca bat output-tensor-meta=1) KHONG the doc ra duoc).

File nay export DUNG SharedBackbone.forward() (shared_backbone.py) --
backbone_layers[0:9] + channel_proj cua pose_classifier.pt Part 1, DUNG,
KHONG train lai/sua trong so -- va khai bao ro luoi dac trung la 1 output ten
"feature_grid" trong do thi ONNX, de Dung (edge) build engine TensorRT rieng,
doc duoc qua output-tensor-meta trong DeepStream, roi moi chay
cross_camera_fuse.py (gop 2 camera) truoc khi phan loai (pool+drop+linear --
buoc nay chay SAU khi gop, KHONG con nam trong file ONNX nay).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_backbone import SharedBackbone  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
CKPT = (AI_ROOT / "Coding" / "training" / "training 3" / "runs" / "classify"
        / "Fall_Detection_Advanced_Loss" / "YOLOv8n_AFCL_Balanced-8" / "weights" / "best.pt")
OUT_ONNX = (AI_ROOT / "Handoff_for_Edge" / "2_Multi_Camera_Modules" / "models"
            / "pose_backbone_feature_grid.onnx")
IMGSZ = 224  # dung DUNG kich thuoc preprocess_crop() cua shared_backbone.py


def main():
    print(f"Nap checkpoint: {CKPT}")
    backbone = SharedBackbone(CKPT).eval()

    dummy = torch.randn(1, 3, IMGSZ, IMGSZ)
    with torch.no_grad():
        grid = backbone(dummy)
    print(f"Grid shape (PyTorch, truoc khi export): {tuple(grid.shape)} "
          f"(ky vong (1,1280,7,7) voi IMGSZ={IMGSZ})")

    OUT_ONNX.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        backbone, dummy, str(OUT_ONNX),
        input_names=["input"], output_names=["feature_grid"],
        opset_version=12, dynamo=False,
        dynamic_axes={"input": {0: "batch"}, "feature_grid": {0: "batch"}},
    )
    print(f"Da luu: {OUT_ONNX}")

    try:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(str(OUT_ONNX), providers=["CPUExecutionProvider"])
        onnx_out = sess.run(None, {"input": dummy.numpy()})[0]
        print(f"Output cua ONNX: shape={onnx_out.shape}")
        diff = float(np.abs(onnx_out - grid.numpy()).max())
        print(f"Sai so toi da ONNX vs PyTorch: {diff:.6f} (ky vong gan 0, < 1e-4)")
        assert diff < 1e-3, "ONNX export lech qua nhieu so voi PyTorch -- KIEM TRA LAI truoc khi giao"
        print("OK -- ONNX khop PyTorch, san sang giao cho Dung (edge) build TensorRT.")
    except ImportError:
        print("(Chua cai onnxruntime -- BO QUA buoc kiem chung. Nen `pip install onnxruntime` "
              "va chay lai truoc khi giao file, de chac chan export dung.)")


if __name__ == "__main__":
    main()
