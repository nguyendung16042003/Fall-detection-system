"""Xuat OSNet_x0.25 Baseline (KHONG Distill) sang ONNX, dung DUNG quy uoc
input/output cua export_osnet_kd_onnx.py (input (1,3,256,128) ten 'input',
output (1,512) ten 'embedding') de dong bo dinh dang voi ban KD."""
import sys
from pathlib import Path

import torch
from torchreid.reid.models import build_model
from torchreid.reid.utils import load_pretrained_weights

HERE = Path(__file__).resolve().parent
CKPT = HERE / "osnet_x0_25_baseline_msmt17_best.pt"
OUT_ONNX = HERE / "osnet_x0_25_baseline_msmt17.onnx"


class EmbeddingOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


def main():
    model = build_model("osnet_x0_25", num_classes=1041, loss="softmax", pretrained=False)
    load_pretrained_weights(model, str(CKPT))
    model.eval()
    wrapped = EmbeddingOnly(model)

    dummy = torch.randn(1, 3, 256, 128)
    torch.onnx.export(
        wrapped, dummy, str(OUT_ONNX),
        input_names=["input"], output_names=["embedding"],
        opset_version=12, dynamo=False,
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
    )
    print(f"Da luu: {OUT_ONNX}")


if __name__ == "__main__":
    main()
