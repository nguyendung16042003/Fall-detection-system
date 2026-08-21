"""Xuat MSINet (doi thu 2, checkpoint pretrained tai ve) sang ONNX, dung DUNG
quy uoc input/output cua export_osnet_kd_onnx.py (input (1,3,256,128) ten
'input', output ten 'embedding') de benchmark Jetson cung 1 quy trinh."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from msinet_model import load_msinet  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_ONNX = HERE / "msinet_msmt17.onnx"


def main():
    model = load_msinet(device="cpu")
    model.eval()

    dummy = torch.randn(1, 3, 256, 128)
    torch.onnx.export(
        model, dummy, str(OUT_ONNX),
        input_names=["input"], output_names=["embedding"],
        opset_version=12, dynamo=False,
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
    )
    print(f"Da luu: {OUT_ONNX}")


if __name__ == "__main__":
    main()
