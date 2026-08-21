"""Do Params/FLOPs cho MSINet (doi thu 2) va OSNet x0.25 KD (bai cua nhom),
dung thop.profile, input (1,3,256,128) -- dung buoc 6 trong
MSINet_benchmark_MSMT17_guide( Mo phong doi thu 2).md."""
import argparse
import sys
from pathlib import Path

import torch
from thop import profile

HERE = Path(__file__).resolve().parent
MSINET_REPO = HERE / "MSINet_repo"
OSNET_CKPT = HERE / "osnet_x0_25_kd_msmt17_best.pt"

sys.path.insert(0, str(MSINET_REPO))
torch.serialization.add_safe_globals([argparse.Namespace])
_orig_load = torch.load


def _patched_load(*a, **k):
    k["weights_only"] = False
    return _orig_load(*a, **k)


torch.load = _patched_load


def measure(name, model):
    dummy = torch.randn(1, 3, 256, 128)
    model.eval()
    flops, params = profile(model, inputs=(dummy,), verbose=False)
    print(f"{name}: Params = {params / 1e6:.2f}M, FLOPs = {flops / 1e9:.3f}G")
    return params, flops


def main():
    from reid.models.msinet import msinet_x1_0
    args = argparse.Namespace(sam_mode="none", source_dataset="msmt17", genotypes="msmt",
                               pretrained=True, pretrain_dir=str(MSINET_REPO / "pretrained"))
    msinet = msinet_x1_0(args, num_classes=1041)
    measure("MSINet (doi thu 2)", msinet)

    sys.path.insert(0, str(HERE))
    from torchreid.reid.models import build_model
    from torchreid.reid.utils import load_pretrained_weights
    osnet = build_model("osnet_x0_25", num_classes=1041, loss="softmax", pretrained=False)
    load_pretrained_weights(osnet, str(OSNET_CKPT))
    measure("OSNet x0.25 KD (bai cua nhom)", osnet)


if __name__ == "__main__":
    main()
