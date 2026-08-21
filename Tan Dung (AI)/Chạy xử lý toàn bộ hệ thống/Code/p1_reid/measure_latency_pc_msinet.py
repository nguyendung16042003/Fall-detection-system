"""Do latency/FPS THAT tren chinh may PC dang dung (khong can Jetson) cho
MSINet vs OSNet x0.25 KD -- cung dieu kien: CPU, input (1,3,256,128), batch=1,
50 lan goi lien tiep (bo qua 5 lan warmup dau)."""
import argparse
import sys
import time
from pathlib import Path

import torch

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

N_WARMUP = 5
N_RUNS = 50


def bench(name, model, dummy):
    model.eval()
    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(dummy)
        t0 = time.perf_counter()
        for _ in range(N_RUNS):
            model(dummy)
        t1 = time.perf_counter()
    ms_per_call = (t1 - t0) / N_RUNS * 1000
    fps = 1000 / ms_per_call
    print(f"{name}: {ms_per_call:.2f} ms/call ({fps:.2f} FPS) -- CPU, PC nay")
    return ms_per_call, fps


def main():
    dummy = torch.randn(1, 3, 256, 128)

    from reid.models.msinet import msinet_x1_0
    args = argparse.Namespace(sam_mode="none", source_dataset="msmt17", genotypes="msmt",
                               pretrained=True, pretrain_dir=str(MSINET_REPO / "pretrained"))
    msinet = msinet_x1_0(args, num_classes=1041)
    bench("MSINet (doi thu 2)", msinet, dummy)

    sys.path.insert(0, str(HERE))
    from torchreid.reid.models import build_model
    from torchreid.reid.utils import load_pretrained_weights
    osnet = build_model("osnet_x0_25", num_classes=1041, loss="softmax", pretrained=False)
    load_pretrained_weights(osnet, str(OSNET_CKPT))
    bench("OSNet x0.25 KD (bai cua nhom)", osnet, dummy)


if __name__ == "__main__":
    main()
