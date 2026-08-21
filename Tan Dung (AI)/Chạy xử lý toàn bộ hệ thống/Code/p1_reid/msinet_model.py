"""Wrapper nap kien truc MSINet that (clone tu github.com/vimar-gu/MSINet) +
checkpoint pretrained tai ve (msinet_msmt.pth.tar, guide buoc 3) -- KHONG tu
train. Cung cap ham embed_fn(crop_bgr) -> np.ndarray (768-d, da L2-normalize),
cung interface voi osnet_embed/shared_embed dang dung trong cac script P1
khac, de tai dung nguyen ham build_gallery/match_gallery/analyze_video.

Tien xu ly DUNG chuan that cua repo (xac nhan qua reid/data/build_data.py
test_transforms): Resize((256,128)) + ToTensor + Normalize ImageNet.

Luu y trung thuc: checkpoint tai ve THIEU nhanh phu f_* (256/768 chieu output)
-- khong co trong state_dict, van random-init. Chi nhanh chinh 512/768 chieu
la co trong so pretrained that. Embedding van chay dung, nhung chat luong co
the kem hon ban checkpoint day du cua paper (khong dung de suy ra Rank-1/mAP).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

HERE = Path(__file__).resolve().parent
MSINET_REPO = HERE / "MSINet_repo"

torch.serialization.add_safe_globals([argparse.Namespace])
_orig_load = torch.load


def _patched_load(*a, **k):
    k["weights_only"] = False
    return _orig_load(*a, **k)


torch.load = _patched_load

MSINET_TRANSFORM = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_msinet(device="cpu"):
    if str(MSINET_REPO) not in sys.path:
        sys.path.insert(0, str(MSINET_REPO))
    from reid.models.msinet import msinet_x1_0

    args = argparse.Namespace(sam_mode="none", source_dataset="msmt17", genotypes="msmt",
                               pretrained=True, pretrain_dir=str(MSINET_REPO / "pretrained"))
    model = msinet_x1_0(args, num_classes=1041).to(device)
    model.eval()
    return model


def make_msinet_embed_fn(device="cpu"):
    model = load_msinet(device)

    def embed(crop_bgr):
        pil_img = Image.fromarray(crop_bgr[:, :, ::-1])
        tensor = MSINET_TRANSFORM(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(tensor)
        feat = torch.nn.functional.normalize(feat, dim=1)
        return feat.cpu().numpy().flatten()

    return embed


if __name__ == "__main__":
    embed = make_msinet_embed_fn()
    dummy_crop = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
    v = embed(dummy_crop)
    print(f"OK, embedding shape={v.shape}, norm={np.linalg.norm(v):.4f}")
