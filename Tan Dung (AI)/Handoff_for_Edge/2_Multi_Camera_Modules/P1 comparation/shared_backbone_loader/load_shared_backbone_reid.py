"""Vi du load Shared Backbone (doi thu 1, KHONG dung trong pipeline that)
-- trich embedding tu 1 anh crop nguoi, dung tham khao/doi chieu neu can.

Chay thu: python load_shared_backbone_reid.py
"""
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reid_model import ReIDModel  # noqa: E402

HERE = Path(__file__).resolve().parent
MULTI_CAM_ROOT = HERE.parents[1]           # 2_Multi_Camera_Modules
HANDOFF_ROOT = HERE.parents[2]             # Handoff_for_Edge

POSE_BACKBONE_CKPT = HANDOFF_ROOT / "1_Single_Camera_Pipeline" / "models" / "pose_classifier.pt"
SHARED_BACKBONE_HEAD_CKPT = MULTI_CAM_ROOT / "P1 comparation" / "models" / "reid_shared_backbone_frozen.pt"
NUM_CLASSES_TRAIN = 1041  # so danh tinh MSMT17 luc train (chi de dung shape, khong dung classifier khi suy luan)

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])


def load_model(device="cpu"):
    model = ReIDModel(str(POSE_BACKBONE_CKPT), num_classes=NUM_CLASSES_TRAIN, freeze_backbone=True)
    state_dict = torch.load(str(SHARED_BACKBONE_HEAD_CKPT), map_location=device)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model


def embed_crop(model, pil_image, device="cpu"):
    tensor = TRANSFORM(pil_image).unsqueeze(0).to(device)
    return model.inference_embedding(tensor).cpu().numpy().flatten()


if __name__ == "__main__":
    m = load_model()
    dummy = Image.new("RGB", (100, 200), color=(128, 128, 128))
    v = embed_crop(m, dummy)
    print(f"OK, embedding shape={v.shape}")
