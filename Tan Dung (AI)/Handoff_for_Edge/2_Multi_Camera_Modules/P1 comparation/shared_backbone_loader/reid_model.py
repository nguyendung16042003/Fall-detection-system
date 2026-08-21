"""
Kien truc Re-ID Head cua "Shared Backbone" (doi thu 1, KHONG phai model dang
dung that trong he thong -- xem README.md o thu muc cha). Dung lai backbone
da train cho Pose Head (YOLOv8n-cls) roi gan them Re-ID head (projection +
BNNeck + classifier).

Ban sao tu Code/p1_reid/reid_model.py (may dev), da sua duong dan import de
tu chua trong dung pham vi Handoff_for_Edge nay.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
MULTI_CAM_ROOT = HERE.parents[1]  # 2_Multi_Camera_Modules

sys.path.insert(0, str(MULTI_CAM_ROOT / "pipeline_code" / "boundary_feature_fusion"))
from shared_backbone import SharedBackbone  # noqa: E402

EMBED_DIM = 256


class ReIDModel(nn.Module):
    def __init__(self, backbone_ckpt, num_classes, freeze_backbone=True):
        super().__init__()
        self.backbone = SharedBackbone(backbone_ckpt)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        backbone_dim = 1280  # dung SharedBackbone (channel_proj -> 1280)
        self.embed_fc = nn.Linear(backbone_dim, EMBED_DIM)
        self.bn = nn.BatchNorm1d(EMBED_DIM)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(EMBED_DIM, num_classes, bias=False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def extract_backbone_vector(self, x):
        ctx = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with ctx:
            grid = self.backbone(x)
            v = grid.mean(dim=[2, 3])
        return v

    def forward(self, x):
        v = self.extract_backbone_vector(x)
        embed = self.embed_fc(v)
        embed_bn = self.bn(embed)
        logits = self.classifier(embed_bn)
        return embed, logits

    @torch.no_grad()
    def inference_embedding(self, x):
        """Dung luc test/suy luan that -- CHI lay embedding (sau BN, chuan
        L2), khong dung classifier (chi co nghia luc train tren 1041 lop
        MSMT17 goc, khong dung de phan loai danh tinh moi)."""
        self.eval()
        v = self.extract_backbone_vector(x)
        embed = self.embed_fc(v)
        embed_bn = self.bn(embed)
        return nn.functional.normalize(embed_bn, dim=1)
