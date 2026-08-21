"""
Kien truc Re-ID Head cho P1, Giai doan A -- theo dung "Huong dan train test.pdf"
muc 5 Cach 1: dung THANG backbone that cua SGIE (YOLOv8n-cls, KHONG DCNv2/EMA
-- da xac nhan qua Context xu ly van de 3 (no_training).md), input 224x224
xuyen suot, KHONG qua buoc "OSNet tam roi chuyen trong so".

QUYET DINH quan trong (khac thu tu ly tuong trong PDF muc 1): PDF gia dinh
Giai doan A (Re-ID/MSMT17) chay TRUOC, backbone hoc tu do, roi Giai doan B
(Pose/URFD) fine-tune tiep. Nhung du an nay backbone da duoc train Pose TRUOC
(tu Giai doan A cua MVP, xong tu lau) va P3 da validate toan bo ket qua tren
DUNG checkpoint nay (khong sua). Vi Phase 2 (P3) da xac nhan KHONG can
Phuong an B (finetune Pose Head), backbone hien tai la BAN CUOI, khong nen
dung lai lam thay doi (se lam sai lech ket qua P3 da bao cao). Nen: dong bang
backbone (giu nguyen checkpoint Pose), CHI train Re-ID head (projection +
BNNeck + classifier) tren dac trung da trich. Day la lua chon AN TOAN hon,
danh doi lay rui ro Rank-1/mAP co the thap hon tai lieu tham khao (74-78%)
neu backbone "huong Pose" thieu dac trung phan biet danh tinh -- se do that
va bao cao trung thuc (Test Tang 1), khong bia so.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p3_cross_camera"))
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
        self.bn.bias.requires_grad_(False)  # BNNeck chuan (bag-of-tricks ReID)
        self.classifier = nn.Linear(EMBED_DIM, num_classes, bias=False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()  # backbone luon eval (dong bang BN stats)
        return self

    def extract_backbone_vector(self, x):
        ctx = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with ctx:
            grid = self.backbone(x)  # (B,1280,h,w)
            v = grid.mean(dim=[2, 3])  # GAP -> (B,1280)
        return v

    def forward(self, x):
        v = self.extract_backbone_vector(x)
        embed = self.embed_fc(v)          # dung cho Triplet Loss (truoc BN)
        embed_bn = self.bn(embed)         # dung cho Cross-Entropy (sau BN, BNNeck)
        logits = self.classifier(embed_bn)
        return embed, logits

    @torch.no_grad()
    def inference_embedding(self, x):
        """Dung luc test/suy luan that -- CHI lay embedding (sau BN, chuan L2),
        khong dung classifier (chi co nghia luc train tren 1041 lop MSMT17)."""
        self.eval()
        v = self.extract_backbone_vector(x)
        embed = self.embed_fc(v)
        embed_bn = self.bn(embed)
        return nn.functional.normalize(embed_bn, dim=1)
