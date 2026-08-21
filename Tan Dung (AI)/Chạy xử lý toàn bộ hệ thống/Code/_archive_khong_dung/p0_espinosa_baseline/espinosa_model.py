"""
Tai tao kien truc CNN cua Espinosa et al. (2019), "A vision-based approach for
fall detection using multiple cameras and convolutional neural networks",
Computers in Biology and Medicine.

Paper goc bi chan (403 tren ScienceDirect/ACM/ResearchGate/Academia.edu) -- kien
truc duoi day duoc suy lai tu Hinh 11 cua bai khao sat Alam et al. 2022
(arXiv:2207.10952), bai nay tai hien lai so do khoi cua Espinosa et al. Khong
tim thay code/weight chinh thuc nao cong khai cho bai nay.

Kien truc xac nhan duoc: Conv2D(128) -> Pool -> Conv2D(128) -> Pool ->
Conv2D(64) -> Pool -> Dense(64) -> Dense(128) -> Dense(256) -> Softmax(2).
Input: 1 anh optical-flow xam 38x51 GOP TU 2 CAMERA (early fusion truoc CNN).

2 diem KHONG xac nhan duoc tu paper goc (gia dinh hop ly, ghi ro o day):
  1. Thuat toan optical flow cu the -- gia dinh dung Farneback (thuat toan
     optical-flow dense pho bien nhat trong OpenCV, lua chon mac dinh hop ly
     khi paper khong noi ro).
  2. Toan tu gop 2 camera thanh "1 anh chung" -- gia dinh XEP KENH (channel-
     stack): moi camera la 1 kenh xam rieng, gop thanh input 2 kenh (38,51,2).
     Day la cach don gian nhat khop mo ta "1 anh xam 38x51 dai dien ca 2 cam"
     tu so do khoi (khac tile ngang/doc, khong xac nhan duoc tu van ban).
"""
import torch
import torch.nn as nn

INPUT_H = 38
INPUT_W = 51
IN_CHANNELS = 2  # gop 2 camera bang channel-stack (gia dinh, xem docstring)
N_CLASSES = 2  # 0=ADL, 1=Fall


class EspinosaCNN(nn.Module):
    def __init__(self, in_channels: int = IN_CHANNELS, n_classes: int = N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 38x51 -> 19x25
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 19x25 -> 9x12
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 9x12 -> 4x6
        )
        flat_dim = 64 * 4 * 6
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    m = EspinosaCNN()
    dummy = torch.randn(4, IN_CHANNELS, INPUT_H, INPUT_W)
    out = m(dummy)
    print("Output shape:", out.shape)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"So tham so: {n_params:,}")
