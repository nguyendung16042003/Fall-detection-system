"""
P3 -- Shared Feature Backbone. Tai nguyen ven checkpoint classifier da co san
tu Giai doan A (YOLOv8n-cls, 5 lop, KHONG DCNv2/EMA -- xem
"Context xu ly van de 3 (no_training).md" muc 2), nhung LAY OUTPUT TRUOC
buoc pool -- giu dang luoi khong gian (B, d, h, w) thay vi vector da GAP.

Kien truc thuc te (kiem tra qua .model[i]):
  layer 0-8: cac block Conv/C2f thuong (khong DCNv2/EMA) -> feature map 256 kenh
  layer 9 "Classify": conv (256->1280, 1x1) -> pool (AdaptiveAvgPool2d) -> drop -> linear (1280->5)

SharedBackbone = layer 0-8 + layer9.conv (BO QUA pool/drop/linear) -> (B,1280,h,w)
PoseHead (tach rieng) = layer9.pool + layer9.drop + layer9.linear, nhan vao
(B,1280) da GAP -- dung lai KHONG SUA khi khong gop (1 cam) HOAC khi da gop
(cross_camera_fuse.fuse_to_single_vector() da tu GAP ve (B,1280) roi).
"""
import torch
import torch.nn as nn


class SharedBackbone(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        from ultralytics import YOLO
        full_model = YOLO(str(checkpoint_path)).model
        # layer 0..8: backbone thuong. layer 9 la Classify head -- chi lay .conv
        # cua no (projection 256->1280), BO .pool/.drop/.linear.
        self.backbone_layers = full_model.model[:9]
        self.channel_proj = full_model.model[9].conv  # Conv2d(256,1280,1x1)+BN+SiLU
        self.pose_head = full_model.model[9]  # giu nguyen ca block de dung lai pool+drop+linear

    def forward(self, x):
        """x: (B,3,H,W) da chuan hoa dung nhu luc train/infer classifier thuong.
        Tra ve (B, d, h, w) -- d=1280, h,w tuy kich thuoc input (vd 224x224 -> 7x7)."""
        for layer in self.backbone_layers:
            x = layer(x)
        x = self.channel_proj(x)
        return x

    def classify_from_vector(self, v):
        """v: (B, d) da GAP (tu 1 cam hoac tu fuse_to_single_vector). Dung lai
        DUNG pool+drop+linear cua checkpoint goc -- KHONG train lai, KHONG sua.
        Neu v da la vector (khong con h,w), bo qua pool, chi qua drop+linear."""
        v = self.pose_head.drop(v)
        return self.pose_head.linear(v)

    def classify_from_grid(self, grid):
        """Duong di binh thuong (1 cam, khong gop): grid (B,d,h,w) -> pool -> classify.
        Dung de kiem chung SharedBackbone cho ra dung ket qua nhu classifier goc."""
        v = self.pose_head.pool(grid).flatten(1)
        return self.classify_from_vector(v)


_CLASSIFY_TRANSFORM = None


def preprocess_crop(crop_bgr, imgsz=224):
    """Crop OpenCV BGR (H,W,3) -> tensor (1,3,imgsz,imgsz), DUNG DUNG transform
    cua chinh Ultralytics classify (Resize giu ti le + CenterCrop + ToTensor,
    KHONG mean/std normalize). Da xac nhan qua test: resize thang bien dang
    anh (khong CenterCrop) cho ra logits SAI hoan toan so voi classifier goc
    (vd top1 lech tu class 2 sang class 1) -- day la 1 loi thuc su da bat duoc,
    khong phai chi tiet vun vat."""
    global _CLASSIFY_TRANSFORM
    if _CLASSIFY_TRANSFORM is None:
        from ultralytics.data.augment import classify_transforms
        _CLASSIFY_TRANSFORM = classify_transforms(size=imgsz)

    from PIL import Image
    pil_img = Image.fromarray(crop_bgr[:, :, ::-1])  # BGR (cv2) -> RGB (PIL)
    tensor = _CLASSIFY_TRANSFORM(pil_img).unsqueeze(0)  # (1,3,imgsz,imgsz)
    return tensor
