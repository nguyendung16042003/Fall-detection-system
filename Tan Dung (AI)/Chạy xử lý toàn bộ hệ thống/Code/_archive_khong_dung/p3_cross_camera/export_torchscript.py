"""
Xuat SharedBackbone sang TorchScript de chay tren Jetson (Python 3.6, khong
co ultralytics -- xem ghi chu trong shared_backbone.py). TorchScript la
"bien dich" san do thi tensor, KHONG can class Python goc (Conv/C2f cua
ultralytics) luc load lai -- chi can torch thuan tuy.

Xuat 2 phan rieng (input shape khac nhau):
  1. backbone_forward.pt  -- (1,3,224,224) -> (1,1280,7,7) grid dac trung
  2. classify_from_vector.pt -- (1,1280) -> (1,5) logits
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_backbone import SharedBackbone  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
CKPT = (AI_ROOT / "Coding" / "training" / "training 3" / "runs" / "classify"
        / "Fall_Detection_Advanced_Loss" / "YOLOv8n_AFCL_Balanced-8" / "weights" / "best.pt")
OUT_DIR = Path(r"C:\Users\ADMIN\AppData\Local\Temp\claude\d--DOWLOAD-FileTaiLieuHocTapCuaDung-Ki9-----n-Fall-detection-system-Fall-detection-system-Tan-Dung--AI-\c85b318a-c9ee-44d9-921e-d40c735bfdbb\scratchpad\torchscript_export")
# LUU Y: torch.jit.save() khong xu ly duoc duong dan co ky tu tieng Viet
# (khac dau) trong ten thu muc goc du an -- loi thuc su (khong phai chi loi
# in console nhu cac cho khac), nen phai luu tam ra scratchpad (duong dan
# ASCII) roi copy/scp file .pt sang Jetson tu day.
OUT_DIR.mkdir(parents=True, exist_ok=True)


class BackboneForwardWrapper(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


class ClassifyFromVectorWrapper(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.pose_head = backbone.pose_head

    def forward(self, v):
        v = self.pose_head.drop(v)
        return self.pose_head.linear(v)


def main():
    print("Nap SharedBackbone...")
    backbone = SharedBackbone(CKPT).eval()

    dummy_img = torch.randn(1, 3, 224, 224)
    dummy_vec = torch.randn(1, 1280)

    print("Trace backbone_forward...")
    wrapper1 = BackboneForwardWrapper(backbone).eval()
    traced1 = torch.jit.trace(wrapper1, dummy_img)
    out1 = traced1(dummy_img)
    print("  output shape:", out1.shape)
    traced1.save(str(OUT_DIR / "backbone_forward.pt"))

    print("Trace classify_from_vector...")
    wrapper2 = ClassifyFromVectorWrapper(backbone).eval()
    traced2 = torch.jit.trace(wrapper2, dummy_vec)
    out2 = traced2(dummy_vec)
    print("  output shape:", out2.shape)
    traced2.save(str(OUT_DIR / "classify_from_vector.pt"))

    # Kiem tra lai: nap tu file (khong dung backbone goc) so voi ket qua that
    reloaded1 = torch.jit.load(str(OUT_DIR / "backbone_forward.pt"))
    reloaded2 = torch.jit.load(str(OUT_DIR / "classify_from_vector.pt"))
    with torch.no_grad():
        real_grid = backbone(dummy_img)
        traced_grid = reloaded1(dummy_img)
        diff = (real_grid - traced_grid).abs().max().item()
        print(f"Sai so toi da giua backbone that vs TorchScript (grid): {diff:.8f}")

        real_logits = backbone.classify_from_vector(real_grid.mean(dim=[2, 3]))
        traced_logits = reloaded2(traced_grid.mean(dim=[2, 3]))
        diff2 = (real_logits - traced_logits).abs().max().item()
        print(f"Sai so toi da (classify_from_vector): {diff2:.8f}")

    print(f"\nDa luu vao: {OUT_DIR}")
    print("File can copy sang Jetson: backbone_forward.pt, classify_from_vector.pt")


if __name__ == "__main__":
    main()
