import os
import torch
import torch.nn as nn
from ultralytics import YOLO

# Khóa cấu hình chia sẻ bộ nhớ tránh xung đột trên Windows
if os.name == 'nt':
    torch.multiprocessing.set_sharing_strategy('file_system')

# =========================================================================
# 1. ĐỊNH NGHĨA HÀM MẤT MÁT ASYMMETRIC FOCAL COGNITIVE LOSS (AFCL)
# =========================================================================
class AdvancedAFCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        weights_val = [1.3903, 1.4013, 0.7561, 1.3129, 0.9958, 0.6763]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_weights = torch.FloatTensor(weights_val).to(self.device)
        self.gamma = 2.0 

    def forward(self, preds, targets):
        probs = torch.softmax(preds, dim=-1)
        one_hot_targets = torch.zeros_like(preds).scatter_(1, targets.unsqueeze(1), 1.0)
        pt = (probs * one_hot_targets).sum(dim=-1)
        batch_weights = self.class_weights[targets]
        loss = - batch_weights * ((1.0 - pt) ** self.gamma) * torch.log(pt + 1e-6)
        return loss.mean()

# =========================================================================
# 2. KHỞI CHẠY TIẾN TRÌNH HUẤN LUYỆN ĐIỀU TỐC CÂN BẰNG
# =========================================================================
def main():
    if not torch.cuda.is_available():
        print("⚠️ Cảnh báo: PyTorch chưa nhận được GPU CUDA!")
        device = "cpu"
    else:
        print(f"🖥️ Thiết bị: {torch.cuda.get_device_name(0)}")
        device = 0

    model = YOLO("yolov8n-cls.pt")
    
    custom_loss = AdvancedAFCLoss()
    model.model.criterion = custom_loss  
    print("🧠 Tích hợp thành công hàm mất mát tùy biến AFCL!")

    # CẤU HÌNH CÂN BẰNG: ĐẢM BẢO TỐC ĐỘ NHANH - MÁY MƯỢT KHÔNG SẬP
    model.train(
        data=r"D:\v4_split_flat", 
        epochs=13,           
        imgsz=224,           
        batch=128,           # 🚀 HẠ XUỐNG 128: Giảm tải áp lực nạp ảnh cho CPU, giúp giải nén nhanh hơn
        workers=4,           # 🚀 TĂNG LÊN 2: Tận dụng 2 luồng CPU bốc ảnh gối đầu trước lên RAM ảo, triệt tiêu độ trễ 3 giây của ổ cứng
        device=device,       
        project="Fall_Detection_Advanced_Loss",
        name="YOLOv8n_AFCL_Balanced",
        optimizer="AdamW",   
        lr0=0.001,           
        cache=False,         # Tuyệt đối giữ False để RAM vật lý luôn trống trải, máy mượt 100%
        plots=True           
    )

    print("💾 Huấn luyện hoàn tất!")

if __name__ == "__main__":
    main()