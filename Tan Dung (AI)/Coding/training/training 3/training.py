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
        # Đã bỏ hẳn class half_person (5/7/2026): nó là class "rác" — data gồm
        # crop bbox ngẫu nhiên/không nhất quán từ COCO (che khuất kiểu khác nhau
        # mỗi ảnh), không phải 1 khái niệm tư thế nhất quán như 5 lớp còn lại.
        # Giữ đúng trọng số gốc của AFCL cho 5 lớp còn lại (bỏ vị trí thứ 3 tương
        # ứng half_person trong danh sách alphabet cũ: bend, exercise, half_person,
        # lie, sit, stand), không tự bịa số mới.
        weights_val = [1.3903, 1.4013, 1.3129, 0.9958, 0.6763]
        # Không chốt cứng device ở đây — batch/preds có thể ở cpu hoặc bất kỳ
        # cuda index nào tuỳ tham số device= của model.train(); luôn chuyển
        # class_weights sang đúng device của preds bên trong forward().
        self.register_buffer("class_weights", torch.FloatTensor(weights_val))
        self.gamma = 2.0

    def forward(self, preds, batch):
        # Ultralytics gọi criterion(preds, batch) với batch là CẢ dict (giống
        # v8ClassificationLoss.__call__), không phải tensor nhãn — phải tự lấy
        # batch["cls"], và phải trả về tuple (loss, loss.detach()) để trainer log
        # đúng, nếu không training sẽ lỗi ngay ở batch đầu tiên.
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        targets = batch["cls"].to(preds.device)
        class_weights = self.class_weights.to(preds.device)
        probs = torch.softmax(preds, dim=-1)
        one_hot_targets = torch.zeros_like(preds).scatter_(1, targets.unsqueeze(1), 1.0)
        pt = (probs * one_hot_targets).sum(dim=-1)
        batch_weights = class_weights[targets]
        loss = - batch_weights * ((1.0 - pt) ** self.gamma) * torch.log(pt + 1e-6)
        loss = loss.mean()
        return loss, loss.detach()

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

    # QUAN TRỌNG: model.train() bên trong Ultralytics tạo LẠI một ClassificationModel
    # mới toanh qua trainer.get_model() và chỉ nạp state_dict trọng số từ model cũ,
    # KHÔNG copy attribute Python (criterion) mà ta gán tay ở đây. Gán trực tiếp
    # model.model.criterion trước khi gọi train() do đó bị "rơi mất" — training thật
    # sự sẽ tự init_criterion() -> v8ClassificationLoss() mặc định, không phải AFCL.
    # Fix: đăng ký callback on_train_start, chạy SAU khi trainer đã dựng xong model
    # training thật (BaseTrainer._setup_train) và TRƯỚC vòng lặp epoch đầu tiên.
    def _attach_afcl_loss(trainer):
        trainer.model.criterion = AdvancedAFCLoss()
        print("🧠 Tích hợp thành công hàm mất mát tùy biến AFCL (attached on_train_start)!")

    model.add_callback("on_train_start", _attach_afcl_loss)

    # CẤU HÌNH CÂN BẰNG: ĐẢM BẢO TỐC ĐỘ NHANH - MÁY MƯỢT KHÔNG SẬP
    model.train(
        data=r"D:\v4_split_flat",
        # 13 epoch (cấu hình gốc) quá mỏng để hội tụ (round 1 cần 120 epoch cho
        # cùng kiến trúc). Tăng lên 40 + patience=15 (early stop nếu val không
        # cải thiện sau 15 epoch liên tiếp) để có baseline đáng tin cậy mà không
        # tốn thời gian thừa nếu hội tụ sớm.
        epochs=40,
        patience=15,
        imgsz=224,
        # Đã thử batch=128/workers=4 (crash sớm) và batch=64/workers=6 (crash muộn
        # hơn nhưng vẫn crash) — cả 2 đều dính lỗi Windows "Couldn't open shared
        # file mapping" (page-file limit) khi đẩy workers*batch lên cao. Cấu hình
        # batch=32/workers=4 là cấu hình DUY NHẤT chưa từng crash qua nhiều lần
        # thử — chấp nhận tốc độ vừa phải (~8 batch/s) để không phải train lại
        # nhiều lần nữa.
        batch=32,
        workers=4,
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