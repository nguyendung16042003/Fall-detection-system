import os
import torch
import numpy as np
import pandas as pd  # <--- Sử dụng thư viện Pandas để quản lý và xuất file CSV
from PIL import Image
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix

def main():
    # 1. Thứ tự 6 lớp chuẩn của mô hình v1 đã huấn luyện
    model_classes = ['bend', 'exercise', 'half_person', 'lie', 'sit', 'stand']
    
    # Mảng ánh xạ nhãn thực tế từ tập v2 sang index của mô hình v1
    class_mapping = {
        'bend': 0,
        'exercise': 1,
        'unknown': 2,       # Ánh xạ lớp unknown của v2 vào lớp nhiễu half_person của v1
        'lie': 3,
        'sit': 4,
        'stand': 5
    }

    # 2. ĐƯỜNG DẪN FILE TRỌNG SỐ VÀ THƯ MỤC GỐC TẬP V2
    model_path = r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)\runs\classify\Fall_Detection_Advanced_Loss\YOLOv8n_AFCL_Balanced-5\weights\best.pt"
    test_data_dir = r"D:\v2_split_flat"
    
    # Tên file CSV đầu ra dự kiến
    csv_output_path = "V2_Predictions_Report.csv"
    
    if not os.path.exists(model_path):
        print(f"❌ Lỗi: Không tìm thấy file trọng số tại: {model_path}")
        return
    if not os.path.exists(test_data_dir):
        print(f"❌ Lỗi: Thư mục dataset không tồn tại tại: {test_data_dir}")
        return

    print("🤖 Đang tải mô hình YOLOv8n-cls cải tiến với hàm Loss AFCL...")
    model = YOLO(model_path)
    
    # Khởi tạo danh sách chứa dữ liệu thô phục vụ xuất file CSV nâng cao
    raw_data_records = []
    
    # Định dạng ảnh hợp lệ
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
    sub_sets = ['train', 'val']
    
    total_images_counter = {cls: 0 for cls in class_mapping.keys()}

    print("\n📸 BẮT ĐẦU CÀN QUÉT VÀ TRÍCH XUẤT DỮ LIỆU TẬP V2 RA FILE CSV:")
    print("-" * 75)
    
    # VÒNG LẶP TẦNG 1: Đi vào từng tập 'train' và 'val'
    for sub_set in sub_sets:
        sub_set_path = os.path.join(test_data_dir, sub_set)
        if not os.path.exists(sub_set_path):
            continue
            
        # VÒNG LẶP TẦNG 2: Đi vào từng thư mục tư thế
        for folder_name in os.listdir(sub_set_path):
            folder_path = os.path.join(sub_set_path, folder_name)
            if not os.path.isdir(folder_path) or folder_name not in class_mapping:
                continue
                
            true_label_idx = class_mapping[folder_name]
            images = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
            
            for img_name in images:
                img_path = os.path.join(folder_path, img_name)
                
                try:
                    # Mở file bằng RAM stream chống lỗi Windows Long Path (> 260 ký tự)
                    with open(img_path, 'rb') as f:
                        with Image.open(f) as img:
                            pil_img = img.convert('RGB')
                            results = model(pil_img, verbose=False)
                    
                    pred_label_idx = results[0].probs.top1
                    
                    # Lấy chuỗi ký tự tên nhãn để hiển thị trực quan trong CSV
                    true_label_name = folder_name
                    pred_label_name = model_classes[pred_label_idx]
                    
                    # Đánh giá xem mô hình đoán Đúng hay Sai cho ảnh này
                    is_correct = "ĐÚNG" if true_label_name == pred_label_name or (true_label_name == "unknown" and pred_label_name == "half_person") else "SAI"
                    
                    # Gom dữ liệu thô vào danh sách dòng của CSV
                    raw_data_records.append({
                        "Folder_Source": sub_set,         # Thuộc tập train hay val của v2
                        "Image_Name": img_name,           # Tên file ảnh thực tế
                        "True_Label_Index": true_label_idx,
                        "True_Label_Name": true_label_name,
                        "Pred_Label_Index": pred_label_idx,
                        "Pred_Label_Name": pred_label_name,
                        "Result": is_correct              # Kết quả đối chứng trực quan
                    })
                    
                    total_images_counter[folder_name] += 1
                    
                except Exception:
                    continue

    print("-" * 75)
    for cls, count in total_images_counter.items():
        print(f"● Đã quét lớp [{cls:<11}] (Train + Val v2): {count:>5} ảnh.")
    print("-" * 75)
    
    if len(raw_data_records) == 0:
        print("❌ Lỗi chí mạng: Không quét được ảnh nào để xuất báo cáo!")
        return

    # =========================================================================
    # 3. TIẾN HÀNH ĐÓNG GÓI VÀ XUẤT FILE CSV VÀO Ổ CỨNG
    # =========================================================================
    print("💾 Đang khởi tạo ma trận Pandas DataFrame và cấu trúc file CSV...")
    df = pd.DataFrame(raw_data_records)
    
    # Xuất file CSV mã hóa UTF-8 để hiển thị được chữ tiếng Việt "ĐÚNG / SAI" không lỗi font trong Excel
    df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
    print(f"🎯 XUẤT FILE THÀNH CÔNG! File báo cáo thô lưu tại: {os.path.abspath(csv_output_path)}")
    print("-" * 75)

    # =========================================================================
    # 4. IN NHANH SỐ LIỆU LÊN TERMIMAL ĐỂ XEM TRƯỚC
    # =========================================================================
    y_true = df["True_Label_Index"].tolist()
    y_pred = df["Pred_Label_Index"].tolist()

    print("\n" + "="*65)
    print("📊 BẢNG CHỈ SỐ HIỆU NĂNG TỔNG HỢP TRÊN TẬP V2 (CLASSIFICATION REPORT)")
    print("="*65)
    print(classification_report(y_true, y_pred, target_names=model_classes, zero_division=0))
    
    print("="*65)
    print("🧱 MA TRẬN NHẦM LẪN TỔNG HỢP (COMBINED CONFUSION MATRIX)")
    print("="*65)
    print(confusion_matrix(y_true, y_pred))
    print("="*65)

if __name__ == "__main__":
    main()