import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# 1. ĐƯỜNG DẪN R"..." TỚI 4 ẢNH (1 CẢNH NGÃ + 1 CẢNH NHẶT ĐỒ) CỦA BẠN
# ==============================================================================
# Cảnh Ngã (Fall Event)
FALL_CAM1_PATH = r"C:\Users\ADMIN\Pictures\Screenshots\Screenshot 2026-08-04 132715.png"  # Cảnh Ngã - Cam 1
FALL_CAM2_PATH = r"C:\Users\ADMIN\Pictures\Screenshots\Screenshot 2026-08-04 132746.png"  # Cảnh Ngã - Cam 2

# Cảnh Nhặt đồ (ADL Picking Up Event)
ADL_CAM1_PATH  = r"C:\Users\ADMIN\Pictures\Screenshots\Screenshot 2026-08-04 132813.png"   # Cảnh Nhặt đồ - Cam 1
ADL_CAM2_PATH  = r"C:\Users\ADMIN\Pictures\Screenshots\Screenshot 2026-08-04 132900.png"   # Cảnh Nhặt đồ - Cam 2

# ==============================================================================
# 2. CẤU HÌNH PHÔNG CHỮ HỌC THUẬT (TIMES NEW ROMAN / SERIF)
# ==============================================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Cambria', 'Georgia']
plt.rcParams['text.color'] = '#1a1a1a'

# ==============================================================================
# 3. HÀM GHÉP LƯỚI 2x2 CHO P3
# ==============================================================================
def create_p3_grid_figure(fall_c1, fall_c2, adl_c1, adl_c2, output_filename):
    img_paths = [fall_c1, fall_c2, adl_c1, adl_c2]
    
    # Tiêu đề được điều chỉnh rõ ràng giữa Fall Event và ADL Picking Up Event
    titles = [
        "(a) Boundary Fall Event - Camera 1 View (Truncated View)",
        "(b) Boundary Fall Event - Camera 2 View (Complementary View)",
        "(c) ADL Picking Up Event - Camera 1 View (Bending Motion)",
        "(d) ADL Picking Up Event - Camera 2 View (Complementary View)"
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    
    for i, ax in enumerate(axes.flat):
        img = cv2.imread(img_paths[i])
        
        # Tạo khung ảnh mẫu dự phòng nếu chưa tìm thấy file
        if img is None:
            img = 200 * np.ones((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, f"Missing Image {i+1}", (180, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        ax.imshow(img)
        ax.set_title(titles[i], fontfamily='serif', fontsize=10, fontweight='semibold', pad=8, color='#222222')
        ax.axis('off')

    plt.suptitle("Figure 6.6: Dual-Camera Synchronized Samples for Boundary Fall vs. ADL Picking Up Event (P3)", 
                 fontfamily='serif', fontsize=11.5, fontweight='bold', y=0.98, color='#111111')
    
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()
    print(f"✅ ĐÃ XUẤT HÌNH FIGURE 6.6 P3 (FALL VS ADL): {output_filename}")

# ==============================================================================
# 4. CHẠY TẠO HÌNH
# ==============================================================================
if __name__ == '__main__':
    create_p3_grid_figure(
        FALL_CAM1_PATH, 
        FALL_CAM2_PATH, 
        ADL_CAM1_PATH, 
        ADL_CAM2_PATH, 
        'Figure_6_6_P3_Boundary_Fall_vs_ADL.png'
    )