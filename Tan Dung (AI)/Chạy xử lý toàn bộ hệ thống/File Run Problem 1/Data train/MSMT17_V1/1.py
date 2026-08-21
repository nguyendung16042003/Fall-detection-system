import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ==============================================================================
# 1. TRUYỀN ĐƯỜNG DẪN TỚI THƯ MỤC MSMT17_V1 CỦA BẠN VÀO ĐÂY (NẾU CÓ)
# ==============================================================================
DATASET_DIR = r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)\Chạy xử lý toàn bộ hệ thống\File Run Problem 1\Data train\MSMT17_V1"  # Ném đường dẫn thư mục MSMT17 vào đây

splits = ['Train Set', 'Query Set', 'Gallery Set']
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def count_images_in_folder(folder_path):
    if not os.path.exists(folder_path):
        return 0
    return sum(1 for f in os.listdir(folder_path) if f.lower().endswith(VALID_EXTENSIONS))

# Tự động quét số liệu nếu có thư mục thực tế, nếu không có thì dùng số liệu chuẩn CVPR quốc tế
train_img_count = count_images_in_folder(os.path.join(DATASET_DIR, 'train'))
query_img_count = count_images_in_folder(os.path.join(DATASET_DIR, 'query'))
gallery_img_count = count_images_in_folder(os.path.join(DATASET_DIR, 'gallery'))

if train_img_count > 0:
    bboxes = [train_img_count, query_img_count, gallery_img_count]
    # Số ID lấy tương ứng
    ids = [1041, 3060, 3060] 
    print(f"✅ Đã quét thành công dữ liệu từ đường dẫn: {DATASET_DIR}")
else:
    # Số liệu gốc chuẩn quốc tế CVPR của MSMT17_V1
    ids = [1041, 3060, 3060]
    bboxes = [32621, 11659, 82161]
    print("⚠️ Dùng số liệu thống kê chuẩn CVPR của bộ MSMT17_V1.")

# ==============================================================================
# 2. THIẾT LẬP VẼ FIGURE 6.2 (SẠCH SẼ, KHÔNG BỊ ĐÈ CHỮ)
# ==============================================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['text.color'] = '#000000'
plt.rcParams['axes.labelcolor'] = '#000000'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

x = np.arange(len(splits))
width = 0.35

fig, ax1 = plt.subplots(figsize=(8, 4.8), dpi=300)

color_ids = '#2b5c8f'      # Xanh lam đậm
color_boxes = '#d95f02'    # Cam đất

# Trục 1 (bên trái): Cột số lượng IDs
rects1 = ax1.bar(x - width/2, ids, width, label='Identities (IDs)', color=color_ids)
ax1.set_ylabel('Number of Identities (IDs)', fontweight='bold', color=color_ids)
ax1.tick_params(axis='y', labelcolor=color_ids)
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{int(y):,}'))
ax1.set_ylim(0, 3800)

# Trục 2 (bên phải): Cột số lượng Bounding Boxes
ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, bboxes, width, label='Bounding Boxes', color=color_boxes)
ax2.set_ylabel('Number of Bounding Box Images', fontweight='bold', color=color_boxes)
ax2.tick_params(axis='y', labelcolor=color_boxes)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{int(y):,}'))
ax2.set_ylim(0, 95000)

# Tiêu đề biểu đồ
ax1.set_title('Figure 6.2: Dataset Breakdown of MSMT17_V1', fontweight='bold', pad=35, color='black')

ax1.set_xticks(x)
ax1.set_xticklabels(splits, fontweight='bold', color='black')
ax1.grid(axis='y', linestyle='--', alpha=0.3)

# Legend đưa ra ngoài phía trên, không lo bị đè
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True, facecolor='white', edgecolor='gray')

# Ghi số liệu chính xác lên đầu từng cột
for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f'{h:,}', xy=(rect.get_x() + rect.get_width()/2, h), 
                 xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8.5, fontweight='bold', color='#1a3654')

for rect in rects2:
    h = rect.get_height()
    ax2.annotate(f'{h:,}', xy=(rect.get_x() + rect.get_width()/2, h), 
                 xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8.5, fontweight='bold', color='#8c3d01')

plt.tight_layout()
plt.savefig('Figure_6_2_msmt17_breakdown.png', bbox_inches='tight')
plt.close()

print("✅ ĐÃ XUẤT HÌNH: Figure_6_2_msmt17_breakdown.png")