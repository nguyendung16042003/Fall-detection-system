# conda activate fall
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model(r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall Detection\Datasets\FA\train\images\-Flashback-Man-falls-and-slips-down-icy-driveway-Winter-fail-compilation-_-Laugh-Down_46_jpg.rf.7NCRbkM7iuFuoaruHwp1.jpg")

print("YOLO hoạt động bình thường")