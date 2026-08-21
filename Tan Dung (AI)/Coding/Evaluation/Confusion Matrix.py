import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

pred_df = pd.read_csv(r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)\Coding\Test\fall_predictions.csv")
gt_df = pd.read_csv(r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)\Coding\Test\ground_truth.csv")

pred_df["video_name"] = pred_df["video_name"].str.replace("\\", "/", regex=False)
gt_df["video_name"] = gt_df["video_name"].str.replace("\\", "/", regex=False)

df = pred_df.merge(gt_df, on="video_name")

cm = confusion_matrix(
    df["ground_truth"],
    df["prediction"],
    labels=["Fall", "NoFall"]
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fall", "NoFall"]
)

disp.plot()

plt.title("Experiment 1 - Confusion Matrix")
plt.savefig("exp1_confusion_matrix.png")
plt.show()