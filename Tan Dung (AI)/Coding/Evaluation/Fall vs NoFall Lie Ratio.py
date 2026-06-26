import pandas as pd
import matplotlib.pyplot as plt

pred_df = pd.read_csv(r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)\Coding\Test\fall_predictions.csv")
gt_df = pd.read_csv(r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)\Coding\Test\ground_truth.csv")

pred_df["video_name"] = pred_df["video_name"].str.replace("\\", "/", regex=False)
gt_df["video_name"] = gt_df["video_name"].str.replace("\\", "/", regex=False)

df = pred_df.merge(gt_df, on="video_name")

fall = df[df["ground_truth"]=="Fall"]["lie_ratio"]

nofall = df[df["ground_truth"]=="NoFall"]["lie_ratio"]

plt.figure(figsize=(8,5))

plt.boxplot(
    [fall, nofall],
    labels=["Fall", "NoFall"]
)

plt.ylabel("Lie Ratio")

plt.title("Experiment 1 - Lie Ratio Comparison")

plt.savefig("exp1_boxplot.png")

plt.show()