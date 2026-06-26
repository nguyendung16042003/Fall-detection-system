import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)\Coding\Test\fall_predictions.csv")

plt.figure(figsize=(10,5))

plt.hist(
    df["lie_ratio"],
    bins=20
)

plt.axvline(
    0.4,
    linestyle="--"
)

plt.xlabel("Lie Ratio")
plt.ylabel("Number of Videos")
plt.title("Experiment 1 - Lie Ratio Distribution")

plt.savefig("exp1_lie_ratio_distribution.png")

plt.show()