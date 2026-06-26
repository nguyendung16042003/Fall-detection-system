import pandas as pd

pred_df = pd.read_csv(r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)\Coding\Test\fall_predictions.csv")

gt_df = pd.read_csv(r"D:\DOWLOAD\FileTaiLieuHocTapCuaDung\Ki9\Đồ án\Fall-detection-system\Fall-detection-system\Tan Dung (AI)\Coding\Test\ground_truth.csv"
)

# =========================
# NORMALIZE PATH
# =========================

pred_df["video_name"] = (
    pred_df["video_name"]
    .astype(str)
    .str.replace("\\", "/", regex=False)
)

gt_df["video_name"] = (
    gt_df["video_name"]
    .astype(str)
    .str.replace("\\", "/", regex=False)
)

# =========================
# MERGE
# =========================

df = pred_df.merge(
    gt_df,
    on="video_name"
)

print("Matched videos:", len(df))

# =========================
# CONFUSION MATRIX
# =========================

TP = 0
TN = 0
FP = 0
FN = 0

for _, row in df.iterrows():

    pred = row["prediction"]
    gt = row["ground_truth"]

    if pred == "Fall" and gt == "Fall":
        TP += 1

    elif pred == "NoFall" and gt == "NoFall":
        TN += 1

    elif pred == "Fall" and gt == "NoFall":
        FP += 1

    elif pred == "NoFall" and gt == "Fall":
        FN += 1

total = TP + TN + FP + FN

accuracy = (TP + TN) / total

precision = TP / (TP + FP) if (TP + FP) > 0 else 0

recall = TP / (TP + FN) if (TP + FN) > 0 else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0
    else 0
)

print("\n========== RESULT ==========")

print(f"TP = {TP}")
print(f"TN = {TN}")
print(f"FP = {FP}")
print(f"FN = {FN}")

print(f"\nAccuracy  = {accuracy:.4f}")
print(f"Precision = {precision:.4f}")
print(f"Recall    = {recall:.4f}")
print(f"F1 Score  = {f1:.4f}")