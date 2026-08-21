"""
Benchmark: PIFR replica (SVM + YOLO11n-pose, sua bug + dung hyperparameter)
vs model moi cua nhom (YOLOv8n detector + classifier 5 lop + rule windowed),
tren CUNG MOT tap test 32 video (6 Fall tu MCFD + 6 NoFall tu URFD ADL) --
dung 32 video da dung o Coding/Test/Test 4/evaluate_pipeline.py.

SVM duoc train tren tap video KHAC (khong trung video test) de tranh data
leakage -- day la diem sua so voi SVM.PY cu (train/test tren cung 1 file).
"""
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, st
                
from fall_rule import detect_fall_events  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[2]
MCFD_DIR = AI_ROOT / "Datasets" / "File Test 2" / "MCFD" / "dataset"
URFD_ADL_DIR = AI_ROOT / "Datasets" / "File Test 2" / "URFD" / "Cam" / "ADL"
OUT_DIR = Path(__file__).resolve().parent

# Tap test. TEST_FALL chi co the la chute01-06 (6 video) vi MCFD chi co 24
# chute, 18 chute con lai da danh cho PIFR train. TEST_ADL mo rong dung HET
# cac video ADL chua tung dung de train PIFR (adl01-06 + adl21-40 = 26 video)
# thay vi chi 6 nhu ban dau -- de so sanh dang tin cay hon (n lon hon).
TEST_FALL = [MCFD_DIR / f"chute{i:02d}" / "cam1.avi" for i in range(1, 7)]
_all_adl = sorted(URFD_ADL_DIR.glob("*.mp4"))
TEST_ADL = _all_adl[:6] + _all_adl[20:]  # adl01-06 + adl21-40, khong trung TRAIN_ADL (adl07-20)

# Tap train SVM -- video KHAC hoan toan, khong trung tap test.
# Dung het chute07-24 (18 video) thay vi chi 10 -- lan dau chi lay 10 video +
# mau 1fps cho ra 155 dong nhung chi 5 dong "lying" (qua it de SVM hoc duoc,
# dan den predict toan "standing" -> Recall 0%). Tang video + lay mau day hon
# (0.3s/lan thay vi 1s/lan) de co du mau lop "lying".
TRAIN_FALL = [MCFD_DIR / f"chute{i:02d}" / "cam1.avi" for i in range(7, 25)]  # chute07-24
TRAIN_ADL = sorted(URFD_ADL_DIR.glob("*.mp4"))[6:20]  # adl thu 7-20


def train_pifr_svm():
    print("[1/3] Trich feature + train PIFR-replica SVM tren video KHONG trung tap test...")
    pose_model = YOLO(str(POSE_MODEL_PATH))

    rows = build_labeled_features(
        TRAIN_FALL + TRAIN_ADL, pose_model,
        is_fall_flags=[True] * len(TRAIN_FALL) + [False] * len(TRAIN_ADL),
        sample_every_s=0.3,
    )
    df = pd.DataFrame(rows, columns=FEATURE_NAMES + ["pose_class"])
    print(f"   Da trich {len(df)} dong feature tu {len(TRAIN_FALL) + len(TRAIN_ADL)} video train")
    print(f"   Phan bo lop: {df['pose_class'].value_counts().to_dict()}")
    df.to_csv(OUT_DIR / "pifr_replica_train_features.csv", index=False, encoding="utf-8-sig")

    X = df[FEATURE_NAMES]
    y = df["pose_class"]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Hyperparameter goc theo Bang 2 bai bao PIFR: C=1.0, kernel=RBF, gamma=0.1.
    # Them class_weight='balanced' -- khong co trong bai bao, nhung can thiet o day
    # vi tap train van lech lop (video Fall da so thoi gian la dang/ngoi, chi 1 doan
    # ngan la nam) -- ghi ro day la bo sung cua nhom, khong phai nguyen ban PIFR.
    clf = SVC(kernel="rbf", C=1.0, gamma=0.1, probability=True, class_weight="balanced")
    clf.fit(Xs, y)

    joblib.dump(clf, OUT_DIR / "pifr_replica_svm.pkl")
    joblib.dump(scaler, OUT_DIR / "pifr_replica_scaler.pkl")
    print("   Da luu pifr_replica_svm.pkl + pifr_replica_scaler.pkl")
    return pose_model, clf, scaler


def run_pifr_on_test_set(pose_model, clf, scaler):
    print("\n[2/3] Chay PIFR-replica tren tap test 32 video...")
    rows = []
    for video_path in TEST_FALL:
        t0 = time.time()
        pred = evaluate_video_fall(video_path, pose_model, clf, scaler)
        rows.append({"video": f"{video_path.parent.name}_{video_path.name}",
                      "ground_truth": "Fall",
                      "prediction": "Fall" if pred else "NoFall",
                      "thoi_gian_s": round(time.time() - t0, 1)})
        print(f"   {rows[-1]}")
    for video_path in TEST_ADL:
        t0 = time.time()
        pred = evaluate_video_fall(video_path, pose_model, clf, scaler)
        rows.append({"video": f"ADL_{video_path.name}",
                      "ground_truth": "NoFall",
                      "prediction": "Fall" if pred else "NoFall",
                      "thoi_gian_s": round(time.time() - t0, 1)})
        print(f"   {rows[-1]}")
    return pd.DataFrame(rows)


def run_our_model_on_test_set():
    print("\n[3/3] Chay model cua nhom (detect+crop+classify+rule) tren tap test...")
    pipeline = DetectClassifyPipeline()
    rows = []
    for video_path, gt in [(v, "Fall") for v in TEST_FALL] + [(v, "NoFall") for v in TEST_ADL]:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        records = pipeline.process_video(video_path, frame_skip=5)
        events = detect_fall_events(records, fps=fps)
        name = f"{video_path.parent.name}_{video_path.name}" if gt == "Fall" else f"ADL_{video_path.name}"
        rows.append({"video": name, "ground_truth": gt, "prediction": "Fall" if events else "NoFall"})
        print(f"   {rows[-1]}")
    return pd.DataFrame(rows)


def compute_metrics(df):
    TP = ((df.ground_truth == "Fall") & (df.prediction == "Fall")).sum()
    FN = ((df.ground_truth == "Fall") & (df.prediction == "NoFall")).sum()
    TN = ((df.ground_truth == "NoFall") & (df.prediction == "NoFall")).sum()
    FP = ((df.ground_truth == "NoFall") & (df.prediction == "Fall")).sum()
    total = TP + FN + TN + FP
    acc = (TP + TN) / total if total else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {"TP": TP, "FN": FN, "TN": TN, "FP": FP,
            "Accuracy": acc, "Precision": precision, "Recall": recall, "F1": f1}


def main():
    pose_model, clf, scaler = train_pifr_svm()
    pifr_df = run_pifr_on_test_set(pose_model, clf, scaler)

    ours_path = OUT_DIR / "our_model_test_results.csv"
    if ours_path.exists():
        print("\n[3/3] Model cua nhom khong doi tu lan chay truoc -- doc lai ket qua da co san.")
        ours_df = pd.read_csv(ours_path)
    else:
        ours_df = run_our_model_on_test_set()
        ours_df.to_csv(ours_path, index=False, encoding="utf-8-sig")

    pifr_df.to_csv(OUT_DIR / "pifr_replica_test_results.csv", index=False, encoding="utf-8-sig")

    pifr_metrics = compute_metrics(pifr_df)
    ours_metrics = compute_metrics(ours_df)

    print("\n" + "=" * 60)
    print("SO SANH TREN CUNG TAP TEST (6 Fall MCFD + 6 NoFall URFD ADL)")
    print("=" * 60)
    compare_df = pd.DataFrame([
        {"He thong": "PIFR replica (sua bug)", **pifr_metrics},
        {"He thong": "Model cua nhom", **ours_metrics},
    ])
    print(compare_df.to_string(index=False))
    compare_df.to_csv(OUT_DIR / "so_sanh_ket_qua.csv", index=False, encoding="utf-8-sig")
    print(f"\nDa luu: {OUT_DIR / 'so_sanh_ket_qua.csv'}")


if __name__ == "__main__":
    main()
