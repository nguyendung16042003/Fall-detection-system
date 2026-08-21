"""
P2 -- So sanh voi Luna et al. 2022 "GNN-CCA" (arXiv:2201.06311), dung DUNG bo
metric clustering ho dung trong Table V cua paper (ARI/AMI/Homogeneity/
Completeness/V-measure, tinh moi frame roi trung binh) -- KHONG chi so sanh
% khop nhu compare_p2_alternatives.py.

Chay tren CA goc LAN augment (4 canh x 5 bien the: goc + sang/toi/nhieu/nen)
-- ban dau chi chay tren goc (168 frame), nguoi dung hoi lai "so luong test
hay van la video test cua toi, da augment chua" -- mo rong day du theo dung
quy uoc an toan cua P2 (KHONG lat/xoay/crop, giu nguyen calib homography).

2 phuong an duoc danh gia tren CUNG data P2 Data 3 that (4 canh, "P2 CO so
sanh ky thuat" trong plan):
  (a) P2 that (Hungarian + homography chan) -- da co san, khong sua.
  (b) "geometrical_association" cua chinh GNN-CCA (threshold khoang cach +
      connected components, KHONG hoc, KHONG can PyTorch Geometric) -- doc
      lai tu inference.py cua ho (dong 628-786), gan nhu giong (a) nhung
      dung threshold+CC thay vi Hungarian toi uu toan cuc.

GROUND TRUTH: KHONG co nhan tay san (P2 Data 3 khong gan ID that). Xay dung
bang CACH: voi moi cap track_id (cam1,cam2) xuat hien qua Hungarian (da
validate dung boi validate.py -- khop ID on dinh, khong lan luc nga), dem
TAN SUAT dong-xuat-hien qua toan bo canh. Cap co tan suat >= MIN_COOCCUR
duoc coi la "cung 1 nguoi that" -- gop thanh Connected Components tren do
thi (cam1_id -- cam2_id). Day la ky thuat bootstrap ground-truth chuan (vote
da so qua nhieu frame loc nhieu 1-2 lan) -- THAY THE cho heuristic ty le
khung hinh da BI HUY trong compare_p2_case_by_case.py (loi do track_id gay
vo lung tung, xem [[feedback_check_unused_camera_files]]) -- o day dung
CHINH tan suat ghep cap (khong phai vi tri/hinh dang) nen khong bi loi do.
"""
import json
import sys
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
import pandas as pd
from sklearn import metrics
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from homography import apply_homography, foot_point_from_bbox  # noqa: E402
from homography_matching import Track, match_two_cameras  # noqa: E402

AI_ROOT = Path(__file__).resolve().parents[3]
P2_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "File Run Problem 2" / "P2 data test"
RESULTS_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p2"

DET_CONF = 0.2
DIST_THRESHOLD = 0.5  # nguong Hungarian da validate (khong doi)
FRAME_SKIP = 3
MIN_COOCCUR = 2  # loc nhieu: cap xuat hien < 2 lan khong tinh la "cung nguoi that"
VARIANTS = ["original", "aug-bright_up", "aug-bright_down", "aug-noise", "aug-compress"]


def video_path(scene_id, cam, variant):
    if variant == "original":
        return P2_DIR / "P2 Data 3" / f"Scene {scene_id}-{cam}.mp4"
    return P2_DIR / "P2 Data 3 (augmented)" / f"Scene {scene_id}-{cam}_{variant}.mp4"

SCENE_NOTE = {
    1: "dung cach 1-1.5m (binh thuong)",
    2: "dung/ngoi SAT nhau (mo ho nhat)",
    3: "A bi che khuat 1 cam (occlusion)",
    4: "A NGA, B dung yen gan do (quan trong nhat)",
}
AMBIGUOUS_SCENES = {2, 3, 4}  # dung sat / che khuat / nga -- dung loai case GNN thiet ke de giai quyet
NORMAL_SCENES = {1}


def load_H():
    with open(RESULTS_DIR / "homography_matrices.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {cam: np.array(H) for cam, H in raw.items()}


def detect_tracks(detector, frame, conf=DET_CONF):
    results = detector.track(frame, classes=[0], conf=conf, persist=True, verbose=False)[0]
    out = []
    if results.boxes is None:
        return out
    for box in results.boxes:
        if box.id is None:
            continue
        out.append({"track_id": int(box.id[0]), "bbox_xyxy": tuple(box.xyxy[0].cpu().numpy().tolist())})
    return out


def geometrical_association_predict(t1_list, t2_list, threshold=DIST_THRESHOLD):
    """Tai hien DUNG logic geometrical_association() cua GNN-CCA (inference.py
    dong 628-786): MOI cap (cam1,cam2) la 1 canh cua do thi 2-phia; canh active
    neu khoang cach < threshold; danh tinh cuoi = Connected Components. KHAC
    Hungarian: khong toi uu toan cuc, 1 track co the "active" voi NHIEU track
    ben kia cung luc (dung dung nhu ho lam, khong sua)."""
    G = nx.Graph()
    for t1 in t1_list:
        G.add_node(("1", t1.track_id))
    for t2 in t2_list:
        G.add_node(("2", t2.track_id))
    for t1 in t1_list:
        for t2 in t2_list:
            dist = np.linalg.norm(np.array(t1.ground_xy) - np.array(t2.ground_xy))
            if dist < threshold:
                G.add_edge(("1", t1.track_id), ("2", t2.track_id))
    return G


def process_scene(scene_id, variant, detector1, detector2, H_floor):
    cap1 = cv2.VideoCapture(str(video_path(scene_id, "CAM 1", variant)))
    cap2 = cv2.VideoCapture(str(video_path(scene_id, "CAM 2", variant)))

    frame_records = []
    frame_idx = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue

        dets1 = detect_tracks(detector1, frame1)
        dets2 = detect_tracks(detector2, frame2)
        if not dets1 or not dets2:
            continue

        t1 = [Track("CAM 1", d["track_id"], d["bbox_xyxy"],
                     ground_xy=apply_homography(H_floor["CAM 1"], foot_point_from_bbox(d["bbox_xyxy"])))
              for d in dets1]
        t2 = [Track("CAM 2", d["track_id"], d["bbox_xyxy"],
                     ground_xy=apply_homography(H_floor["CAM 2"], foot_point_from_bbox(d["bbox_xyxy"])))
              for d in dets2]

        pairs_hungarian = match_two_cameras(t1, t2, distance_threshold=DIST_THRESHOLD)
        G_thresh = geometrical_association_predict(t1, t2, threshold=DIST_THRESHOLD)

        frame_records.append({
            "scene": scene_id, "variant": variant, "frame_idx": frame_idx,
            "cam1_ids": [t.track_id for t in t1], "cam2_ids": [t.track_id for t in t2],
            "hungarian_pairs": [(p[0].track_id, p[1].track_id) for p in pairs_hungarian],
            "hungarian_edges": [(("1", p[0].track_id), ("2", p[1].track_id)) for p in pairs_hungarian],
            "thresh_edges": list(G_thresh.edges()),
        })

    cap1.release()
    cap2.release()
    return frame_records


def build_ground_truth(all_records):
    """Bootstrap GT: dem tan suat cap (cam1_id,cam2_id) qua Hungarian tren
    TOAN BO frame CUNG (scene,variant) -- track_id chi co y nghia trong PHAM
    VI 1 file video (goc hoac 1 bien the augment) rieng, KHONG chung giua cac
    bien the -- giu cap >= MIN_COOCCUR, gop Connected Components ->
    {(cam,id): group_label}."""
    from collections import Counter

    gt_map = {}  # (scene, variant, "1"/"2", track_id) -> group_label
    keys = sorted(set((r["scene"], r["variant"]) for r in all_records))
    for scene_id, variant in keys:
        key_records = [r for r in all_records if r["scene"] == scene_id and r["variant"] == variant]
        cnt = Counter()
        for rec in key_records:
            for a, b in rec["hungarian_pairs"]:
                cnt[(a, b)] += 1
        G = nx.Graph()
        for rec in key_records:
            for tid in rec["cam1_ids"]:
                G.add_node(("1", tid))
            for tid in rec["cam2_ids"]:
                G.add_node(("2", tid))
        for (a, b), c in cnt.items():
            if c >= MIN_COOCCUR:
                G.add_edge(("1", a), ("2", b))
        for i, comp in enumerate(nx.connected_components(G)):
            for node in comp:
                gt_map[(scene_id, variant, node[0], node[1])] = f"S{scene_id}_{variant}_P{i}"
    return gt_map


def score_frame(nodes, gt_map, scene_id, variant, pred_edges):
    """nodes: list (cam,"1"/"2",track_id) trong 1 frame (chi node THUOC gt_map,
    tuc da xac dinh danh tinh that qua bootstrap). pred_edges: list cap
    active theo phuong phap dang xet. Tra ve (y_true_labels, y_pred_labels)
    cung do dai = len(nodes), dung Connected Components cua pred_edges lam
    cum du doan (giong dung cach paper lam -- xem III-G)."""
    node_list = [n for n in nodes if (scene_id, variant, n[0], n[1]) in gt_map]
    if len(node_list) < 2:
        return None, None
    y_true = [gt_map[(scene_id, variant, n[0], n[1])] for n in node_list]

    Gp = nx.Graph()
    for n in node_list:
        Gp.add_node(n)
    for a, b in pred_edges:
        if a in node_list and b in node_list:
            Gp.add_edge(a, b)
    comp_map = {}
    for i, comp in enumerate(nx.connected_components(Gp)):
        for node in comp:
            comp_map[node] = i
    y_pred = [comp_map[n] for n in node_list]
    return y_true, y_pred


def main():
    print("Nap 2 detector YOLOv8n + homography P2...")
    detector1 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    detector2 = YOLO(str(AI_ROOT / "yolov8n.pt"))
    H_floor = load_H()

    all_records = []
    for scene_id in [1, 2, 3, 4]:
        for variant in VARIANTS:
            path1 = video_path(scene_id, "CAM 1", variant)
            if not path1.exists():
                print(f"  CANH BAO: thieu {path1}, bo qua")
                continue
            print(f"=== P2 Data 3 / Scene {scene_id} / {variant} ({SCENE_NOTE[scene_id]}) ===")
            recs = process_scene(scene_id, variant, detector1, detector2, H_floor)
            print(f"  {len(recs)} frame co detect ca 2 cam")
            all_records.extend(recs)

    print("\nXay ground truth (bootstrap tu tan suat dong-xuat-hien Hungarian, rieng tung scene+variant)...")
    gt_map = build_ground_truth(all_records)
    for scene_id in [1, 2, 3, 4]:
        for variant in VARIANTS:
            ids = set(v for k, v in gt_map.items() if k[0] == scene_id and k[1] == variant)
            if ids:
                print(f"  Scene {scene_id}/{variant}: {len(ids)} danh tinh that xac dinh duoc")

    rows = []
    for rec in all_records:
        scene_id, variant = rec["scene"], rec["variant"]
        nodes = [("1", tid) for tid in rec["cam1_ids"]] + [("2", tid) for tid in rec["cam2_ids"]]

        y_true, y_pred_hung = score_frame(nodes, gt_map, scene_id, variant, rec["hungarian_edges"])
        _, y_pred_thresh = score_frame(nodes, gt_map, scene_id, variant, rec["thresh_edges"])
        if y_true is None:
            continue

        row = {"scene": scene_id, "variant": variant, "frame_idx": rec["frame_idx"], "n_nodes": len(y_true)}
        for tag, y_pred in [("hungarian", y_pred_hung), ("thresh_cc", y_pred_thresh)]:
            row[f"{tag}_ari"] = metrics.adjusted_rand_score(y_true, y_pred)
            row[f"{tag}_ami"] = metrics.adjusted_mutual_info_score(y_true, y_pred)
            row[f"{tag}_h"] = metrics.homogeneity_score(y_true, y_pred)
            row[f"{tag}_c"] = metrics.completeness_score(y_true, y_pred)
            row[f"{tag}_v"] = metrics.v_measure_score(y_true, y_pred)
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "compare_p2_gnn_cca_clustering.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    def summarize(sub, label):
        print(f"\n--- {label} (n={len(sub)} frame) ---")
        summary = {}
        for tag in ["hungarian", "thresh_cc"]:
            ari = sub[f"{tag}_ari"].mean() * 100
            ami = sub[f"{tag}_ami"].mean() * 100
            h = sub[f"{tag}_h"].mean() * 100
            c = sub[f"{tag}_c"].mean() * 100
            v = sub[f"{tag}_v"].mean() * 100
            print(f"  [{tag}] ARI={ari:.2f} AMI={ami:.2f} H={h:.2f} C={c:.2f} V-m={v:.2f}")
            summary[tag] = {"ARI": ari, "AMI": ami, "H": h, "C": c, "V-m": v}
        return summary

    print("\n" + "=" * 70)
    print("KET QUA THEO CHUAN TABLE V CUA GNN-CCA (ARI/AMI/H/C/V-measure, tinh")
    print("moi frame roi trung binh, dung ground truth bootstrap tu Hungarian):")
    overall = summarize(df, f"TOAN BO P2 Data 3 (4 canh x {len(VARIANTS)} bien the: goc+augment)")
    original_only = summarize(df[df["variant"] == "original"], "CHI ban goc (khong augment, doi chieu voi lan chay truoc)")
    normal = summarize(df[df["scene"].isin(NORMAL_SCENES)], "Canh BINH THUONG (Scene 1, moi bien the)")
    ambiguous = summarize(df[df["scene"].isin(AMBIGUOUS_SCENES)], "Canh MO HO (Scene 2/3/4, moi bien the)")
    print("=" * 70)

    per_variant = {}
    for variant in VARIANTS:
        sub = df[df["variant"] == variant]
        if len(sub):
            per_variant[variant] = summarize(sub, f"Bien the: {variant}")

    summary_out = {"overall": overall, "original_only": original_only, "normal_scene1": normal,
                   "ambiguous_scene234": ambiguous, "per_variant": per_variant}
    with open(RESULTS_DIR / "compare_p2_gnn_cca_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_out, f, ensure_ascii=False, indent=2)

    print(f"\nDa luu: {out_csv}")
    print(f"Da luu: {RESULTS_DIR / 'compare_p2_gnn_cca_summary.json'}")


if __name__ == "__main__":
    main()
