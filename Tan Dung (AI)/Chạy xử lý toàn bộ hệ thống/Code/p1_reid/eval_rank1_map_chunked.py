"""
Tinh Rank-1/Rank-5/mAP tu embedding DA TRICH SAN (eval_rank1_map.py da chay
xong buoc trich, chi loi o buoc xep hang do torchreid.evaluate_rank can
~22GB RAM cung luc cho ma tran 11659x82161 -- may chi co 16.8GB). Viet lai
danh gia theo tung LO (chunk) query, dung float32, giai phong bo nho sau moi
lo -- dung protocol chuan Market1501 (loai gallery CUNG pid VA CUNG camid
voi query, xem torchreid.metrics.rank.eval_market1501).
"""
import argparse
from pathlib import Path

import numpy as np

AI_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "Results" / "p1"
MSMT17_DIR = (AI_ROOT / "Chạy xử lý toàn bộ hệ thống" / "File Run Problem 1"
              / "Data train" / "MSMT17_V1")

CHUNK = 500
MAX_RANK = 50

parser = argparse.ArgumentParser()
parser.add_argument("--suffix", default="frozen_backbone")
args = parser.parse_args()


def load_pids_camids(list_file):
    pids, camids = [], []
    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel_path, pid = line.split(" ")
            camids.append(int(rel_path.split("_")[2]) - 1)
            pids.append(int(pid))
    return np.array(pids), np.array(camids)


def main():
    print("Nap embedding + pid/camid...")
    q_embeds = np.load(RESULTS_DIR / f"msmt17_query_embeds_{args.suffix}.npy").astype(np.float32)
    g_embeds = np.load(RESULTS_DIR / f"msmt17_gallery_embeds_{args.suffix}.npy").astype(np.float32)
    q_pids, q_camids = load_pids_camids(MSMT17_DIR / "list_query.txt")
    g_pids, g_camids = load_pids_camids(MSMT17_DIR / "list_gallery.txt")
    print(f"Query: {q_embeds.shape} | Gallery: {g_embeds.shape}")
    assert len(q_pids) == len(q_embeds) and len(g_pids) == len(g_embeds)

    n_q = len(q_pids)
    all_cmc = np.zeros(MAX_RANK, dtype=np.float64)
    all_AP = []
    n_valid_q = 0

    for start in range(0, n_q, CHUNK):
        end = min(start + CHUNK, n_q)
        emb_chunk = q_embeds[start:end]
        pid_chunk = q_pids[start:end]
        camid_chunk = q_camids[start:end]

        distmat = 1.0 - emb_chunk @ g_embeds.T  # (chunk, n_gallery), float32
        indices = np.argsort(distmat, axis=1)

        for i in range(len(pid_chunk)):
            q_pid, q_camid = pid_chunk[i], camid_chunk[i]
            order = indices[i]
            # bo qua gallery CUNG pid VA CUNG camid (junk, chuan Market1501)
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = ~remove
            matches_row = (g_pids[order][keep] == q_pid).astype(np.int32)
            if matches_row.sum() == 0:
                continue  # query khong co gallery hop le -- bo qua
            n_valid_q += 1

            cmc_row = matches_row.cumsum()
            cmc_row[cmc_row > 1] = 1
            all_cmc[: min(MAX_RANK, len(cmc_row))] += cmc_row[:MAX_RANK]
            if len(cmc_row) < MAX_RANK:
                all_cmc[len(cmc_row):] += cmc_row[-1]

            n_rel = matches_row.sum()
            tmp_cmc = matches_row.cumsum()
            tmp_cmc = [x / (j + 1.0) for j, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * matches_row
            AP = tmp_cmc.sum() / n_rel
            all_AP.append(AP)

        print(f"  Da xu ly {end}/{n_q} query...")

    cmc = all_cmc / n_valid_q
    mAP = float(np.mean(all_AP))

    print("\n" + "=" * 60)
    print(f"So query hop le: {n_valid_q}/{n_q}")
    print(f"Rank-1: {cmc[0]*100:.2f}%")
    print(f"Rank-5: {cmc[4]*100:.2f}%")
    print(f"mAP:    {mAP*100:.2f}%")
    print("(Moc tham khao literature OSNet_x1.0 tren MSMT17: Rank-1 ~74-78%, mAP ~52-55%)")
    print("=" * 60)

    out_path = RESULTS_DIR / f"msmt17_rank1_map_{args.suffix}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Rank-1: {cmc[0]*100:.2f}%\n")
        f.write(f"Rank-5: {cmc[4]*100:.2f}%\n")
        f.write(f"mAP: {mAP*100:.2f}%\n")
        f.write(f"n_query_valid: {n_valid_q}\n")
        f.write(f"n_query_total: {n_q}\n")
        f.write(f"n_gallery: {len(g_pids)}\n")
    print(f"\nDa luu: {out_path}")


if __name__ == "__main__":
    main()
