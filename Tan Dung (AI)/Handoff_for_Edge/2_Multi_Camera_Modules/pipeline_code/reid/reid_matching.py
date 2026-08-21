"""
Re-Identification -- ghep danh tinh 1 nguoi khi ho di qua "vung chet" (mat
dau 1 camera, xuat hien lai o camera khac hoac cung camera sau 1 khoang thoi
gian). Dung model rieng OSNet_x0.25 (Knowledge Distillation tu OSNet_x1.0,
checkpoint: models/reid_osnet_x0_25_kd.pt) -- KHONG dung chung backbone voi
Pose Head (da thu backbone dung chung, that bai hoan toan -- xem README).

Yeu cau: torchreid (`pip install torchreid` hoac deep-person-reid).

Quy trinh:
  1. build_gallery(): dang ky 1 lan moi nguoi can theo doi (vd luc setup, hoac
     lan dau detect duoc 1 nguoi moi trong nha) -- trung binh embedding qua
     vai frame lien tiep.
  2. Khi 1 track MOI xuat hien (khong lien tuc voi track nao dang co, vd sau
     khi di qua vung chet): gom embedding qua vai frame dau, roi match_gallery()
     de xac dinh day co phai nguoi da biet khong.

Nguong khop MATCH_THRESHOLD=0.6 da hieu chinh tren du lieu that cua nhom (xem
README muc so lieu) -- co the can chinh lai neu goc camera/anh sang khac
nhieu so voi luc test.
"""
from pathlib import Path

import numpy as np

MATCH_THRESHOLD = 0.6
MIN_BUFFER_FRAMES = 5   # can it nhat 5 frame moi tin embedding trung binh
MAX_BUFFER_FRAMES = 10  # khong can gom qua 10 frame, da du on dinh


class ReIDExtractor:
    """Boc torchreid.FeatureExtractor, chi lo phan trich + chuan hoa embedding."""

    def __init__(self, checkpoint_path, device="cpu"):
        from torchreid.reid.utils import FeatureExtractor
        self._extractor = FeatureExtractor(
            model_name="osnet_x0_25", model_path=str(checkpoint_path), device=device)

    def embed(self, crop_bgr):
        """crop_bgr: 1 anh crop nguoi (numpy BGR, tu OpenCV). Tra ve vector
        512-d da L2-normalize (numpy)."""
        import torch
        emb = self._extractor([crop_bgr])
        emb = torch.nn.functional.normalize(emb, dim=1)
        return emb.cpu().numpy().flatten()


class PersonGallery:
    """Gallery cac nguoi da dang ky, moi nguoi 1 vector embedding trung binh."""

    def __init__(self):
        self._gallery = {}  # person_name -> embedding (512-d, normalized)

    def register(self, person_name, embeddings):
        """embeddings: list vector 512-d (vd tu nhieu frame dang ky). Trung
        binh + chuan hoa lai thanh 1 vector dai dien."""
        avg = np.mean(embeddings, axis=0)
        self._gallery[person_name] = avg / np.linalg.norm(avg)

    def match(self, query_embedding, threshold=MATCH_THRESHOLD):
        """Tra ve (person_name, score) neu khop (score > threshold), nguoc
        lai ("NEW", best_score) -- "NEW" nghia la chua co trong gallery,
        co the can dang ky nguoi moi."""
        best_name, best_score = None, -1.0
        for name, gal_emb in self._gallery.items():
            score = float(np.dot(query_embedding, gal_emb))
            if score > best_score:
                best_name, best_score = name, score
        if best_score > threshold:
            return best_name, best_score
        return "NEW", best_score


class TrackEmbeddingBuffer:
    """Gom embedding cho 1 track_id qua nhieu frame, tra ve embedding trung
    binh khi da du (>= MIN_BUFFER_FRAMES). Dung khi 1 track MOI xuat hien va
    can xac dinh day la ai (goi match_gallery voi ket qua tra ve)."""

    def __init__(self):
        self._buffers = {}  # track_id -> list embedding

    def add(self, track_id, embedding):
        buf = self._buffers.setdefault(track_id, [])
        if len(buf) < MAX_BUFFER_FRAMES:
            buf.append(embedding)

    def get_average_if_ready(self, track_id):
        """Tra ve embedding trung binh (da normalize) neu track_id da co du
        frame, nguoc lai None (chua du du lieu de ket luan)."""
        buf = self._buffers.get(track_id, [])
        if len(buf) < MIN_BUFFER_FRAMES:
            return None
        avg = np.mean(buf, axis=0)
        return avg / np.linalg.norm(avg)

    def forget(self, track_id):
        self._buffers.pop(track_id, None)


if __name__ == "__main__":
    # Vi du toi thieu (khong chay duoc truc tiep, can anh crop nguoi that).
    import argparse
    parser = argparse.ArgumentParser(description="Demo API Re-Identification")
    parser.add_argument("--ckpt", default=str(Path(__file__).resolve().parents[2] / "models" / "reid_osnet_x0_25_kd.pt"))
    args = parser.parse_args()
    print(f"Se nap checkpoint: {args.ckpt}")
    print("Xem docstring dau file de biet cach dung ReIDExtractor + PersonGallery + TrackEmbeddingBuffer.")
