"""
P3 -- Cross-Camera Attention Fusion, ban KHONG CAN TRAIN (dung dung theo
"Context xu ly van de 3 (no_training).md" muc 3). Toan bo la cong thuc toan
co dinh, khong co nn.Module/tham so nao can hoc -- chi dung lai Backbone +
Pose Head da train san (xem shared_backbone.py), KHONG sua, KHONG train lai.
"""
import numpy as np
import torch


ASSUMED_HEIGHTS_M = np.linspace(0.0, 2.0, num=8)


def fit_line_least_squares(points):
    """points: list (x,y). Tra ve (a,b,c) cua duong thang ax+by+c=0 khop binh
    phuong toi thieu qua cac diem. Dung SVD (on dinh hon polyfit khi duong
    gan thang dung, luc do slope -> vo cuc)."""
    pts = np.array(points, dtype=np.float64)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    # Duong thang qua centroid, huong la eigenvector ung voi eigenvalue LON
    # nhat cua centered^T centered (huong "trai dai" nhat cua tap diem).
    _, _, vt = np.linalg.svd(centered)
    direction = vt[0]  # (dx, dy)
    # Vector phap tuyen (a,b) vuong goc voi huong duong thang
    a, b = -direction[1], direction[0]
    norm = np.hypot(a, b)
    if norm < 1e-9:
        # cac diem trung nhau -- duong khong xac dinh, tra ve duong ngang qua centroid
        a, b = 0.0, 1.0
        norm = 1.0
    a, b = a / norm, b / norm
    c = -(a * centroid[0] + b * centroid[1])
    return a, b, c


def apply_homography_point(H, point_xy):
    x, y = point_xy
    p = np.array([x, y, 1.0])
    p_proj = H @ p
    p_proj = p_proj / p_proj[2]
    return (float(p_proj[0]), float(p_proj[1]))


def approximate_epipolar_line(p1_floor_point, H_by_height_target_cam,
                               assumed_heights_m=ASSUMED_HEIGHTS_M):
    """
    p1_floor_point: (X,Y) toa do san (met) cua nguoi tuong ung voi pixel dang
        xet ben cam nguon (LAY TU P2, khong tu tinh lai o day).
    H_by_height_target_cam: dict {height_m: H 3x3} CUA CAMERA DICH (noi can
        tim duong epipolar) -- xem homography_by_height.py.
    Tra ve (a,b,c): duong thang ax+by+c=0 xap xi duong epipolar ben cam dich.
    """
    candidate_points = []
    for h in assumed_heights_m:
        H_h = H_by_height_target_cam[float(h)]
        # QUAN TRONG: H_h duoc calib theo chieu PIXEL -> WORLD (xem calibrate.py:
        # calibrate_from_points(pixel_points, world_points)). De tim pixel ben
        # cam dich tuong ung voi 1 diem WORLD da biet, phai dung NGHICH DAO
        # H_h (world -> pixel) -- ap thang H_h (pixel -> world) vao 1 diem world
        # la SAI (da phat hien qua sigma sweep: ket qua khong doi theo sigma o
        # khoang binh thuong, dau hieu ro dist dang bi tinh sai/rac).
        H_h_inv = np.linalg.inv(H_h)
        # Diem tren "cot cao h" tai vi tri (X,Y) tren san -- gia dinh nguoi
        # dung thang, hinh chieu X,Y khong doi theo do cao (xap xi don gian,
        # dung theo dung cong thuc (8) trong dac ta).
        p2_k = apply_homography_point(H_h_inv, p1_floor_point)
        candidate_points.append(p2_k)
    return fit_line_least_squares(candidate_points)


def point_to_line_distance(p2_xy, line_abc):
    a, b, c = line_abc
    x2, y2 = p2_xy
    return abs(a * x2 + b * y2 + c) / np.hypot(a, b)


def epipolar_weight(dist_matrix, sigma=1.0):
    """dist_matrix: (hw, hw) numpy hoac tensor. Tra ve trong so Gaussian da
    chuan hoa (moi hang cong lai = 1), KHONG co tham so hoc."""
    if not torch.is_tensor(dist_matrix):
        dist_matrix = torch.tensor(dist_matrix, dtype=torch.float32)
    raw = torch.exp(-(dist_matrix ** 2) / (2 * sigma ** 2))
    weight = raw / (raw.sum(dim=-1, keepdim=True) + 1e-8)
    return weight


def cross_camera_fuse(F1_grid, F2_grid, dist_1to2, dist_2to1, sigma=1.0):
    """
    F1_grid, F2_grid: (B, hw, d) -- da flatten tu (B,d,h,w), CHUA pool.
    dist_1to2, dist_2to1: (hw,hw) -- 2 chieu KHAC NHAU (khong phai chuyen vi).
    Tra ve F1_enriched, F2_enriched cung shape voi input.
    """
    w_1to2 = epipolar_weight(dist_1to2, sigma)
    w_2to1 = epipolar_weight(dist_2to1, sigma)

    F1_enriched = F1_grid + torch.matmul(w_1to2, F2_grid)
    F2_enriched = F2_grid + torch.matmul(w_2to1, F1_grid)
    return F1_enriched, F2_enriched


def fuse_to_single_vector(F1_enriched, F2_enriched):
    """Trung binh cong don gian, KHONG nn.Linear."""
    v1 = F1_enriched.mean(dim=1)
    v2 = F2_enriched.mean(dim=1)
    return (v1 + v2) / 2.0


def grid_to_flat(grid):
    """(B,d,h,w) -> (B,hw,d)."""
    b, d, h, w = grid.shape
    return grid.flatten(2).transpose(1, 2), (h, w)


def build_distance_matrices(grid_hw, foot_point_cam1, foot_point_cam2,
                             H_by_height_cam1, H_by_height_cam2,
                             bbox_cam1, bbox_cam2):
    """
    Tinh dist_1to2 va dist_2to1 (hw,hw) cho 1 cap nguoi da ghep (tu P2).
    grid_hw: (h,w) cua luoi dac trung (vd (7,7)).
    foot_point_camX: (X,Y) toa do san (met) cua nguoi nay, tu P2.
    H_by_height_camX: dict {height_m: H} cua tung camera.
    bbox_camX: (x1,y1,x2,y2) bbox GOC trong khung hinh day du (cung he quy
        chieu voi luc calib homography) -- BAT BUOC de doi toa do o luoi
        (dang tinh theo ti le trong crop) VE toa do khung hinh day du. Thieu
        buoc nay la 1 loi thuc su da phat hien (dist tinh sai he quy chieu,
        lam sigma sweep cho ket qua khong doi bat thuong o vai gia tri).
        Anh xa GAN DUNG theo ty le vi tri trong bbox (bo qua chi tiet
        Resize+CenterCrop cua tien xu ly) -- hop ly o cung muc xap xi voi
        "1 duong epipolar chung cho ca bbox" da ghi ro ben duoi.

    XAP XI DON GIAN HOA: vi nguoi chi co 1 bbox (1 diem chan), moi o luoi
    trong bbox duoc coi la cung xap xi 1 duong epipolar (tinh 1 lan tu diem
    chan chung, khong tinh rieng tung o -- vi thieu keypoint chi tiet tung
    vi tri co the tren nguoi). Day la don gian hoa THEM so voi cong thuc goc
    (dung 1 duong epipolar cho ca bbox thay vi rieng tung pixel/o luoi) --
    hop ly vi bbox nguoi thuong nho hon nhieu so voi sai so xap xi epipolar
    da co san tu H_by_height. Ghi ro trong Ket qua/han che.
    """
    h, w = grid_hw
    n = h * w

    line_1to2 = approximate_epipolar_line(foot_point_cam1, H_by_height_cam2)
    line_2to1 = approximate_epipolar_line(foot_point_cam2, H_by_height_cam1)

    def grid_to_full_frame_px(bbox):
        x1, y1, x2, y2 = bbox
        bbox_w, bbox_h = x2 - x1, y2 - y1
        return [
            (x1 + (gx + 0.5) / w * bbox_w, y1 + (gy + 0.5) / h * bbox_h)
            for gy in range(h) for gx in range(w)
        ]

    grid_px_cam2 = grid_to_full_frame_px(bbox_cam2)
    grid_px_cam1 = grid_to_full_frame_px(bbox_cam1)

    # Vi dung 1 duong epipolar chung cho ca bbox (xap xi don gian hoa o tren),
    # dist_1to2[i,j] chi phu thuoc j (vi tri ben cam2) -- khong phu thuoc i.
    # Van giu dang ma tran day du (hw,hw) de tuong thich cross_camera_fuse().
    dist_1to2 = np.zeros((n, n), dtype=np.float32)
    dist_2to1 = np.zeros((n, n), dtype=np.float32)
    for j, (px, py) in enumerate(grid_px_cam2):
        dist_1to2[:, j] = point_to_line_distance((px, py), line_1to2)
    for j, (px, py) in enumerate(grid_px_cam1):
        dist_2to1[:, j] = point_to_line_distance((px, py), line_2to1)

    return dist_1to2, dist_2to1


def sgie_forward(backbone, crop1, crop2, has_both_cams,
                  dist_1to2=None, dist_2to1=None, sigma=1.0):
    """
    backbone: SharedBackbone da nap checkpoint.
    crop1, crop2: tensor (1,3,H,W) da preprocess (preprocess_crop), hoac None
        neu cam do khong thay nguoi.
    has_both_cams: bool -- co ghep duoc ca 2 cam khong (tu P2).
    Tra ve v (1,d) -- vector dua vao backbone.classify_from_vector().
    """
    if has_both_cams:
        F1 = backbone(crop1)
        F2 = backbone(crop2)
        F1_flat, hw = grid_to_flat(F1)
        F2_flat, _ = grid_to_flat(F2)
        F1_out, F2_out = cross_camera_fuse(F1_flat, F2_flat, dist_1to2, dist_2to1, sigma)
        v = fuse_to_single_vector(F1_out, F2_out)
    else:
        crop = crop1 if crop1 is not None else crop2
        F = backbone(crop)
        F_flat, _ = grid_to_flat(F)
        v = F_flat.mean(dim=1)
    return v
