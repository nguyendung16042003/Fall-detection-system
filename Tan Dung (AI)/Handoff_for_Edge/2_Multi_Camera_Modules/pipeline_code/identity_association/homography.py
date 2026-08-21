"""
P2 -- Homography Association, Buoc 1: chieu diem chan nguoi tu anh 2D len mat
san chung. Thuan hinh hoc, KHONG dung model (dung theo dac ta
"Context xu ly van de 2.docx").
"""
import numpy as np


def foot_point_from_bbox(bbox_xyxy):
    """Diem cham san cua 1 bbox = trung diem canh duoi (bottom-center)."""
    x1, y1, x2, y2 = bbox_xyxy
    return ((x1 + x2) / 2.0, y2)


def head_point_from_bbox(bbox_xyxy):
    """Diem dinh dau cua 1 bbox = trung diem canh tren (top-center). Dung de
    calib mat phang thu 2 (o do cao dung cua nguoi) cho H_by_height."""
    x1, y1, x2, y2 = bbox_xyxy
    return ((x1 + x2) / 2.0, y1)


def calibrate_from_points(pixel_points, world_points):
    """
    pixel_points: list 4 (x,y) toa do pixel tren anh camera.
    world_points: list 4 (X,Y) toa do met tren san that, cung thu tu.
    Tra ve ma tran homography 3x3 (pixel -> san).
    """
    import cv2
    src = np.array(pixel_points, dtype=np.float32)
    dst = np.array(world_points, dtype=np.float32)
    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise ValueError("Khong tinh duoc homography tu 4 diem da cho")
    return H


def apply_homography(H, point_xy):
    """Chieu 1 diem pixel (x,y) qua homography H, tra ve toa do san (X,Y)."""
    x, y = point_xy
    p = np.array([x, y, 1.0])
    p_proj = H @ p
    p_proj = p_proj / p_proj[2]
    return (float(p_proj[0]), float(p_proj[1]))


def project_bbox_foot(H, bbox_xyxy):
    """Chieu diem chan cua 1 bbox len toa do san that. Ham chinh dung xuyen
    suot P2 (homography_matching.py) va P3 (xap xi epipolar)."""
    foot_px = foot_point_from_bbox(bbox_xyxy)
    return apply_homography(H, foot_px)
