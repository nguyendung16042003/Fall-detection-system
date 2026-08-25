"""Cho phép pytest import module trong src/ mà không cần cài đặt package.

Đặt ở gốc repo nên pytest tự nạp; thêm thư mục src/ vào sys.path.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
