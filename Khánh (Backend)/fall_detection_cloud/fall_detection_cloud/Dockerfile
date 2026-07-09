# 1. Khai báo Image nền (Cần bổ sung)
FROM python:3.13.3

# 2. Thiết lập thư mục làm việc (Cần bổ sung)
WORKDIR /app

# 3. Cài đặt các thư viện hệ thống (Phần bạn đã liệt kê)
RUN apt-get update && apt-get install -y libpq-dev gcc

# 4. Sao chép và cài đặt thư viện Python (Phần bạn đã liệt kê)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Sao chép toàn bộ code (Phần bạn đã liệt kê)
COPY . .

# 6. Lệnh chạy ứng dụng (Phần bạn đã liệt kê)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
