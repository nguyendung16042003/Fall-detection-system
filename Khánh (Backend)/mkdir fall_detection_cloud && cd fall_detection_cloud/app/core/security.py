from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import jwt

# Cấu hình mã hóa mật khẩu
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = "YOUR_SUPER_SECRET_KEY" # Lấy từ .env
ALGORITHM = "HS256"

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)