from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import jwt
from src.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
BCRYPT_MAX_BYTES = 72

def _truncate_password(password: str) -> str:
    if password is None:
        return ""
    pb = password.encode("utf-8")[:BCRYPT_MAX_BYTES]
    return pb.decode("utf-8", errors="ignore")

def hash_password(password: str) -> str:
    safe_pw = _truncate_password(password)
    return pwd_context.hash(safe_pw)

def verify_password(plain: str, hashed: str) -> bool:
    safe_plain = _truncate_password(plain)
    return pwd_context.verify(safe_plain, hashed)

def create_access_token(data: dict, expires_delta: int = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=expires_delta or settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded = jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")
    return encoded
