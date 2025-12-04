import os
from src.core.config import settings

def save_upload(file_obj, filename: str) -> str:
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    path = os.path.join(settings.UPLOAD_DIR, filename)
    with open(path, "wb") as f:
        f.write(file_obj.read())
    return path
