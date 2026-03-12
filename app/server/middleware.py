import hashlib
import hmac
import mimetypes
import tempfile
import time
import uuid
from pathlib import Path

import orjson
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger

from app.utils import g_config

# Persistent directory for storing generated images
IMAGE_STORE_DIR = Path(g_config.storage.images_path)
IMAGE_STORE_DIR.mkdir(parents=True, exist_ok=True)
FILE_STORE_DIR = Path(g_config.storage.path).parent / "files"
FILE_STORE_DIR.mkdir(parents=True, exist_ok=True)


def get_image_store_dir() -> Path:
    """Returns a persistent directory for storing images."""
    return IMAGE_STORE_DIR


def get_file_store_dir() -> Path:
    """返回持久化上传文件目录。"""
    return FILE_STORE_DIR


def get_image_token(filename: str) -> str:
    """Generate a HMAC-SHA256 token for a filename using the API key."""
    secret = g_config.server.api_key
    if not secret:
        return ""

    msg = filename.encode("utf-8")
    secret_bytes = secret.encode("utf-8")
    return hmac.new(secret_bytes, msg, hashlib.sha256).hexdigest()


def verify_image_token(filename: str, token: str | None) -> bool:
    """Verify the provided token against the filename."""
    expected = get_image_token(filename)
    if not expected:
        return True  # No auth required
    if not token:
        return False
    return hmac.compare_digest(token, expected)


def cleanup_expired_images(retention_days: int) -> int:
    """Delete images in IMAGE_STORE_DIR older than retention_days."""
    if retention_days <= 0:
        return 0

    now = time.time()
    retention_seconds = retention_days * 24 * 60 * 60
    cutoff = now - retention_seconds

    count = 0
    for file_path in IMAGE_STORE_DIR.iterdir():
        if not file_path.is_file():
            continue
        try:
            if file_path.stat().st_mtime < cutoff:
                file_path.unlink()
                count += 1
        except Exception as e:
            logger.warning(f"Failed to delete expired image {file_path}: {e}")

    if count > 0:
        logger.info(f"Cleaned up {count} expired images.")
    return count


def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": exc.detail}},
        )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": {"message": str(exc)}},
    )


def get_temp_dir():
    temp_dir = tempfile.TemporaryDirectory()
    try:
        yield Path(temp_dir.name)
    finally:
        temp_dir.cleanup()


def _write_uploaded_file_metadata(file_id: str, metadata: dict[str, str | int | None]) -> None:
    """写入上传文件元数据。"""
    (FILE_STORE_DIR / f"{file_id}.json").write_text(
        orjson.dumps(metadata).decode("utf-8"), encoding="utf-8"
    )


def create_uploaded_file(data: bytes, filename: str, purpose: str) -> dict[str, str | int | None]:
    """持久化上传文件并返回元数据。"""
    file_id = f"file-{uuid.uuid4().hex}"
    suffix = Path(filename).suffix or mimetypes.guess_extension(
        mimetypes.guess_type(filename)[0] or ""
    )
    stored_name = f"{file_id}{suffix or ''}"
    stored_path = FILE_STORE_DIR / stored_name
    stored_path.write_bytes(data)

    created_at = int(time.time())
    metadata = {
        "id": file_id,
        "filename": filename,
        "purpose": purpose,
        "bytes": len(data),
        "created_at": created_at,
        "path": stored_name,
        "client_id": None,
        "gemini_file_url": None,
        "uploaded_at": None,
    }
    _write_uploaded_file_metadata(file_id, metadata)
    return metadata


def update_uploaded_file_metadata(file_id: str, **updates: str | int | None) -> dict[str, str | int | None]:
    """更新上传文件元数据。"""
    metadata = get_uploaded_file_metadata(file_id)
    metadata.update(updates)
    _write_uploaded_file_metadata(file_id, metadata)
    return metadata


def get_uploaded_file_metadata(file_id: str) -> dict[str, str | int | None]:
    """读取上传文件元数据。"""
    metadata_path = FILE_STORE_DIR / f"{file_id}.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return orjson.loads(metadata_path.read_bytes())


def get_uploaded_file_path(file_id: str) -> Path:
    """根据 file_id 定位已上传文件。"""
    metadata = get_uploaded_file_metadata(file_id)
    stored_name = metadata.get("path")
    if not isinstance(stored_name, str):
        raise HTTPException(status_code=500, detail="Invalid file metadata")
    path = FILE_STORE_DIR / stored_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="File content not found")
    return path


def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    if not g_config.server.api_key:
        return ""

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")

    api_key = credentials.credentials
    if api_key != g_config.server.api_key:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Wrong API key")

    return api_key


def add_exception_handler(app: FastAPI):
    app.add_exception_handler(Exception, global_exception_handler)


def add_cors_middleware(app: FastAPI):
    if g_config.cors.enabled:
        cors = g_config.cors
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors.allow_origins,
            allow_credentials=cors.allow_credentials,
            allow_methods=cors.allow_methods,
            allow_headers=cors.allow_headers,
        )
