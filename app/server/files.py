import time
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from app.models import UploadedFileResponse
from app.server.middleware import (
    create_uploaded_file,
    get_uploaded_file_metadata,
    get_uploaded_file_path,
    update_uploaded_file_metadata,
    verify_api_key,
)
from app.services import GeminiClientPool

router = APIRouter()


@router.post("/v1/files", response_model=UploadedFileResponse, tags=["Files"])
async def upload_file(
    file: Annotated[UploadFile, File(...)],
    purpose: Annotated[str, Form()] = "assistants",
    api_key: str = Depends(verify_api_key),
):
    data = await file.read()
    filename = file.filename or "upload.bin"
    metadata = create_uploaded_file(data, filename, purpose)
    path = get_uploaded_file_path(str(metadata["id"]))

    try:
        pool = GeminiClientPool()
        client = await pool.acquire()
        upload_ref = await client.upload_file_reference(path, filename=filename)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to upload file to Gemini: {exc}",
        ) from exc

    metadata = update_uploaded_file_metadata(
        str(metadata["id"]),
        client_id=client.id,
        gemini_file_url=upload_ref.upload_url,
        uploaded_at=int(time.time()),
    )
    return UploadedFileResponse.model_validate(metadata)


@router.get("/v1/files/{file_id}", response_model=UploadedFileResponse, tags=["Files"])
async def retrieve_file(file_id: str, api_key: str = Depends(verify_api_key)):
    metadata = get_uploaded_file_metadata(file_id)
    return UploadedFileResponse.model_validate(metadata)


@router.get("/v1/files/{file_id}/content", tags=["Files"])
async def retrieve_file_content(file_id: str, api_key: str = Depends(verify_api_key)):
    metadata = get_uploaded_file_metadata(file_id)
    path = get_uploaded_file_path(file_id)
    filename = metadata.get("filename")
    return FileResponse(path, filename=filename if isinstance(filename, str) else path.name)
