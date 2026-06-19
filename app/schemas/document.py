from pydantic import BaseModel, Field, field_validator
from typing import Optional


# ---------------------------------------------------------------------------
# Document Schemas
# ---------------------------------------------------------------------------

class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    text_chunk_count: int = 0
    image_count: int = 0
    status: str = "indexed"


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    text_chunk_count: int = 0
    image_count: int = 0
    total_chunks: int = 0
    created_at: str = ""


class ImageInfo(BaseModel):
    image_id: str
    doc_id: str
    file_path: str
    caption: str
    page_num: int = 0


class DocumentDeleteResponse(BaseModel):
    doc_id: str
    deleted: bool
    chunks_removed: int


class ErrorResponse(BaseModel):
    detail: str
