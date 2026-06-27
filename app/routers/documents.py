"""Documents Router — upgraded for multimodal document processing."""

import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.schemas.document import (
    DocumentUploadResponse,
    DocumentInfo,
    DocumentDeleteResponse,
    ImageInfo,
    ErrorResponse,
)
from app.services.parser import (
    parse_document,
    caption_images,
    UnsupportedFileTypeError,
    FileTooLargeError,
    FileContentInvalidError,
    ParseError,
)
from app.services.chunker import chunk_text
from app.services.vector_store import get_vector_store, VectorStoreError
from app.services.embedding import EmbeddingError

router = APIRouter(tags=["documents"])

_FILE_TYPE_MAP = {"pdf": "pdf", "md": "md", "txt": "txt", "docx": "docx"}


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=201,
    responses={
        400: {"model": ErrorResponse, "description": "不支持的文件类型、文件损坏或解析失败"},
        413: {"model": ErrorResponse, "description": "文件超过大小限制 (10MB)"},
        500: {"model": ErrorResponse, "description": "向量存储或 Embedding 服务错误"},
    },
)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document (PDF/DOCX/MD/TXT). For PDFs, extracts embedded images
    and generates Vision LLM captions before indexing."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    # Read file content
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"文件读取失败: {str(e)}")

    # Generate doc_id early
    doc_id = uuid.uuid4().hex
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "txt"
    file_type = _FILE_TYPE_MAP.get(ext, "txt")

    # Parse document (text + images)
    try:
        parsed = await parse_document(file.filename, content, doc_id)
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except FileContentInvalidError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ParseError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Chunk text
    chunks = chunk_text(parsed.full_text)
    if not chunks:
        raise HTTPException(status_code=400, detail="文档解析后无有效文本内容，无法分块")

    text_chunk_dicts = [{"content": c.content, "index": c.index} for c in chunks]

    # Generate image captions via Vision LLM
    image_count = 0
    if parsed.images:
        try:
            parsed.images = await caption_images(parsed.images)
            image_count = len(parsed.images)
        except Exception as e:
            # Image captioning failure is non-fatal — store with empty captions
            image_count = len(parsed.images)

    # Store in ChromaDB
    store = await get_vector_store()

    try:
        # Store text chunks
        text_stored = await store.add_text_chunks(
            doc_id=doc_id,
            filename=file.filename,
            file_type=file_type,
            chunks=text_chunk_dicts,
        )
    except EmbeddingError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except VectorStoreError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文本存储失败: {str(e)}")

    # Store image chunks
    if parsed.images:
        image_dicts = [
            {
                "image_id": img.image_id,
                "caption": img.caption or "",
                "file_path": img.file_path,
                "page_num": img.page_num,
            }
            for img in parsed.images
        ]
        try:
            await store.add_image_chunks(
                doc_id=doc_id,
                filename=file.filename,
                images=image_dicts,
            )
        except Exception as e:
            # Image storage failure is non-fatal
            pass

    return DocumentUploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        file_type=file_type,
        text_chunk_count=text_stored,
        image_count=image_count,
        status="indexed",
    )


@router.get("", response_model=list[DocumentInfo])
async def list_documents():
    """List all indexed documents with chunk counts (text + image)."""
    store = await get_vector_store()
    try:
        docs = store.list_documents()
    except VectorStoreError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return [
        DocumentInfo(
            doc_id=d["doc_id"],
            filename=d["filename"],
            file_type=d.get("file_type", ""),
            text_chunk_count=d.get("text_chunk_count", 0),
            image_count=d.get("image_count", 0),
            total_chunks=d.get("total_chunks", 0),
            created_at=d.get("created_at", ""),
        )
        for d in docs
    ]


@router.get(
    "/{doc_id}/images",
    response_model=list[ImageInfo],
    responses={404: {"model": ErrorResponse, "description": "文档不存在"}},
)
async def get_document_images(doc_id: str):
    """Get all images extracted from a document."""
    store = await get_vector_store()

    if not store.doc_exists(doc_id):
        raise HTTPException(status_code=404, detail=f"文档不存在: {doc_id}")

    try:
        images = store.get_document_images(doc_id)
    except VectorStoreError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return [
        ImageInfo(
            image_id=img["image_id"],
            doc_id=img["doc_id"],
            file_path=img.get("file_path", ""),
            caption=img.get("caption", ""),
            page_num=img.get("page_num", 0),
        )
        for img in images
    ]


@router.delete(
    "/{doc_id}",
    response_model=DocumentDeleteResponse,
    responses={404: {"model": ErrorResponse, "description": "文档不存在"}},
)
async def delete_document(doc_id: str):
    """Delete a document and all its text + image chunks."""
    store = await get_vector_store()

    if not store.doc_exists(doc_id):
        raise HTTPException(status_code=404, detail=f"文档不存在: {doc_id}")

    try:
        chunks_removed = store.delete_document(doc_id)
    except VectorStoreError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return DocumentDeleteResponse(
        doc_id=doc_id,
        deleted=True,
        chunks_removed=chunks_removed,
    )
