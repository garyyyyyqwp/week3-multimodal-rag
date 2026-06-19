"""Tests for multimodal document upload and management."""

import pytest


@pytest.mark.asyncio
async def test_upload_txt_and_list(test_app, sample_txt_content):
    """Upload a TXT file, verify response, then list documents."""
    resp = await test_app.post(
        "/api/v1/documents/upload",
        files={"file": ("test_notes.txt", sample_txt_content, "text/plain")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "indexed"
    assert data["filename"] == "test_notes.txt"
    assert data["file_type"] == "txt"
    assert data["text_chunk_count"] > 0
    doc_id = data["doc_id"]
    assert len(doc_id) == 32  # uuid hex

    # List
    resp = await test_app.get("/api/v1/documents")
    assert resp.status_code == 200
    docs = resp.json()
    assert isinstance(docs, list)
    assert any(d["doc_id"] == doc_id for d in docs)


@pytest.mark.asyncio
async def test_delete_document(test_app, sample_txt_content):
    """Upload a document, delete it, verify it's gone."""
    resp = await test_app.post(
        "/api/v1/documents/upload",
        files={"file": ("to_delete.txt", sample_txt_content, "text/plain")},
    )
    doc_id = resp.json()["doc_id"]

    resp = await test_app.delete(f"/api/v1/documents/{doc_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["deleted"] is True
    assert data["chunks_removed"] > 0

    resp = await test_app.get("/api/v1/documents")
    docs = resp.json()
    assert not any(d["doc_id"] == doc_id for d in docs)


@pytest.mark.asyncio
async def test_unsupported_file_type(test_app):
    """Upload an unsupported file type should return 400."""
    resp = await test_app.post(
        "/api/v1/documents/upload",
        files={"file": ("test.exe", b"binary content", "application/octet-stream")},
    )
    assert resp.status_code == 400
    assert "不支持的文件类型" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_delete_nonexistent_document(test_app):
    """Delete a non-existent document should return 404."""
    resp = await test_app.delete("/api/v1/documents/nonexistent123")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_images_for_nonexistent_doc(test_app):
    """Get images for a non-existent document should return 404."""
    resp = await test_app.get("/api/v1/documents/nonexistent123/images")
    assert resp.status_code == 404
