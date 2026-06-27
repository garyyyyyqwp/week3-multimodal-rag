"""
Dual-collection VectorStore: manages text chunks (Zhipu Embedding) and
image chunks (CLIP Embedding) in separate ChromaDB collections.

Text collection schema:
  - id: {doc_id}_{chunk_index}
  - embedding: Zhipu embedding-2 (1024-d)
  - document: chunk text content
  - metadata: {doc_id, filename, chunk_type="text", chunk_index, ...}

Image collection schema:
  - id: {image_id}
  - embedding: CLIP ViT-B/32 (512-d)
  - document: Vision LLM generated caption (text for display)
  - metadata: {doc_id, image_id, file_path, caption, page_num, chunk_type="image", ...}
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import chromadb

from app.services.embedding import embed_texts
from app.services.clip_embedding import clip_encode_images
from app.utils.config import (
    CHROMA_PERSIST_DIR,
    TEXT_COLLECTION_NAME,
    IMAGE_COLLECTION_NAME,
)


class VectorStoreError(Exception):
    """Raised when a ChromaDB operation fails."""
    pass


class VectorStore:
    """Dual-collection multimodal vector store with Text and Image collections.

    Uses ChromaDB PersistentClient with cosine distance on both collections.
    """

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        text_collection_name: str = TEXT_COLLECTION_NAME,
        image_collection_name: str = IMAGE_COLLECTION_NAME,
    ):
        self._persist_dir = persist_dir
        self._text_collection_name = text_collection_name
        self._image_collection_name = image_collection_name
        self._client: chromadb.PersistentClient | None = None
        self._text_collection: chromadb.Collection | None = None
        self._image_collection: chromadb.Collection | None = None

    @property
    def client(self) -> chromadb.PersistentClient:
        if self._client is None:
            Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self._persist_dir)
        return self._client

    @property
    def text_collection(self):
        if self._text_collection is None:
            self._text_collection = self.client.get_or_create_collection(
                name=self._text_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._text_collection

    @property
    def image_collection(self):
        if self._image_collection is None:
            self._image_collection = self.client.get_or_create_collection(
                name=self._image_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._image_collection

    # ------------------------------------------------------------------
    # Text Chunk Operations
    # ------------------------------------------------------------------

    async def add_text_chunks(
        self,
        doc_id: str,
        filename: str,
        file_type: str,
        chunks: list[dict],  # [{"content": str, "index": int}, ...]
    ) -> int:
        """Add text chunks to the text collection. Returns chunk count."""
        if not chunks:
            return 0

        texts = [c["content"] for c in chunks]

        # Batch embeddings to avoid API token/size limits (Zhipu API limit ~16 items)
        _EMBED_BATCH = 8
        all_embeddings = []
        for b_start in range(0, len(texts), _EMBED_BATCH):
            batch_embs = await embed_texts(texts[b_start : b_start + _EMBED_BATCH])
            all_embeddings.extend(batch_embs)
        embeddings = all_embeddings

        ids = [f"{doc_id}_t{c['index']}" for c in chunks]
        metadatas = [
            {
                "doc_id": doc_id,
                "filename": filename,
                "file_type": file_type,
                "chunk_type": "text",
                "chunk_index": c["index"],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            for c in chunks
        ]

        try:
            self.text_collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
        except Exception as e:
            raise VectorStoreError(f"文本向量存储失败: {str(e)}") from e

        return len(chunks)

    async def add_image_chunks(
        self,
        doc_id: str,
        filename: str,
        images: list[dict],  # [{"image_id": str, "caption": str, "file_path": str, "page_num": int}, ...]
    ) -> int:
        """Add image embeddings (CLIP) to the image collection. Returns count."""
        if not images:
            return 0

        # Images are embedded via CLIP from their saved files
        image_paths = [img["file_path"] for img in images]
        embeddings = await clip_encode_images(image_paths)

        ids = [img["image_id"] for img in images]
        documents = [img["caption"] for img in images]
        metadatas = [
            {
                "doc_id": doc_id,
                "filename": filename,
                "chunk_type": "image",
                "image_id": img["image_id"],
                "file_path": img["file_path"],
                "caption": img["caption"],
                "page_num": img.get("page_num", 0),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            for img in images
        ]

        try:
            self.image_collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as e:
            raise VectorStoreError(f"图片向量存储失败: {str(e)}") from e

        return len(images)

    # ------------------------------------------------------------------
    # Unified Search
    # ------------------------------------------------------------------

    async def search_text(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Search text collection for chunks relevant to the query."""
        from app.services.embedding import embed_single

        if self.text_collection.count() == 0:
            return []

        query_embedding = await embed_single(query)

        try:
            results = self.text_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.text_collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            raise VectorStoreError(f"文本检索失败: {str(e)}") from e

        hits = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                document = results["documents"][0][i] if results["documents"] else ""
                distance = results["distances"][0][i] if results["distances"] else 0.0
                score = 1.0 - distance
                hits.append({
                    "chunk_id": chunk_id,
                    "doc_id": metadata.get("doc_id", ""),
                    "filename": metadata.get("filename", ""),
                    "chunk_type": "text",
                    "chunk_index": metadata.get("chunk_index", 0),
                    "content": document,
                    "content_preview": document[:200] if document else "",
                    "score": round(score, 4),
                })

        return hits

    async def search_image(
        self,
        query: str,
        top_m: int = 3,
    ) -> list[dict]:
        """Search image collection using CLIP text encoding."""
        from app.services.clip_embedding import clip_encode_text

        if self.image_collection.count() == 0:
            return []

        query_embedding = await clip_encode_text(query)

        try:
            results = self.image_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_m, self.image_collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            raise VectorStoreError(f"图片检索失败: {str(e)}") from e

        hits = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                document = results["documents"][0][i] if results["documents"] else ""
                distance = results["distances"][0][i] if results["distances"] else 0.0
                score = 1.0 - distance
                hits.append({
                    "chunk_id": chunk_id,
                    "doc_id": metadata.get("doc_id", ""),
                    "filename": metadata.get("filename", ""),
                    "chunk_type": "image",
                    "image_id": metadata.get("image_id", ""),
                    "file_path": metadata.get("file_path", ""),
                    "caption": metadata.get("caption", ""),
                    "content": document,       # caption text
                    "content_preview": document[:200] if document else "",
                    "page_num": metadata.get("page_num", 0),
                    "score": round(score, 4),
                })

        return hits

    async def search_multimodal(
        self,
        query: str,
        top_k: int = 5,
        top_m: int = 3,
    ) -> dict[str, list[dict]]:
        """Unified multimodal search: returns both text and image results.

        Returns:
            {"texts": [...], "images": [...]}
        """
        texts = await self.search_text(query, top_k=top_k)
        images = await self.search_image(query, top_m=top_m)
        return {"texts": texts, "images": images}

    # ------------------------------------------------------------------
    # Document Administration
    # ------------------------------------------------------------------

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks (text + image) for a document. Returns total removed."""
        total = 0

        # Delete from text collection
        try:
            existing = self.text_collection.get(
                where={"doc_id": doc_id},
                include=["metadatas"],
            )
            count = len(existing["ids"]) if existing["ids"] else 0
            if count > 0:
                self.text_collection.delete(where={"doc_id": doc_id})
            total += count
        except Exception as e:
            raise VectorStoreError(f"文本删除失败: {str(e)}") from e

        # Delete from image collection
        try:
            existing = self.image_collection.get(
                where={"doc_id": doc_id},
                include=["metadatas"],
            )
            count = len(existing["ids"]) if existing["ids"] else 0
            if count > 0:
                self.image_collection.delete(where={"doc_id": doc_id})
            total += count
        except Exception as e:
            raise VectorStoreError(f"图片删除失败: {str(e)}") from e

        return total

    def doc_exists(self, doc_id: str) -> bool:
        """Check if a document exists in either collection."""
        for col in [self.text_collection, self.image_collection]:
            existing = col.get(where={"doc_id": doc_id}, limit=1)
            if len(existing["ids"]) > 0 if existing["ids"] else False:
                return True
        return False

    def list_documents(self) -> list[dict]:
        """List all unique documents with chunk counts across both collections."""
        doc_map: dict[str, dict] = {}

        for col in [self.text_collection, self.image_collection]:
            try:
                all_data = col.get(include=["metadatas"])
            except Exception:
                continue

            if not all_data["metadatas"]:
                continue

            for meta in all_data["metadatas"]:
                did = meta["doc_id"]
                if did not in doc_map:
                    doc_map[did] = {
                        "doc_id": did,
                        "filename": meta.get("filename", ""),
                        "file_type": meta.get("file_type", ""),
                        "text_chunk_count": 0,
                        "image_count": 0,
                        "total_chunks": 0,
                        "created_at": meta.get("created_at", ""),
                    }
                if meta.get("chunk_type") == "image":
                    doc_map[did]["image_count"] += 1
                else:
                    doc_map[did]["text_chunk_count"] += 1
                doc_map[did]["total_chunks"] += 1

        return list(doc_map.values())

    def get_document_images(self, doc_id: str) -> list[dict]:
        """Get all images for a document."""
        try:
            existing = self.image_collection.get(
                where={"doc_id": doc_id},
                include=["metadatas", "documents"],
            )
        except Exception as e:
            raise VectorStoreError(f"获取图片列表失败: {str(e)}") from e

        results = []
        if existing["ids"]:
            for i, img_id in enumerate(existing["ids"]):
                meta = existing["metadatas"][i] if existing["metadatas"] else {}
                results.append({
                    "image_id": img_id,
                    "doc_id": doc_id,
                    "file_path": meta.get("file_path", ""),
                    "caption": meta.get("caption", ""),
                    "page_num": meta.get("page_num", 0),
                })
        return results

    def count(self) -> int:
        """Return total chunks across both collections."""
        return self.text_collection.count() + self.image_collection.count()

    def clear(self):
        """Remove all data from both collections (for testing)."""
        try:
            self.client.delete_collection(self._text_collection_name)
        except Exception:
            pass
        try:
            self.client.delete_collection(self._image_collection_name)
        except Exception:
            pass
        self._text_collection = None
        self._image_collection = None


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

_store: VectorStore | None = None
_store_lock = asyncio.Lock()


async def get_vector_store() -> VectorStore:
    global _store
    if _store is not None:
        return _store
    async with _store_lock:
        if _store is None:
            _store = VectorStore()
        return _store
