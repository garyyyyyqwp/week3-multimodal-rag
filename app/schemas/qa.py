from typing import Optional

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000, description="用户自然语言问题")
    top_k: int = Field(default=5, ge=1, le=50, description="检索的文字 Chunk 数量")
    top_m: int = Field(default=3, ge=0, le=20, description="检索的图片数量")
    strategy: str = Field(default="basic", description="Prompt 策略: basic | fewshot | cot | multimodal_step")
    force_vision: Optional[bool] = Field(default=None, description="强制使用/禁用 Vision LLM（默认自动判断）")


class TextCitation(BaseModel):
    ref_label: str
    doc_id: str
    filename: str
    chunk_index: int
    content_snippet: str


class ImageCitation(BaseModel):
    ref_label: str
    image_id: str
    filename: str
    caption: str


class RetrievedChunkPreview(BaseModel):
    ref_label: str = ""
    doc_id: str
    filename: str
    content_preview: str
    score: float


class ImageChunkPreview(BaseModel):
    ref_label: str = ""
    image_id: str
    filename: str
    caption: str
    score: float


class AskResponse(BaseModel):
    answer: str
    text_citations: list[TextCitation]
    image_citations: list[ImageCitation]
    retrieved_texts: list[RetrievedChunkPreview]
    retrieved_images: list[ImageChunkPreview]
    use_vision: bool = False
