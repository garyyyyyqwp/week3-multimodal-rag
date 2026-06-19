from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os

from app.routers import documents, qa, experiments

app = FastAPI(
    title="多模态 RAG + Prompt 策略评估平台",
    description=(
        "Multimodal RAG Knowledge Base with Prompt Strategy Evaluation. "
        "Upload image-containing PDFs → multimodal retrieval → "
        "multi-strategy prompt comparison → LLM-as-Judge auto-scoring → "
        "comparison report generation."
    ),
    version="0.2.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure data directories exist
for d in ["./data/images", "./data/experiments"]:
    os.makedirs(d, exist_ok=True)

# Static files directory
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)

# Routers
app.include_router(documents.router, prefix="/api/v1/documents")
app.include_router(qa.router, prefix="/api/v1/qa")
app.include_router(experiments.router, prefix="/api/v1/experiments")


@app.get("/", include_in_schema=False)
async def root():
    """Serve the SPA frontend."""
    return FileResponse(static_dir / "index.html")


@app.get("/health", include_in_schema=False)
async def health_check():
    return {
        "status": "ok",
        "service": "Multimodal RAG + Prompt Evaluation Platform",
        "version": "0.2.0",
    }


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    detail = getattr(exc, "detail", "")
    if detail and detail != "Not Found":
        return JSONResponse(status_code=404, content={"detail": detail})

    # SPA fallback — serve index.html for non-API routes
    full_path = request.url.path
    if not full_path.startswith("/api/") and not full_path.startswith("/docs") and not full_path.startswith("/openapi.json") and not full_path.startswith("/health"):
        target = static_dir / "index.html"
        if target.exists():
            return FileResponse(target)

    return JSONResponse(
        status_code=404,
        content={
            "detail": "Not Found",
            "message": "请求的资源不存在。可用接口请查看 /docs",
            "available_endpoints": {
                "swagger_docs": "/docs",
                "health_check": "/health",
                "upload_doc": "POST /api/v1/documents/upload",
                "list_docs": "GET /api/v1/documents",
                "get_doc_images": "GET /api/v1/documents/{id}/images",
                "delete_doc": "DELETE /api/v1/documents/{doc_id}",
                "ask_qa": "POST /api/v1/qa/ask",
                "ask_sync": "POST /api/v1/qa/ask/sync",
                "create_experiment": "POST /api/v1/experiments",
                "list_experiments": "GET /api/v1/experiments",
                "run_experiment": "POST /api/v1/experiments/{id}/run",
                "get_report": "GET /api/v1/experiments/{id}/report",
            },
        },
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误", "message": "请联系管理员"},
    )
