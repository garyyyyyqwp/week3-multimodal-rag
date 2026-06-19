import os
from dotenv import load_dotenv

load_dotenv()


def get_env(key: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(key, default)
    if required and value is None:
        raise ValueError(
            f"Environment variable '{key}' is not set. "
            f"Please set it in .env file or in the environment."
        )
    return value


# --- LLM (Text) ---
OPENAI_API_KEY = get_env("OPENAI_API_KEY", required=True)
OPENAI_MODEL = get_env("OPENAI_MODEL", "glm-4-flash")
OPENAI_BASE_URL = get_env("OPENAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")

# --- Vision LLM (GLM-4.6V-Flash) ---
VISION_MODEL = get_env("VISION_MODEL", "glm-4.6v-flash")
VISION_API_KEY = get_env("VISION_API_KEY", OPENAI_API_KEY)
VISION_BASE_URL = get_env("VISION_BASE_URL", OPENAI_BASE_URL)

# --- Embedding ---
EMBEDDING_PROVIDER = get_env("EMBEDDING_PROVIDER", "openai")
EMBEDDING_MODEL = get_env("EMBEDDING_MODEL", "embedding-2")
EMBEDDING_API_KEY = get_env("EMBEDDING_API_KEY", OPENAI_API_KEY)
EMBEDDING_BASE_URL = get_env("EMBEDDING_BASE_URL", OPENAI_BASE_URL)

# --- CLIP ---
CLIP_MODEL_NAME = get_env("CLIP_MODEL_NAME", "ViT-B/32")
CLIP_DEVICE = get_env("CLIP_DEVICE", "cpu")

# --- ChromaDB ---
CHROMA_PERSIST_DIR = get_env("CHROMA_PERSIST_DIR", "./chroma_data")
TEXT_COLLECTION_NAME = get_env("TEXT_COLLECTION_NAME", "text_collection")
IMAGE_COLLECTION_NAME = get_env("IMAGE_COLLECTION_NAME", "image_collection")

# --- Chunker ---
CHUNK_MAX_TOKENS = int(get_env("CHUNK_MAX_TOKENS", "512"))
CHUNK_OVERLAP_TOKENS = int(get_env("CHUNK_OVERLAP_TOKENS", "50"))

# --- RAG ---
RAG_TOP_K = int(get_env("RAG_TOP_K", "5"))
RAG_TOP_M = int(get_env("RAG_TOP_M", "3"))

# --- Image Processing ---
IMAGE_MAX_DIMENSION = int(get_env("IMAGE_MAX_DIMENSION", "512"))
IMAGE_SAVE_DIR = get_env("IMAGE_SAVE_DIR", "./data/images")

# --- Experiment ---
EXPERIMENT_MAX_CONCURRENCY = int(get_env("EXPERIMENT_MAX_CONCURRENCY", "5"))
EXPERIMENT_DEFAULT_TEST_CASES = int(get_env("EXPERIMENT_DEFAULT_TEST_CASES", "20"))
EXPERIMENT_DATA_DIR = get_env("EXPERIMENT_DATA_DIR", "./data/experiments")
