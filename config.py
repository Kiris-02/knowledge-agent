import os
from pathlib import Path
from dotenv import load_dotenv

# Configure BOOKS_DIR adaptively
env_books_dir = os.getenv("BOOKS_DIR")
if env_books_dir and Path(env_books_dir).exists():
    BOOKS_DIR = Path(env_books_dir)
elif Path(r"d:\BÌnh\Sách Báo Chí").exists():
    BOOKS_DIR = Path(r"d:\BÌnh\Sách Báo Chí")
elif (BASE_DIR.parent / "(khonglao)-nho-giao-tran-trong-kim-dantocking.com.pdf").exists():
    BOOKS_DIR = BASE_DIR.parent
else:
    BOOKS_DIR = BASE_DIR / "books"
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
INDEX_METADATA_FILE = DATA_DIR / "indexed_books.json"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".epub"}

# LLM Providers Configuration
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODELS = ["deepseek-chat", "deepseek-reasoner"]

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-1.5-flash",
]

# Chunking settings
DEFAULT_CHUNK_SIZE = 1000  # characters
DEFAULT_CHUNK_OVERLAP = 150
TOP_K_RESULTS = 5
