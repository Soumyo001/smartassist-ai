import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"

APP_DB = DATA_DIR / "task_manager.sqlite3"

GRAPH_DB = DATA_DIR / "graph_checkpoints.sqlite3"

TIMEZONE_NAME = os.getenv(
    "USER_TIMEZONE",
    "Asia/Dhaka"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:4b"
)

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

PERMISSIONS_FILE = (
    ROOT / "config" / "tool_permissions.json"
)