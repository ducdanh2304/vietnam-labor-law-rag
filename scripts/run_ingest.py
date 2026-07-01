"""
Entry point script để build lại ChromaDB từ dữ liệu thô trong data/raw/.

Usage:
    python scripts/run_ingest.py
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.ingest import ingest_pdf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "luat_lao_dong.txt")

if __name__ == "__main__":
    ingest_pdf(DATA_PATH)