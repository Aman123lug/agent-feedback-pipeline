"""
Adaptive Feedback Pipeline — Entry Point

Run:  python run.py
UI:   http://localhost:8000
API:  POST /api/chat, GET /api/state, GET /api/threads
"""

import logging
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

from pipeline import app  # noqa: E402


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  ADAPTIVE FEEDBACK PIPELINE — Python Backend")
    print("=" * 60)
    print()
    print("  🎯 Dashboard: http://localhost:8000")
    print("  📊 State API: http://localhost:8000/api/state")
    print("  💬 Chat API:  POST http://localhost:8000/api/chat")
    print()
    uvicorn.run(app, host="0.0.0.0", port=8000)
