import os

import uvicorn

from .server import app

if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("WHISPER_DEMO_HOST", "127.0.0.1"),
                port=int(os.environ.get("WHISPER_DEMO_PORT", 8001)))
