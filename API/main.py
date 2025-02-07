import sys
from pathlib import Path

# Add parent directory to Python path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from configs.config import CloudConfig
from API.client import CloudStorageClient

# Validate configuration at startup
CloudConfig.validate()

app = FastAPI()
storage_client = CloudStorageClient()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
