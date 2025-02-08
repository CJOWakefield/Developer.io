from flask import Flask
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.models.predictor import RegionPredictor
from api.client import CloudStorageClient

predictor = None
cloud_client = None

def create_app():
    app = Flask(__name__)
    
    # Initialize global objects
    global predictor, cloud_client
    try:
        predictor = RegionPredictor()
        print("Successfully initialized RegionPredictor")
    except Exception as e:
        print(f"Error initializing RegionPredictor: {str(e)}")
        predictor = None

    try:
        cloud_client = CloudStorageClient()
        print("Successfully initialized CloudStorageClient")
    except Exception as e:
        print(f"Error initializing CloudStorageClient: {str(e)}")
        cloud_client = None
    
    # Ensure required directories exist
    cache_dir = Path(project_root) / 'cache'
    cache_dir.mkdir(exist_ok=True)
    
    # Register blueprints
    from app.routes.main import main_bp
    app.register_blueprint(main_bp)
    
    return app
