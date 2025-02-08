import sys
from pathlib import Path

# Add parent directory to Python path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from configs.config import CloudConfig
from api.client import CloudStorageClient
from flask import Blueprint, render_template, jsonify, request
from app import predictor, cloud_client
import json

# Validate configuration at startup
CloudConfig.validate()

app = FastAPI()
storage_client = CloudStorageClient()

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/search-location', methods=['GET'])
def search_location():
    query = request.args.get('query', '')
    suggestions = get_location_suggestions(query)
    return jsonify(suggestions)

@main_bp.route('/get-region-data', methods=['POST'])
def get_region_data():
    data = request.json
    bounds = data.get('bounds')
    location = data.get('location')
    
    try:
        region = {
            'name': location['name'],
            'coordinates': {
                'north': bounds['north'],
                'south': bounds['south'],
                'east': bounds['east'],
                'west': bounds['west']
            }
        }
        
        prediction_results = predictor.predict_region(region)
        cache_results(location['name'], prediction_results)
        
        return jsonify({
            'status': 'success',
            'predictions': {
                'image_paths': prediction_results['image_paths'],
                'prediction_paths': prediction_results['prediction_paths'],
                'metrics': prediction_results.get('metrics', {})
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def cache_results(location_name: str, predictions: dict):
    cache_dir = Path('cache')
    cache_dir.mkdir(exist_ok=True)
    
    cache_file = cache_dir / f"{location_name}_predictions.json"
    with open(cache_file, 'w') as f:
        json.dump(predictions, f)

def get_location_suggestions(query: str) -> list:
    return [
        {
            'name': 'Example Location',
            'lat': 0,
            'lng': 0
        }
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
