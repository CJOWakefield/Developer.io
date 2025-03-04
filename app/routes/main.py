from flask import Blueprint, render_template, request, jsonify
from app.functions import get_location_suggestions
from src.models.predictor import RegionPredictor
from api.cloud_storage_client import CloudStorageClient
import os
from pathlib import Path
import json

main_bp = Blueprint('main', __name__)

cloud_client = CloudStorageClient()
predictor = RegionPredictor()

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

@main_bp.route('/api/location-suggestions', methods=['GET'])
def location_suggestions():
    query = request.args.get('query')
    if not query:
        return jsonify([])

    suggestions = get_location_suggestions(query)
    return jsonify(suggestions)

@main_bp.route('/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files[]')
    uploaded_urls = []
    
    for file in files:
        if file:
            filename = file.filename
            temp_path = Path('temp') / filename
            os.makedirs('temp', exist_ok=True)
            file.save(temp_path)
            
            try:
                url = cloud_client.upload_file(temp_path, f"uploads/{filename}")
                uploaded_urls.append(url)
                os.remove(temp_path)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    return jsonify({'urls': uploaded_urls})

@main_bp.route('/list-regions', methods=['GET'])
def list_regions():
    regions = {
        'asia': {
            'coordinates': [34.047863, 100.619655],
            'count': len(cloud_client.list_files(prefix='images'))
        }
    }
    return jsonify(regions) 