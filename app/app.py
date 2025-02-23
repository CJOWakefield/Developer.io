from flask import Flask, render_template, request, jsonify
from api.cloud_storage_client import CloudStorageClient
from src.models.predictor import RegionPredictor
from pathlib import Path
import os
import json

app = Flask(__name__)
cloud_client = CloudStorageClient()
predictor = RegionPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search-location', methods=['GET'])
def search_location():
    query = request.args.get('query', '')
    # Implement location search using a geocoding service
    suggestions = get_location_suggestions(query)
    return jsonify(suggestions)

@app.route('/get-region-data', methods=['POST'])
def get_region_data():
    data = request.json
    bounds = data.get('bounds')
    location = data.get('location')
    
    try:
        # Create region dictionary for predictor
        region = {
            'name': location['name'],
            'coordinates': {
                'north': bounds['north'],
                'south': bounds['south'],
                'east': bounds['east'],
                'west': bounds['west']
            }
        }
        
        # Get predictions using RegionPredictor
        prediction_results = predictor.predict_region(region)
        
        # Cache results
        cache_results(location['name'], prediction_results)
        
        return jsonify({
            'status': 'success',
            'predictions': {
                'image_paths': prediction_results['image_paths'],
                'prediction_paths': prediction_results['prediction_paths'],
                'metrics': prediction_results['metrics'] if 'metrics' in prediction_results else {}
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
    # Implement location search
    # For now, return dummy data
    return [
        {
            'name': 'Example Location',
            'lat': 0,
            'lng': 0
        }
    ]

@app.route('/upload', methods=['POST'])
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

@app.route('/list-regions', methods=['GET'])
def list_regions():
    # Example regions data - replace with your actual data
    regions = {
        'asia': {
            'coordinates': [34.047863, 100.619655],
            'count': len(cloud_client.list_files(prefix='images'))
        }
    }
    return jsonify(regions)

if __name__ == '__main__':
    app.run(debug=True) 