from flask import Flask, render_template, request, jsonify, send_file
from api.cloud_storage_client import CloudStorageClient
from src.models.predictor import RegionPredictor
from pathlib import Path
import os
import json
import asyncio
import logging
import io
import cv2
import numpy as np

from run_api import SatelliteAPI
import base64

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger('developerio_app')

app = Flask(__name__, 
    static_folder='static',
    template_folder='templates')
cloud_client = CloudStorageClient()
predictor = RegionPredictor()
satellite_api = SatelliteAPI(mode='single') 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search-location', methods=['GET'])
def search_location():
    query = request.args.get('query', '')
    components = query.split(',')
    components = [c.strip() for c in components]
    
    country = components[0] if len(components) > 0 else ""
    city = components[1] if len(components) > 1 else ""
    postcode = components[2] if len(components) > 2 else None
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(satellite_api.process_location(country, city, postcode))
    loop.close()
    
    if result['status'] == 'success':
        return jsonify({
            'status': 'success',
            'suggestions': [{
                'name': result['address'],
                'lat': result['coordinates']['lat'],
                'lng': result['coordinates']['lon']
            }]
        })
    else:
        return jsonify({
            'status': 'error',
            'message': result['message'],
            'suggestions': []
        })

@app.route('/get-satellite-image', methods=['POST'])
def get_satellite_image():
    data = request.json
    lat = data.get('lat')
    lon = data.get('lng')
    mode = data.get('mode', 'single')
    
    if not lat or not lon:
        return jsonify({'status': 'error', 'message': 'Latitude and longitude are required'})
    
    if satellite_api.mode != mode:
        satellite_api.mode = mode
        app.logger.info(f"Updated satellite API mode to {mode}")
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(satellite_api.get_satellite_image(float(lat), float(lon)))
    loop.close()
    
    if result['status'] == 'success':
        image_urls = []
        for i, path in enumerate(satellite_api.image_paths):
            image_id = f"img_{i}_{os.path.basename(path)}"
            image_urls.append(f"/direct-image/{image_id}")
        
        return jsonify({
            'status': 'success',
            'image_paths': image_urls
        })
    else:
        return jsonify({
            'status': 'error',
            'message': result['message']
        })

@app.route('/direct-image/<path:image_id>', methods=['GET'])
def direct_image(image_id):
    try:
        parts = image_id.split('_', 2)
        if len(parts) < 3:
            return jsonify({'status': 'error', 'message': 'Invalid image ID format'}), 400
        
        index = int(parts[1])
        filename = parts[2]
        
        if index < len(satellite_api.image_bytes):
            return send_file(
                io.BytesIO(satellite_api.image_bytes[index]),
                mimetype='image/jpeg',
                as_attachment=False,
                download_name=filename
            )
        
        for path in satellite_api.image_paths:
            if os.path.basename(path) == filename:
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    return send_file(path, mimetype='image/jpeg')
        
        return jsonify({'status': 'error', 'message': 'Image not found'}), 404
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get-segmentation', methods=['POST'])
def get_segmentation():
    data = request.json
    image_path = data.get('image_path')
    
    if not image_path:
        return jsonify({'status': 'error', 'message': 'Image path is required'})
    
    image_id = image_path.split('/')[-1]
    parts = image_id.split('_', 2)
    
    if len(parts) < 3:
        return jsonify({'status': 'error', 'message': 'Invalid image ID format'}), 400
    
    index = int(parts[1])
    
    if index >= len(satellite_api.image_paths):
        return jsonify({'status': 'error', 'message': 'Image index out of range'})
    
    actual_path = satellite_api.image_paths[index]
    
    if not os.path.exists(actual_path) or os.path.getsize(actual_path) == 0:
        if index < len(satellite_api.image_bytes):
            try:
                os.makedirs(os.path.dirname(actual_path), exist_ok=True)
                with open(actual_path, 'wb') as f:
                    f.write(satellite_api.image_bytes[index])
            except Exception:
                pass
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(satellite_api.get_segmentation_mask(actual_path))
    loop.close()
    
    if result['status'] == 'success':
        try:
            mask_id = f"mask_{index}_{os.path.basename(actual_path).replace('.jpg', '.png')}"
            
            if not hasattr(satellite_api, 'masks'):
                satellite_api.masks = {}
            
            _, buffer = cv2.imencode('.png', cv2.cvtColor(result['mask'], cv2.COLOR_RGB2BGR))
            mask_bytes = buffer.tobytes()
            
            satellite_api.masks[mask_id] = mask_bytes
            
            mask_url = f"/direct-mask/{mask_id}"
            
            return jsonify({
                'status': 'success',
                'mask_path': mask_url
            })
        except Exception as e:
            return jsonify({
                'status': 'error', 
                'message': f'Error processing mask: {str(e)}'
            })
    else:
        return jsonify({
            'status': 'error',
            'message': result['message']
        })

@app.route('/direct-mask/<path:mask_id>', methods=['GET'])
def direct_mask(mask_id):
    try:
        if hasattr(satellite_api, 'masks') and mask_id in satellite_api.masks:
            return send_file(
                io.BytesIO(satellite_api.masks[mask_id]),
                mimetype='image/png',
                as_attachment=False,
                download_name=mask_id.split('_')[-1]
            )
        
        parts = mask_id.split('_', 2)
        if len(parts) >= 3:
            index = int(parts[1])
            
            if index < len(satellite_api.image_paths):
                original_path = satellite_api.image_paths[index]
                mask_path = original_path.replace('.jpg', '_mask.png')
                
                if os.path.exists(mask_path) and os.path.getsize(mask_path) > 0:
                    return send_file(mask_path, mimetype='image/png')
        
        return jsonify({'status': 'error', 'message': 'Mask not found'}), 404
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get-land-proportions', methods=['POST'])
def get_land_proportions():
    data = request.json
    image_path = data.get('image_path')
    
    if not image_path:
        return jsonify({'status': 'error', 'message': 'Image path is required'})
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(satellite_api.get_land_type_proportions(image_path))
    loop.close()
    
    if result['status'] == 'success':
        return jsonify({
            'status': 'success',
            'proportions': result['proportions']
        })
    else:
        return jsonify({
            'status': 'error',
            'message': result['message']
        })

@app.route('/identify-suitable-locations', methods=['POST'])
def identify_suitable_locations():
    data = request.json
    purpose = data.get('purpose')
    min_area_sqm = data.get('min_area_sqm', 1000)
    
    if not purpose:
        return jsonify({'status': 'error', 'message': 'Purpose is required'})
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(satellite_api.identify_suitable_locations(purpose, min_area_sqm))
    loop.close()
    
    if result['status'] == 'success':
        return jsonify({
            'status': 'success',
            'suitable_locations': result['suitable_locations']
        })
    else:
        return jsonify({
            'status': 'error',
            'message': result['message']
        })

@app.route('/reverse-geocode', methods=['GET'])
def reverse_geocode():
    lat = request.args.get('lat')
    lng = request.args.get('lng')
    
    if not lat or not lng:
        return jsonify({'status': 'error', 'message': 'Latitude and longitude are required'})
    
    try:
        # In a real implementation, this would call a geocoding service
        # For now, we'll just return a placeholder response
        location_name = f"Location at {lat}, {lng}"
        
        return jsonify({
            'status': 'success',
            'address': location_name
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

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
    regions = {
        'asia': {
            'coordinates': [34.047863, 100.619655],
            'count': len(cloud_client.list_files(prefix='images'))
        }
    }
    return jsonify(regions)

if __name__ == '__main__':
    # Ensure the static directories exist
    os.makedirs(os.path.join('static', 'css'), exist_ok=True)
    os.makedirs(os.path.join('static', 'js'), exist_ok=True)
    
    # Create symbolic links or copy files if needed
    css_path = os.path.join('static', 'css', 'style.css')
    js_path = os.path.join('static', 'js', 'main.js')
    
    if not os.path.exists(css_path):
        print(f"Warning: CSS file not found at {css_path}. Make sure to create this file.")
    
    if not os.path.exists(js_path):
        print(f"Warning: JS file not found at {js_path}. Make sure to create this file.")
    
    app.run(debug=True)