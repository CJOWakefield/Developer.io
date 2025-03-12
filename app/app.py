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

# We've updated run_api.py, so reimport to get the latest version
from run_api import SatelliteAPI
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger('satellite_app')

app = Flask(__name__)
cloud_client = CloudStorageClient()
predictor = RegionPredictor()
satellite_api = SatelliteAPI(mode='single')  # Default to single mode, can be updated based on user preference

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search-location', methods=['GET'])
def search_location():
    query = request.args.get('query', '')
    # Split query into potential components (country, city, postcode)
    components = query.split(',')
    components = [c.strip() for c in components]
    
    country = components[0] if len(components) > 0 else ""
    city = components[1] if len(components) > 1 else ""
    postcode = components[2] if len(components) > 2 else None
    
    # Use asyncio to run the async function
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
    mode = data.get('mode', 'single')  # 'single' or 'grid'
    
    if not lat or not lon:
        return jsonify({'status': 'error', 'message': 'Latitude and longitude are required'})
    
    # Update API mode if needed
    if satellite_api.mode != mode:
        satellite_api.mode = mode
        app.logger.info(f"Updated satellite API mode to {mode}")
    
    # Use asyncio to run the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(satellite_api.get_satellite_image(float(lat), float(lon)))
    loop.close()
    
    if result['status'] == 'success':
        # Create URLs for the frontend that use our direct image serving endpoint
        image_urls = []
        
        for i, path in enumerate(satellite_api.image_paths):
            # Create a unique identifier for this image
            image_id = f"img_{i}_{os.path.basename(path)}"
            image_urls.append(f"/direct-image/{image_id}")
            app.logger.info(f"Created direct image URL: /direct-image/{image_id}")
        
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
    app.logger.info(f"Requested direct image: {image_id}")
    
    try:
        # Parse the image ID to get the index and filename
        parts = image_id.split('_', 2)
        if len(parts) < 3:
            app.logger.error(f"Invalid image ID format: {image_id}")
            return jsonify({'status': 'error', 'message': 'Invalid image ID format'}), 400
        
        index = int(parts[1])
        filename = parts[2]
        
        # Check if we have the image in memory
        if index < len(satellite_api.image_bytes):
            app.logger.info(f"Serving image directly from memory: index={index}, filename={filename}")
            return send_file(
                io.BytesIO(satellite_api.image_bytes[index]),
                mimetype='image/jpeg',
                as_attachment=False,
                download_name=filename
            )
        
        # If not in memory, try to find it on disk
        for path in satellite_api.image_paths:
            if os.path.basename(path) == filename:
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    app.logger.info(f"Serving image from disk: {path}")
                    return send_file(path, mimetype='image/jpeg')
        
        # If we got here, no image was found
        app.logger.error(f"Image not found: {image_id}")
        return jsonify({'status': 'error', 'message': 'Image not found'}), 404
    
    except Exception as e:
        app.logger.error(f"Error serving direct image {image_id}: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get-segmentation', methods=['POST'])
def get_segmentation():
    data = request.json
    image_path = data.get('image_path')
    
    if not image_path:
        return jsonify({'status': 'error', 'message': 'Image path is required'})
    
    app.logger.info(f"Processing segmentation for image: {image_path}")
    
    # Extract the image ID from the URL
    image_id = image_path.split('/')[-1]
    parts = image_id.split('_', 2)
    
    if len(parts) < 3:
        app.logger.error(f"Invalid image ID format: {image_id}")
        return jsonify({'status': 'error', 'message': 'Invalid image ID format'}), 400
    
    index = int(parts[1])
    
    # Get the actual image path from the API
    if index >= len(satellite_api.image_paths):
        app.logger.error(f"Image index out of range: {index}")
        return jsonify({'status': 'error', 'message': 'Image index out of range'})
    
    actual_path = satellite_api.image_paths[index]
    
    # Ensure the image file exists
    if not os.path.exists(actual_path) or os.path.getsize(actual_path) == 0:
        # Try to recreate it from image_bytes
        if index < len(satellite_api.image_bytes):
            try:
                os.makedirs(os.path.dirname(actual_path), exist_ok=True)
                with open(actual_path, 'wb') as f:
                    f.write(satellite_api.image_bytes[index])
                app.logger.info(f"Recreated image file from bytes: {actual_path}")
            except Exception as e:
                app.logger.error(f"Error recreating image file: {str(e)}")
    
    # Use asyncio to run the async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(satellite_api.get_segmentation_mask(actual_path))
    loop.close()
    
    if result['status'] == 'success':
        try:
            # Create a unique ID for the mask
            mask_id = f"mask_{index}_{os.path.basename(actual_path).replace('.jpg', '.png')}"
            
            # Store the mask in memory for direct serving
            if not hasattr(satellite_api, 'masks'):
                satellite_api.masks = {}
            
            # Convert mask to PNG bytes
            _, buffer = cv2.imencode('.png', cv2.cvtColor(result['mask'], cv2.COLOR_RGB2BGR))
            mask_bytes = buffer.tobytes()
            
            # Store the mask bytes
            satellite_api.masks[mask_id] = mask_bytes
            
            # Create a URL for the frontend
            mask_url = f"/direct-mask/{mask_id}"
            
            return jsonify({
                'status': 'success',
                'mask_path': mask_url
            })
        except Exception as e:
            app.logger.error(f"Error processing mask: {str(e)}")
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
    app.logger.info(f"Requested direct mask: {mask_id}")
    
    try:
        # Check if we have the mask in memory
        if hasattr(satellite_api, 'masks') and mask_id in satellite_api.masks:
            app.logger.info(f"Serving mask directly from memory: {mask_id}")
            return send_file(
                io.BytesIO(satellite_api.masks[mask_id]),
                mimetype='image/png',
                as_attachment=False,
                download_name=mask_id.split('_')[-1]
            )
        
        # If not in memory, try to find it on disk
        parts = mask_id.split('_', 2)
        if len(parts) >= 3:
            index = int(parts[1])
            filename = parts[2]
            
            if index < len(satellite_api.image_paths):
                original_path = satellite_api.image_paths[index]
                mask_path = original_path.replace('.jpg', '_mask.png')
                
                if os.path.exists(mask_path) and os.path.getsize(mask_path) > 0:
                    app.logger.info(f"Serving mask from disk: {mask_path}")
                    return send_file(mask_path, mimetype='image/png')
        
        # If we got here, no mask was found
        app.logger.error(f"Mask not found: {mask_id}")
        return jsonify({'status': 'error', 'message': 'Mask not found'}), 404
    
    except Exception as e:
        app.logger.error(f"Error serving direct mask {mask_id}: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get-land-proportions', methods=['POST'])
def get_land_proportions():
    data = request.json
    image_path = data.get('image_path')
    
    if not image_path:
        return jsonify({'status': 'error', 'message': 'Image path is required'})
    
    # Use asyncio to run the async function
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
    
    # Use asyncio to run the async function
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