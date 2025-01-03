import os
import math
import json
import requests
from PIL import Image
from io import BytesIO
from geopy.geocoders import Nominatim
from dotenv import load_dotenv
import urllib

def load_api_key():
    """
    Load Google Maps API key from .env file
    """
    load_dotenv()
    if not os.getenv('GOOGLE_MAPS_API_KEY'): raise ValueError("API key invalid.")
    return os.getenv('GOOGLE_MAPS_API_KEY')

def ret_city(longitude, latitude) -> str:
    """
    Find the nearest city or town to given coordinates.
    """
    geolocator = Nominatim(user_agent="birds_eye_view_downloader")
    try:
        location = geolocator.reverse(f"{latitude}, {longitude}")
        if location and 'address' in location.raw:
            address = location.raw['address']
            city_names = [address[x] for x in ['city', 'town', 'village'] if x in address]
            if city_names:
                return city_names[0].lower().replace(' ', '_')
    except Exception as e: return f'anon_{latitude:.3f}_{longitude:.3f}'
    return f"location_{latitude:.3f}_{longitude:.3f}"

def ret_zoom(grid_size_km) -> int:
    """
    Convert desired grid input to relevant Google Maps zoom specification.

    Zoom Level Coverage (approximate for 640x640 pixels):
    Zoom 20: ~100m, Zoom 19: ~200m, Zoom 18: ~400m, Zoom 17: ~800m, Zoom 16: ~1.5km, Zoom 15: ~3km, Zoom 14: ~6km
    """
    grid_km = grid_size_km * 1000
    assert grid_km <= 3000
    grid_specs = {100: 20, 200: 19, 400: 18, 800: 17, 1500: 16, 3000: 15}

    zoom_diffs = [abs(grid_km-key) for key in grid_specs]
    return list(grid_specs.values())[zoom_diffs.index(min(zoom_diffs))]

def ret_zoom_coverage(grid_size_km, zoom):
    """
    Prints verification information about the selected zoom level coverage
    """
    coverage_map = {
        20: "~100m",
        19: "~200m",
        18: "~400m",
        17: "~800m",
        16: "~1.5km",
        15: "~3km",
        14: "~6km"
    }
    print(f"\nZoom Level Settings:")
    print(f"Requested grid size: {grid_size_km*1000:.0f}m ---- Zoom level: {zoom}")
    print(f"Approximate coverage at this zoom: {coverage_map.get(zoom, 'unknown')}")

def ret_coords(center_lat, center_lon, grid_size_km, num_images):
    """
    Calculate grid coordinates based on desired grid size and number of images
    """
    # Calculate grid dimensions
    grid_dim = math.ceil(math.sqrt(num_images))
    total_area = grid_size_km * grid_dim
    lat_adj = (total_area / 2) / 111.32
    lon_adj = (total_area / 2) / (111.32 * math.cos(math.radians(center_lat)))
    sub_grid_lat, sub_grid_lon = (2 * lat_adj) / grid_dim, (2 * lon_adj) / grid_dim
    start_lat, start_lon = center_lat + lat_adj - (sub_grid_lat / 2), center_lon - lon_adj + (sub_grid_lon / 2)

    coordinates = []
    for i in range(grid_dim):
        for j in range(grid_dim):
            if len(coordinates) < num_images:  # Only add up to num_images
                lat = start_lat - (i * sub_grid_lat)
                lon = start_lon + (j * sub_grid_lon)
                coordinates.append((lat, lon))

    return coordinates, grid_dim

def image_download(api, latitude, longitude, zoom, size=(640, 640)):
    base_url = "https://maps.googleapis.com/maps/api/staticmap?"
    params = {'center': f'{latitude},{longitude}',
              'zoom': zoom,
              'size': f'{size[0]}x{size[1]}',
              'maptype': 'satellite',
              'key': api,
              'scale': 2}

    response = requests.get(f'{base_url}{urllib.parse.urlencode(params)}')
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        raise Exception(f"Download fail: {response.status_code}")

def download_area(api, center_lat, center_lon, grid_size_km, num_images):
    location = ret_city(center_lon, center_lat)
    zoom = ret_zoom(grid_size_km)
    coordinates, grid_dim = ret_coords(center_lat, center_lon, grid_size_km, num_images)
    total_area_km = grid_size_km * grid_dim

    # Create unique ID starting point (e.g., 100000 to avoid conflicts)
    base_id = 100000

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_generated_data')
    os.makedirs(base_dir, exist_ok=True)

    output_dir = os.path.join(base_dir,
                             f'{location}_{center_lat:.3f}_{center_lon:.3f}_{total_area_km:.1f}km_{grid_size_km*1000:.0f}m')
    os.makedirs(output_dir, exist_ok=True)

    coordinate_info = {
        'metadata': {
            'center_latitude': center_lat,
            'center_longitude': center_lon,
            'grid_size_km': grid_size_km,
            'total_area_km': total_area_km,
            'num_images': num_images,
            'grid_dimension': grid_dim,
            'zoom_level': zoom
        },
        'images': {}
    }

    for i, (lat, lon) in enumerate(coordinates, 1):
        try:
            print(f"Downloading image {i} of {num_images} ---- position: {lat:.5f}, {lon:.5f}")

            img = image_download(api, lat, lon, zoom=zoom)
            img = img.convert('RGB')

            row = (i - 1) // grid_dim
            col = (i - 1) % grid_dim

            # Create unique ID and filename
            image_id = base_id + i
            filename = f'{image_id}_sat.jpg'

            output_path = os.path.join(output_dir, filename)
            img.save(output_path, 'JPEG', quality=95)

            coordinate_info['images'][filename] = {
                'id': image_id,
                'latitude': lat,
                'longitude': lon,
                'grid_position': {
                    'row': row,
                    'col': col
                },
                'path': output_path
            }
        except Exception as e:
            print(f"Image error --- {lat}, {lon}: {e}")

    file = os.path.join(output_dir, 'coordinate_info.json')
    with open(file, 'w') as f:
        json.dump(coordinate_info, f, indent=2)

def process_images(longitude=-122.4194, latitude=37.7749, grid_size_km=0.2, num_images=16):
    """
    Parameters:
        longitude: focal longitude coordinate.
        latitude: focal latitude coordinate.
        grid_size: invidivual image dimensions in km.
        images: # resultant images.
    """
    try:
        api = load_api_key()
        download_area(api, latitude, longitude, grid_size_km, num_images)
    except ValueError as e:
        print(f"API invalid: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
