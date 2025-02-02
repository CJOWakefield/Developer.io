import os
import math
import json
import requests
from PIL import Image
from io import BytesIO
from geopy.geocoders import Nominatim
from dotenv import load_dotenv
import urllib
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

''' ----- Downloader file summary -----

> SatelliteDownloader - Class to leverage GEE API and return satellite image grid for user specified location.
    >> _load_api_key -> Load API key from .env file for GEE API, saved as GOOGLE_MAPS_API_KEY
    >> get_coordinates -> Leverage Nominatim to return lat/lon co-ordinates for user specified country, town (and postcode).
    >> get_zoom -> Return API specific zoom parameter based on user specified image demand and image_size by km.
    >> calculate_grid -> Calculate relevant grid size per image for specified input parameters.
    >> download_image -> Execute API call for specific co-ordinates and zoom.
    >> process_location -> Combine relevant composition functions and execute relevant API call, saving results to ..data/raw folder.

    >>> Inputs: base_directory/save_directory hardcoded in file. country, city, postcode, grid_size_km, num_images in process_location for user specific output.

'''
thread_limit = 16
base_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
save_directory = os.path.join(base_directory, 'data', 'api_images')

class SatelliteDownloader:
    def __init__(self):
        global thread_limit
        load_dotenv()
        self.api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        if not self.api_key: raise ValueError("API key invalid.")
        self.base_dir = save_directory
        self.geolocator = Nominatim(user_agent="birds_eye_view_downloader")

        # Threading maximised at 16 or CPU max processing
        available_cores = multiprocessing.cpu_count()
        self.num_threads = max(thread_limit, available_cores)

    def get_coordinates(self, country, city, postcode=None):
        try:
            query = f"{postcode}, {city}, {country}" if postcode else f"{city}, {country}"
            location = self.geolocator.geocode(query)
            if not location: raise ValueError(f"Could not find location: {query}")
            return location.latitude, location.longitude, location.address
        except Exception as e: raise Exception(f"Location lookup failed: {e}")

    def get_zoom(self, grid_size_km):
        grid_m = grid_size_km * 1000
        assert grid_m <= 3000
        grid_specs = {100: 20, 200: 19, 400: 18, 800: 17, 1500: 16, 3000: 15}
        return list(grid_specs.values())[min(range(len(grid_specs)), key=lambda i: abs(grid_m - list(grid_specs.keys())[i]))]

    def calculate_grid(self, center_lat, center_lon, grid_size_km, num_images):
        grid_dim = math.ceil(math.sqrt(num_images))
        lat_adj = (grid_size_km * grid_dim / 2) / 111.32
        lon_adj = lat_adj / math.cos(math.radians(center_lat))

        sub_lat = (2 * lat_adj) / grid_dim
        sub_lon = (2 * lon_adj) / grid_dim
        start_lat = center_lat + lat_adj - (sub_lat / 2)
        start_lon = center_lon - lon_adj + (sub_lon / 2)

        return [(start_lat - (i * sub_lat), start_lon + (j * sub_lon))
                for i in range(grid_dim)
                for j in range(grid_dim)][:num_images], grid_dim

    def download_image(self, lat, lon, zoom, output_path):
        params = {
            'center': f'{lat},{lon}',
            'zoom': zoom,
            'size': '640x640',
            'maptype': 'satellite',
            'key': self.api_key,
            'scale': 2
        }
        response = requests.get(f'https://maps.googleapis.com/maps/api/staticmap?{urllib.parse.urlencode(params)}')
        if response.status_code == 200:
            Image.open(BytesIO(response.content)).convert('RGB').save(output_path, 'JPEG', quality=95)
            return True
        return False

    def process_location(self, country, city, postcode=None, grid_size_km=0.5, num_images=16):
        try:
            # Ensure base directory exists
            os.makedirs(self.base_dir, exist_ok=True)

            lat, lon, address = self.get_coordinates(country, city, postcode)
            zoom = self.get_zoom(grid_size_km)
            coordinates, grid_dim = self.calculate_grid(lat, lon, grid_size_km, num_images)

            folder_name = f'{city.lower().replace(" ", "_")}_{lat:.3f}_{lon:.3f}_{grid_size_km*grid_dim:.1f}km_{grid_size_km*1000:.0f}m'
            output_dir = os.path.join(self.base_dir, folder_name)

            # Check for existing folder before creating
            if os.path.exists(output_dir):
                print('Images pre-existing.')
                return output_dir

            # Create output directory only if it doesn't exist
            os.makedirs(output_dir)

            metadata = {
                'metadata': {
                    'address': address,
                    'center': {'lat': lat, 'lon': lon},
                    'grid_size_km': grid_size_km,
                    'num_images': num_images,
                    'grid_dim': grid_dim,
                    'zoom': zoom
                },
                'images': {}
            }

            def download_worker(args):
                idx, (lat, lon) = args
                image_id = 100001 + idx
                filename = f'{image_id}_sat.jpg'
                output_path = os.path.join(output_dir, filename)

                if self.download_image(lat, lon, zoom, output_path):
                    return filename, {
                        'id': image_id,
                        'position': {'lat': lat, 'lon': lon},
                        'grid': {'row': idx//grid_dim, 'col': idx%grid_dim},
                        'path': output_path
                    }
                return None

            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                for result in executor.map(download_worker, enumerate(coordinates)):
                    if result:
                        filename, info = result
                        metadata['images'][filename] = info
                        print(f"Downloaded {len(metadata['images'])}/{num_images}")

            with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)

            return output_dir

        except Exception as e:
            print(f"Process error: {e}")
            return None

if __name__ == '__main__':
    downloader = SatelliteDownloader()
    downloader.process_location(country="Indonesia",
                                city="Jakarta",
                                grid_size_km=0.5,
                                num_images=16)
