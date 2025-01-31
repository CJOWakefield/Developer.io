import os
import math
import json
import requests
from PIL import Image
from io import BytesIO
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from dotenv import load_dotenv
import urllib

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

base_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
save_directory = os.path.join(base_directory, 'data', 'api_images')

class SatelliteDownloader:
    def __init__(self):
        self.api_key = self._load_api_key()
        self.base_dir = save_directory
        self.geolocator = Nominatim(user_agent="birds_eye_view_downloader")
        self.zoom_coverage = {20: "~100m", 19: "~200m", 18: "~400m", 17: "~800m",
                            16: "~1.5km", 15: "~3km", 14: "~6km"}
        os.makedirs(self.base_dir, exist_ok=True)

    def _load_api_key(self):
        load_dotenv()
        if not os.getenv('GOOGLE_MAPS_API_KEY'): raise ValueError("API key invalid.")
        return os.getenv('GOOGLE_MAPS_API_KEY')

    def get_coordinates(self, country, city, postcode=None):
        try:
            query = f"{city}, {country}"
            if postcode: query = f"{postcode}, {query}"
            location = self.geolocator.geocode(query)
            if not location: raise ValueError(f"Could not find location: {query}")

            return location.latitude, location.longitude, location.address

        except (GeocoderTimedOut, GeocoderUnavailable) as e: raise Exception(f"Geocoding service error: {e}")
        except Exception as e: raise Exception(f"Location lookup failed: {e}")

    def get_zoom(self, grid_size_km):
        grid_m = grid_size_km * 1000
        assert grid_m <= 3000
        grid_specs = {100: 20, 200: 19, 400: 18, 800: 17, 1500: 16, 3000: 15}
        zoom_diffs = [abs(grid_m-key) for key in grid_specs]
        return list(grid_specs.values())[zoom_diffs.index(min(zoom_diffs))]

    def calculate_grid(self, center_lat, center_lon, grid_size_km, num_images):
        grid_dim = math.ceil(math.sqrt(num_images))
        total_area = grid_size_km * grid_dim
        lat_adj = (total_area / 2) / 111.32
        lon_adj = (total_area / 2) / (111.32 * math.cos(math.radians(center_lat)))

        sub_grid_lat = (2 * lat_adj) / grid_dim
        sub_grid_lon = (2 * lon_adj) / grid_dim
        start_lat = center_lat + lat_adj - (sub_grid_lat / 2)
        start_lon = center_lon - lon_adj + (sub_grid_lon / 2)

        coordinates = []
        for i in range(grid_dim):
            for j in range(grid_dim):
                if len(coordinates) < num_images:
                    coordinates.append((start_lat - (i * sub_grid_lat),
                                     start_lon + (j * sub_grid_lon)))

        return coordinates, grid_dim

    def download_image(self, latitude, longitude, zoom, size=(640, 640)):
        params = {
            'center': f'{latitude},{longitude}',
            'zoom': zoom,
            'size': f'{size[0]}x{size[1]}',
            'maptype': 'satellite',
            'key': self.api_key,
            'scale': 2
        }

        response = requests.get(f'https://maps.googleapis.com/maps/api/staticmap?{urllib.parse.urlencode(params)}')
        if response.status_code != 200:
            raise Exception(f"Download failed: {response.status_code}")
        return Image.open(BytesIO(response.content))

    def process_location(self, country, city, postcode=None, grid_size_km=0.5, num_images=16):
        try:
            latitude, longitude, address = self.get_coordinates(country, city, postcode)
            zoom = self.get_zoom(grid_size_km)
            coordinates, grid_dim = self.calculate_grid(latitude, longitude, grid_size_km, num_images)
            total_area_km = grid_size_km * grid_dim

            safe_city = city.lower().replace(' ', '_')
            folder_name = f'{safe_city}_{latitude:.3f}_{longitude:.3f}_{total_area_km:.1f}km_{grid_size_km*1000:.0f}m'
            output_dir = os.path.join(self.base_dir, folder_name)

            if folder_name in os.listdir(save_directory):
                print('Images pre-existing.')
                return output_dir
            os.makedirs(output_dir, exist_ok=True)

            coordinate_info = {
                'metadata': {
                    'address': address,
                    'center_latitude': latitude,
                    'center_longitude': longitude,
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
                    print(f"Downloading image {i}/{num_images} - position: {lat:.5f}, {lon:.5f}")
                    img = self.download_image(lat, lon, zoom=zoom).convert('RGB')

                    image_id = 100000 + i
                    filename = f'{image_id}_sat.jpg'
                    output_path = os.path.join(output_dir, filename)
                    img.save(output_path, 'JPEG', quality=95)

                    coordinate_info['images'][filename] = {
                        'id': image_id,
                        'latitude': lat,
                        'longitude': lon,
                        'grid_position': {'row': (i-1)//grid_dim, 'col': (i-1)%grid_dim},
                        'path': output_path
                    }
                except Exception as e: print(f"Image error - {lat}, {lon}: {e}")

            with open(os.path.join(output_dir, 'coordinate_info.json'), 'w') as f:
                json.dump(coordinate_info, f, indent=2)
            return output_dir

        except Exception as e:
            print(f"Process error: {e}")
            return None

if __name__ == '__main__':
    downloader = SatelliteDownloader()
    downloader.process_location(country="United States of America",
                                city="Chicago",
                                # postcode="60601",
                                grid_size_km=0.5,
                                num_images=16)
