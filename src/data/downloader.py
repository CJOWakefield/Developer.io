import os
import math
import json
import aiohttp
import asyncio
import multiprocessing
from PIL import Image
from io import BytesIO
from geopy.geocoders import Nominatim
from dotenv import load_dotenv
import yaml
from typing import Tuple, List, Dict, Optional, Any
import shutil
import time
import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from api.cloud_storage_client import CloudStorageClient

thread_limit = 16
base_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
save_directory = os.path.join(base_directory, 'data', 'downloaded')

with open(os.path.join(base_directory, 'configs', 'default_config.yaml'), 'r') as file:
    config = yaml.safe_load(file)

''' ----- Downloader file summary -----

> SatelliteDownloader - Class to leverage GEE API and return satellite image grid for user specified location.
    >> _load_api_key -> Load API key from .env file for GEE API, saved as GOOGLE_MAPS_API_KEY
    >> get_coordinates -> Leverage Nominatim to return lat/lon co-ordinates for user specified country, town (and postcode).
        >>> Inputs: country (str), city (str), postcode (Optional[str])
    >> get_zoom -> Return API specific zoom parameter based on user specified image demand and image_size by km.
        >>> Inputs: grid_size_km (float)
    >> calculate_grid -> Calculate relevant grid size per image for specified input parameters.
        >>> Inputs: center_lat (float), center_lon (float), grid_size_km (float), num_images (int)
    >> download_image -> Execute API call for specific co-ordinates and zoom.
        >>> Inputs: lat (float), lon (float), area_size (Optional[float]), zoom (Optional[int])
    >> _init_session -> Initialize aiohttp session for API calls.
        >>> Inputs: None
    >> _download_image_async -> Asynchronously download image from API or retrieve from cache.
        >>> Inputs: lat (float), lon (float), area_size (Optional[float]), zoom (Optional[int])
    >> _process_batch -> Process a batch of coordinates for downloading.
        >>> Inputs: batch_coords (List[Tuple]), zoom (int), grid_dim (int), metadata (Dict)
    >> push_to_cloud -> Upload pending images to cloud storage.
        >>> Inputs: None
    >> clear_local_cache -> Remove older images from local cache.
        >>> Inputs: keep_recent (int)
    >> process_location -> Combine relevant composition functions and execute relevant API call, saving results to data folder.
        >>> Inputs: country (str), city (str), postcode (Optional[str]), grid_size_km (float), num_images (int), output_dir (Optional[str])
    >> save_to_final_location -> Move cached files to final directory.
        >>> Inputs: final_dir (str)
    >> get_location -> Get address from latitude and longitude.
        >>> Inputs: latitude (float), longitude (float)
    >> _get_cache_key -> Generate cache key from coordinates and zoom.
        >>> Inputs: lat (float), lon (float), zoom (int)
    >> _get_cloud_path -> Generate cloud storage path from cache key.
        >>> Inputs: cache_key (str)
    >> _check_local_cache -> Check if image exists in local cache.
        >>> Inputs: lat (float), lon (float), zoom (int)
    >> _check_cloud_cache -> Check if image exists in cloud cache.
        >>> Inputs: lat (float), lon (float), zoom (int)
    >> _save_cache_index -> Save cache index to disk.
        >>> Inputs: None

    >>> Inputs: base_directory/save_directory hardcoded in file. country, city, postcode, grid_size_km, num_images in process_location for user specific output.

'''

class SatelliteDownloader:
    def __init__(self, batch_size=4, thread_limit=16, save_directory=None, testing=False):
        load_dotenv()
        self.api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        if not self.api_key: raise ValueError("API key invalid.")
        
        self.cloud_client = None
        if not testing:
            try:
                self.cloud_client = CloudStorageClient()
            except Exception as e:
                print(f"Warning: Could not initialize cloud storage: {e}")
        
        if save_directory is None:
            save_directory = config['data']['output_dir']
            
        if not os.path.isabs(save_directory):
            save_directory = os.path.join(base_directory, save_directory)
        
        self.base_dir = save_directory
        self.cache_dir = os.path.join(self.base_dir, 'cache')
        
        self.local_cache_dir = os.path.join(self.base_dir, 'local_image_cache')
        os.makedirs(self.local_cache_dir, exist_ok=True)
        
        self.cache_index_path = os.path.join(self.local_cache_dir, 'cache_index.json')
        if os.path.exists(self.cache_index_path):
            with open(self.cache_index_path, 'r') as f:
                self.cache_index = json.load(f)
        else:
            self.cache_index = {}
            self._save_cache_index()
            
        os.makedirs(self.cache_dir, exist_ok=True)
        self.geolocator = Nominatim(user_agent="birds_eye_view_downloader")
        self.batch_size = batch_size
        self.session = None
        self.testing = testing
        
        self.pending_uploads = set()
        
        available_cores = multiprocessing.cpu_count()
        self.num_threads = min(thread_limit, available_cores * 2)
            
    def _save_cache_index(self) -> None:
        with open(self.cache_index_path, 'w') as f:
            json.dump(self.cache_index, f, indent=2)
            
    def _get_cache_key(self, lat: float, lon: float, zoom: int) -> str:
        return f"{lat:.6f}_{lon:.6f}_{zoom}"
    
    def _get_cloud_path(self, cache_key: str) -> str:
        return f"downloaded/{cache_key.replace('.', '_')}.jpg"
    
    def _check_local_cache(self, lat: float, lon: float, zoom: int) -> Optional[str]:
        cache_key = self._get_cache_key(lat, lon, zoom)
        if cache_key in self.cache_index and self.cache_index[cache_key].get('local_path'):
            local_path = self.cache_index[cache_key]['local_path']
            if os.path.exists(local_path):
                return local_path
        return None
    
    async def _check_cloud_cache(self, lat: float, lon: float, zoom: int) -> Optional[bytes]:
        if self.testing or not self.cloud_client:
            return None
            
        cache_key = self._get_cache_key(lat, lon, zoom)
        cloud_path = self._get_cloud_path(cache_key)
        
        try:
            cloud_files = self.cloud_client.list_files(prefix=cloud_path)
            if not cloud_files:
                return None
                
            local_filename = os.path.join(self.local_cache_dir, f"{cache_key.replace('.', '_')}.jpg")
            
            def download_file():
                self.cloud_client.download_file(cloud_path, local_filename)
                return True
                
            success = await asyncio.to_thread(download_file)
            
            if success:
                if cache_key not in self.cache_index:
                    self.cache_index[cache_key] = {}
                self.cache_index[cache_key].update({
                    'local_path': local_filename,
                    'cloud_path': cloud_path,
                    'last_accessed': time.time(),
                    'in_cloud': True
                })
                self._save_cache_index()
                
                with open(local_filename, 'rb') as f:
                    return f.read()
        except Exception as e:
            print(f"Cloud cache check error: {e}")
        return None

    async def _init_session(self) -> None:
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def _download_image_async(self, lat: float, 
                                  lon: float, 
                                  area_size: Optional[float] = None,
                                  zoom: Optional[int] = None) -> Optional[bytes]:
        
        if area_size:
            zoom = self.get_zoom(area_size)
            
        cache_key = self._get_cache_key(lat, lon, zoom)
            
        local_path = self._check_local_cache(lat, lon, zoom)
        if local_path:
            if cache_key in self.cache_index:
                self.cache_index[cache_key]['last_accessed'] = time.time()
                self._save_cache_index()
            
            with open(local_path, 'rb') as f:
                return f.read()
        
        cloud_data = await self._check_cloud_cache(lat, lon, zoom)
        if cloud_data:
            return cloud_data
        
        params = {
            'center': f'{lat},{lon}',
            'zoom': zoom,
            'size': '640x640',
            'maptype': 'satellite',
            'key': self.api_key,
            'scale': 2
        }

        try:
            async with self.session.get('https://maps.googleapis.com/maps/api/staticmap', params=params) as response:
                if response.status == 200:
                    image_data = await response.read()
                    
                    local_filename = f"{cache_key.replace('.', '_')}.jpg"
                    local_path = os.path.join(self.local_cache_dir, local_filename)
                    
                    with open(local_path, 'wb') as f:
                        f.write(image_data)
                    
                    cloud_path = self._get_cloud_path(cache_key)
                    if cache_key not in self.cache_index:
                        self.cache_index[cache_key] = {}
                    
                    self.cache_index[cache_key].update({
                        'local_path': local_path,
                        'cloud_path': cloud_path,
                        'last_accessed': time.time(),
                        'in_cloud': False
                    })
                    self._save_cache_index()
                    
                    self.pending_uploads.add(cache_key)
                    
                    return image_data
        except Exception as e:
            print(f"Download error: {e}")
        return None

    async def _process_batch(self, batch_coords: List[Tuple[int, Tuple[float, float]]], 
                           zoom: int, 
                           grid_dim: int, 
                           metadata: Dict[str, Any]) -> List[Image.Image]:
        tasks = []
        images = []
        for idx, (lat, lon) in batch_coords:
            tasks.append(self._download_image_async(lat, lon, zoom=zoom))

        batch_results = await asyncio.gather(*tasks)

        for (idx, (lat, lon)), img_data in zip(batch_coords, batch_results):
            if img_data:
                image_id = 100001 + idx
                filename = f'{image_id}_sat.jpg'
                cache_path = os.path.join(self.cache_dir, filename)

                img = Image.open(BytesIO(img_data)).convert('RGB')
                img.save(cache_path, 'JPEG', quality=95)
                images.append(img)

                cache_key = self._get_cache_key(lat, lon, zoom)
                
                metadata['images'][filename] = {
                    'id': image_id,
                    'position': {'lat': lat, 'lon': lon},
                    'grid': {'row': idx//grid_dim, 'col': idx%grid_dim},
                    'path': cache_path,
                    'cache_key': cache_key
                }

        return images

    async def push_to_cloud(self) -> int:
        if self.testing or not self.cloud_client:
            return 0
            
        upload_count = 0
        
        for cache_key in list(self.pending_uploads):
            if cache_key in self.cache_index and 'local_path' in self.cache_index[cache_key]:
                local_path = self.cache_index[cache_key]['local_path']
                cloud_path = self.cache_index[cache_key]['cloud_path']
                
                if os.path.exists(local_path):
                    try:
                        def upload_file():
                            self.cloud_client.upload_file(local_path, cloud_path)
                            return True
                            
                        success = await asyncio.to_thread(upload_file)
                        
                        if success:
                            self.cache_index[cache_key]['in_cloud'] = True
                            self.pending_uploads.remove(cache_key)
                            upload_count += 1
                    except Exception as e:
                        print(f"Cloud upload error for {cache_key}: {e}")
        
        self._save_cache_index()
        return upload_count
        
    async def clear_local_cache(self, keep_recent: int = 50) -> int:
        await self.push_to_cloud()
        
        sorted_entries = sorted(
            [(k, v) for k, v in self.cache_index.items() if v.get('local_path') and os.path.exists(v['local_path'])],
            key=lambda x: x[1].get('last_accessed', 0),
            reverse=True
        )
        
        to_keep = sorted_entries[:keep_recent]
        to_remove = sorted_entries[keep_recent:]
        
        removed_count = 0
        for cache_key, entry in to_remove:
            if entry.get('local_path') and os.path.exists(entry['local_path']):
                if entry.get('in_cloud', False):
                    os.remove(entry['local_path'])
                    self.cache_index[cache_key]['local_path'] = None
                    removed_count += 1
        
        self._save_cache_index()
        return removed_count

    async def process_location(self, 
                             country: str, 
                             city: str, 
                             postcode: Optional[str] = None, 
                             grid_size_km: float = 0.5, 
                             num_images: int = 16,
                             output_dir: Optional[str] = None) -> Tuple[Optional[str], List[Image.Image]]:
        try:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            os.makedirs(self.cache_dir)

            lat, lon, address = self.get_coordinates(country, city, postcode)
            zoom = self.get_zoom(grid_size_km)
            coordinates, grid_dim = self.calculate_grid(lat, lon, grid_size_km, num_images)
            
            if output_dir is None:
                folder_name = f'{city.lower().replace(" ", "_")}_{lat:.3f}_{lon:.3f}_{grid_size_km*grid_dim:.1f}km_{grid_size_km*1000:.0f}m'
                final_dir = os.path.join(self.base_dir, folder_name)
            else:
                final_dir = output_dir

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

            await self._init_session()
            coords_with_idx = list(enumerate(coordinates))
            batches = [coords_with_idx[i:i + self.batch_size] for i in range(0, len(coords_with_idx), self.batch_size)]

            all_images = []
            for batch in batches:
                images = await self._process_batch(batch, zoom, grid_dim, metadata)
                all_images.extend(images)

            with open(os.path.join(self.cache_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
                
            if not self.testing and self.cloud_client:
                await self.push_to_cloud()
                await self.clear_local_cache()

            return final_dir, all_images

        except Exception as e:
            print(f"Process error: {e}")
            return None, []
        finally:
            if self.session:
                await self.session.close()
                self.session = None

    def save_to_final_location(self, final_dir: str) -> None:
        if not os.path.exists(final_dir):
            os.makedirs(final_dir)
        for file in os.listdir(self.cache_dir):
            shutil.move(os.path.join(self.cache_dir, file), os.path.join(final_dir, file))
        
        if not self.testing and self.cloud_client:
            try:
                cloud_prefix = os.path.basename(final_dir)
                self.cloud_client.upload_directory(final_dir, cloud_prefix)
                print(f"Synced {final_dir} to cloud storage under prefix {cloud_prefix}")
            except Exception as e:
                print(f"Error syncing to cloud: {e}")
                
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        os.makedirs(self.cache_dir)
        
    def get_location(self, latitude: float, longitude: float) -> str:
        location = self.geolocator.reverse((latitude, longitude))
        return location.address

    def get_coordinates(self, country: str, 
                        city: str, 
                        postcode: Optional[str] = None) -> Tuple[float, float, str]:
        try:
            query = f"{postcode}, {city}, {country}" if postcode else f"{city}, {country}"
            location = self.geolocator.geocode(query)
            if not location: raise ValueError(f"Could not find location: {query}")
            return location.latitude, location.longitude, location.address
        except Exception as e: raise Exception(f"Location lookup failed: {e}")

    def get_zoom(self, grid_size_km: float) -> int:
        grid_m = grid_size_km * 1000
        if grid_m > 3000:
            raise ValueError(f'Area size too large. Must be <= 3km. Got {grid_size_km:.1f}km')
        grid_specs = {100: 20, 200: 19, 400: 18, 800: 17, 1500: 16, 3000: 15}
        return list(grid_specs.values())[min(range(len(grid_specs)), key=lambda i: abs(grid_m - list(grid_specs.keys())[i]))]

    def calculate_grid(self, center_lat: float, center_lon: float, grid_size_km: float, num_images: int) -> Tuple[List[Tuple[float, float]], int]:
        grid_dim = math.ceil(math.sqrt(num_images))
        lat_adj = (grid_size_km * grid_dim / 2) / 111.32
        lon_adj = lat_adj / math.cos(math.radians(center_lat))
        
        start_lat = center_lat + lat_adj
        start_lon = center_lon - lon_adj
        
        lat_step = 2 * lat_adj / grid_dim
        lon_step = 2 * lon_adj / grid_dim
        
        coordinates = []
        for i in range(grid_dim):
            for j in range(grid_dim):
                if len(coordinates) < num_images:
                    lat = start_lat - (i + 0.5) * lat_step
                    lon = start_lon + (j + 0.5) * lon_step
                    coordinates.append((lat, lon))
        
        return coordinates, grid_dim
if __name__ == '__main__':
    downloader = SatelliteDownloader(batch_size=config['training']['batch_size'], testing=True)
    lat, lon, address = downloader.get_coordinates(country="Germany", city="Augsburg")
    
    async def main():
        await downloader._init_session()
        result = await downloader._download_image_async(lat, lon, area_size=0.5)
        
        if result:
            img = Image.open(BytesIO(result))
            img = img.convert('RGB')
            img.save('test_image.jpg')
            print("Image downloaded and saved as 'test_image.jpg'")
        else:
            print("Failed to download image")
            
        await downloader.session.close()

    asyncio.run(main())
