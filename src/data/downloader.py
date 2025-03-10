import os
import math
import json
import aiohttp
import asyncio
import multiprocessing
import logging
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
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

logger = logging.getLogger('satellite_downloader')

project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from api.cloud_storage_client import CloudStorageClient

thread_limit = 16
base_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
save_directory = os.path.join(base_directory, 'data', 'downloaded')

with open(os.path.join(base_directory, 'configs', 'default_config.yaml'), 'r') as file:
    config = yaml.safe_load(file)

class SatelliteDownloader:
    """
    Class to download satellite imagery from Google Maps Static API.
    
    This class handles coordinate calculation, image downloading, caching,
    and cloud storage integration for satellite imagery.
    """
    
    def __init__(self, batch_size=4, thread_limit=16, save_directory=None, testing=False):
        """
        Initialize the SatelliteDownloader.
        
        Args:
            batch_size: Number of images to download in parallel
            thread_limit: Maximum number of threads to use
            save_directory: Directory to save downloaded images
            testing: If True, disable cloud storage and some features for testing
        """
        load_dotenv()
        self.api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        if not self.api_key:
            raise ValueError("API key not found in environment variables.")
        
        self.cloud_client = None
        if not testing:
            try:
                self.cloud_client = CloudStorageClient()
            except Exception as e:
                logger.warning(f"Could not initialize cloud storage: {e}")
        
        if save_directory is None:
            save_directory = config['data']['output_dir']
            
        if not os.path.isabs(save_directory):
            save_directory = os.path.join(base_directory, save_directory)
        
        self.base_dir = save_directory
        self.save_directory = save_directory
        self.cache_dir = os.path.join(self.base_dir, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.local_cache_dir = os.path.join(self.base_dir, 'local_image_cache')
        os.makedirs(self.local_cache_dir, exist_ok=True)
        
        self.cache_index_path = os.path.join(self.local_cache_dir, 'cache_index.json')
        if os.path.exists(self.cache_index_path):
            with open(self.cache_index_path, 'r') as f:
                self.cache_index = json.load(f)
        else:
            self.cache_index = {}
            self._save_cache_index()
            
        self.geolocator = Nominatim(user_agent="birds_eye_view_downloader")
        self.batch_size = batch_size
        self.session = None
        self.testing = testing
        
        self.pending_uploads = set()
        
        available_cores = multiprocessing.cpu_count()
        self.num_threads = min(thread_limit, available_cores * 2)
        
        self.metadata_db_path = os.path.join(base_directory, 'data', 'metadata.csv')
        self._init_metadata_db()

    # Direct download functions
    
    async def single_download(self, 
                              lat: float, 
                              lon: float, 
                              area_size: Optional[float] = 0.5, 
                              save: bool = False) -> Optional[bytes]:
        """
        Download a single satellite image for specified coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            area_size: Size of area in km
            save: Whether to save the image to disk
            
        Returns:
            Image data as bytes if successful, None otherwise
        """
        try:
            await self._init_session()
            zoom = self.get_zoom(area_size)
            image_data = await self._download_image_async(lat, lon, area_size=area_size, zoom=zoom)
            
            if image_data and save:
                cloud_path = self._get_cloud_path(lat, lon, zoom)
                local_path = os.path.join(self.save_directory, os.path.basename(cloud_path))
                
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                with open(local_path, 'wb') as f:
                    f.write(image_data)
                
                if self.cloud_client:
                    self.cloud_client.upload_file(local_path, cloud_path)
            
            return image_data
        except Exception as e:
            logger.error(f"Single download error: {e}")
            return None
        finally:
            if self.session:
                await self.session.close()
                self.session = None

    async def grid_download(self, 
                           lat: float, 
                           lon: float, 
                           grid_size_km: float = 0.5, 
                           num_images: int = 4,
                           save: bool = False) -> List[bytes]:
        """
        Download a grid of satellite images around specified coordinates.
        
        Args:
            lat: Center latitude
            lon: Center longitude
            grid_size_km: Size of each grid cell in km
            num_images: Number of images to download
            save: Whether to save images to disk
            
        Returns:
            List of image data as bytes
        """
        try:
            await self._init_session()
            zoom = self.get_zoom(grid_size_km)
            coordinates, grid_dim = self.calculate_grid(lat, lon, grid_size_km, num_images)
            
            metadata = {
                'metadata': {
                    'center': {'lat': lat, 'lon': lon},
                    'grid_size_km': grid_size_km,
                    'num_images': num_images,
                    'grid_dim': grid_dim,
                    'zoom': zoom
                },
                'images': {}
            }
            
            coords_with_idx = list(enumerate(coordinates))
            batches = [coords_with_idx[i:i + self.batch_size] 
                      for i in range(0, len(coords_with_idx), self.batch_size)]
            
            all_image_bytes = []
            for batch in batches:
                image_bytes = await self._process_batch(batch, zoom, grid_dim, metadata, save=save)
                all_image_bytes.extend(image_bytes)
            
            with open(os.path.join(self.cache_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
                
            if save and self.cloud_client:
                await self.push_to_cloud()
                await self.clear_local_cache()
            
            return all_image_bytes
        except Exception as e:
            logger.error(f"Grid download error: {e}")
            return []
        finally:
            if self.session:
                await self.session.close()
                self.session = None

    async def location_download(self, 
                             country: str, 
                             city: str, 
                             postcode: Optional[str] = None, 
                             grid_size_km: float = 0.5, 
                             num_images: int = 16,
                             output_dir: Optional[str] = None) -> Tuple[Optional[str], List[bytes]]:
        """
        Process a location by downloading satellite imagery for a grid around it.
        
        Args:
            country: Country name
            city: City name
            postcode: Optional postal code
            grid_size_km: Size of each grid cell in km
            num_images: Number of images to download
            output_dir: Optional output directory
            
        Returns:
            Tuple of (final directory path, list of image data as bytes)
        """
        try:
            existing_search = self.find_existing_search(
                country, city, postcode, grid_size_km, num_images
            )
            
            if existing_search is not None and len(existing_search['file_paths']) > 0:
                logger.info(f"Found existing search: {existing_search['id']}")
                
                file_paths = existing_search['file_paths']
                all_files_exist = True
                
                if self.cloud_client:
                    all_files_exist = all(
                        path in self.cloud_client.list_files() 
                        for path in file_paths
                    )
                
                if all_files_exist:
                    image_bytes = []
                    for cloud_path in file_paths:
                        local_path = os.path.join(self.save_directory, os.path.basename(cloud_path))
                        
                        if self.cloud_client:
                            self.cloud_client.download_file(cloud_path, local_path)
                        
                        if os.path.exists(local_path):
                            with open(local_path, 'rb') as f:
                                image_bytes.append(f.read())
                    
                    logger.info(f"Retrieved {len(image_bytes)} images from existing search")
                    return existing_search['id'], image_bytes
            
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            os.makedirs(self.cache_dir, exist_ok=True)

            lat, lon, address = self.get_coordinates(country, city, postcode)
            zoom = self.get_zoom(grid_size_km)
            
            if output_dir is None:
                folder_name = f'{city.lower().replace(" ", "_")}_{lat:.3f}_{lon:.3f}_{grid_size_km*math.ceil(math.sqrt(num_images)):.1f}km_{grid_size_km*1000:.0f}m'
                final_dir = os.path.join(self.base_dir, folder_name)
            else:
                final_dir = output_dir
                
            image_bytes = await self.grid_download(lat, lon, grid_size_km, num_images, save=True)
            
            if image_bytes:
                coordinates, grid_dim = self.calculate_grid(lat, lon, grid_size_km, num_images)
                file_paths = [self._get_cloud_path(coord[0], coord[1], zoom) for coord in coordinates[:len(image_bytes)]]
                
                search_id = self.add_search_to_db(
                    country, city, lat, lon, postcode, grid_size_km, num_images, zoom, file_paths
                )
                
                self.save_to_final_location(final_dir)
                return final_dir, image_bytes
            
            return None, []
        except Exception as e:
            logger.error(f"Location download error: {e}")
            return None, []
        finally:
            if self.session:
                await self.session.close()
                self.session = None

    def save_to_final_location(self, final_dir: str) -> None:
        """
        Move cached files to final directory.
        
        Args:
            final_dir: Final directory path
        """
        if not os.path.exists(final_dir): os.makedirs(final_dir)
        for file in os.listdir(self.cache_dir):
            shutil.move(os.path.join(self.cache_dir, file), os.path.join(final_dir, file))
        
        if not self.testing and self.cloud_client:
            try:
                cloud_prefix = os.path.basename(final_dir)
                self.cloud_client.upload_directory(final_dir, cloud_prefix)
                logger.info(f"Synced {final_dir} to cloud storage under prefix {cloud_prefix}")
            except Exception as e:
                logger.error(f"Error syncing to cloud: {e}")
                
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    # Download helper functions

    # Convert lat/lon co-ordinates to address.
    def get_location(self, latitude: float, longitude: float) -> str:
        location = self.geolocator.reverse((latitude, longitude))
        return location.address

    # Convert address to lat/lon co-ordinates for API call.
    def get_coordinates(self, country: str, 
                        city: str, 
                        postcode: Optional[str] = None) -> Tuple[float, float, str]:
        
        query = f"{city}, {country}"
        if postcode: query = f"{postcode}, {query}"
        try:
            location = self.geolocator.geocode(query)
            if location is None:
                raise ValueError(f"Location not found: {query}")
            return location.latitude, location.longitude, location.address
        except Exception as e:
            logger.error(f"Location lookup failed: {e}")
            raise ValueError(f"Location lookup failed: {e}")

    # Convert specified grid size to appropriate zoom level for API call.
    def get_zoom(self, grid_size_km: float) -> int:
        grid_m = grid_size_km * 1000
        if grid_m > 3000:
            raise ValueError(f'Area size too large. Must be <= 3km. Got {grid_size_km:.1f}km')
        grid_specs = {100: 20, 200: 19, 400: 18, 800: 17, 1500: 16, 3000: 15}
        return list(grid_specs.values())[min(range(len(grid_specs)), key=lambda i: abs(grid_m - list(grid_specs.keys())[i]))]

    # Calculate grid size and dimensions for area grid search/download.
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
    
    async def _init_session(self) -> None:
        """Initialize aiohttp session for API calls."""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def _download_image_async(self, lat: float, 
                                  lon: float, 
                                  area_size: Optional[float] = None,
                                  zoom: Optional[int] = None) -> Optional[bytes]:
        """
        Asynchronously download image from API or retrieve from cache.
        
        Args:
            lat: Latitude
            lon: Longitude
            area_size: Size of area in km (used to calculate zoom if zoom not provided)
            zoom: Zoom level (if not provided, calculated from area_size)
            
        Returns:
            Image data as bytes if successful, None otherwise
        """
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
                    
                    cloud_path = self._get_cloud_path(lat, lon, zoom)
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
                else:
                    logger.error(f"API request failed with status {response.status}: {await response.text()}")
        except Exception as e:
            logger.error(f"Download error: {e}")
        return None

    async def _process_batch(self, batch_coords: List[Tuple[int, Tuple[float, float]]], 
                           zoom: int, 
                           grid_dim: int, 
                           metadata: Dict[str, Any],
                           save: bool = False) -> List[bytes]:
        """
        Process a batch of coordinates for downloading.
        
        Args:
            batch_coords: List of (index, (lat, lon)) tuples
            zoom: Zoom level
            grid_dim: Grid dimension
            metadata: Metadata dictionary to update
            save: Whether to save images to disk
            
        Returns:
            List of image data as bytes
        """
        tasks = []
        image_bytes_list = []
        
        for idx, (lat, lon) in batch_coords:
            tasks.append(self._download_image_async(lat, lon, zoom=zoom))

        batch_results = await asyncio.gather(*tasks)

        for (idx, (lat, lon)), img_data in zip(batch_coords, batch_results):
            if img_data:
                image_id = 100001 + idx
                filename = f'{image_id}_sat.jpg'
                cache_path = os.path.join(self.cache_dir, filename)
                
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)

                img = Image.open(BytesIO(img_data)).convert('RGB')
                img.save(cache_path, 'JPEG', quality=95)
                image_bytes_list.append(img_data)

                cache_key = self._get_cache_key(lat, lon, zoom)
                
                metadata['images'][filename] = {
                    'id': image_id,
                    'position': {'lat': lat, 'lon': lon},
                    'grid': {'row': idx//grid_dim, 'col': idx%grid_dim},
                    'path': cache_path,
                    'cache_key': cache_key
                }
                
                if save and self.cloud_client:
                    cloud_path = self._get_cloud_path(lat, lon, zoom)
                    self.cloud_client.upload_file(cache_path, cloud_path)

        return image_bytes_list

    # Database functions
    
    def _init_metadata_db(self) -> None:
        """Initialize the metadata database if it doesn't exist."""
        if not os.path.exists(self.metadata_db_path):
            os.makedirs(os.path.dirname(self.metadata_db_path), exist_ok=True)
            df = pd.DataFrame(columns=[
                'id', 'country', 'city', 'postcode', 'latitude', 'longitude',
                'grid_size_km', 'num_images', 'zoom', 'timestamp', 'file_paths'
            ])
            df.to_csv(self.metadata_db_path, index=False)
    
    def _load_metadata_db(self) -> pd.DataFrame:
        """Load the metadata database from disk."""
        try:
            return pd.read_csv(self.metadata_db_path, converters={'file_paths': eval})
        except (FileNotFoundError, pd.errors.EmptyDataError):
            self._init_metadata_db()
            return pd.DataFrame(columns=[
                'id', 'country', 'city', 'postcode', 'latitude', 'longitude',
                'grid_size_km', 'num_images', 'zoom', 'timestamp', 'file_paths'
            ])
    
    def _save_metadata_db(self, df: pd.DataFrame) -> None:
        """Save the metadata database to disk."""
        df.to_csv(self.metadata_db_path, index=False)
    
    def _generate_search_id(self) -> str:
        """Generate a unique search ID based on timestamp."""
        timestamp = int(time.time())
        return f"search_{timestamp}"
    
    def find_existing_search(self, 
                             country: str, 
                             city: str, 
                             postcode: Optional[str] = None,
                             grid_size_km: float = 0.5,
                             num_images: int = 16) -> Optional[pd.Series]:
        """
        Find an existing search in the metadata database.
        
        Args:
            country: Country name
            city: City name
            postcode: Optional postal code
            grid_size_km: Size of grid cell in km
            num_images: Number of images
            
        Returns:
            DataFrame row if found, None otherwise
        """
        df = self._load_metadata_db()
        if df.empty: 
            return None
            
        if postcode is None: 
            postcode_match = df['postcode'].isna()
        else: 
            postcode_match = df['postcode'] == postcode
        
        matches = df[(df['country'].str.lower() == country.lower()) &
                     (df['city'].str.lower() == city.lower()) &
                     postcode_match &
                     (abs(df['grid_size_km'] - grid_size_km) < 0.001) &
                     (df['num_images'] == num_images)]
        
        if not matches.empty:
            return matches.sort_values('timestamp', ascending=False).iloc[0]
        
        return None
    
    def add_search_to_db(self, 
                         country: str, 
                         city: str, 
                         lat: float,
                         lon: float,
                         postcode: Optional[str] = None,
                         grid_size_km: float = 0.5,
                         num_images: int = 16,
                         zoom: int = 18,
                         file_paths: List[str] = None) -> str:
        """
        Add a new search to the metadata database.
        
        Args:
            country: Country name
            city: City name
            lat: Latitude
            lon: Longitude
            postcode: Optional postal code
            grid_size_km: Size of grid cell in km
            num_images: Number of images
            zoom: Zoom level
            file_paths: List of file paths
            
        Returns:
            Search ID
        """
        df = self._load_metadata_db()
        search_id = self._generate_search_id()
        
        new_row = pd.DataFrame([{
            'id': search_id,
            'country': country,
            'city': city,
            'postcode': postcode,
            'latitude': lat,
            'longitude': lon,
            'grid_size_km': grid_size_km,
            'num_images': num_images,
            'zoom': zoom,
            'timestamp': time.time(),
            'file_paths': file_paths or []
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_metadata_db(df)
        
        return search_id

    # Caching functions
    
    def _save_cache_index(self) -> None:
        """Save the cache index to disk."""
        with open(self.cache_index_path, 'w') as f:
            json.dump(self.cache_index, f, indent=2)

    def _get_cache_key(self, lat: float, lon: float, zoom: int) -> str:
        """
        Generate a unique cache key for the given coordinates and zoom level.
        
        Args:
            lat: Latitude
            lon: Longitude
            zoom: Zoom level
            
        Returns:
            String cache key
        """
        return f"{lat:.6f}_{lon:.6f}_{zoom}"
    
    def _get_cloud_path(self, lat: float, lon: float, zoom: int) -> str:
        """
        Generate cloud storage path from coordinates and zoom.
        
        Args:
            lat: Latitude
            lon: Longitude
            zoom: Zoom level
            
        Returns:
            Cloud storage path
        """
        return f"satellite_images/{lat:.6f}_{lon:.6f}_z{zoom}.jpg"
    
    def _check_local_cache(self, lat: float, lon: float, zoom: int) -> Optional[str]:
        """
        Check if image exists in local cache.
        
        Args:
            lat: Latitude
            lon: Longitude
            zoom: Zoom level
            
        Returns:
            Path to local file if exists, None otherwise
        """
        cache_key = self._get_cache_key(lat, lon, zoom)
        if cache_key in self.cache_index and self.cache_index[cache_key].get('local_path'):
            local_path = self.cache_index[cache_key]['local_path']
            if os.path.exists(local_path):
                return local_path
        return None
    
    async def _check_cloud_cache(self, lat: float, lon: float, zoom: int) -> Optional[bytes]:
        """
        Check if image exists in cloud cache and download if found.
        
        Args:
            lat: Latitude
            lon: Longitude
            zoom: Zoom level
            
        Returns:
            Image data as bytes if found in cloud, None otherwise
        """
        if self.testing or not self.cloud_client:
            return None
            
        cloud_path = self._get_cloud_path(lat, lon, zoom)
        
        try:
            cloud_files = self.cloud_client.list_files(prefix=cloud_path)
            if not cloud_files:
                return None
                
            cache_key = self._get_cache_key(lat, lon, zoom)
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
            logger.error(f"Cloud cache check error: {e}")
        return None

    async def push_to_cloud(self) -> int:
        """
        Upload pending images to cloud storage.
        
        Returns:
            Number of images uploaded
        """
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
                        logger.error(f"Cloud upload error for {cache_key}: {e}")
        
        self._save_cache_index()
        return upload_count
        
    async def clear_local_cache(self, keep_recent: int = 50) -> int:
        """
        Remove older images from local cache.
        
        Args:
            keep_recent: Number of recent images to keep
            
        Returns:
            Number of images removed
        """
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
