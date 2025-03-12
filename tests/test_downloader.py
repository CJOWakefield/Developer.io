import os
import sys
import time
import asyncio
import unittest
import logging
from io import BytesIO
from pathlib import Path
from PIL import Image
import shutil
import json

project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.data.downloader import SatelliteDownloader

# Configure test logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_satellite_downloader')

class TestSatelliteDownloader(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testing')
        os.makedirs(self.test_dir, exist_ok=True)
        
        self.test_cache_dir = os.path.join(self.test_dir, 'test_cache')
        os.makedirs(self.test_cache_dir, exist_ok=True)
        
        self.test_metadata_db_path = os.path.join(self.test_dir, 'test_metadata.csv')
        
        self.downloader = SatelliteDownloader(
            batch_size=2, 
            thread_limit=4,
            save_directory=self.test_dir,
            testing=True
        )
        
        # Override paths for testing
        self.downloader.cache_dir = os.path.join(self.test_dir, 'cache')
        self.downloader.local_cache_dir = os.path.join(self.test_dir, 'local_image_cache')
        self.downloader.cache_index_path = os.path.join(self.test_dir, 'cache_index.json')
        self.downloader.metadata_db_path = self.test_metadata_db_path
        
        os.makedirs(self.downloader.cache_dir, exist_ok=True)
        os.makedirs(self.downloader.local_cache_dir, exist_ok=True)
        
        # Initialize cache index
        self.downloader.cache_index = {}
        self.downloader._save_cache_index()
        
        # Initialize metadata DB
        self.downloader._init_metadata_db()
        
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_get_coordinates(self):
        lat, lon, address = self.downloader.get_coordinates("United States", "New York")
        self.assertIsInstance(lat, float)
        self.assertIsInstance(lon, float)
        self.assertIsInstance(address, str)
        logger.info(f"✓ get_coordinates: {address} ({lat}, {lon})")
    
    def test_get_zoom(self):
        zoom = self.downloader.get_zoom(0.5)
        self.assertIsInstance(zoom, int)
        self.assertTrue(15 <= zoom <= 20)
        logger.info(f"✓ get_zoom: {zoom} for 0.5km")
        
        for size in [0.1, 0.2, 0.5, 1.0, 2.0]:
            zoom = self.downloader.get_zoom(size)
            logger.info(f"  - Grid size {size}km -> zoom level {zoom}")
    
    def test_calculate_grid(self):
        coordinates, grid_dim = self.downloader.calculate_grid(40.7128, -74.0060, 0.5, 4)
        self.assertEqual(len(coordinates), 4)
        self.assertEqual(grid_dim, 2)
        logger.info(f"✓ calculate_grid: Generated {len(coordinates)} coordinates with grid dimension {grid_dim}")
        
        for num_images in [4, 9, 16]:
            coordinates, grid_dim = self.downloader.calculate_grid(40.7128, -74.0060, 0.5, num_images)
            self.assertEqual(len(coordinates), num_images)
            logger.info(f"  - {num_images} images -> grid dimension {grid_dim}")
    
    def test_get_location(self):
        address = self.downloader.get_location(40.7128, -74.0060)
        self.assertIsInstance(address, str)
        logger.info(f"✓ get_location: {address}")
    
    def test_cache_key_generation(self):
        cache_key = self.downloader._get_cache_key(40.7128, -74.0060, 18)
        self.assertIsInstance(cache_key, str)
        logger.info(f"✓ _get_cache_key: {cache_key}")
        
        cache_key2 = self.downloader._get_cache_key(40.71281, -74.00601, 18)
        self.assertNotEqual(cache_key, cache_key2)
        logger.info(f"  - Different coordinates produce different keys")
    
    def test_cloud_path_generation(self):
        cloud_path = self.downloader._get_cloud_path(40.7128, -74.0060, 18)
        self.assertIsInstance(cloud_path, str)
        self.assertTrue(cloud_path.startswith("satellite_images/"))
        logger.info(f"✓ _get_cloud_path: {cloud_path}")
    
    def test_metadata_db_functions(self):
        # Test initialization
        self.assertTrue(os.path.exists(self.downloader.metadata_db_path))
        
        # Test adding search
        search_id = self.downloader.add_search_to_db(
            "United States", "New York", 40.7128, -74.0060, 
            grid_size_km=0.5, num_images=4, zoom=18
        )
        self.assertIsInstance(search_id, str)
        
        # Test finding search
        result = self.downloader.find_existing_search(
            "United States", "New York", grid_size_km=0.5, num_images=4
        )
        self.assertIsNotNone(result)
        self.assertEqual(result['id'], search_id)
        
        logger.info(f"✓ metadata_db_functions: Successfully added and retrieved search {search_id}")
    
    def test_save_cache_index(self):
        test_key = "test_key"
        self.downloader.cache_index[test_key] = {"test": "data"}
        self.downloader._save_cache_index()
        
        self.assertTrue(os.path.exists(self.downloader.cache_index_path))
        
        with open(self.downloader.cache_index_path, 'r') as f:
            loaded_index = json.load(f)
        
        self.assertIn(test_key, loaded_index)
        logger.info(f"✓ _save_cache_index: Successfully saved and loaded cache index")
    
    async def async_test_single_download(self):
        try:
            await self.downloader._init_session()
            image_data = await self.downloader.single_download(40.7128, -74.0060, area_size=0.5, save=True)
            self.assertIsNotNone(image_data)
            
            img = Image.open(BytesIO(image_data))
            self.assertIsInstance(img, Image.Image)
            logger.info(f"✓ single_download: Downloaded image of size {img.size}")
            
            zoom = self.downloader.get_zoom(0.5)
            cache_key = self.downloader._get_cache_key(40.7128, -74.0060, zoom)
            self.assertIn(cache_key, self.downloader.cache_index)
            self.assertIn('local_path', self.downloader.cache_index[cache_key])
            logger.info(f"  - Image was properly cached with key {cache_key}")
            
            cached_image_data = await self.downloader.single_download(40.7128, -74.0060, area_size=0.5)
            self.assertIsNotNone(cached_image_data)
            logger.info(f"  - Successfully retrieved image from cache")
            
        finally:
            if self.downloader.session:
                await self.downloader.session.close()
                self.downloader.session = None
    
    async def async_test_batch_download(self):
        try:
            await self.downloader._init_session()
            lat, lon = 40.7128, -74.0060
            coordinates = [(0, (lat, lon)), (1, (lat + 0.001, lon)), (2, (lat, lon + 0.001)), (3, (lat + 0.001, lon + 0.001))]
            
            metadata = {'images': {}}
            images = await self.downloader._process_batch(coordinates[:2], 18, 2, metadata, save=True)
            
            self.assertEqual(len(images), 2)
            self.assertEqual(len(metadata['images']), 2)
            logger.info(f"✓ _process_batch: Processed {len(images)} images")
            
            for filename, img_metadata in metadata['images'].items():
                self.assertIn('id', img_metadata)
                self.assertIn('position', img_metadata)
                self.assertIn('grid', img_metadata)
                self.assertIn('path', img_metadata)
                self.assertIn('cache_key', img_metadata)
            logger.info(f"  - Metadata correctly generated for all images")
            
        finally:
            if self.downloader.session:
                await self.downloader.session.close()
                self.downloader.session = None
    
    async def async_test_grid_download(self):
        try:
            lat, lon = 40.7128, -74.0060
            images = await self.downloader.grid_download(lat, lon, grid_size_km=0.2, num_images=4, save=True)
            
            self.assertEqual(len(images), 4)
            logger.info(f"✓ grid_download: Downloaded {len(images)} images in grid")
            
            # Test with different grid sizes
            for num in [4, 9]:
                images = await self.downloader.grid_download(lat, lon, grid_size_km=0.2, num_images=num)
                self.assertEqual(len(images), num)
                logger.info(f"  - Successfully downloaded grid with {num} images")
                
        except Exception as e:
            self.fail(f"grid_download raised exception: {e}")
        finally:
            if self.downloader.session:
                await self.downloader.session.close()
                self.downloader.session = None
    
    async def async_test_location_download(self):
        test_location_dir = os.path.join(self.test_dir, 'test_location')
        if os.path.exists(test_location_dir):
            shutil.rmtree(test_location_dir)
        
        try:
            output_dir, images = await self.downloader.location_download(
                country="United States",
                city="New York",
                postcode=None,
                grid_size_km=0.2,
                num_images=4,
                output_dir=test_location_dir
            )
            
            self.assertIsNotNone(output_dir)
            self.assertEqual(len(images), 4)
            logger.info(f"✓ location_download: Processed location with {len(images)} images")
            
            self.assertTrue(os.path.exists(test_location_dir))
            self.assertTrue(os.path.exists(os.path.join(test_location_dir, 'metadata.json')))
            
            # Verify metadata structure
            with open(os.path.join(test_location_dir, 'metadata.json'), 'r') as f:
                metadata = json.load(f)
                self.assertIn('metadata', metadata)
                self.assertIn('images', metadata)
                self.assertEqual(len(metadata['images']), 4)
                logger.info(f"  - Metadata file contains correct structure")
            
        except Exception as e:
            self.fail(f"location_download raised exception: {e}")
        finally:
            if self.downloader.session:
                await self.downloader.session.close()
                self.downloader.session = None
    
    async def async_test_save_to_final_location(self):
        final_dir = os.path.join(self.test_dir, 'final_location')
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
            
        test_file = os.path.join(self.downloader.cache_dir, 'test_file.txt')
        with open(test_file, 'w') as f:
            f.write('test content')
            
        self.downloader.save_to_final_location(final_dir)
        
        self.assertTrue(os.path.exists(os.path.join(final_dir, 'test_file.txt')))
        logger.info(f"✓ save_to_final_location: Successfully moved files to final location")
    
    async def async_test_cache_management(self):
        # Clear any existing cache entries first
        self.downloader.cache_index = {}
        
        for i in range(10):
            lat, lon = 40.7128 + (i * 0.001), -74.0060 + (i * 0.001)
            cache_key = self.downloader._get_cache_key(lat, lon, 18)
            local_path = os.path.join(self.downloader.local_cache_dir, f"test_image_{i}.jpg")
            
            img = Image.new('RGB', (100, 100), color=(i*20, 100, 100))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            img.save(local_path)
            
            self.downloader.cache_index[cache_key] = {
                'local_path': local_path,
                'cloud_path': self.downloader._get_cloud_path(lat, lon, 18),
                'last_accessed': time.time() - (i * 100),
                'in_cloud': True
            }
            
            if i >= 7:
                self.downloader.pending_uploads.add(cache_key)
        
        self.downloader._save_cache_index()
        
        # Test push_to_cloud
        uploaded = await self.downloader.push_to_cloud()
        logger.info(f"✓ push_to_cloud: Attempted to upload {uploaded} images to cloud")
        
        # Test clear_local_cache
        removed = await self.downloader.clear_local_cache(keep_recent=5)
        logger.info(f"✓ clear_local_cache: Removed {removed} images from local cache")
        
        cache_entries = [(k, v) for k, v in self.downloader.cache_index.items() 
                         if v.get('local_path') and os.path.exists(v['local_path'])]
        
        self.assertLessEqual(len(cache_entries), 5)
        logger.info(f"  - {len(cache_entries)} images remain in local cache")
    
    async def async_test_cloud_cache(self):
        # Test _check_local_cache
        lat, lon = 40.7128, -74.0060
        zoom = 18
        cache_key = self.downloader._get_cache_key(lat, lon, zoom)
        local_path = os.path.join(self.downloader.local_cache_dir, f"test_image_cloud.jpg")
        
        img = Image.new('RGB', (100, 100), color=(255, 0, 0))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        img.save(local_path)
        
        self.downloader.cache_index[cache_key] = {
            'local_path': local_path,
            'cloud_path': self.downloader._get_cloud_path(lat, lon, zoom),
            'last_accessed': time.time(),
            'in_cloud': True
        }
        self.downloader._save_cache_index()
        
        result = self.downloader._check_local_cache(lat, lon, zoom)
        self.assertEqual(result, local_path)
        logger.info(f"✓ _check_local_cache: Successfully retrieved local cache path")
        
        cloud_result = await self.downloader._check_cloud_cache(lat, lon, zoom)
        logger.info(f"✓ _check_cloud_cache: {'Retrieved from cloud' if cloud_result else 'Skipped in testing mode'}")
    
    def run_async_tests(self):
        loop = asyncio.get_event_loop()
        
        logger.info("\nTesting image download...")
        loop.run_until_complete(self.async_test_single_download())
        
        logger.info("\nTesting batch processing...")
        loop.run_until_complete(self.async_test_batch_download())
        
        logger.info("\nTesting grid download...")
        loop.run_until_complete(self.async_test_grid_download())
        
        logger.info("\nTesting save to final location...")
        loop.run_until_complete(self.async_test_save_to_final_location())
        
        logger.info("\nTesting cache management...")
        loop.run_until_complete(self.async_test_cache_management())
        
        logger.info("\nTesting cloud cache functions...")
        loop.run_until_complete(self.async_test_cloud_cache())
        
        logger.info("\nTesting full location processing...")
        loop.run_until_complete(self.async_test_location_download())


def run_tests():
    test = TestSatelliteDownloader()
    test.setUp()
    
    try:
        logger.info("\n=== Running Synchronous Tests ===")
        test.test_get_coordinates()
        test.test_get_zoom()
        test.test_calculate_grid()
        test.test_get_location()
        test.test_cache_key_generation()
        test.test_cloud_path_generation()
        test.test_metadata_db_functions()
        test.test_save_cache_index()
        
        logger.info("\n=== Running Asynchronous Tests ===")
        test.run_async_tests()
    finally:
        test.tearDown()
    
    logger.info("\n=== All Tests Completed ===")


if __name__ == "__main__":
    run_tests()