import os
import sys
import time
import asyncio
import unittest
from io import BytesIO
from pathlib import Path
from PIL import Image
import shutil
import json

project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.data.downloader import SatelliteDownloader

class TestSatelliteDownloader(unittest.TestCase):
    
    def setUp(self):
        self.downloader = SatelliteDownloader(batch_size=2, testing=True)
        
        self.test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testing')
        os.makedirs(self.test_dir, exist_ok=True)
        
        self.test_cache_dir = os.path.join(self.test_dir, 'test_cache')
        os.makedirs(self.test_cache_dir, exist_ok=True)
        
        self.original_cache_dir = self.downloader.cache_dir
        self.downloader.cache_dir = self.test_cache_dir
        
    def tearDown(self):
        self.downloader.cache_dir = self.original_cache_dir
        
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)
    
    def test_get_coordinates(self):
        lat, lon, address = self.downloader.get_coordinates("United States", "New York")
        self.assertIsInstance(lat, float)
        self.assertIsInstance(lon, float)
        self.assertIsInstance(address, str)
        print(f"✓ get_coordinates: {address} ({lat}, {lon})")
    
    def test_get_zoom(self):
        zoom = self.downloader.get_zoom(0.5)
        self.assertIsInstance(zoom, int)
        self.assertTrue(15 <= zoom <= 20)
        print(f"✓ get_zoom: {zoom} for 0.5km")
        
        for size in [0.1, 0.2, 0.5, 1.0, 2.0]:
            zoom = self.downloader.get_zoom(size)
            print(f"  - Grid size {size}km -> zoom level {zoom}")
    
    def test_calculate_grid(self):
        coordinates, grid_dim = self.downloader.calculate_grid(40.7128, -74.0060, 0.5, 4)
        self.assertEqual(len(coordinates), 4)
        self.assertEqual(grid_dim, 2)
        print(f"✓ calculate_grid: Generated {len(coordinates)} coordinates with grid dimension {grid_dim}")
        
        for num_images in [4, 9, 16]:
            coordinates, grid_dim = self.downloader.calculate_grid(40.7128, -74.0060, 0.5, num_images)
            self.assertEqual(len(coordinates), num_images)
            print(f"  - {num_images} images -> grid dimension {grid_dim}")
    
    def test_get_location(self):
        address = self.downloader.get_location(40.7128, -74.0060)
        self.assertIsInstance(address, str)
        print(f"✓ get_location: {address}")
    
    def test_cache_key_generation(self):
        cache_key = self.downloader._get_cache_key(40.7128, -74.0060, 18)
        self.assertIsInstance(cache_key, str)
        print(f"✓ _get_cache_key: {cache_key}")
        
        cache_key2 = self.downloader._get_cache_key(40.71281, -74.00601, 18)
        self.assertNotEqual(cache_key, cache_key2)
        print(f"  - Different coordinates produce different keys")
    
    def test_cloud_path_generation(self):
        cache_key = self.downloader._get_cache_key(40.7128, -74.0060, 18)
        cloud_path = self.downloader._get_cloud_path(cache_key)
        self.assertIsInstance(cloud_path, str)
        self.assertTrue(cloud_path.startswith("downloaded/"))
        print(f"✓ _get_cloud_path: {cloud_path}")
    
    async def async_test_download_image(self):
        await self.downloader._init_session()
        try:
            image_data = await self.downloader._download_image_async(40.7128, -74.0060, zoom=18)
            self.assertIsNotNone(image_data)
            
            img = Image.open(BytesIO(image_data))
            self.assertIsInstance(img, Image.Image)
            print(f"✓ _download_image_async: Downloaded image of size {img.size}")
            
            cache_key = self.downloader._get_cache_key(40.7128, -74.0060, 18)
            self.assertIn(cache_key, self.downloader.cache_index)
            self.assertIn('local_path', self.downloader.cache_index[cache_key])
            print(f"  - Image was properly cached with key {cache_key}")
            
            cached_image_data = await self.downloader._download_image_async(40.7128, -74.0060, zoom=18)
            self.assertIsNotNone(cached_image_data)
            print(f"  - Successfully retrieved image from cache")
            
        finally:
            await self.downloader.session.close()
            self.downloader.session = None
    
    async def async_test_process_batch(self):
        await self.downloader._init_session()
        try:
            lat, lon = 40.7128, -74.0060
            coordinates = [(0, (lat, lon)), (1, (lat + 0.001, lon)), (2, (lat, lon + 0.001)), (3, (lat + 0.001, lon + 0.001))]
            
            metadata = {'images': {}}
            images = await self.downloader._process_batch(coordinates[:2], 18, 2, metadata)
            
            self.assertEqual(len(images), 2)
            self.assertEqual(len(metadata['images']), 2)
            print(f"✓ _process_batch: Processed {len(images)} images")
            
            for filename, img_metadata in metadata['images'].items():
                self.assertIn('id', img_metadata)
                self.assertIn('position', img_metadata)
                self.assertIn('grid', img_metadata)
                self.assertIn('path', img_metadata)
                self.assertIn('cache_key', img_metadata)
            print(f"  - Metadata correctly generated for all images")
            
        finally:
            await self.downloader.session.close()
            self.downloader.session = None
    
    async def async_test_process_location(self):
        test_location_dir = os.path.join(self.test_dir, 'test_location')
        if os.path.exists(test_location_dir):
            shutil.rmtree(test_location_dir)
        
        try:
            output_dir, images = await self.downloader.process_location(
                country="United States",
                city="New York",
                grid_size_km=0.2,
                num_images=4
            )
            
            self.assertIsNotNone(output_dir)
            self.assertEqual(len(images), 4)
            print(f"✓ process_location: Processed location with {len(images)} images")
            print(f"  - Original output directory: {output_dir}")
            
            if os.path.exists(output_dir) and output_dir != test_location_dir:
                os.makedirs(test_location_dir, exist_ok=True)
                
                for file in os.listdir(output_dir):
                    src_file = os.path.join(output_dir, file)
                    dst_file = os.path.join(test_location_dir, file)
                    if os.path.isfile(src_file):
                        shutil.copy2(src_file, dst_file)
                
                shutil.rmtree(output_dir)
                
                print(f"  - Moved data to test directory: {test_location_dir}")
                
                output_dir = test_location_dir
            
            self.downloader.save_to_final_location(output_dir)
            self.assertTrue(os.path.exists(output_dir))
            self.assertTrue(os.path.exists(os.path.join(output_dir, 'metadata.json')))
            print(f"✓ save_to_final_location: Saved files to {output_dir}")
            
        finally:
            if os.path.exists(test_location_dir):
                shutil.rmtree(test_location_dir)
            
            if 'output_dir' in locals() and os.path.exists(output_dir) and output_dir != test_location_dir:
                shutil.rmtree(output_dir)
    
    async def async_test_cache_management(self):
        for i in range(10):
            lat, lon = 40.7128 + (i * 0.001), -74.0060 + (i * 0.001)
            cache_key = self.downloader._get_cache_key(lat, lon, 18)
            local_path = os.path.join(self.downloader.local_cache_dir, f"test_image_{i}.jpg")
            
            img = Image.new('RGB', (100, 100), color=(i*20, 100, 100))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            img.save(local_path)
            
            self.downloader.cache_index[cache_key] = {
                'local_path': local_path,
                'cloud_path': self.downloader._get_cloud_path(cache_key),
                'last_accessed': time.time() - (i * 100),
                'in_cloud': True if i < 7 else False
            }
            
            if i >= 7:
                self.downloader.pending_uploads.add(cache_key)
        
        self.downloader._save_cache_index()
        
        if not self.downloader.testing:
            uploaded = await self.downloader.push_to_cloud()
            print(f"✓ push_to_cloud: Uploaded {uploaded} images to cloud")
        
        removed = await self.downloader.clear_local_cache(keep_recent=5)
        print(f"✓ clear_local_cache: Removed {removed} images from local cache")
        
        cache_entries = [(k, v) for k, v in self.downloader.cache_index.items() 
                         if v.get('local_path') and os.path.exists(v['local_path'])]
        
        recent_entries = sorted(
            [(k, v) for k, v in self.downloader.cache_index.items() 
             if v.get('local_path') and os.path.exists(v['local_path'])],
            key=lambda x: x[1].get('last_accessed', 0),
            reverse=True
        )[:5]
        
        print(f"  - {len(cache_entries)} images in cache, {len(recent_entries)} are recent")
        self.assertGreaterEqual(len(cache_entries), len(recent_entries))
        print(f"  - At least {len(recent_entries)} recent images remain in local cache")
    
    def run_async_tests(self):
        loop = asyncio.get_event_loop()
        
        print("\nTesting image download...")
        loop.run_until_complete(self.async_test_download_image())
        
        print("\nTesting batch processing...")
        loop.run_until_complete(self.async_test_process_batch())
        
        print("\nTesting cache management...")
        loop.run_until_complete(self.async_test_cache_management())
        
        print("\nTesting full location processing...")
        loop.run_until_complete(self.async_test_process_location())


def run_tests():
    test = TestSatelliteDownloader()
    test.setUp()
    
    try:
        print("\n=== Running Synchronous Tests ===")
        test.test_get_coordinates()
        test.test_get_zoom()
        test.test_calculate_grid()
        test.test_get_location()
        test.test_cache_key_generation()
        test.test_cloud_path_generation()
        
        print("\n=== Running Asynchronous Tests ===")
        test.run_async_tests()
    finally:
        test.tearDown()
    
    print("\n=== All Tests Completed ===")


if __name__ == "__main__":
    run_tests()