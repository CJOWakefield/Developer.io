import os
import sys
import asyncio
import unittest
import logging
from unittest.mock import patch, MagicMock, AsyncMock
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image

project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from run_api import SatelliteAPI
from src.data.downloader import SatelliteDownloader
from src.models.predictor import RegionPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_satellite_api')

class TestSatelliteAPI(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api_testing')
        os.makedirs(self.test_dir, exist_ok=True)
        
        self.test_image_path = os.path.join(self.test_dir, 'test_image.jpg')
        test_image = Image.new('RGB', (256, 256), color=(73, 109, 137))
        test_image.save(self.test_image_path)
        
        self.api = SatelliteAPI(mode='grid')
        
        self.api.image_paths = [self.test_image_path]
        self.api.mask = np.zeros((256, 256, 3), dtype=np.uint8)
        
    def tearDown(self):
        if os.path.exists(self.test_dir):
            import shutil
            shutil.rmtree(self.test_dir)
    
    @patch('run_api.SatelliteDownloader')
    def test_init(self, mock_downloader):
        """Test API initialization"""
        api = SatelliteAPI(mode='single')
        self.assertEqual(api.mode, 'single')
        self.assertIsNone(api.lat)
        self.assertIsNone(api.lon)
        self.assertIsNone(api.address)
        self.assertEqual(api.image_paths, [])
        self.assertEqual(api.image_bytes, [])
        logger.info("✓ API initialization successful")
    
    @patch('run_api.SatelliteDownloader')
    async def async_test_process_location(self, mock_downloader):
        """Test processing a location"""
        self.api.downloader.get_coordinates = MagicMock(return_value=(40.7128, -74.0060, "New York, USA"))
        
        result = await self.api.process_location(country="United States", city="New York")
        
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['coordinates']['lat'], 40.7128)
        self.assertEqual(result['coordinates']['lon'], -74.0060)
        self.assertEqual(result['address'], "New York, USA")
        self.assertEqual(self.api.lat, 40.7128)
        self.assertEqual(self.api.lon, -74.0060)
        self.assertEqual(self.api.address, "New York, USA")
        logger.info("✓ process_location: Successfully processed location")
        
        self.api.downloader.get_coordinates = MagicMock(side_effect=Exception("Location not found"))
        
        result = await self.api.process_location(country="Unknown", city="Nowhere")
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('message', result)
        logger.info("✓ process_location: Properly handled error case")
    
    @patch('run_api.SatelliteDownloader')
    async def async_test_get_satellite_image_grid(self, mock_downloader):
        """Test getting satellite images in grid mode"""
        self.api.image_paths = []
        self.api.image_bytes = []

        self.api.mode = 'grid'
        self.api.downloader._init_session = AsyncMock()
        
        image_bytes = [b'image1', b'image2', b'image3', b'image4']
        self.api.downloader.grid_download = AsyncMock(return_value=image_bytes)
        
        self.api.downloader.get_zoom = MagicMock(return_value=18)
        coordinates = [(40.7128, -74.0060), (40.7129, -74.0060), (40.7128, -74.0061), (40.7129, -74.0061)]
        self.api.downloader.calculate_grid = MagicMock(return_value=(coordinates, 2))
        
        self.api.downloader.cache_index = {}
        for i, coord in enumerate(coordinates):
            cache_key = f"{coord[0]:.6f}_{coord[1]:.6f}_18"
            local_path = os.path.join(self.test_dir, f"image_{i}.jpg")
            Image.new('RGB', (100, 100), color=(i*50, 100, 150)).save(local_path)
            self.api.downloader.cache_index[cache_key] = {'local_path': local_path}
        
        self.api.downloader._get_cache_key = lambda lat, lon, zoom: f"{lat:.6f}_{lon:.6f}_{zoom}"
        self.api.downloader.session = MagicMock()
        self.api.downloader.session.close = AsyncMock()
        
        result = await self.api.get_satellite_image(40.7128, -74.0060)
        
        self.assertEqual(result['status'], 'success')
        self.assertEqual(len(self.api.image_bytes), 4)
        self.assertEqual(len(self.api.image_paths), 4)
        logger.info(f"✓ get_satellite_image (grid): Downloaded {len(self.api.image_bytes)} images")
        logger.info(f"  - Found {len(self.api.image_paths)} image paths in cache")
    
    @patch('run_api.SatelliteDownloader')
    async def async_test_get_satellite_image_single(self, mock_downloader):
        """Test getting a single satellite image"""
        single_test_dir = os.path.join(self.test_dir, 'single_test')
        os.makedirs(single_test_dir, exist_ok=True)
        
        self.api.image_paths = []
        self.api.image_bytes = []
        
        self.api.downloader.cache_dir = single_test_dir
        
        self.api.mode = 'single'
        self.api.downloader._init_session = AsyncMock()
        
        test_image = Image.new('RGB', (100, 100), color=(73, 109, 137))
        img_byte_arr = BytesIO()
        test_image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        self.api.downloader.single_download = AsyncMock(return_value=img_byte_arr)
        self.api.downloader.get_zoom = MagicMock(return_value=18)
        
        local_path = os.path.join(single_test_dir, "test_image.jpg")
        test_image.save(local_path)
        
        with patch('os.listdir', return_value=["test_image.jpg"]):
            result = await self.api.get_satellite_image(40.7128, -74.0060)
        
        self.assertEqual(result['status'], 'success')
        self.assertEqual(len(self.api.image_bytes), 1)
        self.assertEqual(len(self.api.image_paths), 1)
        self.assertEqual(self.api.image_paths[0], local_path)
        logger.info("✓ get_satellite_image (single): Downloaded single image")
        logger.info(f"  - Image path: {self.api.image_paths[0]}")
        
        self.api.image_paths = []
        self.api.image_bytes = []
        self.api.downloader.single_download = AsyncMock(return_value=None)
        
        result = await self.api.get_satellite_image(40.7128, -74.0060)
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('message', result)
        logger.info("✓ get_satellite_image (single): Properly handled error case")
    
    @patch('run_api.RegionPredictor')
    async def async_test_get_segmentation_mask(self, mock_predictor):
        """Test generating segmentation mask"""
        mock_tensor = MagicMock()
        mock_raw_mask = np.zeros((256, 256), dtype=np.uint8)
        mock_colored_mask = np.zeros((256, 256, 3), dtype=np.uint8)
        mock_colored_mask[50:100, 50:100, 0] = 255  # Add some features
        
        self.api.predictor.tensor_from_file = MagicMock(return_value=mock_tensor)
        self.api.predictor.predict_from_tensor = MagicMock(return_value={
            'raw_mask': mock_raw_mask,
            'colored_mask': mock_colored_mask
        })
        
        result = await self.api.get_segmentation_mask(self.test_image_path)
        
        self.assertEqual(result['status'], 'success')
        self.assertIs(result['mask'], mock_colored_mask)
        self.assertIs(self.api.raw_mask, mock_raw_mask)
        self.assertIs(self.api.colored_mask, mock_colored_mask)
        logger.info("✓ get_segmentation_mask: Successfully generated mask")
        
        self.api.predictor.tensor_from_file = MagicMock(side_effect=Exception("Model error"))
        
        result = await self.api.get_segmentation_mask(self.test_image_path)
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('message', result)
        logger.info("✓ get_segmentation_mask: Properly handled error case")
    
    @patch('run_api.RegionPredictor')
    async def async_test_get_land_type_proportions(self, mock_predictor):
        """Test calculating land type proportions"""
        mock_proportions = {
            'urban': 0.15,
            'agriculture': 0.25,
            'rangeland': 0.10,
            'forest': 0.30,
            'water': 0.20,
            'barren': 0.05,
            'unknown': 0.05,
            'vegetated': 0.65,
            'developed': 0.20
        }
        
        mock_tensor = MagicMock()
        mock_raw_mask = np.zeros((256, 256), dtype=np.uint8)
        mock_colored_mask = np.zeros((256, 256, 3), dtype=np.uint8)
        
        self.api.predictor.tensor_from_file = MagicMock(return_value=mock_tensor)
        self.api.predictor.predict_from_tensor = MagicMock(return_value={
            'raw_mask': mock_raw_mask,
            'colored_mask': mock_colored_mask
        })
        
        self.api.predictor.get_land_proportions = MagicMock(return_value=mock_proportions)
        
        result = await self.api.get_land_type_proportions(self.test_image_path)
        
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['proportions'], mock_proportions)
        self.assertEqual(self.api.proportions, mock_proportions)
        logger.info("✓ get_land_type_proportions: Successfully calculated proportions")
        logger.info(f"  - Proportions: {mock_proportions}")
        
        self.api.raw_mask = mock_raw_mask
        result = await self.api.get_land_type_proportions(self.test_image_path)
        self.assertEqual(result['status'], 'success')
        
        self.api.raw_mask = None
        self.api.predictor.get_land_proportions = MagicMock(side_effect=Exception("Analysis error"))
        
        result = await self.api.get_land_type_proportions(self.test_image_path)
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('message', result)
        logger.info("✓ get_land_type_proportions: Properly handled error case")
    
    @patch('run_api.RegionPredictor')
    async def async_test_identify_suitable_locations(self, mock_predictor):
        """Test identifying suitable locations"""
        mock_locations = [
            {'id': 0, 'area_sqm': 2500, 'centroid': (100, 100), 'contour': [[[100, 100]], [[150, 100]], [[150, 150]], [[100, 150]]]},
            {'id': 1, 'area_sqm': 900, 'centroid': (200, 200), 'contour': [[[200, 200]], [[230, 200]], [[230, 230]], [[200, 230]]]}
        ]
        
        self.api.predictor.identify_locations = MagicMock(return_value=mock_locations)
        
        self.api.mask = np.zeros((256, 256, 3), dtype=np.uint8)
        
        result = await self.api.identify_suitable_locations(purpose='solar_panels', min_area_sqm=500)
        
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['suitable_locations'], mock_locations)
        logger.info("✓ identify_suitable_locations: Successfully identified locations")
        logger.info(f"  - Found {len(mock_locations)} suitable locations")
        
        self.api.mask = None
        
        result = await self.api.identify_suitable_locations(purpose='wind_turbines')
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('message', result)
        logger.info("✓ identify_suitable_locations: Properly handled missing mask case")
        
        self.api.mask = np.zeros((256, 256, 3), dtype=np.uint8)
        self.api.predictor.identify_locations = MagicMock(side_effect=Exception("Analysis error"))
        
        result = await self.api.identify_suitable_locations(purpose='solar_panels')
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('message', result)
        logger.info("✓ identify_suitable_locations: Properly handled error case")
    
    def run_async_tests(self):
        loop = asyncio.get_event_loop()
        
        logger.info("\nTesting process_location...")
        loop.run_until_complete(self.async_test_process_location())
        
        logger.info("\nTesting get_satellite_image (grid mode)...")
        loop.run_until_complete(self.async_test_get_satellite_image_grid())
        
        logger.info("\nTesting get_satellite_image (single mode)...")
        loop.run_until_complete(self.async_test_get_satellite_image_single())
        
        logger.info("\nTesting get_segmentation_mask...")
        loop.run_until_complete(self.async_test_get_segmentation_mask())
        
        logger.info("\nTesting get_land_type_proportions...")
        loop.run_until_complete(self.async_test_get_land_type_proportions())


def run_tests():
    test = TestSatelliteAPI()
    test.setUp()
    
    try:
        logger.info("\n=== Running Synchronous Tests ===")
        test.test_init()
        
        logger.info("\n=== Running Asynchronous Tests ===")
        test.run_async_tests()
    finally:
        test.tearDown()
    
    logger.info("\n=== All Tests Completed ===")

if __name__ == "__main__":
    run_tests()
