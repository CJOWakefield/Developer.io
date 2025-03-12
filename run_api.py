from src.data.downloader import SatelliteDownloader
from src.models.predictor import RegionPredictor
import asyncio
import logging
import numpy as np
import os
from typing import Dict, List, Optional, Any, Union
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('satellite_api')

## Required api calls for frontend:
# - 1. Process location. Given a country, city (and postcode), retrieve the coordinates and a list of suggested locations.
# - 2. Given co-ordinates, call to the GoogleMaps API to download image data and cache locally. 
# - 3. Using cached images, use model to predict segmentation mask and return overlayed image.
# - 4. Calculate relevant proportions of certain land types.
# - 5. Identify suitable locations for a given purpose, e.g. wind turbines by calculating the available area 

class SatelliteAPI:
    """
    API for satellite image processing and analysis.
    
    This class provides methods for downloading satellite imagery,
    analyzing land types, and identifying suitable locations for
    specific purposes.
    """
    
    def __init__(self, mode='grid'):
        """
        Initialize the SatelliteAPI.
        
        Args:
            mode: 'grid' for multiple images or 'single' for a single image
        """
        self.downloader = SatelliteDownloader()
        self.predictor = RegionPredictor()
        self.mode = mode
        self.lat = None
        self.lon = None
        self.address = None
        self.image_paths = []
        self.image_bytes = []
        self.mask = None
        self.proportions = None
        logger.info(f"SatelliteAPI initialized in {mode} mode")

    async def process_location(self, country: str, city: str, postcode: str = None) -> Dict[str, Any]:
        """
        Process a location to get coordinates and address.
        
        Args:
            country: Country name
            city: City name
            postcode: Optional postal code
            
        Returns:
            Dictionary with status, coordinates, and address
        """
        try:
            lat, lon, address = self.downloader.get_coordinates(country, city, postcode)
            self.lat, self.lon, self.address = lat, lon, address
            logger.info(f"Processed location: {address} ({lat}, {lon})")
            return {'status': 'success', 'coordinates': {'lat': lat, 'lon': lon}, 'address': address}
        except Exception as e:
            logger.error(f"Error processing location: {e}")
            return {'status': 'error', 'message': str(e)}

    async def get_satellite_image(self, lat: float, lon: float, cache: bool = True) -> Dict[str, Any]:
        try:
            await self.downloader._init_session()
            
            if self.mode == 'grid':
                grid_size_km = 0.5
                num_images = 4
                
                logger.info(f"Downloading grid of {num_images} images around ({lat}, {lon})")
                self.image_bytes = await self.downloader.grid_download(
                    lat, lon, grid_size_km=grid_size_km, num_images=num_images, save=cache
                )
                
                if not self.image_bytes:
                    return {'status': 'error', 'message': 'Failed to download images'}
                
                self.image_paths = []
                for file in os.listdir(self.downloader.cache_dir):
                    if file.endswith('.jpg') and not file.startswith('metadata'):
                        path = os.path.join(self.downloader.cache_dir, file)
                        self.image_paths.append(path)
                
                logger.info(f"Downloaded {len(self.image_bytes)} images, found {len(self.image_paths)} in cache")
                
            else:
                # Single image mode
                logger.info(f"Downloading single image at ({lat}, {lon})")
                image_data = await self.downloader.single_download(lat, lon, area_size=0.5, save=cache)
                
                if image_data:
                    self.image_bytes = [image_data]
                    self.image_paths = []
                    for file in os.listdir(self.downloader.cache_dir):
                        if file.endswith('.jpg') and not file.startswith('metadata'):
                            path = os.path.join(self.downloader.cache_dir, file)
                            self.image_paths.append(path)
                            
                    if self.image_paths:
                        logger.info(f"Downloaded image and found in cache at {self.image_paths[0]}")
                    else:
                        return {'status': 'error', 'message': 'Image downloaded but not found in cache'}
                else:
                    return {'status': 'error', 'message': 'Failed to download image'}
            
            if self.image_paths: 
                return {'status': 'success', 'image_paths': self.image_paths}
            else: 
                return {'status': 'error', 'message': 'No images found in cache'}
        except Exception as e:
            logger.error(f"Error in get_satellite_image: {str(e)}")
            return {'status': 'error', 'message': str(e)}
        finally:
            if self.downloader.session:
                await self.downloader.session.close()
                self.downloader.session = None

    async def get_segmentation_mask(self, image_path: str) -> Dict[str, Any]:
        """
        Generate segmentation mask for the specified image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with status and mask
        """
        try:
            logger.info(f"Generating segmentation mask for {image_path}")
            image_tensor = self.predictor.tensor_from_file(image_path)
            prediction_result = self.predictor.predict_from_tensor(image_tensor)
            
            self.raw_mask = prediction_result['raw_mask']
            self.colored_mask = prediction_result['colored_mask']
            
            logger.info(f"Generated mask with shape {self.colored_mask.shape}")
            return {'status': 'success', 'mask': self.colored_mask}
        except Exception as e:
            logger.error(f"Error generating segmentation mask: {e}")
            return {'status': 'error', 'message': str(e)}

    async def get_land_type_proportions(self, image_path: str) -> Dict[str, Any]:
        """
        Calculate land type proportions for the specified image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with status and proportions
        """
        try:
            if not hasattr(self, 'raw_mask') or self.raw_mask is None:
                mask_result = await self.get_segmentation_mask(image_path)
                if mask_result['status'] != 'success':
                    return {'status': 'error', 'message': 'Failed to generate mask'}
            
            logger.info(f"Calculating land type proportions")
            proportions = self.predictor.get_land_proportions(self.raw_mask)
            self.proportions = proportions
            
            logger.info(f"Land proportions: {proportions}")
            return {'status': 'success', 'proportions': proportions}
        except Exception as e:
            logger.error(f"Error in get_land_proportions: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    async def identify_suitable_locations(self, purpose: str, min_area_sqm: float = 1000) -> Dict[str, Any]:
        """
        Identify suitable locations for a specific purpose.
        
        Args:
            purpose: Purpose (e.g., 'wind_turbines', 'solar_panels')
            min_area_sqm: Minimum area in square meters
            
        Returns:
            Dictionary with status and suitable locations
        """
        try:
            if not self.mask:
                return {'status': 'error', 'message': 'No segmentation mask available. Run get_segmentation_mask first.'}
            
            logger.info(f"Identifying suitable locations for {purpose}")
            suitable_locations = self.predictor.identify_suitable_locations(self.mask, purpose, min_area_sqm)
            return {'status': 'success', 'suitable_locations': suitable_locations}
        except Exception as e:
            logger.error(f"Error identifying suitable locations: {e}")
            return {'status': 'error', 'message': str(e)}

if __name__ == '__main__':
    api = SatelliteAPI(mode='single')
    async def test_api():
        try:
            location_result = await api.process_location(country='Jamaica', city='Kingston')
            print(f"Location: {location_result}")
            if location_result['status'] == 'success':
                image_result = await api.get_satellite_image(api.lat, api.lon)
                print(f"Image paths: {api.image_paths}")
                print(f"Number of images: {len(api.image_bytes)}")
                if image_result['status'] == 'success' and api.image_paths:
                    mask_result = await api.get_segmentation_mask(api.image_paths[0])
                    if mask_result['status'] == 'success':
                        mask_path = os.path.join(api.downloader.cache_dir, 
                                               os.path.basename(api.image_paths[0]).replace('.jpg', '_mask.png'))
                        cv2.imwrite(mask_path, cv2.cvtColor(mask_result['mask'], cv2.COLOR_RGB2BGR))
                        print(f"Saved mask to: {mask_path}")
                        
                    proportions_result = await api.get_land_type_proportions(api.image_paths[0])
                    if proportions_result['status'] == 'success':
                        print(f"Land type proportions: {api.proportions}")
        except Exception as e:
            print(f"Error in test_api: {e}")
    
    asyncio.run(test_api())
