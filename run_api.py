from src.data.downloader import SatelliteDownloader
from src.models.predictor import RegionPredictor
import asyncio
import logging
import numpy as np
import os
from typing import Dict, Any
import cv2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('satellite_api')

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
        os.makedirs(self.downloader.cache_dir, exist_ok=True)

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
                for i, img_bytes in enumerate(self.image_bytes):
                    # Create a filename
                    filename = f"satellite_{lat}_{lon}_grid_{i}.jpg"
                    file_path = os.path.join(self.downloader.cache_dir, filename)
                    
                    try:
                        with open(file_path, 'wb') as f:
                            f.write(img_bytes)
                        self.image_paths.append(file_path)
                        logger.info(f"Saved image to {file_path}")
                    except Exception as e:
                        logger.error(f"Error saving image: {e}")
                
                logger.info(f"Downloaded {len(self.image_bytes)} images, saved {len(self.image_paths)} files")
                
            else:
                logger.info(f"Downloading single image at ({lat}, {lon})")
                image_data = await self.downloader.single_download(lat, lon, area_size=0.5, save=cache)
                
                if image_data:
                    self.image_bytes = [image_data]
                    filename = f"satellite_{lat}_{lon}.jpg"
                    file_path = os.path.join(self.downloader.cache_dir, filename)
                    
                    try:
                        with open(file_path, 'wb') as f:
                            f.write(image_data)
                        self.image_paths = [file_path]
                        logger.info(f"Saved image to {file_path}")
                    except Exception as e:
                        logger.error(f"Error saving image: {e}")
                        return {'status': 'error', 'message': f'Error saving image: {str(e)}'}
                else:
                    return {'status': 'error', 'message': 'Failed to download image'}
            
            if self.image_paths: 
                return {'status': 'success', 'image_paths': self.image_paths}
            else: 
                return {'status': 'error', 'message': 'No images saved to cache'}
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
            
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return {'status': 'error', 'message': 'Image file not found'}
            
            if os.path.getsize(image_path) == 0:
                logger.error(f"Image file is empty: {image_path}")
                
                if hasattr(self, 'image_bytes') and self.image_bytes:
                    index = -1
                    for i, path in enumerate(self.image_paths):
                        if os.path.normpath(path) == os.path.normpath(image_path):
                            index = i
                            break
                    
                    if index >= 0 and index < len(self.image_bytes):
                        with open(image_path, 'wb') as f:
                            f.write(self.image_bytes[index])
                        logger.info(f"Restored image from bytes: {image_path}")
                    else:
                        return {'status': 'error', 'message': 'Image file is empty and no bytes available'}

            try:
                img = cv2.imread(image_path)
                if img is None:
                    logger.error(f"Failed to read image with OpenCV: {image_path}")
                    with open(image_path, 'rb') as f:
                        img_bytes = f.read()
                        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                    
                    if img is None:
                        return {'status': 'error', 'message': 'Image file is corrupted or invalid format'}
            except Exception as e:
                logger.error(f"Error reading image: {e}")
                return {'status': 'error', 'message': f'Error reading image: {str(e)}'}
            
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