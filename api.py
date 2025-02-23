from src.data.downloader import SatelliteDownloader
from src.models.predictor import RegionPredictor
import asyncio
## Required api calls for frontend:
# - 1. Process location. Given a country, city (and postcode), retrieve the coordinates and a list of suggested locations.
# - 2. Given co-ordinates, call to the GoogleMaps API to download image data and cache locally. 
# - 3. Using cached images, use model to predict segmentation mask and return overlayed image.
# - 4. Calculate relevant proportions of certain land types.
# - 5. Identify suitable locations for a given purpose, e.g. wind turbines by calculating the available area 

class SatelliteAPI:
    def __init__(self, mode='grid'):
        self.downloader = SatelliteDownloader()
        self.predictor = RegionPredictor()
        self.mode = mode
        self.lat = None
        self.lon = None
        self.address = None
        self.images = None
        self.mask = None
        self.proportions = None

    ## Return and cache coordinates and address for a given location.
    async def process_location(self, country, city, postcode=None):
        try:
            lat, lon, address = self.downloader.get_coordinates(country, city, postcode)
            self.lat, self.lon, self.address = lat, lon, address
            return {'status': 'success', 'coordinates': {'lat': lat, 'lon': lon}, 'address': address}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    ## Return and cache satellite image/images for a given location.
    async def get_satellite_image(self, lat, lon, cache=True):
        try:
            if self.mode == 'grid':
                image_paths = self.downloader.download_grid(lat, lon, cache)
                return {'status': 'success', 'image_paths': image_paths}
            elif self.mode == 'single':
                image_paths = self.downloader._download_image_async(lat, lon, cache)
                return {'status': 'success', 'image_path': image_paths}
            self.images = image_paths
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    ## Return and cache segmentation mask for a given image/images.
    async def get_segmentation_mask(self, image_path):
        try:
            mask = self.predictor.predict_mask(image_path)
            self.mask = mask
            return {'status': 'success', 'mask': mask}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    ## Return and cache land type proportions for a given image/images.
    async def get_land_type_proportions(self, image_path):
        try:
            proportions = self.predictor.calculate_proportions(image_path)
            self.proportions = proportions
            return {'status': 'success', 'proportions': proportions}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

if __name__ == '__main__':
    api = SatelliteAPI(mode='single')
    asyncio.run(api.process_location(country='Jamaica', city='Kingston'))
    asyncio.run(api.get_satellite_image(api.lat, api.lon))
    # asyncio.run(api.get_segmentation_mask(api.images[0]))
    # asyncio.run(api.get_land_type_proportions(api.images[0]))
    print(api.lat, api.lon)
    print(api.images)