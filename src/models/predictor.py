import os
import re
import cv2
import torch
import numpy as np
import asyncio
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import threading
import yaml
from google.cloud import storage
from io import BytesIO
from typing import Union, List, Optional, Dict
import sys

base_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_directory)
train_directory = os.path.join(base_directory, 'data', 'train')
model_directory = os.path.join(base_directory, 'data', 'models')

from src.models.trainer import transformer, UNet

with open(os.path.join(base_directory, 'configs', 'default_config.yaml'), 'r') as file:
    config = yaml.safe_load(file)

''' ----- Predictor file summary -----

> visualise_pred - Predict and visualise satellite image and mask selection from most recent model.

    >>> Inputs: Model path & data path both hardcoded in file. N_samples and image_ids completely optional and not overly important.

> RegionPredictor - Class allowing for more comprehensive model prediction visualisation. Identifies chunk regions with high percentage of specific land type.
    >> predict_region - Highlights specific region with high % specific land type with annotation.

'''

class ImageBatchDataset(Dataset):
    def __init__(self, data_path, image_files):
        self.data_path = data_path
        self.image_files = image_files

    def __getitem__(self, idx):
        file = self.image_files[idx]
        image = cv2.cvtColor(cv2.imread(os.path.join(self.data_path, file)), cv2.COLOR_BGR2RGB)
        image_id = re.match(r'(\d+)', file).group(1)
        return transformer(image), image, image_id

    def __len__(self):
        return len(self.image_files)

class RegionPredictor:
    def __init__(self, base_dir=base_directory, model_version=None, cache_size=100):
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.generated_dir = os.path.join(self.base_dir, 'data', 'downloaded')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.thread_lock = threading.Lock()
        self.storage_client = storage.Client()
        self.cache_size = cache_size
        self.image_cache = {}
        self.prediction_cache = {}
        self.classes = {'urban': (0, 255, 255),    # urban land - Cyan
                        'agricultural': (255, 255, 0),    # agricultural land - Yellow
                        'rangeland': (255, 0, 255),    # rangeland - Magenta
                        'forest': (0, 255, 0),      # forest land - Green
                        'water': (0, 0, 255),      # water - Dark blue
                        'barren': (255, 255, 255),  # barren land - White
                        'unidentified': (0, 0, 0)}         # unknown - Black

        if not model_version:
            versions = [d for d in os.listdir(model_directory) if os.path.isdir(os.path.join(model_directory, d)) and d.startswith('v_')]
            if not versions: raise ValueError('No models available.')
            self.model_version = sorted(versions, key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2])))[-1]
        self.model = UNet().to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(self.base_dir, 'data', 'models', self.model_version, 'model.pt'), weights_only=True)['model_state_dict'])
        self.model.eval()
        
    @torch.no_grad()
    def predict_from_tensor(self, image_tensor):
        with torch.no_grad():
            prediction = self.model(image_tensor)['segmentation']
            predicted_classes = torch.argmax(prediction, dim=1)[0].cpu().numpy()
        
        pred_rgb = np.zeros((*predicted_classes.shape, 3), dtype=np.uint8)

        class_colors = {
            0: (0, 255, 255),  # urban - Cyan
            1: (255, 255, 0),  # agriculture - Yellow
            2: (255, 0, 255),  # rangeland - Magenta
            3: (0, 255, 0),    # forest - Green
            4: (0, 0, 255),    # water - Blue
            5: (255, 255, 255), # barren - White
            6: (0, 0, 0)        # unknown - Black
        }
        
        for class_idx, color in class_colors.items():
            pred_rgb[predicted_classes == class_idx] = color

        return {
            'raw_mask': predicted_classes,
            'colored_mask': pred_rgb
        }

    @torch.no_grad()
    def predict_from_file(self, file_path: str) -> np.ndarray:
        """
        Predict segmentation mask from an image file.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Segmentation mask as numpy array
        """
        try:
            image = cv2.imread(file_path)
            if image is None: raise ValueError(f"Failed to load image from {file_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_height, original_width = image.shape[:2]
            if hasattr(self, 'input_size'): resized_image = cv2.resize(image, self.input_size)
            else: resized_image = image
            
            image_tensor = torch.from_numpy(resized_image.transpose(2, 0, 1)).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            result = self.predict_from_tensor(image_tensor)
        
            if hasattr(self, 'input_size') and (original_height, original_width) != result['colored_mask'].shape[:2]:
                result['colored_mask'] = cv2.resize(result['colored_mask'], (original_width, original_height), interpolation=cv2.INTER_NEAREST)
                raw_mask_resized = cv2.resize(result['raw_mask'].astype(float), (original_width, original_height), interpolation=cv2.INTER_NEAREST)
                result['raw_mask'] = raw_mask_resized.astype(result['raw_mask'].dtype)
            return result
            
        except Exception as e:
            return {
                'raw_mask': np.zeros((512, 512), dtype=np.uint8),
                'colored_mask': np.zeros((512, 512, 3), dtype=np.uint8),
                'unique_classes': []
            }

    @torch.no_grad()
    def predict_from_bucket(self, bucket_name: str, blob_path: str) -> np.ndarray:
        cache_key = ('bucket', bucket_name, blob_path)
        cached_prediction = self._get_from_cache(cache_key)
        if cached_prediction is not None:
            return cached_prediction

        try:
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            image_bytes = BytesIO()
            blob.download_to_file(image_bytes)
            image_bytes.seek(0)
            image_array = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = transformer(image).unsqueeze(0).to(self.device)
            prediction = self.predict_from_tensor(image_tensor)
            self._add_to_cache(cache_key, image, prediction)
            return prediction

        except Exception as e:
            raise ValueError(f"Failed to process image from bucket: {str(e)}")
        
    @torch.no_grad()
    async def predict_region(self, images: Union[List[torch.Tensor], List[str]], save_dir: Optional[str] = None) -> Dict:
        predictions = {}
        if not images:
            raise ValueError("No images provided")
            
        if isinstance(images[0], str):
            for img_path in images:
                if not img_path.endswith('_sat.jpg'):
                    continue
                    
                image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                image_tensor = transformer(image).unsqueeze(0).to(self.device)
                pred_rgb = self.predict_from_tensor(image_tensor)
                
                image_id = os.path.basename(img_path).replace('_sat.jpg', '')
                predictions[image_id] = {
                    'original': image,
                    'mask': pred_rgb,
                    'path': img_path
                }
                
        elif isinstance(images[0], torch.Tensor):
            for idx, img_tensor in enumerate(images):
                if img_tensor.dim() == 3:
                    img_tensor = img_tensor.unsqueeze(0)
                pred_rgb = self.predict_from_tensor(img_tensor)
                
                predictions[f'tensor_{idx}'] = {
                    'original': img_tensor.cpu().numpy(),
                    'mask': pred_rgb
                }
        
        if save_dir: await self.save_predictions(predictions, save_dir)
        return predictions

    def get_land_proportions(self, mask: np.ndarray) -> Dict[str, float]:
            """
            Calculate land type proportions from a segmentation mask.
            
            Args:
                mask: Segmentation mask
            
            """
            
            class_labels = {0: 'urban', 
                            1: 'agriculture', 
                            2: 'rangeland', 
                            3: 'forest', 
                            4: 'water', 
                            5: 'barren', 
                            6: 'unknown'}
            try:
                # Handle different input types
                if isinstance(mask, dict) and 'raw_mask' in mask:
                    # Already have raw class indices
                    raw_mask = mask['raw_mask']
                elif len(mask.shape) == 2:
                    # Already have raw class indices 
                    raw_mask = mask
                elif len(mask.shape) == 3 and mask.shape[2] == 3:
                    # Create a raw mask array from colored mask
                    if torch.is_tensor(mask):
                        # If it's a PyTorch tensor, handle accordingly
                        raw_mask = torch.zeros((mask.shape[0], mask.shape[1]), dtype=torch.long, device=mask.device)
                        for class_idx, color in self.classes.items():
                            color_tensor = torch.tensor(color, device=mask.device).view(1, 1, 3)
                            raw_mask[torch.all(mask == color_tensor, dim=2)] = list(self.classes.keys()).index(class_idx)
                        raw_mask = raw_mask.cpu().numpy()
                    else:
                        # Handle numpy array
                        raw_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
                        for class_idx, color in self.classes.items():
                            color_array = np.array(color).reshape(1, 1, 3)
                            raw_mask[np.all(mask == color_array, axis=2)] = list(self.classes.keys()).index(class_idx)
                elif hasattr(self, 'raw_mask') and self.raw_mask is not None:
                    raw_mask = self.raw_mask
                else:
                    # Handle direct prediction from model output
                    with torch.no_grad():
                        if hasattr(self, 'model') and torch.is_tensor(mask):
                            prediction = self.model(mask)['segmentation']
                            raw_mask = torch.argmax(prediction, dim=1)[0].cpu().numpy()
                        else:
                            raise ValueError("Invalid mask format")
                
                # Calculate proportions
                total_pixels = raw_mask.size
                proportions = {}
                for class_idx, label in class_labels.items():
                    pixels_in_class = np.sum(raw_mask == class_idx)
                    proportion = float(pixels_in_class) / total_pixels
                    proportions[label] = round(proportion, 4)
                    
                proportions['vegetated'] = proportions['forest'] + proportions['rangeland'] + proportions['agriculture']
                proportions['developed'] = proportions['urban'] + proportions['barren']
                return proportions
                    
            except Exception as e:
                print(f"Error in get_land_proportions: {str(e)}")
                return {
                    'urban': 0,
                    'agriculture': 0,
                    'rangeland': 0,
                    'forest': 0,
                    'water': 0,
                    'barren': 0,
                    'unknown': 1,
                    'vegetated': 0,
                    'developed': 0
                }
    
    async def save_predictions(self, predictions: Dict, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        
        def save_prediction(pred_data):
            img_id, data = pred_data
            with self.thread_lock:
                if 'path' not in data:
                    orig_path = os.path.join(save_dir, f'{img_id}_sat.jpg')
                    cv2.imwrite(orig_path, cv2.cvtColor(data['original'], cv2.COLOR_RGB2BGR))
                
                mask_path = os.path.join(save_dir, f'{img_id}_mask.png')
                Image.fromarray(data['mask'].astype(np.uint8)).save(mask_path, format='PNG')
        
        with ThreadPoolExecutor(max_workers=config['data']['num_workers']) as executor:
            list(executor.map(save_prediction, predictions.items()))

    def _add_to_cache(self, key: tuple, image: np.ndarray, prediction: np.ndarray):
        if len(self.image_cache) >= self.cache_size:
            oldest_key, oldest_pred = self.prediction_cache.popitem(last=False)
            _, oldest_img = self.image_cache.popitem(last=False)
            self._save_cached_data(oldest_key, oldest_img, oldest_pred)

        self.image_cache[key] = image
        self.prediction_cache[key] = prediction

    def _save_cached_data(self, key: tuple, image: np.ndarray, prediction: np.ndarray):
        cache_dir = os.path.join(self.generated_dir, 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        if key[0] == 'file':
            base_filename = os.path.basename(key[1]).replace('_sat.jpg', '')
        else:
            base_filename = key[2].replace('/', '_').replace('_sat.jpg', '')
        
        image_path = os.path.join(cache_dir, f"{base_filename}_sat.jpg")
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        pred_path = os.path.join(cache_dir, f"{base_filename}_mask.png")
        Image.fromarray(prediction.astype(np.uint8)).save(pred_path, format='PNG')

    def predict_mask(self, image_path: str) -> np.ndarray:
        """
        Generate segmentation mask for the specified image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Segmentation mask as numpy array
        """
        try:
            result = self.predict_from_file(image_path)
            self.raw_mask = result['raw_mask']
            return result['colored_mask']
        except Exception as e:
            return np.zeros((512, 512, 3), dtype=np.uint8)
    
    def identify_suitable_locations(self, mask: np.ndarray, purpose: str, min_area_sqm: float = 1000) -> List[Dict]:
        """
        Identify suitable locations for a specific purpose.
        
        Args:
            mask: Segmentation mask
            purpose: Purpose (e.g., 'wind_turbines', 'solar_panels')
            min_area_sqm: Minimum area in square meters
            
        Returns:
            List of suitable locations with coordinates and areas
        """
        suitable_locations = []
        
        # Map purposes to land types
        purpose_land_map = {
            'wind_turbines': ['rangeland', 'barren'],
            'solar_panels': ['barren', 'rangeland', 'agricultural'],
            'agriculture': ['agricultural', 'rangeland'],
            'urban_development': ['barren', 'rangeland'],
            'conservation': ['forest', 'water', 'rangeland']
        }
        
        if purpose not in purpose_land_map:
            raise ValueError(f"Unsupported purpose: {purpose}. Supported purposes: {list(purpose_land_map.keys())}")
        
        suitable_land_types = purpose_land_map[purpose]
        
        # Create a binary mask for suitable land types
        binary_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        
        for land_type in suitable_land_types:
            if land_type in self.classes:
                color = self.classes[land_type]
                binary_mask |= np.all(mask == color, axis=-1)
        
        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate pixel to square meter conversion (approximate)
        # Assuming a standard satellite image covers about 0.5km x 0.5km
        pixel_to_sqm = (500 * 500) / (mask.shape[0] * mask.shape[1])
        
        for i, contour in enumerate(contours):
            area_pixels = cv2.contourArea(contour)
            area_sqm = area_pixels * pixel_to_sqm
            
            if area_sqm >= min_area_sqm:
                # Get centroid of the contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0
                
                suitable_locations.append({
                    'id': i,
                    'area_sqm': area_sqm,
                    'centroid': (cx, cy),
                    'contour': contour.tolist()
                })
        
        return suitable_locations
    
    def visualise(self, image_path: str) -> tuple:
        """
        Visualize prediction results with land type proportions.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple containing (figure, prediction_result, proportions)
        """
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        image_tensor = transformer(image).unsqueeze(0).to(self.device)
        
        prediction_result = self.predict_from_tensor(image_tensor)
        proportions = self.get_land_proportions(prediction_result)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(prediction_result['colored_mask'])
        axes[1].set_title('Predicted Mask')
        axes[1].axis('off')
        
        land_types = [lt for lt in proportions if lt not in ['vegetated', 'developed']]
        values = [proportions[lt] * 100 for lt in land_types]
        colors = ['cyan', 'yellow', 'magenta', 'green', 'blue', 'white', 'black']
        
        axes[2].bar(land_types, values, color=colors[:len(land_types)])
        axes[2].set_title('Land Type Proportions (%)')
        axes[2].set_ylim(0, 100)
        axes[2].set_ylabel('Percentage')
        plt.setp(axes[2].get_xticklabels(), rotation=45, ha='right')
        
        for land_type, proportion in proportions.items(): print(f"{land_type}: {proportion*100:.2f}%")
        
        plt.tight_layout()
        plt.show()
        return fig, prediction_result, proportions

''' _______________________________________________________________________________________________________'''

## Visualisation function. Executes prediction on random training images and visualises w/ plt.
@torch.no_grad()
def visualise_pred(model_path=model_directory, data_path=train_directory, model_version=None, n_samples=3, image_ids=None, batch_size=config['training']['batch_size']):
    versions = [d for d in os.listdir(model_directory) if os.path.isdir(os.path.join(model_directory, d)) and d.startswith('v_')]
    if not versions: raise ValueError('Model not found.')

    if model_version: model_path = os.path.join(model_directory, model_version, 'model.pt')
    else:
        latest_version = sorted(versions, key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2])))[-1]
        model_path = os.path.join(model_directory, latest_version, 'model.pt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True)['model_state_dict'])
    model.eval()

    if image_ids:
        image_ids = [image_ids] if isinstance(image_ids, (int, str)) else image_ids
        sat_files = [f for id in image_ids for f in os.listdir(data_path) if f.endswith(f"_sat.jpg")]
        if not sat_files: raise ValueError("No matching images found")
        n_samples = len(sat_files)
    else:
        sat_files = np.random.choice([f for f in os.listdir(data_path) if f.endswith('_sat.jpg')], size=n_samples, replace=False)

    dataset = ImageBatchDataset(data_path, sat_files)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=config['data']['num_workers'], pin_memory=True)

    class_colors = {
        0: (0, 255, 255),    # urban land - Cyan
        1: (255, 255, 0),    # agricultural land - Yellow
        2: (255, 0, 255),    # rangeland - Magenta
        3: (0, 255, 0),      # forest land - Green
        4: (0, 0, 255),      # water - Dark blue
        5: (255, 255, 255),  # barren land - White
        6: (0, 0, 0)         # unknown - Black
    }

    results = []
    for batch_tensors, batch_images, batch_ids in dataloader:
        batch_tensors = batch_tensors.to(device)
        predictions = torch.argmax(model(batch_tensors)['segmentation'], dim=1).cpu()

        for pred, orig_img, img_id in zip(predictions, batch_images, batch_ids):
            pred_rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)
            for class_idx, color in class_colors.items():
                pred_rgb[pred == class_idx] = color
            results.append((orig_img, pred_rgb, img_id))

    rows = 1
    cols = n_samples
    plt.figure(figsize=(6*cols, 8*rows))

    for idx, (sat_img, pred_rgb, image_id) in enumerate(results):
        plt.subplot(rows*2, cols, idx+1)
        plt.title(f'Original Image (ID: {image_id})')
        plt.imshow(sat_img)
        plt.axis('off')

        plt.subplot(rows*2, cols, idx+1+cols)
        plt.title(f'Predicted Mask (ID: {image_id})')
        plt.imshow(pred_rgb)
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    return results

def visualize_prediction_results(result_dict, original_image_path=None):
    """
    Visualize prediction results from predict_from_file output.
    
    Args:
        result_dict: Dictionary containing 'raw_mask' and 'colored_mask'
        original_image_path: Optional path to the original image
    """
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot original image if provided
    if original_image_path:
        original = cv2.imread(original_image_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        axes[0].imshow(original)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
    
    # Plot colored mask
    axes[1].imshow(result_dict['colored_mask'])
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    
    # Plot raw mask with colormap
    raw_mask_display = result_dict['raw_mask'].copy()
    axes[2].imshow(raw_mask_display, cmap='nipy_spectral')
    axes[2].set_title('Raw Class Indices')
    axes[2].axis('off')
    
    # Add colorbar for the raw mask
    cbar = plt.colorbar(axes[2].imshow(raw_mask_display, cmap='nipy_spectral'), 
                        ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Class Index')
    
    # Add legend for the colored mask
    class_names = {
        0: 'Urban (Cyan)',
        1: 'Agriculture (Yellow)',
        2: 'Rangeland (Magenta)',
        3: 'Forest (Green)',
        4: 'Water (Blue)',
        5: 'Barren (White)',
        6: 'Unknown (Black)'
    }
    
    legend_elements = []
    for class_idx, name in class_names.items():
        color = result_dict['colored_mask'][0, 0].copy()
        if class_idx == 0: color = np.array([0, 255, 255])
        if class_idx == 1: color = np.array([255, 255, 0])
        if class_idx == 2: color = np.array([255, 0, 255])
        if class_idx == 3: color = np.array([0, 255, 0])
        if class_idx == 4: color = np.array([0, 0, 255])
        if class_idx == 5: color = np.array([255, 255, 255])
        if class_idx == 6: color = np.array([0, 0, 0])
        
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=color/255.0, label=name))
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    plt.show()
    
    return fig

if __name__ == '__main__':
    # predictor = RegionPredictor()
    # res = predictor.predict_from_file('data/downloaded/chicago_41.876_-87.624_2.0km_500m/100002_sat.jpg')
    # visualize_prediction_results(res)
    predictor = RegionPredictor()
    predictor.visualise('data/downloaded/aragon_41.379_-0.764_2.0km_500m/100007_sat.jpg')
    # visualise_pred(data_path='data/downloaded/aragon_41.379_-0.764_2.0km_500m', n_samples=1, image_ids=100007)