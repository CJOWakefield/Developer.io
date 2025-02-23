import os
import re
import cv2
import torch
import numpy as np
import asyncio
import matplotlib.pyplot as plt
from src.models.trainer import transformer, UNet
from PIL import Image
from src.data.downloader import SatelliteDownloader
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import threading
import yaml
from google.cloud import storage
from io import BytesIO
from typing import Union, List, Optional, Dict

base_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
train_directory = os.path.join(base_directory, 'data', 'train')
model_directory = os.path.join(base_directory, 'data', 'models')

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
        sat_files = np.random.choice([f for f in os.listdir(data_path) if f.endswith('_sat.jpg')],
                                   size=n_samples, replace=False)

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

        if not model_version:
            versions = [d for d in os.listdir(model_directory) if os.path.isdir(os.path.join(model_directory, d)) and d.startswith('v_')]
            if not versions: raise ValueError('No models available.')
            self.model_version = sorted(versions, key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2])))[-1]
        self.model = UNet().to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(self.base_dir, 'models', self.model_version, 'model.pt'), weights_only=True)['model_state_dict'])
        self.model.eval()
        
    @torch.no_grad()
    def predict_from_tensor(self, image_tensor):
        prediction = torch.argmax(self.model(image_tensor)['segmentation'], dim=1)[0].cpu()
        class_colors = {
            0: (0, 255, 255),    # urban land - Cyan
            1: (255, 255, 0),    # agricultural land - Yellow
            2: (255, 0, 255),    # rangeland - Magenta
            3: (0, 255, 0),      # forest land - Green
            4: (0, 0, 255),      # water - Dark blue
            5: (255, 255, 255),  # barren land - White
            6: (0, 0, 0)         # unknown - Black
        }
        
        pred_rgb = np.zeros((*prediction.shape, 3), dtype=np.uint8)
        for class_idx, color in class_colors.items():
            pred_rgb[prediction == class_idx] = color
            
        return pred_rgb

    @torch.no_grad()
    def predict_from_file(self, file_path: str) -> np.ndarray:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
            
        if not file_path.endswith('_sat.jpg'):
            raise ValueError("File must be a satellite image with '_sat.jpg' extension")
            
        image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
        image_tensor = transformer(image).unsqueeze(0).to(self.device)
        prediction = self.predict_from_tensor(image_tensor)
        cache_key = ('file', file_path)
        self._add_to_cache(cache_key, image, prediction)
        return prediction

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
        
        if save_dir:
            await self.save_predictions(predictions, save_dir)
            
        return predictions
    
    async def save_predictions(self, predictions: Dict, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        
        def save_prediction(pred_data):
            img_id, data = pred_data
            with self.thread_lock:
                # Save original image if from file
                if 'path' not in data:
                    orig_path = os.path.join(save_dir, f'{img_id}_sat.jpg')
                    cv2.imwrite(orig_path, cv2.cvtColor(data['original'], cv2.COLOR_RGB2BGR))
                
                # Save mask
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

if __name__ == '__main__':
    predictor = RegionPredictor()
    asyncio.run(predictor.predict_region(country='Jamaica',
                                         city='Kingston',
                                         postcode='JN1'))
