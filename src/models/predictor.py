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

# Load the configuration file
with open('configs/default_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

''' ----- Predictor file summary -----

> visualise_pred - Predict and visualise satellite image and mask selection from most recent model.

    >>> Inputs: Model path & data path both hardcoded in file. N_samples and image_ids completely optional and not overly important.

> RegionPredictor - Class allowing for more comprehensive model prediction visualisation. Identifies chunk regions with high percentage of specific land type.
    >> predict_region - Highlights specific region with high % specific land type with annotation.

'''

base_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
train_directory = os.path.join(base_directory, 'data', 'train')
model_directory = os.path.join(base_directory, 'models')

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
    def __init__(self, base_dir=base_directory, model_version=None):
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.generated_dir = os.path.join(self.base_dir, 'data', 'downloaded')
        self.model_version = model_version
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.thread_lock = threading.Lock()

    @torch.no_grad()
    async def predict_region(self, country, city, postcode=None, grid_size_km=0.5, num_images=16):
        downloader = SatelliteDownloader()
        # Await the async process_location call
        directory = await downloader.process_location(country, city, postcode, grid_size_km, num_images)

        if directory is None:
            raise ValueError("Failed to download images")

        data_dir = os.path.join(self.generated_dir, directory)

        if not [f for f in os.listdir(data_dir) if f.endswith('_sat.jpg')]:
            raise ValueError(f"No satellite images found in {data_dir}")

        predictions = visualise_pred(data_path=data_dir,
                                     model_version=self.model_version,
                                     n_samples=num_images,
                                     batch_size=config['training']['batch_size'])

        def save_prediction(pred_data):
            _, mask, id = pred_data
            with self.thread_lock:
                output_path = os.path.join(data_dir, f'{id}_mask.png')
                Image.fromarray(mask.astype(np.uint8)).save(output_path, format='PNG')

        with ThreadPoolExecutor(max_workers=config['data']['num_workers']) as executor:
            list(executor.map(save_prediction, predictions))

        return data_dir

if __name__ == '__main__':
    predictor = RegionPredictor()
    asyncio.run(predictor.predict_region(country='Jamaica',
                                         city='Kingston',
                                         postcode='JN1'))
