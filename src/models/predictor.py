from prev_files.images import *
from prev_files.model import *
import os
import random
# import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class RegionPredictor:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.generated_dir = os.path.join(self.base_dir, '_generated_data')

    def predict_region(self, latitude=None, longitude=None, grid_size_km=0.2, num_images=16):
        if not latitude or not longitude:
            latitude = random.triangular(36.0, 43.5, 40.0)
            longitude = random.triangular(-9.3, 3.0, -3.7)
            print(f"Random coordinates: {latitude:.4f}, {longitude:.4f}")

        process_images(longitude, latitude, grid_size_km, num_images)

        latest_dir = sorted([d for d in os.listdir(self.generated_dir)], key=lambda x: os.path.getmtime(os.path.join(self.generated_dir, x)))[-1]
        data_dir = os.path.join(self.generated_dir, latest_dir)

        if not [f for f in os.listdir(data_dir) if f.endswith('_sat.jpg')]: raise ValueError(f"No satellite images found in {data_dir}")

        predictions = visualise_pred(data_dir=data_dir)
        for _, mask, id in predictions:
            mask_img = Image.fromarray(mask.astype(np.uint8))
            output_path = os.path.join(data_dir, f'{id}_mask')
            mask_img.save(output_path, 'PNG', quality=95)
        return data_dir
