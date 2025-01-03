from images import *
from model import *
import os
import random

def image_to_mask(latitude=None, longitude=None, grid_size_km=0.2, num_images=16):
    """Run satellite image download and prediction pipeline"""
    # Get coordinates
    if not latitude or not longitude:
        latitude = random.triangular(36.0, 43.5, 40.0)  # Center bias around Madrid
        longitude = random.triangular(-9.3, 3.0, -3.7)
        print(f"Random coordinates: {latitude:.4f}, {longitude:.4f}")

    # Download images
    process_images(longitude, latitude, grid_size_km, num_images)

    # Get directory with downloaded images
    base_dir = os.path.dirname(os.path.abspath(__file__))
    generated_dir = os.path.join(base_dir, '_generated_data')
    latest_dir = sorted([d for d in os.listdir(generated_dir)], key=lambda x: os.path.getmtime(os.path.join(generated_dir, x)))[-1]
    data_dir = os.path.join(generated_dir, latest_dir)

    # Select random sample of images for prediction
    sat_files = [f for f in os.listdir(data_dir) if f.endswith('_sat.jpg')]
    if not sat_files:
        raise ValueError(f"No satellite images found in {data_dir}")

    # Extract IDs and make predictions
    # model_path = os.path.join('_data', 'models', 'v_0_02', 'model.pt')
    # sample_files = random.sample(sat_files, min(3, len(sat_files)))
    # image_ids = [int(f.split('_')[0]) for f in sample_files]
    # print(f"\nMaking predictions for images: {sample_files}")

    visualise_pred(data_dir=data_dir)

if __name__ == '__main__':
    image_to_mask()
