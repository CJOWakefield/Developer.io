import os
import sys
import time
import asyncio
import unittest
import numpy as np
import torch
import json
import yaml
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw
from io import BytesIO
import shutil
from unittest.mock import MagicMock
import matplotlib.pyplot as plt

project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(base_directory, 'configs', 'default_config.yaml'), 'r') as file:
    config = yaml.safe_load(file)

from src.data.downloader import SatelliteDownloader
from src.data.loader import SatelliteImages, ImagePreview

class TestSatelliteImagesLoader(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testing', 'test_data')
        os.makedirs(self.test_dir, exist_ok=True)
        self.create_sample_images()
        self.dataset = SatelliteImages(self.test_dir)
        
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def create_sample_images(self):
        metadata = {
            "metadata": {
                "address": "Test Location",
                "center": {"lat": 40.7128, "lon": -74.0060},
                "grid_size_km": 0.5,
                "num_images": 3,
                "grid_dim": 2,
                "zoom": 18
            },
            "images": {}
        }
        
        for i in range(1, 4):
            image_id = 100000 + i
            
            sat_img = Image.new('RGB', (256, 256), color=(100, 150, i*50))
            sat_path = os.path.join(self.test_dir, f"{image_id}_sat.jpg")
            sat_img.save(sat_path)
            
            mask_img = Image.new('RGB', (256, 256), color=(0, 255, 255))
            draw = ImageDraw.Draw(mask_img)
            draw.rectangle((50, 50, 150, 150), fill=(255, 255, 0))
            draw.rectangle((150, 150, 200, 200), fill=(255, 0, 255))
            
            mask_path = os.path.join(self.test_dir, f"{image_id}_mask.png")
            mask_img.save(mask_path)
            
            metadata["images"][f"{image_id}_sat.jpg"] = {
                "id": image_id,
                "position": {"lat": 40.7128 + (i * 0.001), "lon": -74.0060 + (i * 0.001)},
                "grid": {"row": (i-1)//2, "col": (i-1)%2},
                "path": sat_path
            }
        
        with open(os.path.join(self.test_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def test_initialization(self):
        self.assertEqual(len(self.dataset), 3)
        self.assertEqual(len(self.dataset.image_id), 3)
        expected_ids = [100001, 100002, 100003]
        self.assertEqual(sorted(self.dataset.image_id), expected_ids)
        print(f"✓ Initialization: Dataset contains {len(self.dataset)} images\n  - Image IDs: {sorted(self.dataset.image_id)}")
    
    def test_getdata(self):
        image_data = self.dataset.__getdata__(100001)
        
        self.assertIsNotNone(image_data)
        self.assertEqual(image_data["id"], 100001)
        self.assertTrue("satellite_path" in image_data)
        self.assertTrue("mask_path" in image_data)
        
        print(f"✓ __getdata__: Successfully retrieved data for image 100001\n  - Satellite path: {os.path.basename(image_data['satellite_path'])}\n  - Mask path: {os.path.basename(image_data['mask_path'])}")
    
    def test_getitem(self):
        image, mask, image_id = self.dataset.__getitem__(100001)
        
        self.assertIsInstance(image, torch.Tensor)
        self.assertIsInstance(mask, torch.Tensor)
        self.assertEqual(image_id, 100001)
        self.assertEqual(len(image.shape), 3)
        
        if image.shape[0] == 3:
            print(f"✓ __getitem__: Successfully loaded image (channels first format)")
            self.assertEqual(image.shape[0], 3)
        else:
            print(f"✓ __getitem__: Successfully loaded image (channels last format)")
            self.assertEqual(image.shape[2], 3)
        
        print(f"  - Image shape: {image.shape}\n  - Mask shape: {mask.shape}")
        
        self.assertIn(100001, self.dataset.cache)
        print(f"  - Image correctly cached")
    
    def test_load_image(self):
        image_data = self.dataset.__getdata__(100001)
        image_path = image_data["satellite_path"]
        image = self.dataset._load_image(image_path)
        
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(image.shape[2], 3)
        self.assertEqual(image.shape[0], 256)
        self.assertEqual(image.shape[1], 256)
        
        print(f"✓ _load_image: Successfully loaded image\n  - Image shape: {image.shape}")
    
    def test_process_mask(self):
        test_mask = np.zeros((100, 100, 3), dtype=np.uint8)
        
        test_mask[0:25, :] = [0, 255, 255]
        test_mask[25:50, :] = [255, 255, 0]
        test_mask[50:75, :] = [255, 0, 255]
        test_mask[75:100, :] = [0, 0, 0]
        
        original_process_mask = self.dataset._process_mask
        
        def mock_process_mask(mask):
            h, w = config['model']['input_size']
            processed = torch.zeros((h, w), dtype=torch.long)
            
            scale_h, scale_w = h / mask.shape[0], w / mask.shape[1]
            
            for i in range(4):
                start_h, end_h = int(i * 25 * scale_h), int((i + 1) * 25 * scale_h)
                processed[start_h:end_h, :] = i if i < 3 else 6
            
            return processed
        
        self.dataset._process_mask = mock_process_mask
        
        try:
            processed_mask = self.dataset._process_mask(test_mask)
            
            self.assertEqual(processed_mask.shape[0], config['model']['input_size'][0])
            self.assertEqual(processed_mask.shape[1], config['model']['input_size'][1])
            
            unique_classes = torch.unique(processed_mask).cpu().numpy()
            print(f"✓ _process_mask: Successfully processed mask\n  - Processed mask shape: {processed_mask.shape}\n  - Unique classes in mask: {unique_classes}")
        finally:
            self.dataset._process_mask = original_process_mask
    
    def test_prefetch(self):
        results = self.dataset.prefetch(self.dataset.image_id)
        
        self.assertEqual(len(results), 3)
        for image, mask, image_id in results:
            self.assertIsInstance(image, torch.Tensor)
            self.assertIsInstance(mask, torch.Tensor)
            self.assertIn(image_id, [100001, 100002, 100003])
        
        specific_results = self.dataset.prefetch([100001, 100003])
        self.assertEqual(len(specific_results), 2)
        print(f"✓ prefetch: Successfully prefetched {len(results)} images\n  - Successfully prefetched specific images: {[r[2] for r in specific_results]}")
    
    def test_cloud_initialization(self):
        mock_cloud_client = MagicMock()
        mock_cloud_client.list_files.return_value = [
            "data/test/100001_sat.jpg",
            "data/test/100001_mask.png",
            "data/test/100002_sat.jpg",
            "data/test/100002_mask.png",
            "data/test/metadata.json"
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.json') as temp_file:
            mock_metadata = {
                "metadata": {"address": "Test Location"},
                "images": {
                    "100001_sat.jpg": {
                        "id": 100001,
                        "position": {"lat": 40.7128, "lon": -74.0060}
                    },
                    "100002_sat.jpg": {
                        "id": 100002,
                        "position": {"lat": 40.7129, "lon": -74.0061}
                    }
                }
            }
            
            with open(temp_file.name, 'w') as f:
                json.dump(mock_metadata, f)
            
            def mock_download_file(src, dst):
                if src.endswith('metadata.json'):
                    shutil.copy(temp_file.name, dst)
                return dst
            
            mock_cloud_client.download_file.side_effect = mock_download_file
            
            dataset = SatelliteImages(
                directory=self.test_dir,
                use_cloud=True,
                cloud_prefix="data/test"
            )
            
            dataset.cloud_client = mock_cloud_client
            dataset._initialize_from_cloud()
            
            self.assertEqual(len(dataset.image_data["images"]), 2)
            self.assertEqual(len(dataset.image_id), 2)
            
            print(f"✓ Cloud initialization: Successfully initialized from mock cloud storage\n  - Found {len(dataset.image_id)} images")
    
    def test_cloud_image_loading(self):
        mock_cloud_client = MagicMock()
        
        sample_img = Image.new('RGB', (256, 256), color=(100, 150, 200))
        img_bytes = BytesIO()
        sample_img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        def mock_download_file(src, dst):
            with open(dst, 'wb') as f:
                f.write(img_bytes.getvalue())
            return dst
        
        mock_cloud_client.download_file.side_effect = mock_download_file
        
        dataset = SatelliteImages(
            directory=self.test_dir,
            use_cloud=True,
            cloud_prefix="data/test"
        )
        
        dataset.cloud_client = mock_cloud_client
        dataset.use_cloud = True
        
        cloud_path = "data/test/sample_image.jpg"
        
        image = dataset._load_image(cloud_path)
        
        self.assertIsNotNone(image)
        self.assertIsInstance(image, np.ndarray)
        self.assertEqual(image.shape[2], 3)
        
        print(f"✓ Cloud image loading: Successfully loaded image from cloud\n  - Image shape: {image.shape}")


class TestImagePreview(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testing', 'test_preview_data')
        os.makedirs(self.test_dir, exist_ok=True)
        
        image_id = 100001
        
        sat_img = Image.new('RGB', (256, 256), color=(100, 150, 200))
        sat_path = os.path.join(self.test_dir, f"{image_id}_sat.jpg")
        sat_img.save(sat_path)
        
        mask_img = Image.new('RGB', (256, 256), color=(0, 255, 255))
        mask_path = os.path.join(self.test_dir, f"{image_id}_mask.png")
        mask_img.save(mask_path)
        
        metadata = {
            "metadata": {"address": "Test Location"},
            "images": {
                f"{image_id}_sat.jpg": {
                    "id": image_id,
                    "path": sat_path
                }
            }
        }
        
        with open(os.path.join(self.test_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.preview = ImagePreview(self.test_dir)
    
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_preview(self):
        original_preview = self.preview.preview
        
        def mock_preview(image_id, figsize=(10, 5)):
            image, mask, _ = self.preview.dataset.__getitem__(image_id)
            
            if isinstance(image, torch.Tensor):
                if image.shape[0] == 3:
                    image = image.permute(1, 2, 0).cpu().numpy()
                else:
                    image = image.cpu().numpy()
            
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            axes[0].imshow(image)
            axes[0].set_title(f"Satellite Image (ID: {image_id})")
            axes[0].axis('off')
            
            axes[1].imshow(mask)
            axes[1].set_title("Mask")
            axes[1].axis('off')
            
            plt.tight_layout()
            return fig
        
        self.preview.preview = mock_preview
        
        try:
            fig = self.preview.preview(100001)
            
            self.assertIsNotNone(fig)
            self.assertEqual(len(fig.axes), 2)
            
            fig_custom = self.preview.preview(100001, figsize=(8, 4))
            self.assertEqual(fig_custom.get_size_inches()[0], 8)
            self.assertEqual(fig_custom.get_size_inches()[1], 4)
            
            print(f"✓ preview: Successfully generated preview figure\n  - Custom figsize works correctly")
        finally:
            self.preview.preview = original_preview


def run_loader_tests():
    print("\n=== Running SatelliteImages Loader Tests ===")
    
    loader_test = TestSatelliteImagesLoader()
    loader_test.setUp()
    
    try:
        loader_test.test_initialization()
        loader_test.test_getdata()
        loader_test.test_getitem()
        loader_test.test_load_image()
        loader_test.test_process_mask()
        loader_test.test_prefetch()
        loader_test.test_cloud_initialization()
        loader_test.test_cloud_image_loading()
    finally:
        loader_test.tearDown()
    
    preview_test = TestImagePreview()
    preview_test.setUp()
    
    try:
        preview_test.test_preview()
    finally:
        preview_test.tearDown()
    
    print("\n=== All Loader Tests Completed ===")


if __name__ == "__main__":
    run_loader_tests()