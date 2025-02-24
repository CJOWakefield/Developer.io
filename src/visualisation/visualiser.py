import sys
import os
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Tuple, Dict
from src.data.loader import SatelliteImages

## Accessory functions for visualisation
class LandTypeHighlighter:
    def __init__(self) -> None:
        self.class_colors: Dict[str, Tuple[int, int, int]] = {
            'urban': (0, 255, 255),
            'agriculture': (255, 255, 0),
            'rangeland': (255, 0, 255),
            'forest': (0, 255, 0),
            'water': (0, 0, 255),
            'barren': (255, 255, 255),
            'unknown': (0, 0, 0)
        }

    # Highlight regions where specified land types exceed threshold
    def analyze_regions(self, image: np.ndarray, mask: np.ndarray, valid_types: List[str], threshold: float = 0.9) -> np.ndarray:
        
        result = image.copy()
        binary_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        for land_type in valid_types:
            color = self.class_colors[land_type]
            binary_mask |= np.all(mask == color, axis=2)

        contours, _ = cv2.findContours(binary_mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            contour_mask = np.zeros_like(binary_mask)
            cv2.drawContours(contour_mask, [contour], -1, 1, -1)
            total_pixels = np.sum(contour_mask)
            if total_pixels == 0: continue

            valid_pixels = np.sum(binary_mask & contour_mask)
            percentage = (valid_pixels / total_pixels) * 100

            if percentage >= threshold * 100:
                cv2.drawContours(result, [contour], -1, (0, 0, 255), 2)
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                    text = f'{percentage:.1f}%'
                    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(result, (cx-w//2-5, cy-h-5), (cx+w//2+5, cy+5), (255, 255, 255), -1)
                    cv2.putText(result, text, (cx-w//2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return result

    # Display original, mask, and analyzed images
    def visualize(self, image: np.ndarray, mask: np.ndarray, valid_types: List[str], threshold: float = 0.9) -> None:
        highlighted = self.analyze_regions(image, mask, valid_types, threshold)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax2.imshow(mask)
        ax3.imshow(cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB))

        ax1.set_title('Original Image')
        ax2.set_title('Land Type Mask')
        ax3.set_title(f'Regions with ≥{threshold*100}% of\n{", ".join(valid_types)}')

        for ax in (ax1, ax2, ax3):
            ax.axis('off')

        plt.tight_layout()
        plt.show()


class ImagePreview:
    # Initializes preview class with directory path and testing flag
    def __init__(self, directory: str, testing: bool = False) -> None:
        self.directory = directory
        self.testing = testing
        self.class_to_rgb: Dict[int, Tuple[int, int, int]] = {
            0: (0, 255, 255), 
            1: (255, 255, 0), 
            2: (255, 0, 255),
            3: (0, 255, 0), 
            4: (0, 0, 255), 
            5: (255, 255, 255), 
            6: (0, 0, 0)
        }

    # Displays satellite image and optionally its mask using matplotlib
    def preview(self, img_id: int, show_mask: bool = False) -> None:
        images = SatelliteImages(self.directory)

        if self.testing: print(images.__len__())
        idx = images.image_id.index(img_id)
        sat, mask, _ = images[idx]

        sat = sat.cpu().numpy() if torch.is_tensor(sat) else sat
        mask = mask.cpu().numpy() if torch.is_tensor(mask) else mask

        if show_mask:
            plt.subplot(1, 2, 1)
            plt.imshow(sat)
            plt.subplot(1, 2, 2)
            plt.imshow(mask)
            plt.show()
        else:
            plt.imshow(sat)
            plt.title('Satellite Image')
            plt.show()

if __name__ == '__main__':
    highlighter = LandTypeHighlighter()
    image: np.ndarray = cv2.imread(os.path.join(base_dir, 'data', 'train', '119_sat.jpg'))
    mask: np.ndarray = cv2.imread(os.path.join(base_dir, 'data', 'train', '119_mask.png'))
    valid_types: List[str] = ['agriculture', 'rangeland']
    highlighter.visualize(image, mask, valid_types, threshold=0.9)