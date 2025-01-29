import cv2
import numpy as np
import matplotlib.pyplot as plt

## Accessory functions for visualisation
class LandTypeHighlighter:
    def __init__(self):
        self.class_colors = {
            'urban': (0, 255, 255),
            'agriculture': (255, 255, 0),
            'rangeland': (255, 0, 255),
            'forest': (0, 255, 0),
            'water': (0, 0, 255),
            'barren': (255, 255, 255),
            'unknown': (0, 0, 0)
        }

    def analyze_regions(self, image, mask, valid_types, threshold=0.9):
        """Highlight regions where specified land types exceed threshold"""
        result = image.copy()

        # Create binary mask for all valid types
        binary_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        for land_type in valid_types:
            color = self.class_colors[land_type]
            binary_mask |= np.all(mask == color, axis=2)

        # Find contours
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Analyze each contour
        for contour in contours:
            # Create mask for current contour
            contour_mask = np.zeros_like(binary_mask)
            cv2.drawContours(contour_mask, [contour], -1, 1, -1)

            # Calculate percentage of valid types within contour
            total_pixels = np.sum(contour_mask)
            if total_pixels == 0: continue

            valid_pixels = np.sum(binary_mask & contour_mask)
            percentage = (valid_pixels / total_pixels) * 100

            # Draw contour and percentage if above threshold
            if percentage >= threshold * 100:
                cv2.drawContours(result, [contour], -1, (0, 0, 255), 2)

                # Add percentage text
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                    text = f'{percentage:.1f}%'

                    # Get text size for background
                    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    # Draw text background and text
                    cv2.rectangle(result, (cx-w//2-5, cy-h-5), (cx+w//2+5, cy+5), (255, 255, 255), -1)
                    cv2.putText(result, text, (cx-w//2, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return result

    def visualize(self, image, mask, valid_types, threshold=0.9):
        """Display original, mask, and analyzed images"""
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

# Example usage:
"""
highlighter = LandTypeHighlighter()

# Read images
image = cv2.imread('satellite_image.jpg')
mask = cv2.imread('mask.png')

# Analyze agricultural areas that are at least 90% agriculture or rangeland
valid_types = ['agriculture', 'rangeland']
highlighter.visualize_threshold_analysis(
    image, mask, 'agriculture', valid_types, threshold=0.9)
"""
