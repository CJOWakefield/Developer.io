# Satellite Land Classification

A deep learning-based tool for automated land classification using satellite imagery from Google Earth Engine. This project helps property developers identify and analyze different types of land regions using state-of-the-art computer vision techniques.

## Features

- 🛰️ **Real-time Satellite Data**: Automated retrieval of current satellite imagery using Google Earth Engine API
- 🤖 **Advanced Deep Learning**: Custom UNet CNN architecture for precise land region segmentation
- 🎯 **Region Classification**: Identifies and segments:
  - Urban areas
  - Agricultural land
  - Rangeland
  - Forest regions
  - Water bodies
  - Barren land
- 📈 **Data Augmentation**: Smart augmentation pipeline to enhance training data for specific regions
- 🎨 **Visualization Tools**: Built-in tools for result visualization and analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/satellite-land-classification.git
cd satellite-land-classification

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
project_root/
├── src/
│   ├── data/           # Dataset and data loading utilities
│   ├── models/         # UNet model implementation
│   ├── training/       # Training pipeline
│   └── visualization/  # Visualization tools
├── data/
│   ├── raw/           # Original satellite images
│   ├── processed/     # Processed and augmented data
│   ├── training/      # Training dataset
│   └── validation/    # Validation dataset
├── scripts/           # Training and prediction scripts
├── configs/           # Configuration files
└── models/           # Saved model checkpoints
```

## Usage

### 1. Downloading Satellite Images

```python
from src.data.image_loader import download_satellite_images

# Download images for a specific region
download_satellite_images(
    latitude=40.4168,
    longitude=-3.7038,
    grid_size_km=0.2,
    num_images=16
)
```

### 2. Training the Model

```python
from scripts.train import train_model

# Train with default parameters
train_model(epochs=5)

# Train with custom parameters
train_model(
    epochs=10,
    batch_size=32,
    learning_rate=0.001
)
```

### 3. Making Predictions

```python
from src.visualization.predictor import visualise_pred

# Predict land regions for specific images
results = visualise_pred(
    model_path='models/latest/model.pt',
    data_dir='data/test',
    n_samples=3
)
```

## Data Augmentation

The project includes a robust data augmentation pipeline to enhance the training dataset:

- Random rotations and flips
- Brightness and contrast adjustments
- Zoom variations
- Custom region-specific augmentations

## Model Architecture

The UNet architecture is specifically adapted for satellite imagery:

- Residual blocks for better gradient flow
- Focal loss for handling class imbalance
- Multi-scale feature extraction
- Skip connections for fine detail preservation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Training data from [DeepGlobe Land Cover Classification Dataset](https://www.kaggle.com/balraj98/deepglobe-land-cover-classification-dataset)
- Google Earth Engine for satellite imagery
- PyTorch for deep learning framework

## Contact

Your Name - CJOWakefield@outlook.com
Project Link: [https://github.com/CJOWakefield/Developer.io](https://github.com/CJOWakefield/Developer.io)

---
