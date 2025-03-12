# Satellite Land Classification

A deep learning-based tool for automated land classification using satellite imagery from Google Earth Engine. This project helps property developers identify and analyze different types of land regions using state-of-the-art computer vision techniques.

## Features

- 🛰️ **Real-time Satellite Data**: Automated retrieval of current satellite imagery using Google Maps Static API
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
- 🔍 **Location Analysis**: Identify suitable locations for specific purposes (solar panels, wind turbines, etc.)
- ☁️ **Cloud Integration**: Google Cloud Storage support for image caching and retrieval

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
├── api/                  # API integration with cloud services
│   ├── __init__.py
│   ├── cloud_storage_client.py
│   └── cloud_data_management.py
├── app/                  # Web application
│   ├── __init__.py
│   ├── app.py
│   ├── functions.py
│   ├── routes/
│   ├── static/
│   └── templates/
├── src/                  # Core functionality
│   ├── data/             # Dataset and data loading utilities
│   │   ├── __init__.py
│   │   ├── augment.py
│   │   ├── downloader.py
│   │   └── loader.py
│   ├── models/           # UNet model implementation
│   │   ├── __init__.py
│   │   ├── predictor.py
│   │   └── trainer.py
│   └── visualisation/    # Visualization tools
│       ├── __init__.py
│       └── visualiser.py
├── scripts/              # Training and prediction scripts
│   ├── download.py
│   ├── predict.py
│   └── train.py
├── configs/              # Configuration files
│   ├── __init__.py
│   ├── config.py
│   ├── default_config.yaml
│   └── gcp_config.yaml
├── tests/                # Unit and integration tests
│   ├── test_api.py
│   ├── test_downloader.py
│   └── test_loader.py
├── run.py                # Main application entry point
├── run_api.py            # API server entry point
└── requirements.txt      # Project dependencies
```

## Usage

### 1. Downloading Satellite Images

```python
from src.data.downloader import SatelliteDownloader
import asyncio

async def download_images():
    downloader = SatelliteDownloader()
    
    # Download a grid of images around a location
    images = await downloader.grid_download(
        latitude=40.4168,
        longitude=-3.7038,
        grid_size_km=0.5,
        num_images=16,
        save=True
    )
    
    # Or download a single image
    image = await downloader.single_download(
        latitude=40.4168,
        longitude=-3.7038,
        area_size=0.5,
        save=True
    )

# Run the async function
asyncio.run(download_images())
```

### 2. Using the API

```python
from run_api import SatelliteAPI
import asyncio

async def analyze_location():
    api = SatelliteAPI(mode='grid')
    
    # Process a location to get coordinates
    location = await api.process_location(
        country='United States',
        city='New York'
    )
    
    # Download satellite images
    images = await api.get_satellite_image(
        lat=location['coordinates']['lat'],
        lon=location['coordinates']['lon']
    )
    
    # Generate segmentation mask
    mask = await api.get_segmentation_mask(api.image_paths[0])
    
    # Get land type proportions
    proportions = await api.get_land_type_proportions(api.image_paths[0])
    
    # Find suitable locations for solar panels
    locations = await api.identify_suitable_locations(
        purpose='solar_panels',
        min_area_sqm=1000
    )
    
    print(f"Land proportions: {proportions['proportions']}")
    print(f"Found {len(locations['suitable_locations'])} suitable locations")

# Run the async function
asyncio.run(analyze_location())
```

### 3. Training the Model

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

### 4. Making Predictions

```python
from src.models.predictor import RegionPredictor, visualise_pred

# Visualize predictions on sample images
results = visualise_pred(
    model_path='data/models/v_1_0/model.pt',
    data_path='data/downloaded',
    n_samples=3
)

# Or use the RegionPredictor class for more control
predictor = RegionPredictor()
result = predictor.visualise('data/downloaded/sample_image.jpg')
```

## Model Architecture

The UNet architecture is specifically adapted for satellite imagery:

- Residual blocks for better gradient flow
- Focal loss for handling class imbalance
- Multi-scale feature extraction
- Skip connections for fine detail preservation

## Land Classification

The model classifies land into the following categories:

| Land Type | Description | Color |
|-----------|-------------|-------|
| Urban | Developed areas, buildings, roads | Cyan |
| Agricultural | Cropland, pastures | Yellow |
| Rangeland | Natural grasslands, shrublands | Magenta |
| Forest | Forests, woodlands | Green |
| Water | Lakes, rivers, oceans | Blue |
| Barren | Deserts, exposed rock | White |
| Unidentified | Clouds, shadows, unknown | Black |

## Testing

Run the test suite to verify functionality:

```bash
# Run all tests
python -m unittest discover tests

# Run specific test
python tests/test_api.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google Maps Static API for satellite imagery
- PyTorch for deep learning framework
- Google Cloud Storage for cloud integration

## Contact

Your Name - CJOWakefield@outlook.com
Project Link: [https://github.com/CJOWakefield/Developer.io](https://github.com/CJOWakefield/Developer.io)

---
