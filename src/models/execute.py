import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import json
import gc
import time
import traceback
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

# Set up paths
base_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_directory)
model_directory = os.path.join(base_directory, 'data', 'models')

# Configure logging
logging.basicConfig(level=logging.WARNING)
for logger_name in ['src.data.loader', 'src.models.trainer']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Import required modules
from src.data.loader import SatelliteImages
from src.models.trainer import UNet, UNetConfig, Train, FocalLoss, SpatialConsistencyLoss, transformer
from src.models.predictor import RegionPredictor

# Load configuration
with open(os.path.join(base_directory, 'configs', 'default_config.yaml'), 'r') as file:
    config = yaml.safe_load(file)

# Set environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

def cleanup_resources() -> None:
    """
    Clean up resources to prevent memory leaks.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def train_model(epochs: int = 5, 
                model_params: Optional[Dict[str, Any]] = None, 
                loss_params: Optional[Dict[str, Any]] = None, 
                optimizer_params: Optional[Dict[str, Any]] = None, 
                training_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Train a model with customizable parameters.
    
    Args:
        epochs: Number of training epochs
        model_params: Dictionary of model parameters
        loss_params: Dictionary of loss function parameters
        optimizer_params: Dictionary of optimizer parameters
        training_params: Dictionary of training parameters
        
    Returns:
        Dictionary with training results
    """
    model_params = model_params or {}
    loss_params = loss_params or {}
    optimizer_params = optimizer_params or {}
    training_params = training_params or {}
    
    batch_size = training_params.get('batch_size', 16)
    
    try:
        config = UNetConfig(
            n_classes=model_params.get('n_classes', 7),
            input_dim=model_params.get('input_dim', 3),
            dropout=model_params.get('dropout', 0.1)
        )
        confidence_threshold = model_params.get('confidence_threshold', 0.3)
        model = UNet(config, confidence_threshold)
        
        training_data = SatelliteImages(os.path.join(base_directory, 'data', 'train'), transform=transformer)
        
        class_weights = loss_params.get('class_weights', None)
        if class_weights is None:
            class_weights = torch.ones(7)
            class_weights[1] = 1.0
            class_weights[0] = 1.0
            class_weights[2] = 1.0
            class_weights[3] = 1.0
        
        focal_loss = FocalLoss(
            gamma=loss_params.get('gamma', 2.0),
            class_weights=class_weights,
            temperature=loss_params.get('temperature', 1.2),
            entropy_weight=loss_params.get('entropy_weight', 0.05)
        )
        
        spatial_loss = SpatialConsistencyLoss(
            weight=loss_params.get('spatial_weight', 0.1),
            focal_loss=focal_loss
        )
        
        optimiser = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_params.get('lr', 0.001),
            weight_decay=optimizer_params.get('weight_decay', 0.01)
        )
        
        trainer = Train(
            model=model,
            data=training_data,
            optimiser=optimiser,
            loss=spatial_loss,
            epochs=epochs,
            batch_size=batch_size,
            distributed=False,
            local_rank=-1,
            patience=training_params.get('patience', 8),
            min_delta=training_params.get('min_delta', 0.0001)
        )
        
        results = trainer.train()
        
        return results
        
    except Exception as e:
        print(f"Error in train_model: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise
    finally:
        cleanup_resources()

def grid_search(test_image_paths: Optional[List[str]] = None, epochs: int = 10) -> List[Dict[str, Any]]:
    """
    Perform grid search over key model parameters and visualize results.
    
    Args:
        test_image_paths: List of paths to test images for visualization
        epochs: Number of epochs to train each model
        
    Returns:
        List of dictionaries containing results for each parameter combination
    """
    global model_directory
    
    if test_image_paths is None:
        test_image_paths = [
            'data/test/14397_sat.jpg',
            'data/test/26961_sat.jpg',
            'data/test/83692_sat.jpg'
        ]

    param_grid = {
        'confidence_threshold': [0.2, 0.3],
        'entropy_weight': [0.01, 0.05],
        'class_weights': [
            {'agricultural': 1.0, 'urban': 1.0, 'rangeland': 1.0, 'forest': 1.0},
            {'agricultural': 0.8, 'urban': 1.2, 'rangeland': 1.0, 'forest': 1.0},
        ]
    }
    
    results = []
    successful_models = 0
    
    n_combinations = (len(param_grid['confidence_threshold']) * 
                     len(param_grid['entropy_weight']) * 
                     len(param_grid['class_weights']))
    
    fig_rows = int(np.ceil(np.sqrt(n_combinations)))
    fig_cols = int(np.ceil(n_combinations / fig_rows))
    fig = plt.figure(figsize=(5*fig_cols, 5*fig_rows))
    
    combination_idx = 0
    
    models_dir = os.path.join(base_directory, 'data', 'models')
    existing_versions = [d for d in os.listdir(models_dir) if d.startswith('grid_search_v')]
    if existing_versions:
        version_numbers = []
        for v in existing_versions:
            try:
                version_str = v.replace('grid_search_v', '')
                version_str = version_str.replace('v', '')
                version_numbers.append(int(version_str))
            except ValueError:
                continue
                
        next_version = max(version_numbers) + 1 if version_numbers else 1
    else:
        next_version = 1
    
    grid_search_version = f"grid_search_v{next_version:02d}"
    grid_search_dir = os.path.join(models_dir, grid_search_version)
    os.makedirs(grid_search_dir, exist_ok=True)
    
    for conf_thresh in param_grid['confidence_threshold']:
        for ent_weight in param_grid['entropy_weight']:
            for class_weight_dict in param_grid['class_weights']:
                subfolder_name = f"ct{conf_thresh}_ew{ent_weight}_ag{class_weight_dict['agricultural']}"
                subfolder_path = os.path.join(grid_search_dir, subfolder_name)
                os.makedirs(subfolder_path, exist_ok=True)
                
                class_weights = torch.ones(7)
                class_weights[1] = class_weight_dict['agricultural']
                class_weights[0] = class_weight_dict['urban']
                class_weights[2] = class_weight_dict['rangeland']
                class_weights[3] = class_weight_dict['forest']
                
                model_params = {
                    'n_classes': 7,
                    'input_dim': 3,
                    'dropout': 0.2,
                    'confidence_threshold': conf_thresh
                }
                
                loss_params = {
                    'gamma': 3.0,
                    'temperature': 1.5,
                    'entropy_weight': ent_weight,
                    'spatial_weight': 0.15,
                    'class_weights': class_weights
                }
                
                optimizer_params = {
                    'lr': 0.003,
                    'weight_decay': 0.03
                }
                
                training_params = {
                    'batch_size': 16,
                    'patience': 6,
                    'min_delta': 0.0005,
                    'use_validation': True
                }
                
                try:
                    original_model_dir = model_directory
                    model_directory = subfolder_path
                    
                    training_results = train_model(
                        epochs=epochs,
                        model_params=model_params,
                        loss_params=loss_params,
                        optimizer_params=optimizer_params,
                        training_params=training_params
                    )
                    
                    model_directory = original_model_dir
                    
                    if training_results is None:
                        raise Exception("Training failed - no results returned")
                        
                    predictor = RegionPredictor(model_version=training_results['version'])
                    
                    test_fig, test_axes = plt.subplots(2, len(test_image_paths), figsize=(5*len(test_image_paths), 10))
                    if len(test_image_paths) == 1:
                        test_axes = [[test_axes[0]], [test_axes[1]]]
                    
                    prediction_results = []
                    for idx, test_image_path in enumerate(test_image_paths):
                        try:
                            raw_image = plt.imread(test_image_path)
                            test_axes[1][idx].imshow(raw_image)
                            test_axes[1][idx].set_title(f"Raw Image {idx+1}")
                            test_axes[1][idx].axis('off')
                            
                            test_image_tensor = predictor.tensor_from_file(test_image_path)
                            prediction_result = predictor.predict_from_tensor(test_image_tensor)
                            prediction_results.append(prediction_result)
                            
                            test_axes[0][idx].imshow(prediction_result['colored_mask'])
                            test_axes[0][idx].set_title(f"Prediction {idx+1}")
                            test_axes[0][idx].axis('off')
                            
                            plt.figure(figsize=(10, 10))
                            plt.imshow(prediction_result['colored_mask'])
                            plt.title(f"Test Image {idx+1}\nConfidence Threshold: {conf_thresh}\nEntropy Weight: {ent_weight}\nAgricultural Weight: {class_weight_dict['agricultural']}")
                            plt.axis('off')
                            plt.savefig(os.path.join(subfolder_path, f'prediction_{idx+1}.png'))
                            plt.close()
                            
                        except Exception as e:
                            print(f"Error making prediction for test image {idx+1}: {e}")
                            test_axes[0][idx].text(0.5, 0.5, f"Failed\n{os.path.basename(test_image_path)}", 
                                    horizontalalignment='center', verticalalignment='center')
                            test_axes[0][idx].axis('off')
                            test_axes[1][idx].text(0.5, 0.5, f"Failed\n{os.path.basename(test_image_path)}", 
                                    horizontalalignment='center', verticalalignment='center')
                            test_axes[1][idx].axis('off')
                    
                    plt.figure(test_fig.number)
                    plt.tight_layout()
                    plt.savefig(os.path.join(subfolder_path, 'all_predictions.png'))
                    plt.close()
                        
                    results.append({
                        'params': {
                            'confidence_threshold': conf_thresh,
                            'entropy_weight': ent_weight,
                            'class_weights': class_weight_dict
                        },
                        'training_results': training_results,
                        'model_version': training_results['version'],
                        'subfolder': subfolder_name,
                        'test_images': [os.path.basename(path) for path in test_image_paths]
                    })
                    
                    plt.figure(fig.number)
                    plt.subplot(fig_rows, fig_cols, combination_idx + 1)
                    plt.imshow(prediction_results[0]['colored_mask'])
                    plt.title(f"CT:{conf_thresh}\nEW:{ent_weight}\nAg:{class_weight_dict['agricultural']}")
                    plt.axis('off')
                    
                    successful_models += 1
                    
                except Exception as e:
                    print(f"Error training model with parameters: {e}")
                    print(f"Traceback: {traceback.format_exc()}")
                    plt.figure(fig.number)
                    plt.subplot(fig_rows, fig_cols, combination_idx + 1)
                    plt.text(0.5, 0.5, f"Training Failed\nCT:{conf_thresh}\nEW:{ent_weight}\nAg:{class_weight_dict['agricultural']}", 
                            horizontalalignment='center', verticalalignment='center')
                    plt.axis('off')
                    
                    results.append({
                        'params': {
                            'confidence_threshold': conf_thresh,
                            'entropy_weight': ent_weight,
                            'class_weights': class_weight_dict
                        },
                        'training_error': str(e),
                        'subfolder': subfolder_name
                    })
                    
                    intermediate_results_path = os.path.join(grid_search_dir, 'grid_search_intermediate_results.json')
                    with open(intermediate_results_path, 'w') as f:
                        json.dump(results, f, indent=4)
                
                combination_idx += 1
                
                cleanup_resources()
                time.sleep(1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(grid_search_dir, 'grid_search_results.png'))
    plt.close()
    
    results_path = os.path.join(grid_search_dir, 'grid_search_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Grid search completed. {successful_models}/{n_combinations} models successfully trained and predicted.")
    print(f"All results saved in {grid_search_dir}")
    return results

if __name__ == "__main__":
    grid_search(epochs=3) 