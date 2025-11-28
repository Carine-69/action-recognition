"""
Prediction Module
Handles model loading and inference for animal action recognition
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple
import json
from PIL import Image
from torchvision import transforms
import time


class AnimalActionPredictor:
    """
    Predictor class for animal action recognition
    """
    
    def __init__(self, model_path: str = "checkpoints/best_model.pth", 
                 device: str = None):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Initializing predictor on {self.device}")
        
        # Load model
        self._load_model(model_path)
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Statistics
        self.prediction_count = 0
        self.total_inference_time = 0.0
    
    def _load_model(self, model_path: str):
        """Load model from checkpoint"""
        from model import CNNLSTM
        
        checkpoint = torch.load(model_path, map_location=self.device, 
                               weights_only=False)
        
        # Extract model configuration
        self.num_animals = checkpoint['num_animals']
        self.num_actions = checkpoint['num_actions']
        self.animal_to_idx = checkpoint['animal_to_idx']
        self.action_to_idx = checkpoint['action_to_idx']
        self.idx_to_animal = checkpoint['idx_to_animal']
        self.idx_to_action = checkpoint['idx_to_action']
        
        # Initialize and load model
        self.model = CNNLSTM(
            num_animals=self.num_animals,
            num_actions=self.num_actions,
            hidden_size=128
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded: {self.num_animals} animals, {self.num_actions} actions")
    
    def preprocess_frames(self, frames: List[Image.Image]) -> torch.Tensor:
        """
        Preprocess frames for model input
        
        Args:
            frames: List of PIL Images
        
        Returns:
            Tensor of shape (1, num_frames, 3, 224, 224)
        """
        processed_frames = []
        for frame in frames:
            tensor = self.transform(frame)
            processed_frames.append(tensor)
        
        frames_tensor = torch.stack(processed_frames)
        frames_tensor = frames_tensor.unsqueeze(0)  # Add batch dimension
        
        return frames_tensor
    
    def predict(self, frames: List[Image.Image], top_k: int = 5) -> Dict:
        """
        Make prediction on video frames
        
        Args:
            frames: List of PIL Images
            top_k: Number of top predictions to return
        
        Returns:
            Dictionary with predictions and confidence scores
        """
        start_time = time.time()
        
        # Preprocess frames
        frames_tensor = self.preprocess_frames(frames)
        frames_tensor = frames_tensor.to(self.device)
        
        # Get sequence length
        seq_len = torch.tensor([frames_tensor.shape[1]], dtype=torch.long).to(self.device)
        
        # Run inference
        with torch.no_grad():
            animal_logits, action_logits = self.model(frames_tensor, seq_len)
        
        # Get predictions
        animal_probs = torch.softmax(animal_logits, dim=1)[0]
        action_probs = torch.softmax(action_logits, dim=1)[0]
        
        # Get top-k predictions
        animal_topk_probs, animal_topk_indices = torch.topk(
            animal_probs, min(top_k, self.num_animals)
        )
        action_topk_probs, action_topk_indices = torch.topk(
            action_probs, min(top_k, self.num_actions)
        )
        
        # Calculate inference time
        inference_time = time.time() - start_time
        self.prediction_count += 1
        self.total_inference_time += inference_time
        
        # Format results
        result = {
            'animal': {
                'prediction': self.idx_to_animal[animal_topk_indices[0].item()],
                'confidence': float(animal_topk_probs[0].item()),
                'top_k': [
                    {
                        'label': self.idx_to_animal[idx.item()],
                        'confidence': float(prob.item())
                    }
                    for idx, prob in zip(animal_topk_indices, animal_topk_probs)
                ]
            },
            'action': {
                'prediction': self.idx_to_action[action_topk_indices[0].item()],
                'confidence': float(action_topk_probs[0].item()),
                'top_k': [
                    {
                        'label': self.idx_to_action[idx.item()],
                        'confidence': float(prob.item())
                    }
                    for idx, prob in zip(action_topk_indices, action_topk_probs)
                ]
            },
            'inference_time': inference_time,
            'num_frames': len(frames)
        }
        
        return result
    
    def predict_from_folder(self, folder_path: str, max_frames: int = 32, 
                           top_k: int = 5) -> Dict:
        """
        Predict from a folder of images
        
        Args:
            folder_path: Path to folder containing images
            max_frames: Maximum frames to load
            top_k: Number of top predictions
        
        Returns:
            Prediction results
        """
        folder = Path(folder_path)
        
        # Load images
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(folder.glob(f'*{ext}'))
        
        image_files = sorted(image_files)[:max_frames]
        
        frames = []
        for img_path in image_files:
            img = Image.open(img_path).convert('RGB')
            frames.append(img)
        
        if not frames:
            raise ValueError(f"No images found in {folder_path}")
        
        return self.predict(frames, top_k=top_k)
    
    def get_statistics(self) -> Dict:
        """Get prediction statistics"""
        avg_time = (self.total_inference_time / self.prediction_count 
                   if self.prediction_count > 0 else 0)
        
        return {
            'total_predictions': self.prediction_count,
            'total_inference_time': self.total_inference_time,
            'average_inference_time': avg_time,
            'predictions_per_second': 1 / avg_time if avg_time > 0 else 0
        }
    
    def reset_statistics(self):
        """Reset prediction statistics"""
        self.prediction_count = 0
        self.total_inference_time = 0.0


def batch_predict(predictor: AnimalActionPredictor, 
                 folder_paths: List[str], 
                 max_frames: int = 32) -> List[Dict]:
    """
    Make predictions on multiple folders
    
    Args:
        predictor: AnimalActionPredictor instance
        folder_paths: List of folder paths
        max_frames: Maximum frames per video
    
    Returns:
        List of prediction results
    """
    results = []
    
    for folder_path in folder_paths:
        try:
            result = predictor.predict_from_folder(folder_path, max_frames)
            result['folder'] = str(folder_path)
            results.append(result)
        except Exception as e:
            results.append({
                'folder': str(folder_path),
                'error': str(e)
            })
    
    return results


def save_predictions(predictions: List[Dict], output_path: str):
    """
    Save predictions to JSON file
    
    Args:
        predictions: List of prediction dictionaries
        output_path: Path to save JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Predictions saved to {output_path}")


def load_predictions(input_path: str) -> List[Dict]:
    """
    Load predictions from JSON file
    
    Args:
        input_path: Path to JSON file
    
    Returns:
        List of prediction dictionaries
    """
    with open(input_path, 'r') as f:
        predictions = json.load(f)
    
    return predictions


if __name__ == "__main__":
    print("Prediction module ready!")