import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
import os
import sys
from model import CNNLSTM

class AnimalActionPredictor:
    def __init__(self, model_path="checkpoints/best_model.pth", device=None):
        
        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Extract model info
        self.num_animals = checkpoint['num_animals']
        self.num_actions = checkpoint['num_actions']
        self.animal_to_idx = checkpoint['animal_to_idx']
        self.action_to_idx = checkpoint['action_to_idx']
        self.idx_to_animal = checkpoint['idx_to_animal']
        self.idx_to_action = checkpoint['idx_to_action']
        
        # Initialize model
        self.model = CNNLSTM(
            num_animals=self.num_animals, 
            num_actions=self.num_actions,
            hidden_size=128,
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Define transforms (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded successfully!")
        print(f"Can predict {self.num_animals} animals and {self.num_actions} actions")
    
    def extract_frames_from_video(self, video_path, max_frames=32, sample_rate=1):
        """
        Extract frames from video file
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            sample_rate: Take every Nth frame (1 = all frames, 2 = every 2nd frame)
        
        Returns:
            List of PIL Images
        """
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        frame_count = 0
        
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_count % sample_rate == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
            
            frame_count += 1
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError("No frames extracted from video")
        
        print(f"Extracted {len(frames)} frames from video")
        return frames
    
    def load_images_from_folder(self, folder_path, max_frames=32):
        """
        Load images from a folder (for pre-extracted frames)
        
        Args:
            folder_path: Path to folder containing images
            max_frames: Maximum number of images to load
        
        Returns:
            List of PIL Images
        """
        folder = Path(folder_path)
        
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Get all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(folder.glob(f'*{ext}'))
            image_files.extend(folder.glob(f'*{ext.upper()}'))
        
        # Sort by filename
        image_files = sorted(image_files)[:max_frames]
        
        if len(image_files) == 0:
            raise ValueError(f"No images found in {folder_path}")
        
        # Load images
        frames = []
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert('RGB')
                frames.append(img)
            except Exception as e:
                print(f"Warning: Could not load {img_path}: {e}")
        
        print(f"Loaded {len(frames)} images from folder")
        return frames
    
    def preprocess_frames(self, frames):
        """
        Preprocess frames for model input
        
        Args:
            frames: List of PIL Images
        
        Returns:
            Tensor of shape (1, num_frames, 3, 224, 224)
        """
        processed_frames = []
        
        for frame in frames:
            # Apply transforms
            tensor = self.transform(frame)
            processed_frames.append(tensor)
        
        # Stack frames
        frames_tensor = torch.stack(processed_frames)
        
        # Add batch dimension
        frames_tensor = frames_tensor.unsqueeze(0)  # (1, num_frames, 3, 224, 224)
        
        return frames_tensor
    
    def predict(self, frames_tensor):
        """
        Make prediction on preprocessed frames
        
        Args:
            frames_tensor: Tensor of shape (1, num_frames, 3, 224, 224)
        
        Returns:
            Dictionary with prediction results
        """
        # Move to device
        frames_tensor = frames_tensor.to(self.device)
        
        # Get sequence length
        seq_len = torch.tensor([frames_tensor.shape[1]], dtype=torch.long).to(self.device)
        
        # Run inference
        with torch.no_grad():
            animal_logits, action_logits = self.model(frames_tensor, seq_len)
        
        # Get predictions
        animal_pred_idx = torch.argmax(animal_logits, dim=1).item()
        action_pred_idx = torch.argmax(action_logits, dim=1).item()
        
        # Get probabilities
        animal_probs = torch.softmax(animal_logits, dim=1)[0]
        action_probs = torch.softmax(action_logits, dim=1)[0]
        
        # Get top 5 predictions for each
        animal_top5_probs, animal_top5_indices = torch.topk(animal_probs, min(5, self.num_animals))
        action_top5_probs, action_top5_indices = torch.topk(action_probs, min(5, self.num_actions))
        
        # Format results
        result = {
            'animal': {
                'prediction': self.idx_to_animal[animal_pred_idx],
                'confidence': animal_probs[animal_pred_idx].item(),
                'top_5': [
                    {
                        'label': self.idx_to_animal[idx.item()],
                        'confidence': prob.item()
                    }
                    for idx, prob in zip(animal_top5_indices, animal_top5_probs)
                ]
            },
            'action': {
                'prediction': self.idx_to_action[action_pred_idx],
                'confidence': action_probs[action_pred_idx].item(),
                'top_5': [
                    {
                        'label': self.idx_to_action[idx.item()],
                        'confidence': prob.item()
                    }
                    for idx, prob in zip(action_top5_indices, action_top5_probs)
                ]
            }
        }
        
        return result
    
    def predict_from_video(self, video_path, max_frames=32, sample_rate=1):
        """
        End-to-end prediction from video file
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to use
            sample_rate: Frame sampling rate
        
        Returns:
            Dictionary with prediction results
        """
        print(f"Processing video: {video_path}")
        
        # Extract frames
        frames = self.extract_frames_from_video(video_path, max_frames, sample_rate)
        
        # Preprocess
        frames_tensor = self.preprocess_frames(frames)
        
        # Predict
        result = self.predict(frames_tensor)
        
        return result
    
    def predict_from_images(self, image_folder, max_frames=32):
        """
        End-to-end prediction from folder of images
        
        Args:
            image_folder: Path to folder containing images
            max_frames: Maximum frames to use
        
        Returns:
            Dictionary with prediction results
        """
        print(f"Processing images from: {image_folder}")
        
        # Load images
        frames = self.load_images_from_folder(image_folder, max_frames)
        
        # Preprocess
        frames_tensor = self.preprocess_frames(frames)
        
        # Predict
        result = self.predict(frames_tensor)
        
        return result
    
    def predict_from_frame_list(self, frames):
        """
        End-to-end prediction from list of PIL Images
        
        Args:
            frames: List of PIL Images
        
        Returns:
            Dictionary with prediction results
        """
        print(f"Processing {len(frames)} frames")
        
        # Preprocess
        frames_tensor = self.preprocess_frames(frames)
        
        # Predict
        result = self.predict(frames_tensor)
        
        return result


def print_results(result):
    """
    Pretty print prediction results
    """
    print(f"\n{'='*60}")
    print("PREDICTION RESULTS")
    print('='*60)
    
    print(f"\n🐾 Animal Prediction: {result['animal']['prediction']}")
    print(f"   Confidence: {result['animal']['confidence']:.2%}")
    
    print(f"\n🎬 Action Prediction: {result['action']['prediction']}")
    print(f"   Confidence: {result['action']['confidence']:.2%}")
    
    print("\n📊 Top 5 Animal Predictions:")
    for i, pred in enumerate(result['animal']['top_5'], 1):
        bar = '█' * int(pred['confidence'] * 20)
        print(f"  {i}. {pred['label']:<30} {bar} {pred['confidence']:.2%}")
    
    print("\n📊 Top 5 Action Predictions:")
    for i, pred in enumerate(result['action']['top_5'], 1):
        bar = '█' * int(pred['confidence'] * 20)
        print(f"  {i}. {pred['label']:<30} {bar} {pred['confidence']:.2%}")
    
    print('='*60 + '\n')


def main():
    """
    Command-line interface for predictions
    """
    # Initialize predictor
    predictor = AnimalActionPredictor(
        model_path="checkpoints/best_model.pth"
    )
    
    print()  # Add blank line after model loading
    
    # Check command line arguments
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        
        if os.path.isfile(input_path):
            # It's a video file
            print("="*60)
            print(f"Predicting from video: {input_path}")
            print("="*60)
            
            try:
                result = predictor.predict_from_video(input_path, max_frames=32, sample_rate=2)
                print_results(result)
            except Exception as e:
                print(f"Error processing video: {e}")
            
        elif os.path.isdir(input_path):
            # It's a folder of images
            print("="*60)
            print(f"Predicting from image folder: {input_path}")
            print("="*60)
            
            try:
                result = predictor.predict_from_images(input_path, max_frames=32)
                print_results(result)
            except Exception as e:
                print(f"Error processing images: {e}")
        else:
            print(f"❌ Error: Path not found: {input_path}")
            print_usage()
    else:
        print_usage()


def print_usage():
    """Print usage instructions"""
    print("Usage:")
    print("  python predict.py <video_file>        # Predict from video")
    print("  python predict.py <image_folder>      # Predict from image folder")
    print("\nExamples:")
    print("  python predict.py ../dataset/video/cat_jumping.mp4")
    print("  python predict.py ../dataset/frames/sequence_001/")
    print("  python predict.py /path/to/my/video.mp4")
    print("\nSupported video formats: .mp4, .avi, .mov, .mkv")
    print("Supported image formats: .jpg, .jpeg, .png, .bmp, .tiff")


if __name__ == "__main__":
    main()
