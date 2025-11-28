"""
Data Preprocessing Module
Handles data loading, preprocessing, and augmentation for animal action recognition
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Dict
import random


class VideoSequenceDataset(Dataset):
    """
    Dataset class for video sequences stored as image folders
    """
    def __init__(self, root_dir: str, max_frames: int = 32, 
                 transform=None, labels_file: str = None):
        """
        Args:
            root_dir: Root directory containing video sequence folders
            max_frames: Maximum number of frames to load per sequence
            transform: Optional transform to apply to frames
            labels_file: Optional JSON file with labels for each sequence
        """
        self.root_dir = Path(root_dir)
        self.max_frames = max_frames
        self.transform = transform
        
        # Get all video sequence folders
        self.sequences = [d for d in self.root_dir.iterdir() if d.is_dir()]
        
        # Load labels if provided
        self.labels = None
        if labels_file and Path(labels_file).exists():
            with open(labels_file, 'r') as f:
                self.labels = json.load(f)
        
        print(f"Found {len(self.sequences)} video sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Load a video sequence and return frames + labels
        """
        sequence_folder = self.sequences[idx]
        
        # Load images
        frames = self._load_frames(sequence_folder)
        
        # Get labels if available
        if self.labels and sequence_folder.name in self.labels:
            animal_label = self.labels[sequence_folder.name]['animal']
            action_label = self.labels[sequence_folder.name]['action']
        else:
            animal_label = -1  # Unknown
            action_label = -1
        
        return {
            'frames': frames,
            'animal_label': animal_label,
            'action_label': action_label,
            'sequence_name': sequence_folder.name,
            'num_frames': len(frames)
        }
    
    def _load_frames(self, folder: Path) -> torch.Tensor:
        """Load and preprocess frames from folder"""
        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(folder.glob(f'*{ext}'))
        
        # Sort and sample frames
        image_files = sorted(image_files)[:self.max_frames]
        
        frames = []
        for img_path in image_files:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        
        return torch.stack(frames) if frames else torch.empty(0)


def get_data_transforms(augment: bool = False):
    """
    Get data transformation pipeline
    
    Args:
        augment: Whether to apply data augmentation
    
    Returns:
        torchvision transforms
    """
    if augment:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/test transforms without augmentation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def extract_frames_from_video(video_path: str, max_frames: int = 32, 
                              sample_rate: int = 2) -> List[Image.Image]:
    """
    Extract frames from a video file
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract
        sample_rate: Extract every Nth frame
    
    Returns:
        List of PIL Images
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_rate == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
        
        frame_count += 1
    
    cap.release()
    return frames


def save_frames_to_folder(frames: List[Image.Image], output_folder: str):
    """
    Save extracted frames to a folder
    
    Args:
        frames: List of PIL Images
        output_folder: Folder to save frames
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i, frame in enumerate(frames):
        frame.save(output_path / f"frame_{i:04d}.jpg")
    
    print(f"Saved {len(frames)} frames to {output_folder}")


def preprocess_uploaded_video(video_path: str, output_folder: str, 
                              max_frames: int = 32) -> str:
    """
    Preprocess an uploaded video for prediction or training
    
    Args:
        video_path: Path to uploaded video
        output_folder: Where to save extracted frames
        max_frames: Maximum frames to extract
    
    Returns:
        Path to folder containing extracted frames
    """
    # Extract frames
    frames = extract_frames_from_video(video_path, max_frames=max_frames)
    
    # Save to folder
    save_frames_to_folder(frames, output_folder)
    
    return output_folder


def collate_fn(batch):
    """
    Custom collate function for DataLoader to handle variable length sequences
    """
    # Filter out empty sequences
    batch = [item for item in batch if item['frames'].size(0) > 0]
    
    if len(batch) == 0:
        return None
    
    # Get max sequence length in batch
    max_len = max([item['num_frames'] for item in batch])
    
    # Pad sequences
    padded_frames = []
    seq_lengths = []
    animal_labels = []
    action_labels = []
    
    for item in batch:
        frames = item['frames']
        num_frames = frames.size(0)
        
        # Pad if necessary
        if num_frames < max_len:
            padding = torch.zeros(max_len - num_frames, *frames.shape[1:])
            frames = torch.cat([frames, padding], dim=0)
        
        padded_frames.append(frames)
        seq_lengths.append(num_frames)
        animal_labels.append(item['animal_label'])
        action_labels.append(item['action_label'])
    
    return {
        'frames': torch.stack(padded_frames),
        'seq_lengths': torch.tensor(seq_lengths),
        'animal_labels': torch.tensor(animal_labels),
        'action_labels': torch.tensor(action_labels)
    }


def create_dataloaders(train_dir: str, val_dir: str, batch_size: int = 8, 
                       num_workers: int = 4, max_frames: int = 32):
    """
    Create training and validation dataloaders
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        batch_size: Batch size
        num_workers: Number of worker threads
        max_frames: Maximum frames per sequence
    
    Returns:
        train_loader, val_loader
    """
    # Get transforms
    train_transform = get_data_transforms(augment=True)
    val_transform = get_data_transforms(augment=False)
    
    # Create datasets
    train_dataset = VideoSequenceDataset(
        train_dir, 
        max_frames=max_frames,
        transform=train_transform
    )
    
    val_dataset = VideoSequenceDataset(
        val_dir,
        max_frames=max_frames,
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    print("Preprocessing module ready!")