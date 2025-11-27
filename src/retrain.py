import os
import json
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
import subprocess
import threading
from typing import Dict, Optional
from model import CNNLSTM, AnimalBehaviorDataset, collate_fn  # Now we know these exist!
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class RetrainingManager:
    """Manages model retraining process"""
    
    def __init__(self, base_dir: Path = None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        
        self.base_dir = base_dir
        self.retrain_queue_dir = base_dir / "data" / "retrain_queue"
        self.models_dir = base_dir / "models"
        self.checkpoints_dir = base_dir / "models" / "checkpoints" 
        self.status_file = base_dir / "monitoring" / "retrain_status.json"
        
        # Create directories if they don't exist
        self.retrain_queue_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"✓ Retraining manager initialized")
        print(f"  Base dir: {self.base_dir}")
        print(f"  Queue dir: {self.retrain_queue_dir}")
        print(f"  Checkpoints: {self.checkpoints_dir}")
    
    def prepare_retraining_data(self) -> Optional[pd.DataFrame]:
        """
        Prepare data from retraining queue
        Returns DataFrame with new training data
        """
        if not self.retrain_queue_dir.exists():
            print(f" Retrain queue directory not found: {self.retrain_queue_dir}")
            print(f"   Creating directory...")
            self.retrain_queue_dir.mkdir(parents=True, exist_ok=True)
            return None
        
        new_data = []
        
        # Read all videos in queue
        video_dirs = [d for d in self.retrain_queue_dir.iterdir() if d.is_dir()]
        
        if len(video_dirs) == 0:
            print(f"  No video directories in queue: {self.retrain_queue_dir}")
            print(f"   To add data for retraining:")
            print(f"   1. Create folder: {self.retrain_queue_dir}/video_001/")
            print(f"   2. Add images: frame_0001.jpg, frame_0002.jpg, etc.")
            print(f"   3. Add metadata.json with animal and action labels")
            return None
        
        print(f"Found {len(video_dirs)} video directories in queue")
        
        for video_dir in video_dirs:
            # Load metadata
            metadata_file = video_dir / "metadata.json"
            if not metadata_file.exists():
                print(f" Warning: No metadata for {video_dir.name}, skipping...")
                continue
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Get all images in this video directory
            image_files = sorted([
                f for f in video_dir.iterdir() 
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
            ])
            
            if len(image_files) == 0:
                print(f"  Warning: No images in {video_dir.name}, skipping...")
                continue
            
            print(f"  ✓ {video_dir.name}: {len(image_files)} frames")
            
            # Add each frame to dataset
            for idx, img_path in enumerate(image_files, 1):
                new_data.append({
                    'image_path': str(img_path),
                    'video_id': metadata.get('video_id', video_dir.name),
                    'frame_number': idx,
                    'animal': metadata['animal'],
                    'action': metadata['action'],
                    'type': 'retrain'
                })
        
        if len(new_data) == 0:
            print(" No valid data found in retrain queue")
            return None
        
        print(f"✓ Prepared {len(new_data)} frames from {len(video_dirs)} videos")
        return pd.DataFrame(new_data)
    
    def merge_with_existing_data(self, new_df: pd.DataFrame) -> pd.DataFrame:
        """Merge new data with existing training data"""
        # Load existing dataset
        existing_csv = self.base_dir / "dataset" / "filtered_dataset.csv"
        
        if existing_csv.exists():
            existing_df = pd.read_csv(existing_csv)
            
            # Change new data type to 'train'
            new_df['type'] = 'train'
            
            # Combine
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            print(f"✓ Combined: {len(existing_df)} existing + {len(new_df)} new = {len(combined_df)} total")
        else:
            print("  No existing dataset found, using only new data")
            new_df['type'] = 'train'
            combined_df = new_df
        
        return combined_df
    
    def update_status(self, status: str, progress: float = 0.0, **kwargs):
        """Update retraining status file"""
        status_data = {
            'status': status,
            'progress': progress,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        
        with open(self.status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
        
        print(f"Status: {status} ({progress*100:.0f}%)")
    
    def retrain_model(
        self, 
        epochs: int = 5, 
        batch_size: int = 2,
        learning_rate: float = 1e-4,
        max_seq_len: int = 32
    ) -> Dict:
        """
        Retrain model with new data
        Returns training results
        """
        try:
            print("\n" + "="*60)
            print("STARTING MODEL RETRAINING")
            print("="*60)
            
            self.update_status('running', progress=0.1, message='Preparing data...')
            
            # 1. Prepare new data
            new_df = self.prepare_retraining_data()
            if new_df is None:
                self.update_status('failed', error='No new data found')
                return {'success': False, 'error': 'No new data'}
            
            # 2. Merge with existing data
            combined_df = self.merge_with_existing_data(new_df)
            
            # 3. Filter to training data
            train_df = combined_df[combined_df['type'] == 'train'].reset_index(drop=True)
            
            # 4. Split train/val (80/20)
            train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
            split_idx = int(0.8 * len(train_df))
            train_subset = train_df.iloc[:split_idx].reset_index(drop=True)
            val_subset = train_df.iloc[split_idx:].reset_index(drop=True)
            
            print(f"✓ Training: {len(train_subset)} samples")
            print(f"✓ Validation: {len(val_subset)} samples")
            
            self.update_status('running', progress=0.2, message='Creating datasets...')
            
            # 5. Create datasets
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            train_dataset = AnimalBehaviorDataset(
                train_subset,
                max_sequence_length=max_seq_len,
                transform=transform
            )
            
            val_dataset = AnimalBehaviorDataset(
                val_subset,
                max_sequence_length=max_seq_len,
                transform=transform
            )
            
            # Share label mappings
            val_dataset.animal_to_idx = train_dataset.animal_to_idx
            val_dataset.action_to_idx = train_dataset.action_to_idx
            val_dataset.idx_to_animal = train_dataset.idx_to_animal
            val_dataset.idx_to_action = train_dataset.idx_to_action
            
            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                collate_fn=collate_fn
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                collate_fn=collate_fn
            )
            
            print(f"✓ Train batches: {len(train_loader)}")
            print(f"✓ Val batches: {len(val_loader)}")
            
            self.update_status('running', progress=0.3, message='Loading model...')
            
            # 6. Load existing model or create new one
            checkpoint_path = self.checkpoints_dir / "best_model.pth"
            
            num_animals = len(train_dataset.animal_to_idx)
            num_actions = len(train_dataset.action_to_idx)
            
            print(f"✓ Animals: {num_animals}, Actions: {num_actions}")
            
            model = CNNLSTM(
                num_animals=num_animals,
                num_actions=num_actions,
                hidden_size=128,
                num_layers=2
            )
            
            # Load existing weights if available
            if checkpoint_path.exists():
                print(f"✓ Loading existing model from: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                try:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print("✓ Loaded existing weights successfully")
                except Exception as e:
                    print(f"  Warning: Could not load weights: {e}")
                    print("   Training from scratch...")
            else:
                print(f"  No existing checkpoint found at: {checkpoint_path}")
                print("   Training from scratch...")
            
            model = model.to(self.device)
            
            # 7. Setup training
            criterion_animal = nn.CrossEntropyLoss()
            criterion_action = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            self.update_status('running', progress=0.4, message=f'Training for {epochs} epochs...')
            
            # 8. Training loop
            best_val_loss = float('inf')
            
            for epoch in range(epochs):
                print(f"\n{'='*60}")
                print(f"EPOCH {epoch+1}/{epochs}")
                print('='*60)
                
                # Training
                model.train()
                train_loss = 0
                train_batches = 0
                
                for images, animal_labels, action_labels, seq_lens in tqdm(
                    train_loader, 
                    desc=f"Training"
                ):
                    images = images.to(self.device)
                    animal_labels = animal_labels.to(self.device)
                    action_labels = action_labels.to(self.device)
                    seq_lens = seq_lens.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    animal_out, action_out = model(images, seq_lens)
                    
                    loss_animal = criterion_animal(animal_out, animal_labels)
                    loss_action = criterion_action(action_out, action_labels)
                    loss = loss_animal + loss_action
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_batches += 1
                
                avg_train_loss = train_loss / max(train_batches, 1)
                
                # Validation
                model.eval()
                val_loss = 0
                val_animal_correct = 0
                val_action_correct = 0
                val_total = 0
                val_batches = 0
                
                with torch.no_grad():
                    for images, animal_labels, action_labels, seq_lens in tqdm(val_loader, desc="Validation"):
                        images = images.to(self.device)
                        animal_labels = animal_labels.to(self.device)
                        action_labels = action_labels.to(self.device)
                        seq_lens = seq_lens.to(self.device)
                        
                        animal_out, action_out = model(images, seq_lens)
                        
                        loss_animal = criterion_animal(animal_out, animal_labels)
                        loss_action = criterion_action(action_out, action_labels)
                        loss = loss_animal + loss_action
                        
                        val_loss += loss.item()
                        val_batches += 1
                        
                        animal_pred = torch.argmax(animal_out, dim=1)
                        action_pred = torch.argmax(action_out, dim=1)
                        val_animal_correct += (animal_pred == animal_labels).sum().item()
                        val_action_correct += (action_pred == action_labels).sum().item()
                        val_total += animal_labels.size(0)
                
                avg_val_loss = val_loss / max(val_batches, 1)
                val_animal_acc = val_animal_correct / max(val_total, 1)
                val_action_acc = val_action_correct / max(val_total, 1)
                
                progress = 0.4 + (0.5 * (epoch + 1) / epochs)
                self.update_status(
                    'running',
                    progress=progress,
                    message=f'Epoch {epoch+1}/{epochs}',
                    train_loss=avg_train_loss,
                    val_loss=avg_val_loss,
                    val_animal_acc=val_animal_acc,
                    val_action_acc=val_action_acc
                )
                
                print(f"\nResults:")
                print(f"  Train Loss: {avg_train_loss:.4f}")
                print(f"  Val Loss: {avg_val_loss:.4f}")
                print(f"  Animal Accuracy: {val_animal_acc*100:.2f}%")
                print(f"  Action Accuracy: {val_action_acc*100:.2f}%")
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    
                    print(f"  ✓ New best model! Saving...")
                    
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'animal_to_idx': train_dataset.animal_to_idx,
                        'action_to_idx': train_dataset.action_to_idx,
                        'idx_to_animal': train_dataset.idx_to_animal,
                        'idx_to_action': train_dataset.idx_to_action,
                        'num_animals': num_animals,
                        'num_actions': num_actions,
                        'best_val_loss': best_val_loss,
                        'val_animal_acc': val_animal_acc,
                        'val_action_acc': val_action_acc,
                        'retrain_timestamp': datetime.now().isoformat()
                    }, checkpoint_path)
            
            # 9. Save model metadata
            metadata = {
                'version': f"retrained_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'timestamp': datetime.now().isoformat(),
                'num_animals': num_animals,
                'num_actions': num_actions,
                'val_animal_acc': val_animal_acc,
                'val_action_acc': val_action_acc,
                'training_samples': len(train_subset),
                'validation_samples': len(val_subset),
                'new_samples_added': len(new_df)
            }
            
            metadata_file = self.models_dir / "model_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.update_status(
                'completed',
                progress=1.0,
                message='Retraining completed successfully',
                animal_acc=val_animal_acc,
                action_acc=val_action_acc,
                model_path=str(checkpoint_path)
            )
            
            print("\n" + "="*60)
            print("✓ RETRAINING COMPLETED SUCCESSFULLY")
            print("="*60)
            print(f"Animal Accuracy: {val_animal_acc*100:.2f}%")
            print(f"Action Accuracy: {val_action_acc*100:.2f}%")
            print(f"Model saved to: {checkpoint_path}")
            print("="*60)
            
            return {
                'success': True,
                'animal_acc': val_animal_acc,
                'action_acc': val_action_acc,
                'model_path': str(checkpoint_path),
                'metadata': metadata
            }
        
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"\n ERROR during retraining:")
            print(error_msg)
            
            self.update_status(
                'failed',
                error=str(e),
                traceback=error_msg
            )
            
            return {
                'success': False,
                'error': str(e)
            }
    
    def clear_retrain_queue(self):
        """Clear the retraining queue after successful training"""
        import shutil
        
        if self.retrain_queue_dir.exists():
            for item in self.retrain_queue_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            print("✓ Cleared retrain queue")


def trigger_retraining(
    epochs: int = 5,
    batch_size: int = 2,
    async_mode: bool = True
) -> Dict:
    """
    Trigger model retraining
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        async_mode: If True, run in background thread
    
    Returns:
        Status dict
    """
    manager = RetrainingManager()
    
    if async_mode:
        # Run in background thread
        thread = threading.Thread(
            target=manager.retrain_model,
            kwargs={'epochs': epochs, 'batch_size': batch_size}
        )
        thread.daemon = True
        thread.start()
        
        return {
            'status': 'started',
            'message': 'Retraining started in background'
        }
    else:
        # Run synchronously
        return manager.retrain_model(epochs=epochs, batch_size=batch_size)


def check_retraining_status() -> Dict:
    """Check current retraining status"""
    status_file = Path(__file__).parent.parent / "monitoring" / "retrain_status.json"
    
    if status_file.exists():
        try:
            with open(status_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            return {
                'status': 'error',
                'error': f'Could not read status file: {e}'
            }
    
    return {
        'status': 'idle',
        'message': 'No retraining in progress'
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Retrain animal behavior model')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--async', dest='async_mode', action='store_true', help='Run in background')
    
    args = parser.parse_args()
    
    print("="*60)
    print("MODEL RETRAINING")
    print("="*60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Async mode: {args.async_mode}")
    print("="*60)
    
    result = trigger_retraining(
        epochs=args.epochs,
        batch_size=args.batch_size,
        async_mode=args.async_mode
    )
    
    print("\nResult:", json.dumps(result, indent=2))
