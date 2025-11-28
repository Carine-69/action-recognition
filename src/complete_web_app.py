"""
Complete Animal Action Recognition Web Application
- Model prediction
- Data visualizations
- Bulk upload and retraining
- Model monitoring
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
import tempfile
import os
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import threading

# Import custom modules
import sys
sys.path.append('.')
from model import CNNLSTM
from src.preprocessing import preprocess_uploaded_video, get_data_transforms, VideoSequenceDataset, collate_fn

# Global variables for monitoring
PREDICTION_LOG = []
MODEL_START_TIME = time.time()
RETRAIN_STATUS = {"status": "idle", "progress": 0, "message": ""}


class ModelMonitor:
    """Monitor model performance and uptime"""
    def __init__(self):
        self.start_time = time.time()
        self.prediction_count = 0
        self.total_inference_time = 0
        self.predictions_by_hour = defaultdict(int)
        self.errors = []
    
    def log_prediction(self, inference_time, success=True):
        self.prediction_count += 1
        self.total_inference_time += inference_time
        hour = datetime.now().strftime("%Y-%m-%d %H:00")
        self.predictions_by_hour[hour] += 1
        
        if not success:
            self.errors.append({
                'time': datetime.now().isoformat(),
                'message': 'Prediction failed'
            })
    
    def get_uptime(self):
        uptime_seconds = time.time() - self.start_time
        hours = int(uptime_seconds // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        seconds = int(uptime_seconds % 60)
        return f"{hours}h {minutes}m {seconds}s"
    
    def get_avg_response_time(self):
        if self.prediction_count == 0:
            return 0
        return self.total_inference_time / self.prediction_count
    
    def get_stats(self):
        return {
            'uptime': self.get_uptime(),
            'total_predictions': self.prediction_count,
            'avg_response_time': f"{self.get_avg_response_time():.3f}s",
            'predictions_per_hour': dict(self.predictions_by_hour),
            'error_count': len(self.errors)
        }


# Initialize monitor
monitor = ModelMonitor()


class AnimalActionSystem:
    """Complete system for animal action recognition"""
    
    def __init__(self, model_path="checkpoints/best_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing on {self.device}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.num_animals = checkpoint['num_animals']
        self.num_actions = checkpoint['num_actions']
        self.animal_to_idx = checkpoint['animal_to_idx']
        self.action_to_idx = checkpoint['action_to_idx']
        self.idx_to_animal = checkpoint['idx_to_animal']
        self.idx_to_action = checkpoint['idx_to_action']
        
        self.model = CNNLSTM(
            num_animals=self.num_animals,
            num_actions=self.num_actions,
            hidden_size=128
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("Model loaded!")
    
    def extract_frames_from_video(self, video_path, max_frames=32, sample_rate=2):
        """Extract frames from video"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        frame_count = 0
        
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def predict(self, video_file, max_frames=32, sample_rate=2):
        """Make prediction on video"""
        start_time = time.time()
        
        try:
            if video_file is None:
                return "Please upload a video", {}, {}
            
            # Extract frames
            frames = self.extract_frames_from_video(video_file, max_frames, sample_rate)
            
            if not frames:
                return "No frames extracted", {}, {}
            
            # Preprocess
            processed_frames = []
            for frame in frames:
                tensor = self.transform(frame)
                processed_frames.append(tensor)
            
            frames_tensor = torch.stack(processed_frames).unsqueeze(0).to(self.device)
            seq_len = torch.tensor([len(frames)], dtype=torch.long).to(self.device)
            
            # Predict
            with torch.no_grad():
                animal_logits, action_logits = self.model(frames_tensor, seq_len)
            
            animal_probs = torch.softmax(animal_logits, dim=1)[0]
            action_probs = torch.softmax(action_logits, dim=1)[0]
            
            animal_top5_probs, animal_top5_indices = torch.topk(animal_probs, 5)
            action_top5_probs, action_top5_indices = torch.topk(action_probs, 5)
            
            animal_preds = {
                self.idx_to_animal[idx.item()]: float(prob.item())
                for idx, prob in zip(animal_top5_indices, animal_top5_probs)
            }
            
            action_preds = {
                self.idx_to_action[idx.item()]: float(prob.item())
                for idx, prob in zip(action_top5_indices, action_top5_probs)
            }
            
            main_animal = self.idx_to_animal[animal_top5_indices[0].item()]
            main_action = self.idx_to_action[action_top5_indices[0].item()]
            
            inference_time = time.time() - start_time
            monitor.log_prediction(inference_time, success=True)
            
            result_text = f"🐾 **Animal:** {main_animal} ({animal_top5_probs[0].item():.1%})\n\n🎬 **Action:** {main_action} ({action_top5_probs[0].item():.1%})\n\n⏱️ **Inference Time:** {inference_time:.3f}s"
            
            return result_text, animal_preds, action_preds
        
        except Exception as e:
            monitor.log_prediction(0, success=False)
            return f"Error: {str(e)}", {}, {}
    
    def retrain_model(self, uploaded_files, epochs=5, batch_size=4, learning_rate=0.0001):
        """Retrain model with new data"""
        global RETRAIN_STATUS
        
        RETRAIN_STATUS = {"status": "running", "progress": 0, "message": "Starting retraining..."}
        
        try:
            if not uploaded_files or len(uploaded_files) == 0:
                RETRAIN_STATUS = {"status": "error", "progress": 0, "message": "No files uploaded"}
                return "No files uploaded for retraining"
            
            # Create temporary directory for new data
            temp_dir = Path(tempfile.mkdtemp())
            
            RETRAIN_STATUS["message"] = f"Processing {len(uploaded_files)} files..."
            
            # Process uploaded videos
            for i, file_path in enumerate(uploaded_files):
                folder_name = f"new_sequence_{i}"
                output_folder = temp_dir / folder_name
                
                try:
                    frames = self.extract_frames_from_video(file_path, max_frames=32)
                    output_folder.mkdir(parents=True, exist_ok=True)
                    
                    for j, frame in enumerate(frames):
                        frame.save(output_folder / f"frame_{j:04d}.jpg")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    continue
                
                progress = int((i + 1) / len(uploaded_files) * 30)
                RETRAIN_STATUS["progress"] = progress
            
            RETRAIN_STATUS["message"] = "Creating dataset..."
            
            # Create dataset
            transform = get_data_transforms(augment=True)
            dataset = VideoSequenceDataset(str(temp_dir), max_frames=32, transform=transform)
            
            if len(dataset) == 0:
                RETRAIN_STATUS = {"status": "error", "progress": 0, "message": "No valid sequences"}
                return "No valid sequences found"
            
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            
            RETRAIN_STATUS["message"] = "Training model..."
            
            # Set model to training mode
            self.model.train()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            criterion_animal = nn.CrossEntropyLoss()
            criterion_action = nn.CrossEntropyLoss()
            
            # Training loop
            for epoch in range(epochs):
                epoch_loss = 0
                batch_count = 0
                
                for batch in dataloader:
                    if batch is None:
                        continue
                    
                    frames = batch['frames'].to(self.device)
                    seq_lengths = batch['seq_lengths'].to(self.device)
                    
                    optimizer.zero_grad()
                    
                    animal_logits, action_logits = self.model(frames, seq_lengths)
                    
                    # For unsupervised retraining, use predictions as pseudo-labels
                    with torch.no_grad():
                        animal_pseudo_labels = torch.argmax(animal_logits, dim=1)
                        action_pseudo_labels = torch.argmax(action_logits, dim=1)
                    
                    loss_animal = criterion_animal(animal_logits, animal_pseudo_labels)
                    loss_action = criterion_action(action_logits, action_pseudo_labels)
                    loss = loss_animal + loss_action
                    
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                
                avg_loss = epoch_loss / max(batch_count, 1)
                progress = 30 + int((epoch + 1) / epochs * 60)
                RETRAIN_STATUS["progress"] = progress
                RETRAIN_STATUS["message"] = f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}"
            
            # Set back to eval mode
            self.model.eval()
            
            # Save retrained model
            RETRAIN_STATUS["message"] = "Saving model..."
            checkpoint_path = Path("checkpoints") / f"model_retrained_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            checkpoint_path.parent.mkdir(exist_ok=True)
            
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'num_animals': self.num_animals,
                'num_actions': self.num_actions,
                'animal_to_idx': self.animal_to_idx,
                'action_to_idx': self.action_to_idx,
                'idx_to_animal': self.idx_to_animal,
                'idx_to_action': self.idx_to_action
            }, checkpoint_path)
            
            RETRAIN_STATUS = {"status": "complete", "progress": 100, "message": f"Retraining complete! Model saved to {checkpoint_path.name}"}
            
            return f"✅ Retraining complete!\n\n📊 Trained on {len(dataset)} sequences\n🔄 {epochs} epochs completed\n💾 Model saved to {checkpoint_path.name}"
        
        except Exception as e:
            RETRAIN_STATUS = {"status": "error", "progress": 0, "message": str(e)}
            return f"❌ Error during retraining: {str(e)}"


# Initialize system
print("Loading model...")
system = AnimalActionSystem()


# Create visualizations
def create_dataset_visualizations():
    """Create dataset analysis visualizations"""
    data_root = Path("../dataset/image")
    
    if not data_root.exists():
        return None
    
    # Sample folders
    all_folders = [d for d in data_root.iterdir() if d.is_dir()]
    sample_folders = all_folders[:1000]
    
    # Count images per folder
    folder_sizes = []
    for folder in sample_folders:
        count = len(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))
        folder_sizes.append(count)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Distribution of sequence lengths
    axes[0, 0].hist(folder_sizes, bins=50, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Frames per Sequence')
    axes[0, 0].set_xlabel('Number of Frames')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Statistics
    stats_text = f"""
    Dataset Statistics:
    
    Total Sequences: {len(all_folders):,}
    Sampled: {len(sample_folders):,}
    
    Frames per Sequence:
    Mean: {np.mean(folder_sizes):.1f}
    Median: {np.median(folder_sizes):.1f}
    Min: {np.min(folder_sizes)}
    Max: {np.max(folder_sizes)}
    """
    axes[0, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center', family='monospace')
    axes[0, 1].axis('off')
    
    # 3. Box plot
    axes[1, 0].boxplot(folder_sizes, vert=True)
    axes[1, 0].set_title('Box Plot of Sequence Lengths')
    axes[1, 0].set_ylabel('Number of Frames')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Cumulative distribution
    sorted_sizes = np.sort(folder_sizes)
    cumulative = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)
    axes[1, 1].plot(sorted_sizes, cumulative, linewidth=2, color='green')
    axes[1, 1].set_title('Cumulative Distribution')
    axes[1, 1].set_xlabel('Number of Frames')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def get_model_status():
    """Get current model status"""
    stats = monitor.get_stats()
    
    status_text = f"""
    🟢 **Model Status: ONLINE**
    
    ⏱️ **Uptime:** {stats['uptime']}
    📊 **Total Predictions:** {stats['total_predictions']}
    ⚡ **Avg Response Time:** {stats['avg_response_time']}
    ❌ **Errors:** {stats['error_count']}
    """
    
    return status_text


def get_retrain_status():
    """Get retraining status"""
    status = RETRAIN_STATUS
    
    if status["status"] == "idle":
        return "⭕ No retraining in progress"
    elif status["status"] == "running":
        return f"🔄 Retraining... {status['progress']}%\n{status['message']}"
    elif status["status"] == "complete":
        return f"✅ {status['message']}"
    elif status["status"] == "error":
        return f"❌ Error: {status['message']}"


# Create Gradio interface
with gr.Blocks(title="Animal Action Recognition System") as demo:
    gr.Markdown("""
    # 🐾 Animal Action Recognition System
    ### Complete ML Pipeline with Training, Prediction, and Monitoring
    """)
    
    with gr.Tabs():
        # Tab 1: Prediction
        with gr.Tab("🎯 Prediction"):
            gr.Markdown("Upload a video to identify the animal and its action")
            
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Upload Video")
                    
                    with gr.Accordion("Settings", open=False):
                        max_frames_slider = gr.Slider(8, 64, value=32, step=1, label="Max Frames")
                        sample_rate_slider = gr.Slider(1, 5, value=2, step=1, label="Sample Rate")
                    
                    predict_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    prediction_output = gr.Markdown(label="Results")
                    
                    with gr.Row():
                        animal_output = gr.Label(label="Top 5 Animals", num_top_classes=5)
                        action_output = gr.Label(label="Top 5 Actions", num_top_classes=5)
            
            predict_btn.click(
                fn=system.predict,
                inputs=[video_input, max_frames_slider, sample_rate_slider],
                outputs=[prediction_output, animal_output, action_output]
            )
        
        # Tab 2: Data Visualizations
        with gr.Tab("📊 Visualizations"):
            gr.Markdown("### Dataset Analysis and Statistics")
            
            viz_btn = gr.Button("Generate Visualizations", variant="primary")
            viz_output = gr.Plot(label="Dataset Visualizations")
            
            viz_btn.click(fn=create_dataset_visualizations, inputs=[], outputs=[viz_output])
        
        # Tab 3: Retraining
        with gr.Tab("🔄 Retrain Model"):
            gr.Markdown("""
            ### Upload New Training Data
            Upload multiple video files to retrain the model with new data.
            """)
            
            upload_files = gr.File(label="Upload Videos (Multiple)", file_count="multiple", file_types=["video"])
            
            with gr.Row():
                epochs_slider = gr.Slider(1, 10, value=5, step=1, label="Epochs")
                batch_size_slider = gr.Slider(1, 16, value=4, step=1, label="Batch Size")
                lr_slider = gr.Number(value=0.0001, label="Learning Rate")
            
            retrain_btn = gr.Button("🚀 Start Retraining", variant="primary", size="lg")
            retrain_output = gr.Textbox(label="Retraining Status", lines=5)
            
            retrain_btn.click(
                fn=system.retrain_model,
                inputs=[upload_files, epochs_slider, batch_size_slider, lr_slider],
                outputs=[retrain_output]
            )
        
        # Tab 4: Monitoring
        with gr.Tab("📈 Monitoring"):
            gr.Markdown("### Model Performance and Uptime")
            
            status_output = gr.Markdown(value=get_model_status())
            retrain_status_output = gr.Markdown(value=get_retrain_status())
            
            refresh_btn = gr.Button("🔄 Refresh Status")
            
            refresh_btn.click(
                fn=lambda: (get_model_status(), get_retrain_status()),
                inputs=[],
                outputs=[status_output, retrain_status_output]
            )
    
    gr.Markdown("""
    ---
    **Model Info:** 799 Animals | 133 Actions | CNN-LSTM Architecture
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )