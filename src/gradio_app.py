import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
import tempfile
import os
from model import CNNLSTM

class AnimalActionPredictor:
    def __init__(self, model_path="checkpoints/best_model.pth", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Loading model on {self.device}...")
        
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
            hidden_size=128,
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("Model loaded successfully!")
    
    def extract_frames_from_video(self, video_path, max_frames=32, sample_rate=2):
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
        
        if len(frames) == 0:
            raise ValueError("No frames extracted from video")
        
        return frames
    
    def preprocess_frames(self, frames):
        processed_frames = []
        for frame in frames:
            tensor = self.transform(frame)
            processed_frames.append(tensor)
        
        frames_tensor = torch.stack(processed_frames)
        frames_tensor = frames_tensor.unsqueeze(0)
        return frames_tensor
    
    def predict(self, frames_tensor):
        frames_tensor = frames_tensor.to(self.device)
        seq_len = torch.tensor([frames_tensor.shape[1]], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            animal_logits, action_logits = self.model(frames_tensor, seq_len)
        
        animal_pred_idx = torch.argmax(animal_logits, dim=1).item()
        action_pred_idx = torch.argmax(action_logits, dim=1).item()
        
        animal_probs = torch.softmax(animal_logits, dim=1)[0]
        action_probs = torch.softmax(action_logits, dim=1)[0]
        
        animal_top5_probs, animal_top5_indices = torch.topk(animal_probs, min(5, self.num_animals))
        action_top5_probs, action_top5_indices = torch.topk(action_probs, min(5, self.num_actions))
        
        # Format for Gradio display
        animal_predictions = {
            self.idx_to_animal[idx.item()]: float(prob.item())
            for idx, prob in zip(animal_top5_indices, animal_top5_probs)
        }
        
        action_predictions = {
            self.idx_to_action[idx.item()]: float(prob.item())
            for idx, prob in zip(action_top5_indices, action_top5_probs)
        }
        
        main_prediction = f"🐾 **Animal:** {self.idx_to_animal[animal_pred_idx]} ({animal_probs[animal_pred_idx].item():.1%})\n\n🎬 **Action:** {self.idx_to_action[action_pred_idx]} ({action_probs[action_pred_idx].item():.1%})"
        
        return main_prediction, animal_predictions, action_predictions
    
    def predict_from_video(self, video_path, max_frames=32, sample_rate=2):
        frames = self.extract_frames_from_video(video_path, max_frames, sample_rate)
        frames_tensor = self.preprocess_frames(frames)
        return self.predict(frames_tensor)


# Initialize predictor
print("Initializing predictor...")
predictor = AnimalActionPredictor(model_path="checkpoints/best_model.pth")

def predict_video(video_file, max_frames, sample_rate):
    """Process uploaded video and return predictions"""
    try:
        if video_file is None:
            return "Please upload a video file", {}, {}
        
        # Gradio provides the file path directly
        main_pred, animal_preds, action_preds = predictor.predict_from_video(
            video_file, 
            max_frames=int(max_frames), 
            sample_rate=int(sample_rate)
        )
        
        return main_pred, animal_preds, action_preds
    
    except Exception as e:
        return f"Error processing video: {str(e)}", {}, {}


# Create Gradio interface
with gr.Blocks(title="Animal Action Recognition", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🐾 Animal Action Recognition System
        Upload a video to identify the animal and its action!
        
        **Model Info:** Can recognize 799 animals and 133 actions
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload Video")
            
            with gr.Accordion("Advanced Settings", open=False):
                max_frames = gr.Slider(
                    minimum=8, 
                    maximum=64, 
                    value=32, 
                    step=1,
                    label="Maximum Frames to Process"
                )
                sample_rate = gr.Slider(
                    minimum=1, 
                    maximum=5, 
                    value=2, 
                    step=1,
                    label="Frame Sample Rate (1=all frames, 2=every 2nd frame)"
                )
            
            predict_btn = gr.Button("🔍 Analyze Video", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            main_output = gr.Markdown(label="Main Prediction")
            
            with gr.Row():
                with gr.Column():
                    animal_output = gr.Label(label="Top 5 Animal Predictions", num_top_classes=5)
                with gr.Column():
                    action_output = gr.Label(label="Top 5 Action Predictions", num_top_classes=5)
    
    gr.Markdown(
        """
        ### 📝 Tips:
        - Supported formats: MP4, AVI, MOV, MKV
        - Shorter videos process faster
        - Increase sample rate for faster processing
        - Higher frame count = more accurate predictions (but slower)
        
        ### 🎯 Examples to Try:
        - Birds in flight
        - Mammals running or jumping
        - Animals eating or drinking
        """
    )
    
    # Example videos (if you have some)
    gr.Examples(
        examples=[
            # Add paths to example videos here if available
            # ["examples/bird_flying.mp4", 32, 2],
            # ["examples/cat_jumping.mp4", 32, 2],
        ],
        inputs=[video_input, max_frames, sample_rate],
        label="Example Videos"
    )
    
    # Connect the button
    predict_btn.click(
        fn=predict_video,
        inputs=[video_input, max_frames, sample_rate],
        outputs=[main_output, animal_output, action_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True to create a public link
        show_error=True
    )
