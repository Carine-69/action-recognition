from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import tempfile
import os
from pathlib import Path
from model import CNNLSTM
from typing import Optional
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Animal Action Recognition API",
    description="API for identifying animals and their actions in videos",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        
        result = {
            'animal': {
                'prediction': self.idx_to_animal[animal_pred_idx],
                'confidence': float(animal_probs[animal_pred_idx].item()),
                'top_5': [
                    {
                        'label': self.idx_to_animal[idx.item()],
                        'confidence': float(prob.item())
                    }
                    for idx, prob in zip(animal_top5_indices, animal_top5_probs)
                ]
            },
            'action': {
                'prediction': self.idx_to_action[action_pred_idx],
                'confidence': float(action_probs[action_pred_idx].item()),
                'top_5': [
                    {
                        'label': self.idx_to_action[idx.item()],
                        'confidence': float(prob.item())
                    }
                    for idx, prob in zip(action_top5_indices, action_top5_probs)
                ]
            }
        }
        
        return result
    
    def predict_from_video(self, video_path, max_frames=32, sample_rate=2):
        frames = self.extract_frames_from_video(video_path, max_frames, sample_rate)
        frames_tensor = self.preprocess_frames(frames)
        return self.predict(frames_tensor)


# Initialize predictor globally
print("Initializing predictor...")
predictor = AnimalActionPredictor(model_path="checkpoints/best_model.pth")


@app.get("/")
async def root():
    """API root endpoint with information"""
    return {
        "message": "Animal Action Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Upload video for prediction",
            "/health": "GET - Check API health",
            "/info": "GET - Get model information"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "device": str(predictor.device)
    }


@app.get("/info")
async def model_info():
    """Get model information"""
    return {
        "num_animals": predictor.num_animals,
        "num_actions": predictor.num_actions,
        "device": str(predictor.device),
        "animals": list(predictor.idx_to_animal.values())[:10],  # First 10
        "actions": list(predictor.idx_to_action.values())[:10],  # First 10
        "total_animals": len(predictor.idx_to_animal),
        "total_actions": len(predictor.idx_to_action)
    }


@app.post("/predict")
async def predict_video(
    video: UploadFile = File(..., description="Video file to analyze"),
    max_frames: int = Form(32, description="Maximum number of frames to process"),
    sample_rate: int = Form(2, description="Frame sampling rate")
):
    """
    Predict animal and action from uploaded video
    
    Parameters:
    - video: Video file (MP4, AVI, MOV, MKV)
    - max_frames: Maximum frames to process (default: 32)
    - sample_rate: Sample every Nth frame (default: 2)
    
    Returns:
    - JSON with animal and action predictions
    """
    
    # Validate file type
    if not video.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        # Write uploaded video to temp file
        content = await video.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Make prediction
        result = predictor.predict_from_video(
            tmp_file_path,
            max_frames=max_frames,
            sample_rate=sample_rate
        )
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    
    finally:
        # Clean up temp file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


@app.post("/predict-batch")
async def predict_batch(
    videos: list[UploadFile] = File(..., description="Multiple video files"),
    max_frames: int = Form(32),
    sample_rate: int = Form(2)
):
    """
    Predict animal and action from multiple videos
    
    Parameters:
    - videos: List of video files
    - max_frames: Maximum frames to process per video
    - sample_rate: Frame sampling rate
    
    Returns:
    - List of predictions for each video
    """
    results = []
    
    for video in videos:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            content = await video.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            result = predictor.predict_from_video(
                tmp_file_path,
                max_frames=max_frames,
                sample_rate=sample_rate
            )
            
            results.append({
                "filename": video.filename,
                "prediction": result
            })
        
        except Exception as e:
            results.append({
                "filename": video.filename,
                "error": str(e)
            })
        
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
    
    return JSONResponse(content={"results": results})


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
