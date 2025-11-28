"""
Prepare videos for retraining queue
Takes original videos and their extracted frames, creates proper structure
"""

import json
import shutil
from pathlib import Path
import pandas as pd
import argparse
from tqdm import tqdm


def prepare_retrain_queue_from_images(
    image_folders_dir: str = "dataset/image",
    metadata_csv: str = "dataset/filtered_dataset.csv",
    output_dir: str = "data/retrain_queue",
    num_videos: int = 10,
    min_frames: int = 10
):
    """
    Prepare retraining queue from existing image folders
    
    Args:
        image_folders_dir: Directory containing image folders (e.g., ZWMDVKOU/)
        metadata_csv: CSV with video metadata (animal, action labels)
        output_dir: Where to create retrain queue
        num_videos: Number of videos to add to queue
        min_frames: Minimum frames required per video
    """
    
    print("="*60)
    print("PREPARING RETRAIN QUEUE FROM IMAGE FOLDERS")
    print("="*60)
    
    image_base = Path(image_folders_dir)
    output_base = Path(output_dir)
    
    # Clear existing queue
    if output_base.exists():
        print(f"Clearing existing queue: {output_base}")
        shutil.rmtree(output_base)
    
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Load metadata if available
    metadata_df = None
    if Path(metadata_csv).exists():
        print(f"Loading metadata from: {metadata_csv}")
        metadata_df = pd.read_csv(metadata_csv)
        print(f"  Found {len(metadata_df)} entries")
    else:
        print(f"⚠️  No metadata CSV found at: {metadata_csv}")
        print("   Will use folder names as video IDs")
    
    # Get all image folders
    all_folders = [d for d in image_base.iterdir() if d.is_dir()]
    print(f"\nFound {len(all_folders)} image folders")
    
    # Filter folders with enough frames
    valid_folders = []
    for folder in all_folders:
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        if len(images) >= min_frames:
            valid_folders.append(folder)
    
    print(f"Folders with >={min_frames} frames: {len(valid_folders)}")
    
    # Select random subset
    import random
    random.seed(42)
    selected_folders = random.sample(valid_folders, min(num_videos, len(valid_folders)))
    
    print(f"\nSelected {len(selected_folders)} folders for retraining")
    print("="*60)
    
    # Process each folder
    videos_added = 0
    
    for idx, folder in enumerate(tqdm(selected_folders, desc="Preparing videos")):
        folder_name = folder.name
        
        # Get frames
        image_files = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))
        
        # Get metadata for this video
        if metadata_df is not None:
            # Find entries for this video
            video_entries = metadata_df[metadata_df['video_id'] == folder_name]
            
            if len(video_entries) == 0:
                # Try finding by image path
                sample_path = str(image_files[0]) if len(image_files) > 0 else ""
                video_entries = metadata_df[metadata_df['image_path'].str.contains(folder_name)]
            
            if len(video_entries) > 0:
                # Get animal and action from first entry
                animal = video_entries.iloc[0]['animal']
                action = video_entries.iloc[0]['action']
            else:
                print(f"⚠️  No metadata found for {folder_name}, skipping...")
                continue
        else:
            # Default values if no metadata
            animal = "Unknown"
            action = "Unknown"
        
        # Create output directory
        video_id = f"video_{idx+1:04d}"
        output_folder = output_base / video_id
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Copy images
        for i, img_path in enumerate(image_files, 1):
            dest_path = output_folder / f"frame_{i:04d}{img_path.suffix}"
            shutil.copy2(img_path, dest_path)
        
        # Create metadata.json
        metadata = {
            "video_id": video_id,
            "original_folder": folder_name,
            "animal": animal,
            "action": action,
            "num_frames": len(image_files),
            "source": "original_dataset"
        }
        
        with open(output_folder / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        videos_added += 1
        
        # Print sample
        if idx < 3:
            print(f"\n  Sample {idx+1}:")
            print(f"    Folder: {folder_name}")
            print(f"    Video ID: {video_id}")
            print(f"    Animal: {animal}")
            print(f"    Action: {action}")
            print(f"    Frames: {len(image_files)}")
    
    print("\n" + "="*60)
    print(f"✅ RETRAIN QUEUE PREPARED")
    print("="*60)
    print(f"Videos added: {videos_added}")
    print(f"Output directory: {output_base}")
    print(f"Total frames: {sum(len(list(d.glob('frame_*.jpg'))) + len(list(d.glob('frame_*.png'))) for d in output_base.iterdir() if d.is_dir())}")
    print("="*60)
    print("\nNext step:")
    print(f"  cd scripts")
    print(f"  python retrain.py --epochs 5")
    print("="*60)
    
    return videos_added


def prepare_retrain_queue_from_videos(
    videos_dir: str = "dataset/video",
    output_dir: str = "data/retrain_queue",
    num_videos: int = 10
):
    """
    Prepare retraining queue from actual video files
    Extracts frames from videos
    
    Args:
        videos_dir: Directory containing video files (.mp4, .avi, etc.)
        output_dir: Where to create retrain queue
        num_videos: Number of videos to process
    """
    import cv2
    
    print("="*60)
    print("PREPARING RETRAIN QUEUE FROM VIDEO FILES")
    print("="*60)
    
    videos_base = Path(videos_dir)
    output_base = Path(output_dir)
    
    if not videos_base.exists():
        print(f"❌ Video directory not found: {videos_base}")
        return 0
    
    # Clear existing queue
    if output_base.exists():
        print(f"Clearing existing queue: {output_base}")
        shutil.rmtree(output_base)
    
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Get all video files
    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_files.extend(list(videos_base.glob(f'*{ext}')))
        video_files.extend(list(videos_base.glob(f'*{ext.upper()}')))
    
    print(f"Found {len(video_files)} video files")
    
    if len(video_files) == 0:
        print("❌ No video files found")
        return 0
    
    # Select random subset
    import random
    random.seed(42)
    selected_videos = random.sample(video_files, min(num_videos, len(video_files)))
    
    print(f"Selected {len(selected_videos)} videos")
    print("="*60)
    
    videos_added = 0
    
    for idx, video_path in enumerate(tqdm(selected_videos, desc="Extracting frames")):
        video_id = f"video_{idx+1:04d}"
        output_folder = output_base / video_id
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Extract frames
        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0
        extracted = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample every 2nd frame
            if frame_count % 2 == 0:
                # Save frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                output_path = output_folder / f"frame_{extracted+1:04d}.jpg"
                cv2.imwrite(str(output_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                extracted += 1
            
            frame_count += 1
            
            # Limit frames
            if extracted >= 50:
                break
        
        cap.release()
        
        if extracted == 0:
            print(f"⚠️  No frames extracted from {video_path.name}, skipping...")
            shutil.rmtree(output_folder)
            continue
        
        # Create metadata (you'll need to manually set animal/action)
        metadata = {
            "video_id": video_id,
            "original_file": video_path.name,
            "animal": "Unknown",  # TODO: Set manually or from filename
            "action": "Unknown",   # TODO: Set manually or from filename
            "num_frames": extracted,
            "source": "video_file"
        }
        
        with open(output_folder / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        videos_added += 1
        
        if idx < 3:
            print(f"\n  Sample {idx+1}:")
            print(f"    Video: {video_path.name}")
            print(f"    Frames extracted: {extracted}")
    
    print("\n" + "="*60)
    print(f"✅ RETRAIN QUEUE PREPARED")
    print("="*60)
    print(f"Videos processed: {videos_added}")
    print(f"Output directory: {output_base}")
    print("\n⚠️  NOTE: Edit metadata.json files to add correct animal/action labels!")
    print("="*60)
    
    return videos_added


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare retraining queue')
    parser.add_argument('--source', choices=['images', 'videos'], default='images',
                       help='Source type: images (from folders) or videos (from files)')
    parser.add_argument('--image-dir', default='../dataset/image',
                       help='Directory with image folders')
    parser.add_argument('--video-dir', default='../dataset/video',
                       help='Directory with video files')
    parser.add_argument('--metadata', default='../dataset/filtered_dataset.csv',
                       help='Metadata CSV file')
    parser.add_argument('--output', default='../data/retrain_queue',
                       help='Output directory for retrain queue')
    parser.add_argument('--num-videos', type=int, default=10,
                       help='Number of videos to add')
    parser.add_argument('--min-frames', type=int, default=10,
                       help='Minimum frames per video')
    
    args = parser.parse_args()
    
    if args.source == 'images':
        prepare_retrain_queue_from_images(
            image_folders_dir=args.image_dir,
            metadata_csv=args.metadata,
            output_dir=args.output,
            num_videos=args.num_videos,
            min_frames=args.min_frames
        )
    else:
        prepare_retrain_queue_from_videos(
            videos_dir=args.video_dir,
            output_dir=args.output,
            num_videos=args.num_videos
        )
