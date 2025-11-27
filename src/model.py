import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Custom collate function to handle variable-length sequences
def collate_fn(batch):
    """
    Collate function to pad sequences to the same length in a batch
    """
    images_list, animals, actions, seq_lens = zip(*batch)
    
    # Find max sequence length in this batch
    max_len = max(seq_lens)
    
    # Pad all sequences to max_len
    padded_images = []
    for imgs in images_list:
        current_len = imgs.shape[0]
        if current_len < max_len:
            # Pad with zeros
            padding = torch.zeros(max_len - current_len, *imgs.shape[1:])
            imgs = torch.cat([imgs, padding], dim=0)
        padded_images.append(imgs)
    
    # Stack into batch
    images_batch = torch.stack(padded_images)
    animals_batch = torch.stack(animals)
    actions_batch = torch.stack(actions)
    seq_lens_batch = torch.tensor(seq_lens, dtype=torch.long)
    
    return images_batch, animals_batch, actions_batch, seq_lens_batch

# Dataset class
class AnimalBehaviorDataset(Dataset):
    def __init__(self, df, max_sequence_length=None, transform=None):
        self.df = df
        self.max_sequence_length = max_sequence_length  
        self.transform = transform
        self.image_groups = df.groupby('video_id')
        
        # Extract all unique animals and actions from the dataframe
        all_animals = df['animal'].unique().tolist()
        all_actions = df['action'].unique().tolist()
        
        # Create label mappings
        self.animal_to_idx = {animal: idx for idx, animal in enumerate(sorted(all_animals))}
        self.action_to_idx = {action: idx for idx, action in enumerate(sorted(all_actions))}
        
        self.idx_to_animal = {idx: animal for animal, idx in self.animal_to_idx.items()}
        self.idx_to_action = {idx: action for action, idx in self.action_to_idx.items()}
        
        print(f"Found {len(self.animal_to_idx)} unique animals")
        print(f"Found {len(self.action_to_idx)} unique actions")

    def __len__(self):
        return len(self.image_groups)

    def __getitem__(self, idx):
        folder_ids = list(self.image_groups.groups.keys())
        folder_id = folder_ids[idx % len(folder_ids)]
        group = self.image_groups.get_group(folder_id).sort_values('frame_number')
        
        # Get frames (all or up to max_sequence_length)
        if self.max_sequence_length:
            images_list = group.iloc[:self.max_sequence_length]
        else:
            images_list = group
        
        # Load images
        images = []
        for img_path in images_list['image_path']:
            try:
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
        
        # Stack images
        if len(images) > 0:
            images = torch.stack(images)
        else:
            # If no images loaded, create a dummy frame
            images = torch.zeros(1, 3, 224, 224)
        
        # Get labels
        animal_label = images_list['animal'].iloc[0]
        action_label = images_list['action'].iloc[0]
        
        # Convert to indices
        animal_idx = self.animal_to_idx[animal_label]
        action_idx = self.action_to_idx[action_label]
        
        # Convert to tensors
        animal = torch.tensor(animal_idx, dtype=torch.long)
        action = torch.tensor(action_idx, dtype=torch.long)
        
        # Return sequence length as well
        seq_len = images.shape[0]
        
        return images, animal, action, seq_len

# CNN + LSTM model with masking support
class CNNLSTM(nn.Module):
    def __init__(self, num_animals, num_actions, hidden_size=128, num_layers=2):
        super().__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, 
                           num_layers=num_layers, batch_first=True)
        self.fc_animal = nn.Linear(hidden_size, num_animals)
        self.fc_action = nn.Linear(hidden_size, num_actions)

    def forward(self, x, seq_lens=None):
        batch, seq_len, c, h, w = x.shape
        
        # Extract CNN features
        x = x.view(batch*seq_len, c, h, w)
        features = self.cnn(x)
        features = features.view(batch, seq_len, -1)
        
        # Use packed sequences for efficiency (optional but recommended)
        if seq_lens is not None:
            # Sort by length (required for pack_padded_sequence)
            seq_lens_sorted, perm_idx = seq_lens.sort(descending=True)
            features_sorted = features[perm_idx]
            
            # Pack sequences
            packed_features = pack_padded_sequence(
                features_sorted, 
                seq_lens_sorted.cpu(), 
                batch_first=True,
                enforce_sorted=True
            )
            
            # LSTM
            packed_output, (hidden, cell) = self.lstm(packed_features)
            
            # Unpack
            lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
            
            # Restore original order
            _, unperm_idx = perm_idx.sort()
            lstm_out = lstm_out[unperm_idx]
            
            # Get last valid output for each sequence
            idx = (seq_lens - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, lstm_out.size(2))
            lstm_last = lstm_out.gather(1, idx).squeeze(1)
        else:
            # Simple forward without packing
            lstm_out, _ = self.lstm(features)
            lstm_last = lstm_out[:, -1, :]
        
        animal_out = self.fc_animal(lstm_last)
        action_out = self.fc_action(lstm_last)
        
        return animal_out, action_out
