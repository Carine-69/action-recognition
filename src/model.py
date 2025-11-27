import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

# Dataset class
class AnimalBehaviorDataset(Dataset):
    def __init__(self, df, sequence_length=16, transform=None):
        self.df = df
        self.sequence_length = sequence_length
        self.transform = transform
        self.image_groups = df.groupby('video_id')
        
        # Extract all unique animals and actions from the dataframe
        all_animals = set()
        all_actions = set()
        
        for animals_list in df['list_animal']:
            if isinstance(animals_list, list):
                all_animals.update(animals_list)
        
        for actions_list in df['list_animal_action']:
            if isinstance(actions_list, list):
                all_actions.update(actions_list)
        
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
        images_list = group.iloc[:self.sequence_length]
        
        # Load images
        images = []
        for img_path in images_list['image_path']:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)
        images = torch.stack(images)
        
        # Get animals and actions lists
        animals_list = images_list['list_animal'].iloc[0]
        actions_list = images_list['list_animal_action'].iloc[0]
        
        # Multi-label encoding: create binary vectors
        animal_vector = torch.zeros(len(self.animal_to_idx))
        for animal in animals_list:
            if animal in self.animal_to_idx:
                animal_vector[self.animal_to_idx[animal]] = 1
        
        action_vector = torch.zeros(len(self.action_to_idx))
        for action in actions_list:
            if action in self.action_to_idx:
                action_vector[self.action_to_idx[action]] = 1
        
        return images, animal_vector, action_vector

# CNN + LSTM model
class CNNLSTM(nn.Module):
    def __init__(self, num_animals, num_actions, hidden_size=128, num_layers=2):
        super().__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc_animal = nn.Linear(hidden_size, num_animals)
        self.fc_action = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        batch, seq_len, c, h, w = x.shape
        x = x.view(batch*seq_len, c, h, w)
        features = self.cnn(x)
        features = features.view(batch, seq_len, -1)
        lstm_out, _ = self.lstm(features)
        lstm_last = lstm_out[:, -1, :]
        animal_out = self.fc_animal(lstm_last)
        action_out = self.fc_action(lstm_last)
        return animal_out, action_out
