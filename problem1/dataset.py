"""
Dataset loader for font generation task.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
import numpy as np

class FontDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        """
        Initialize the font dataset.
        
        Args:
            data_dir: Path to font dataset directory
            split: 'train' or 'val'
        """
        assert split in ('train', 'val')
        self.data_dir = data_dir
        self.split = split
        
        # TODO: Load metadata from fonts_metadata.json
        # Expected structure:
        # {
        #   "train": [{"path": "A/font1_A.png", "letter": "A", "font": 1}, ...],
        #   "val": [...]
        # }
        metadata_path = os.path.join(data_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        key_map = {'train': 'train_samples', 'val': 'val_samples'}
        key = key_map[split]
        if key not in metadata:
            raise KeyError(f"Split '{split}' not found in metadata. Available keys: {list(metadata.keys())}")
        self.samples = metadata[key]

        if os.path.basename(self.data_dir) == 'fonts':
            self.fonts_root = os.path.join(self.data_dir, split)
        else:
            self.fonts_root = os.path.join(self.data_dir, 'fonts', split)

        if not os.path.isdir(self.fonts_root):
            raise FileNotFoundError(f"Fonts split folder not found: {self.fonts_root}")


    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Return a sample from the dataset.
        
        Returns:
            image: Tensor of shape [1, 28, 28]
            letter_id: Integer 0-25 representing A-Z
        """
        sample = self.samples[idx]
        
        # TODO: Load and process image
        # 1. Load image from sample['path']
        # 2. Convert to grayscale if needed
        # 3. Resize to 28x28 if needed
        # 4. Normalize to [0, 1]
        # 5. Convert to tensor

        filename = sample["filename"]
        label_id = int(sample['letter_idx'])

        image_path = os.path.join(self.fonts_root, filename)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path).convert('L').resize((28, 28))
        arr = np.asarray(img, dtype=np.float32) / 255.0 # [0,1]
        arr = arr * 2.0 - 1.0                           # [-1,1]
        x = torch.from_numpy(arr).unsqueeze(0)

        return x, label_id