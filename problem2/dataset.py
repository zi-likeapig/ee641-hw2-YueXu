"""
Dataset loader for drum pattern generation task.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os

class DrumPatternDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        """
        Initialize drum pattern dataset.
        
        Args:
            data_dir: Path to drum dataset directory
            split: 'train' or 'val'
        """
        assert split in ('train', 'val')
        self.data_dir = data_dir
        self.split = split
        
        # Load patterns from patterns.npz 
        data_path = os.path.join(data_dir, 'patterns.npz')
        data = np.load(data_path)
        
        # Your data is already pre-split into train/val
        if split == 'train':
            self.patterns = data['train_patterns']
            self.styles = data['train_styles']
        else:
            self.patterns = data['val_patterns']
            self.styles = data['val_styles']
        
        # Standard instrument names (since you don't have metadata file)
        self.instrument_names = [
            "Kick", "Snare", "Closed Hi-hat", "Open Hi-hat",
            "Tom1", "Tom2", "Crash", "Ride", "Clap"
        ]
        self.style_names = [f"Style{i}" for i in range(5)]  # Assuming 5 styles (0-4)

    
    def __len__(self):
        return len(self.patterns)
    
    def __getitem__(self, idx):
        """
        Return a drum pattern sample.
        
        Returns:
            pattern: Binary tensor of shape [16, 9]
            style: Integer style label (0-4)
            density: Float indicating pattern density (for analysis)
        """
        pattern = self.patterns[idx]
        style = self.styles[idx]
        
        # Convert to tensor
        pattern_tensor = torch.from_numpy(pattern).float()
        
        # Compute density metric (fraction of active hits)
        density = pattern.sum() / (16 * 9)

        return pattern_tensor, style, density
    
    def pattern_to_pianoroll(self, pattern):
        """
        Convert pattern to visual piano roll representation.
        
        Args:
            pattern: Binary array [16, 9] or tensor
            
        Returns:
            pianoroll: Visual representation for plotting
        """
        if torch.is_tensor(pattern):
            pattern = pattern.cpu().numpy()
        
        # Create visual representation with instrument labels
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each active hit
        for t in range(16):
            for i in range(9):
                if pattern[t, i] > 0.5:
                    rect = patches.Rectangle((t, i), 1, 1, 
                                            linewidth=1, 
                                            edgecolor='black',
                                            facecolor='blue')
                    ax.add_patch(rect)
        
        # Add grid
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 9)
        ax.set_xticks(range(17))
        ax.set_yticks(range(10))
        ax.set_yticklabels([''] + self.instrument_names)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Instrument')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        
        return fig