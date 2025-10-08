"""
GAN training implementation with mode collapse analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('..') 

from provided.metrics import mode_coverage_score
from provided.visualize import plot_alphabet_grid

def train_gan(generator, discriminator, data_loader, num_epochs=100, device='cuda',
              snapshot_epochs=(10, 30, 50, 100), snapshot_dir="results/visualizations/ckpts"):
    """
    Standard GAN training implementation.
    
    Uses vanilla GAN objective which typically exhibits mode collapse.
    
    Args:
        generator: Generator network
        discriminator: Discriminator network
        data_loader: DataLoader for training data
        num_epochs: Number of training epochs
        device: Device for computation
        
    Returns:
        dict: Training history and metrics
    """
    # Initialize optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Training history
    history = defaultdict(list)

    Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
    vis_dir = Path(snapshot_dir).parent / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(num_epochs):
        for batch_idx, (real_images, labels) in enumerate(data_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            if real_images.min().item() >= 0.0: 
                real_images =   real_images * 2.0 - 1.0
            
            # Labels for loss computation
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # ========== Train Discriminator ==========
            # TODO: Implement discriminator training step
            # 1. Zero gradients
            d_optimizer.zero_grad()
            # 2. Forward pass on real images
            real_outputs = discriminator(real_images)
            # 3. Compute real loss
            d_loss_real = criterion(real_outputs, real_labels)
            # 4. Generate fake images from random z
            z = torch.randn(batch_size, generator.z_dim).to(device)
            fake_images = generator(z)
            # 5. Forward pass on fake images (detached)
            fake_outputs = discriminator(fake_images.detach())
            # 6. Compute fake loss
            d_loss_fake = criterion(fake_outputs, fake_labels)
            # 7. Backward and optimize
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # ========== Train Generator ==========
            # TODO: Implement generator training step
            # 1. Zero gradients
            g_optimizer.zero_grad()
            # 2. Generate fake images
            z = torch.randn(batch_size, generator.z_dim).to(device)
            fake_images = generator(z)
            # 3. Forward pass through discriminator
            fake_outputs = discriminator(fake_images)
            # 4. Compute adversarial loss
            g_loss = criterion(fake_outputs, real_labels)   # what D to predict 1 for G's fake images
            # 5. Backward and optimize
            g_loss.backward()
            g_optimizer.step()
            
            # Log metrics
            if batch_idx % 10 == 0:
                history['d_loss'].append(d_loss.item())
                history['g_loss'].append(g_loss.item())
                history['epoch'].append(epoch + batch_idx/len(data_loader))
        
        # Analyze mode collapse every 10 epochs
        if epoch % 10 == 0:
            mode_coverage = analyze_mode_coverage(generator, device)
            history['mode_coverage'].append(mode_coverage)
            print(f"Epoch {epoch}: Mode coverage = {mode_coverage:.2f}")

        # Save model checkpoints and visualizations at specified epochs
        if (epoch + 1) in set(snapshot_epochs):
            torch.save(generator.state_dict(), Path(snapshot_dir) / f"G_{epoch+1:04d}.pth")
            try:
                fig = plot_alphabet_grid(generator, device=device, z_dim=generator.z_dim, seed=641)
                fig.savefig(vis_dir / f"alphabet_epoch_{epoch+1}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                print(f"[WARN] failed to save alphabet grid at epoch {epoch+1}: {e}")
    
    return history

def analyze_mode_coverage(generator, device, n_samples=1000):
    """
    Measure mode coverage by counting unique letters in generated samples.
    
    Args:
        generator: Trained generator network
        device: Device for computation
        n_samples: Number of samples to generate
        
    Returns:
        float: Coverage score (unique letters / 26)
    """
    # TODO: Generate n_samples images
    # Use provided letter classifier to identify generated letters
    # Count unique letters produced
    # Return coverage score (0 to 1)

    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, generator.z_dim, device=device)
        fake_images = generator(z)  # [N,1,28,28]
        fake_images01 = (fake_images + 1) / 2
        result = mode_coverage_score(fake_images01)
        return result["coverage_score"]
                

def visualize_mode_collapse(history, save_path):
    """
    Visualize mode collapse progression over training.
    
    Args:
        history: Training metrics dictionary
        save_path: Output path for visualization
    """
    # TODO: Plot mode coverage over time
    # Show which letters survive and which disappear

    plt.figure(figsize=(6,4))
    if "mode_coverage" in history:
        plt.plot(range(0, len(history["mode_coverage"])*10, 10),
                 history["mode_coverage"], marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Mode coverage")
        plt.title("Mode collapse analysis")
        plt.grid(True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()