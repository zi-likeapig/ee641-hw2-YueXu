"""
Main training script for hierarchical VAE experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
from pathlib import Path
import numpy as np

from dataset import DrumPatternDataset
from hierarchical_vae import HierarchicalDrumVAE
from training_utils import kl_anneal_schedule



def compute_hierarchical_elbo(recon_x, x, mu_low, logvar_low, mu_high, logvar_high, 
                             model, z_high=None, beta_low=1.0, beta_high=0.2):
    """
    Compute Evidence Lower Bound (ELBO) for hierarchical VAE.
    
    ELBO = E[log p(x|z_low,z_high)] - beta * KL(q(z_low|x) || p(z_low|z_high)) 
           - beta * KL(q(z_high|x) || p(z_high))
    
    Args:
        recon_x: Reconstructed pattern logits [batch, 16, 9]
        x: Original patterns [batch, 16, 9]
        mu_low, logvar_low: Low-level latent parameters
        mu_high, logvar_high: High-level latent parameters
        model: Model instance to get conditional prior
        z_high: Sampled z_high for conditional prior computation
        beta: KL weight for beta-VAE
        
    Returns:
        loss: Total loss
        recon_loss: Reconstruction component
        kl_low: KL divergence for low-level latent
        kl_high: KL divergence for high-level latent
    """
    if recon_x.dim() == 4 and recon_x.shape[1] == 1:
        recon_x = recon_x.squeeze(1)
    if recon_x.shape[-2:] == (9, 16):
        recon_x = recon_x.permute(0, 2, 1).contiguous()

    if x.dim() == 4 and x.shape[1] == 1:
        x = x.squeeze(1)
    if x.shape[-2:] == (9, 16):
        x = x.permute(0, 2, 1).contiguous()
    x = x.float()

    if z_high is None:
        z_high = model.reparameterize(mu_high, logvar_high)

    recon_loss = F.binary_cross_entropy_with_logits(
        torch.flatten(recon_x, start_dim=1),
        torch.flatten(x,        start_dim=1),
        reduction='sum'
    ) / x.size(0)

    
    # KL_high: q(z_h|.) vs N(0,I)
    kl_high = -0.5 * (1 + logvar_high - mu_high.pow(2) - logvar_high.exp()).sum() / x.size(0)

    # KL_low: q(z_l|x) vs p(z_l|z_h)
    mu_p, logvar_p = model.cond_prior_low(z_high) 
    kl_low = 0.5 * (
        logvar_p - logvar_low
        + (logvar_low.exp() + (mu_low - mu_p).pow(2)) / logvar_p.exp()
        - 1
    ).sum() / x.size(0)

    
    free_bits = 0.01
    kl_low  = torch.max(kl_low, torch.tensor(free_bits, device=kl_low.device))
    kl_high = torch.max(kl_high, torch.tensor(free_bits, device=kl_high.device))


    total = recon_loss + beta_low * kl_low + beta_high * kl_high
    return total, recon_loss, kl_low, kl_high

def train_epoch(model, data_loader, optimizer, epoch, device, config):
    """
    Train model for one epoch with annealing schedules.
    
    Returns:
        Dictionary of average metrics for the epoch
    """
    model.train()
    
    # Metrics tracking
    metrics = {
        'total_loss': 0,
        'recon_loss': 0,
        'kl_low': 0,
        'kl_high': 0,
        'style_reg': 0
    }
    
    # Get annealing parameters for this epoch - more conservative approach
    # Start with small but non-zero beta values to prevent collapse
    beta_low  = 0.1 + 0.9 * min(1.0, epoch / 50.0)     # 从0.1开始，50 epoch慢慢到1.0
    beta_high = 0.05 + 0.15 * min(1.0, epoch / 60.0)    # 从0.05开始，60 epoch慢慢到0.2

    for batch_idx, (patterns, styles, densities) in enumerate(data_loader):
        patterns = patterns.to(device).float()
        # Convert styles to tensor if needed
        if not torch.is_tensor(styles):
            styles = torch.tensor(styles, dtype=torch.long).to(device)
        else:
            styles = styles.to(device).long()
        optimizer.zero_grad()
        
        # Forward pass
        recon, mu_h, logvar_h, mu_l, logvar_l = model(patterns)
        z_h = model.reparameterize(mu_h, logvar_h)
        
        # Compute simple ELBO loss
        loss, recon_loss, kl_low, kl_high = compute_hierarchical_elbo(
            recon, patterns, mu_l, logvar_l, mu_h, logvar_h,
            model, z_h, beta_low=beta_low, beta_high=beta_high
        )
        
        # Simple ELBO loss only (starter code style)
        total_loss = loss        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Update metrics
        metrics['total_loss'] += total_loss.item()
        metrics['recon_loss'] += recon_loss.item()
        metrics['kl_low'] += kl_low.item()
        metrics['kl_high'] += kl_high.item()
        metrics['style_reg'] += 0.0  # No style regularization
        

        # Log progress
        # if batch_idx % 10 == 0:
        #     print(f'Epoch {epoch:3d} [{batch_idx:3d}/{len(data_loader)}] '
        #           f'Loss: {total_loss.item()/len(patterns):.4f} '
        #           f'Beta: {beta:.3f} Temp: {temperature:.2f} '
        #           f'StyleReg: {style_reg_loss:.4f}' if isinstance(style_reg_loss, torch.Tensor) else f'StyleReg: {style_reg_loss:.4f}')
    
    # Average metrics
    n_samples = len(data_loader.dataset)
    for key in metrics:
        metrics[key] /= n_samples
    
    return metrics

def main():
    """
    Main training entry point for hierarchical VAE experiments.
    """
    # Configuration - Back to starter code settings
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'z_high_dim': 4,  # Back to original starter code
        'z_low_dim': 12,  # Back to original starter code  
        'kl_anneal_method': 'linear',
        'data_dir': '../data/drums',  # Back to relative path
        'checkpoint_dir': 'checkpoints',
        'results_dir': 'results'
    }

    
    # Create directories
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['results_dir']).mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset and dataloader
    train_dataset = DrumPatternDataset(config['data_dir'], split='train')
    val_dataset = DrumPatternDataset(config['data_dir'], split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    # Initialize model and optimizer
    model = HierarchicalDrumVAE(
        z_high_dim=config['z_high_dim'],
        z_low_dim=config['z_low_dim']
    ).to(config['device'])
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training history
    history = {
        'train': [],
        'val': [],
        'config': config
    }
    
    # Training loop
    for epoch in range(config['num_epochs']):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, epoch, 
            config['device'], config
        )
        history['train'].append(train_metrics)
        
        # Validate every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            val_metrics = {
                'total_loss': 0,
                'recon_loss': 0,
                'kl_low': 0,
                'kl_high': 0,
                'style_reg': 0
            }
            
            with torch.no_grad():
                for patterns, styles, densities in val_loader:
                    patterns = patterns.to(config['device']).float()
                    recon, mu_high, logvar_high, mu_low, logvar_low = model(patterns)
                    z_high = model.reparameterize(mu_high, logvar_high)
                    
                    # Simple validation
                    beta_low = 0.1 + 0.9 * min(1.0, epoch / 50.0)  # 从0.1开始，50 epoch慢慢到1.0
                    beta_high = 0.05 + 0.15 * min(1.0, epoch / 60.0)  # 从0.05开始，60 epoch慢慢到0.2
                    loss, recon_loss, kl_low, kl_high = compute_hierarchical_elbo(
                        recon, patterns, mu_low, logvar_low, mu_high, logvar_high,
                        model, z_high, beta_low=beta_low, beta_high=beta_high
                    )
                    
                    val_metrics['total_loss'] += loss.item()
                    val_metrics['recon_loss'] += recon_loss.item()
                    val_metrics['kl_low'] += kl_low.item()
                    val_metrics['kl_high'] += kl_high.item()
            
            # Average validation metrics
            n_val = len(val_dataset)
            for key in val_metrics:
                val_metrics[key] /= n_val
            
            history['val'].append(val_metrics)
            
            print(f"Epoch {epoch:3d} Validation - "
                  f"Loss: {val_metrics['total_loss']:.4f} "
                  f"KL_high: {val_metrics['kl_high']:.4f} "
                  f"KL_low: {val_metrics['kl_low']:.4f} "
                  f"beta_high: {beta_high:.4f} "
                  f"beta_low: {beta_low:.4f}")
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, f"{config['checkpoint_dir']}/checkpoint_epoch_{epoch+1}.pth")
    
    # Save final model and history
    torch.save(model.state_dict(), f"{config['results_dir']}/best_model.pth")

    history['config']['device'] = str(config['device'])
    with open(f"{config['results_dir']}/training_log.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Training complete. Results saved to {config['results_dir']}/")

if __name__ == '__main__':
    main()