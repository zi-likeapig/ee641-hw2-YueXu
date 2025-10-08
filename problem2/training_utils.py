"""
Training implementations for hierarchical VAE with posterior collapse prevention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

def kl_anneal_schedule(epoch, total_epochs=100, method="linear"):
    """
    KL annealing schedule for hierarchical VAE.
    Start with beta ≈ 0, gradually increase to 1.0
    Conservative approach to prevent z_high collapse.
    """
    if method == "linear":
        # Conservative linear: 0 -> 0.5 over total epochs (starter code style)
        t = min(1.0, max(0.0, epoch / max(1, total_epochs - 1)))
        return float(t)
    elif method == "cyclical":
        # Cyclical annealing with 4 cycles
        cycles = 4
        period = max(1, total_epochs // cycles)
        phase = (epoch % period) / period
        tri = 2 * phase if phase < 0.5 else 2 * (1 - phase)
        return float(0.5 * tri)
    elif method == "warmup":
        # Warm-up schedule: stay low for first 30% then increase
        if epoch < 0.3 * total_epochs:
            return 0.01
        else:
            t = (epoch - 0.3 * total_epochs) / (0.7 * total_epochs)
            return float(0.01 + 0.79 * t)
    else:
        # Default linear
        t = min(1.0, max(0.0, epoch / max(1, total_epochs - 1)))
        return float(0.8 * t)
    
def temperature_schedule(epoch, total_epochs=100, t_start=1.0, t_end=0.7):
    """Monotonic small decay of logits temperature"""
    t = min(1.0, max(0.0, epoch / max(1, total_epochs - 1)))
    return float(t_start + (t_end - t_start) * t)

def gaussian_kl(mu, logvar, reduce=None):
    """
    KL(q||p) for q=N(mu, diag(exp(logvar))) and p=N(0,I).
    Return per-dimension KL (no sum) if reduce=None.
    """
    # KL per dim: -0.5 * (1 + logvar - mu^2 - exp(logvar))
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    if reduce == "sum":
        return kl.sum()
    if reduce == "mean":
        return kl.mean()
    return kl  # [B, D]

def apply_free_bits(kl_per_dim, free_bits=0.5):
    """
    free_bits in nats (not bits). Clamp per-dim KL to at least `free_bits`.
    Accepts shape [B,D] or [D].
    """
    return torch.clamp(kl_per_dim, min=float(free_bits))

def _move_to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        patterns = batch[0]
    else:
        patterns = batch
    return patterns.to(device)



def train_hierarchical_vae(model, data_loader, num_epochs=100, device='cuda'):
    """
    Train hierarchical VAE with KL annealing and other tricks.
    
    Implements several techniques to prevent posterior collapse:
    1. KL annealing (gradual beta increase)
    2. Free bits (minimum KL per dimension)
    3. Temperature annealing for discrete outputs
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # KL annealing schedule (moved inside function like starter code)
    def kl_anneal_schedule_inner(epoch):
        """Conservative KL annealing to prevent z_high collapse."""
        t = min(1.0, max(0.0, epoch / max(1, num_epochs - 1)))
        return 0.5 * t  # Conservative: 0 → 0.5
    
    # Free bits threshold
    free_bits = 0.5  # Minimum nats per latent dimension
    
    history = defaultdict(list)
    
    for epoch in range(num_epochs):
        beta = kl_anneal_schedule_inner(epoch)
        
        for batch_idx, batch in enumerate(data_loader):
            # Handle both tuple and tensor data
            if isinstance(batch, (list, tuple)):
                patterns = batch[0]
            else:
                patterns = batch
            patterns = patterns.to(device)
            
            # Forward pass through hierarchical VAE
            recon_logits, mu_high, logvar_high, mu_low, logvar_low = model(patterns, beta=beta)
            
            # Compute reconstruction loss
            recon_loss = F.binary_cross_entropy_with_logits(
                recon_logits, patterns, reduction='sum'
            ) / patterns.size(0)
            
            # Compute KL divergences (both levels)
            kl_low = -0.5 * torch.sum(1 + logvar_low - mu_low.pow(2) - logvar_low.exp())
            kl_high = -0.5 * torch.sum(1 + logvar_high - mu_high.pow(2) - logvar_high.exp())
            
            # Apply free bits to prevent collapse
            kl_low = kl_low / patterns.size(0)  # per sample
            kl_high = kl_high / patterns.size(0)  # per sample
            
            # Apply free bits minimum
            kl_low = torch.clamp(kl_low, min=free_bits)
            kl_high = torch.clamp(kl_high, min=free_bits)
            
            # Total loss = recon_loss + beta * kl_loss
            total_loss = recon_loss + beta * (kl_low + kl_high)
            
            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        # Record epoch statistics
        history['train'].append({
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_low': kl_low.item(),
            'kl_high': kl_high.item(),
            'style_reg': 0.0  # Not used in this simplified version
        })
        
        # Log progress
        if epoch % 1 == 0:  # Log every epoch for short runs
            print(f'Epoch {epoch:3d} Loss: {total_loss.item():.4f} '
                  f'Beta: {beta:.3f} KL_high: {kl_high.item():.4f}')
    
    return history

@torch.no_grad()
def sample_diverse_patterns(model, n_styles=5, n_variations=10, device='cuda', temperature=0.8):
    """
    Generate diverse drum patterns using the hierarchy.
    
    TODO:
    1. Sample n_styles from z_high prior
    2. For each style, sample n_variations from conditional p(z_low|z_high)
    3. Decode to patterns
    4. Organize in grid showing style consistency
    """
    device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
    model.to(device)
    model.eval()

    out = []
    for _ in range(n_styles):
        z_high = torch.randn(1, model.z_high_dim, device=device)
        row = []
        for _ in range(n_variations):
            logits = model.decode_hierarchy(z_high, z_low=None)  # [1,16,9]
            probs = torch.sigmoid(logits)
            patt = (probs > 0.5).float().cpu()  # binarize for clean visualization
            row.append(patt.squeeze(0))         # [16,9]
        out.append(torch.stack(row, dim=0))      # [n_variations,16,9]
    return torch.stack(out, dim=0)               # [n_styles,n_variations,16,9]



@torch.no_grad()
def analyze_posterior_collapse(model, data_loader, device='cuda'):
    """
    Diagnose which latent dimensions are being used.
    
    TODO:
    1. Encode validation data
    2. Compute KL divergence per dimension
    3. Identify collapsed dimensions (KL ≈ 0)
    4. Return utilization statistics
    """
    device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
    model.to(device)
    model.eval()

    kl_low_all = []
    kl_high_all = []

    n_samples = 0
    for batch in data_loader:
        x = _move_to_device(batch, device)
        recon_logits, mu_h, logvar_h, mu_l, logvar_l = model(x)
        kl_low_all.append(gaussian_kl(mu_l, logvar_l, reduce=None).mean(dim=0))   # [z_low]
        kl_high_all.append(gaussian_kl(mu_h, logvar_h, reduce=None).mean(dim=0))  # [z_high]
        n_samples += x.size(0)

    kl_low_per_dim = torch.stack(kl_low_all, dim=0).mean(dim=0).cpu().numpy()     # [z_low]
    kl_high_per_dim = torch.stack(kl_high_all, dim=0).mean(dim=0).cpu().numpy()   # [z_high]

    thresh = 1e-2
    collapsed_low_idx = np.where(kl_low_per_dim < thresh)[0].tolist()
    collapsed_high_idx = np.where(kl_high_per_dim < thresh)[0].tolist()

    return {
        "kl_low_per_dim": kl_low_per_dim.tolist(),
        "kl_high_per_dim": kl_high_per_dim.tolist(),
        "collapsed_low_idx": collapsed_low_idx,
        "collapsed_high_idx": collapsed_high_idx,
        "threshold": float(thresh),
    }