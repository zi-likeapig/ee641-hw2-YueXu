"""
Hierarchical VAE for drum pattern generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalDrumVAE(nn.Module):
    def __init__(self, z_high_dim=4, z_low_dim=12):
        """
        Two-level VAE for drum patterns.
        
        The architecture uses a hierarchy of latent variables where z_high
        encodes style/genre information and z_low encodes pattern variations.
        
        Args:
            z_high_dim: Dimension of high-level latent (style)
            z_low_dim: Dimension of low-level latent (variation)
        """
        super().__init__()
        self.z_high_dim = z_high_dim
        self.z_low_dim = z_low_dim
        
        # Encoder: pattern → z_low → z_high
        # We use 1D convolutions treating the pattern as a sequence
        self.encoder_low = nn.Sequential(
            nn.Conv1d(9, 32, kernel_size=3, padding=1),  # [16, 9] → [16, 32]
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # → [8, 64]
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),  # → [4, 128]
            nn.ReLU(),
            nn.Flatten()  # → [512]
        )

        # 新增：低层的条件先验 p(z_low | z_high)
        self.prior_low = nn.Sequential(
            nn.Linear(z_high_dim, 64), nn.ReLU(),
        )
        self.prior_mu_low     = nn.Linear(64, z_low_dim)
        self.prior_logvar_low = nn.Linear(64, z_low_dim)

        # Low-level latent parameters
        self.fc_mu_low = nn.Linear(512, z_low_dim)
        self.fc_logvar_low = nn.Linear(512, z_low_dim)

        self.zlow_dropout = nn.Dropout(p=0.25)  # Dropout for z_low during training
        
        # Encoder from z_low to z_high (simpler architecture)
        self.encoder_high = nn.Sequential(
            nn.Linear(z_low_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # High-level latent parameters
        self.fc_mu_high = nn.Linear(32, z_high_dim)
        self.fc_logvar_high = nn.Linear(32, z_high_dim)
        
        # Decoder: z_high → z_low → pattern  
        # Simple decoder structure matching starter code
        self.decoder = nn.Sequential(
            nn.Linear(z_high_dim + z_low_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        
        # Reshape and deconv layers
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),  # 4→8
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),   # 8→16
            nn.ReLU(),
            nn.Conv1d(32, 9, kernel_size=3, padding=1)                       # → 9 channels
        )
    
    def cond_prior_low(self, z_high):
        h = self.prior_low(z_high)
        mu_p = self.prior_mu_low(h)
        logvar_p = self.prior_logvar_low(h).clamp(min=-6., max=6.)
        return mu_p, logvar_p


    def encode_hierarchy(self, x):
        """
        Encode pattern to both latent levels.
        
        Args:
            x: Drum patterns [batch_size, 16, 9]
            
        Returns:
            mu_high, logvar_high: Parameters for q(z_high|z_low)  
            mu_low, logvar_low: Parameters for q(z_low|x)
        """
        # Reshape for Conv1d: [batch, 16, 9] → [batch, 9, 16]
        x = x.transpose(1, 2).float()
        
        # Encode to z_low parameters
        h = self.encoder_low(x)
        mu_low = self.fc_mu_low(h)
        logvar_low = self.fc_logvar_low(h)
        
        # Sample z_low using reparameterization
        z_low = self.reparameterize(mu_low, logvar_low)
        
        # Encode z_low to z_high parameters (following starter structure)
        h_high = self.encoder_high(z_low)
        mu_high = self.fc_mu_high(h_high)
        logvar_high = self.fc_logvar_high(h_high)

        return mu_high, logvar_high, mu_low, logvar_low
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for sampling.
        
        TODO: Implement
        z = mu + eps * std where eps ~ N(0,1)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode_hierarchy(self, z_high, z_low=None, temperature=0.6):
        """
        Decode from latent variables to pattern.
        
        Args:
            z_high: High-level latent code
            z_low: Low-level latent code (if None, sample from standard prior)
            
        Returns:
            pattern_logits: Logits for binary pattern [batch, 16, 9]
        """
        # If z_low is None, sample from standard prior p(z_low) = N(0,I)
        if z_low is None:
            mu_p, logvar_p = self.cond_prior_low(z_high)
            std_p = torch.exp(0.5 * logvar_p)
            eps = torch.randn_like(std_p)
            z_low = mu_p + eps * std_p * temperature  # [B, z_low_dim]
        if self.training:
            z_low = self.zlow_dropout(z_low)
        
        # Combine z_high and z_low
        z_combined = torch.cat([z_high, z_low], dim=1)
        
        # Decode to pattern logits
        h = self.decoder(z_combined)  # [B, 512]
        h = h.view(h.size(0), 128, 4)  # [B, 128, 4]
        
        # Deconvolution to get [B, 9, 16]
        h = self.deconv(h)  # [B, 9, 16]
        
        # Transpose to [B, 16, 9]
        logits = h.transpose(1, 2)
            
        return logits
    
    def forward(self, x, beta=1.0):
        """
        Full forward pass with loss computation.
        
        Args:
            x: Input patterns [batch_size, 16, 9]
            beta: KL weight for beta-VAE (use < 1 to prevent collapse)
            
        Returns:
            recon: Reconstructed patterns
            mu_high, logvar_high, mu_low, logvar_low: Latent parameters (corrected order)
        """
        # TODO: Encode, decode, compute losses
        mu_high, logvar_high, mu_low, logvar_low = self.encode_hierarchy(x)
        z_low = self.reparameterize(mu_low, logvar_low)
        z_high = self.reparameterize(mu_high, logvar_high)
        recon_logits = self.decode_hierarchy(z_high, z_low)
        return recon_logits, mu_high, logvar_high, mu_low, logvar_low