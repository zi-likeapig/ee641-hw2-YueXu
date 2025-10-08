"""
GAN stabilization techniques to combat mode collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import sys
sys.path.append('..')
from pathlib import Path
import matplotlib.pyplot as plt
from provided.visualize import plot_alphabet_grid

def train_gan_with_fix(generator, discriminator, data_loader, 
                       num_epochs=100, fix_type='feature_matching', 
                       snapshot_epochs=(10, 30, 50, 100), snapshot_dir="results_fixed/ckpts"):
    """
    Train GAN with mode collapse mitigation techniques.
    
    Args:
        generator: Generator network
        discriminator: Discriminator network
        data_loader: DataLoader for training data
        num_epochs: Number of training epochs
        fix_type: Stabilization method ('feature_matching', 'unrolled', 'minibatch')
        snapshot_epochs: Tuple of epochs to save checkpoints and visualizations
        snapshot_dir: Directory to save checkpoints
        
    Returns:
        dict: Training history with metrics
    """
    
    if fix_type == 'feature_matching':
        # Feature matching: Match statistics of intermediate layers
        # instead of just final discriminator output
        
        def feature_matching_loss(real_images, fake_images, discriminator):
            """
            TODO: Implement feature matching loss
            
            Extract intermediate features from discriminator
            Match mean statistics: ||E[f(x)] - E[f(G(z))]||²
            Use discriminator.features (before final classifier)
            """
            total_loss = 0.0
            num_layers = 0
            
            def get_layer_features(x, model):
                features = []
                current = x
                for i, layer in enumerate(model.features):
                    current = layer(current)
                    if isinstance(layer, (nn.Conv2d, nn.LeakyReLU)):
                        if isinstance(layer, nn.Conv2d):
                            features.append(current)
                return features
            
            with torch.no_grad():
                real_features = get_layer_features(real_images, discriminator)
            
            fake_features = get_layer_features(fake_images, discriminator)
            
            for real_feat, fake_feat in zip(real_features, fake_features):
                if real_feat.dim() == 4:  # [B, C, H, W]
                    real_mean = real_feat.mean(dim=(0, 2, 3))  # [C]
                    fake_mean = fake_feat.mean(dim=(0, 2, 3))  # [C]
                else:  # [B, C]
                    real_mean = real_feat.mean(dim=0)
                    fake_mean = fake_feat.mean(dim=0)
                
                # L2 loss between means
                layer_loss = F.mse_loss(fake_mean, real_mean)
                total_loss += layer_loss
                num_layers += 1
            
            return total_loss / max(num_layers, 1)
        
    elif fix_type == 'unrolled':
        # Unrolled GANs: Look ahead k discriminator updates
        
        def unrolled_discriminator(discriminator, real_data, fake_data, k=5):
            """
            TODO: Implement k-step unrolled discriminator
            
            Create temporary discriminator copy
            Update it k times
            Compute generator loss through updated discriminator
            """
            pass
            
    elif fix_type == 'minibatch':
        # Minibatch discrimination: Let discriminator see batch statistics
        
        class MinibatchDiscrimination(nn.Module):
            """
            TODO: Add minibatch discrimination layer to discriminator
            
            Compute L2 distance between samples in batch
            Concatenate statistics to discriminator features
            """
            pass
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator.to(device)
    discriminator.to(device)

    # Create directories for snapshots and visualizations
    Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
    vis_dir = Path(snapshot_dir).parent / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCELoss()

    history = {'epoch': [], 'd_loss': [], 'g_loss': []}

    for epoch in range(num_epochs):
        running_d, running_g, n_batches = 0.0, 0.0, 0

        for batch_idx, (real_images, _) in enumerate(data_loader):
            real_images = real_images.to(device)
            bsz = real_images.size(0)

            valid = torch.ones(bsz, 1, device=device)
            fake  = torch.zeros(bsz, 1, device=device)

            # ===== Train D (vanilla BCE) =====
            d_optimizer.zero_grad(set_to_none=True)

            d_real = bce(discriminator(real_images), valid)

            z = torch.randn(bsz, generator.z_dim, device=device)  
            fake_images = generator(z)
            d_fake = bce(discriminator(fake_images.detach()), fake)

            d_loss = d_real + d_fake
            d_loss.backward()
            d_optimizer.step()

            # ===== Train G (feature matching + adversarial) =====
            g_optimizer.zero_grad(set_to_none=True)

            if fix_type == 'feature_matching':
                # 1. basic adversarial loss
                g_adv_loss = bce(discriminator(fake_images), valid)
                
                # 2. feature matching loss
                g_fm_loss = feature_matching_loss(real_images, fake_images, discriminator)
                
                # 3. safe diversity loss
                def safe_diversity_loss(fake_images):
                    if fake_images.size(0) < 2:
                        return torch.tensor(0.0, device=fake_images.device)
                    
                    flat_images = fake_images.view(fake_images.size(0), -1)  # [B, H*W]
                    pairwise_distances = torch.pdist(flat_images, p=2)
                    avg_distance = pairwise_distances.mean()
                    similarity_penalty = 1.0 / (avg_distance + 0.1)
                    
                    return similarity_penalty
                
                g_div_loss = safe_diversity_loss(fake_images)
                
                g_loss = 0.5 * g_adv_loss + 1.8 * g_fm_loss + 0.3 * g_div_loss
            else:
                raise NotImplementedError(f"{fix_type} is not implemented in this minimal solution.")

            g_loss.backward()
            g_optimizer.step()

            running_d += float(d_loss.item())
            running_g += float(g_loss.item())
            n_batches += 1

            if batch_idx % 10 == 0:
                history['epoch'].append(epoch + batch_idx / max(1, len(data_loader)))
                history['d_loss'].append(float(d_loss.item()))
                history['g_loss'].append(float(g_loss.item()))

        if (epoch + 1) % 5 == 0:
            print(f"[Epoch {epoch + 1:03d}/{num_epochs}] "
                f"D: {running_d / max(1, n_batches):.4f} | G(FM): {running_g / max(1, n_batches):.4f}")

        # Analyze mode coverage every 10 epochs
        if (epoch + 1) % 10 == 0:
            from training_dynamics import analyze_mode_coverage
            cov = analyze_mode_coverage(generator, device)
            history.setdefault('mode_coverage', []).append(float(cov))
            print(f"Epoch {(epoch + 1)}: Mode coverage = {cov:.2f}")
        
        # Save model checkpoints and visualizations at specified epochs
        if (epoch + 1) in set(snapshot_epochs):
            torch.save(generator.state_dict(), Path(snapshot_dir) / f"G_{epoch+1:04d}.pth")
            try:
                fig = plot_alphabet_grid(generator, device=device, z_dim=generator.z_dim, seed=641)
                fig.savefig(vis_dir / f"alphabet_epoch_{epoch+1}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                # print(f"----Saved checkpoint and visualization for epoch {epoch+1}----")
            except Exception as e:
                print(f"----[WARN] Failed to save alphabet grid at epoch {epoch+1}: {e}----")



    return history