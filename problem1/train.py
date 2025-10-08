"""
Main training script for GAN experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

from dataset import FontDataset
from models import Generator, Discriminator
from training_dynamics import train_gan, analyze_mode_coverage, visualize_mode_collapse
from fixes import train_gan_with_fix
import sys
sys.path.append('..')
from provided.visualize import plot_alphabet_grid
from provided.metrics import mode_coverage_score
from evaluate import interpolation_experiment

def save_mode_coverage_histogram(generator, device, z_dim, save_path, n_samples=1000):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, z_dim, device=device)
        imgs = generator(z)
        imgs01 = (imgs + 1) / 2  # [-1,1] → [0,1]
        stats = mode_coverage_score(imgs01)

    counts = stats.get("letter_counts", {})  # key is 0-25, not 'A'-'Z'

    xs = np.arange(26)
    values = np.array([counts.get(i, 0) for i in range(26)], dtype=int) # use int keys 0-25
    labels = [chr(65 + i) for i in range(26)]     # ['A','B',...,'Z']

    plt.figure(figsize=(10, 3.2))
    plt.bar(xs, values, width=0.8)
    plt.xticks(xs, labels)
    plt.ylabel("Count")
    plt.title("Mode coverage histogram (which letters survive)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_vanilla_vs_fixed_compare(vanilla_G, fixed_G, device, z_dim, save_path):
    import matplotlib.pyplot as plt
    import sys
    sys.path.append('..')  # 添加上级目录到路径
    from provided.metrics import mode_coverage_score
    import torch

    @torch.no_grad()
    def cov(G):
        G.eval()
        z = torch.randn(1000, z_dim, device=device)
        imgs = G(z)
        return float(mode_coverage_score(imgs)["coverage_score"])

    cov_v, cov_f = cov(vanilla_G), cov(fixed_G)
    plt.figure(figsize=(4,3))
    plt.bar(["vanilla", "fixed"], [cov_v, cov_f], width=0.6)
    plt.ylim(0, 1.05); plt.ylabel("Coverage"); plt.title("Mode coverage: vanilla vs fixed")
    for i, v in enumerate([cov_v, cov_f]):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()


def main():
    """
    Main training entry point for GAN experiments.
    """
    # Configuration
    config = {
        "experiment": "vanilla",  # 'vanilla' or 'fixed'
        'fix_type': 'feature_matching',  # Used if experiment='fixed'
        'results_dir': 'results_vanilla',
        'checkpoint_dir': "results_vanilla/ckpts",
        "compare_with": "",

        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'batch_size': 64,
        'num_epochs': 100,
        'z_dim': 100,
        'learning_rate': 0.0002,
        'data_dir': '../data/fonts',
        'snapshot_epochs': (10, 30, 50, 100),
    }
    
    # Create directories
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['results_dir']).mkdir(parents=True, exist_ok=True)
    Path(f'{config["results_dir"]}/visualizations').mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset and dataloader
    train_dataset = FontDataset(config['data_dir'], split='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2
    )
    
    # Initialize models
    generator = Generator(z_dim=config['z_dim']).to(config['device'])
    discriminator = Discriminator().to(config['device'])
    
    # Train model
    if config['experiment'] == 'vanilla':
        print("Training vanilla GAN (expect mode collapse)...")
        history = train_gan(
            generator, 
            discriminator, 
            train_loader,
            num_epochs=config['num_epochs'],
            device=config['device'],
            snapshot_epochs=config['snapshot_epochs'],
            snapshot_dir=config['checkpoint_dir']
        )
    else:
        print(f"Training GAN with {config['fix_type']} fix...")
        history = train_gan_with_fix(
            generator,
            discriminator, 
            train_loader,
            num_epochs=config['num_epochs'],
            fix_type=config['fix_type'],
            snapshot_epochs=config['snapshot_epochs'],
            snapshot_dir=config['checkpoint_dir']
        )
    
    # Save results
    # TODO: Save training history to JSON
    with open(f"{config['results_dir']}/training_log.json", 'w') as f:
        json.dump(history, f, indent=2)

    # Save mode collapse analysis
    visualize_mode_collapse(history, save_path=f"{config['results_dir']}/mode_collapse_analysis.png")

    # TODO: Save final model checkpoint
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'config': config,
        'final_epoch': config['num_epochs']
    }, f"{config['results_dir']}/best_generator.pth")

    # Save final alphabet grid
    fig = plot_alphabet_grid(generator, device=config["device"], z_dim=config["z_dim"], seed=641)
    fig.savefig(f"{config['results_dir']}/visualizations/alphabet_final.png",
                    dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save interpolation results
    interpolation_experiment(generator, config["device"], steps=11,
                             save_path=os.path.join(config['results_dir'], "visualizations", "interpolation.png"))

    # Save mode coverage histogram
    save_mode_coverage_histogram(generator, config["device"], config["z_dim"],
                                 save_path=os.path.join(config['results_dir'], "visualizations", "mode_coverage_hist.png"),
                                 n_samples=1000)

    print(f"Training complete. Results saved to {config['results_dir']}/")

if __name__ == '__main__':
    main()