"""
Analysis and evaluation experiments for trained GAN models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('..')

from provided.metrics import mode_coverage_score, font_consistency_score
from provided.visualize import plot_alphabet_grid

def interpolation_experiment(generator, device, steps=11, save_path=None):
    """
    Interpolate between latent codes to generate smooth transitions.
    
    TODO:
    1. Find latent codes for specific letters (via optimization)
    2. Interpolate between them
    3. Visualize the path from A to Z
    """
    generator.eval()
    z_dim = getattr(generator, "z_dim", 100)

    z0 = torch.randn(1, z_dim, device=device)
    z1 = torch.randn(1, z_dim, device=device)

    alphas = torch.linspace(0, 1, steps, device=device).view(-1, 1)
    Z = (1 - alphas) * z0 + alphas * z1  # [steps, z_dim]

    with torch.no_grad():                               # ← 关键1
        if getattr(generator, "conditional", False):
            y = torch.zeros(steps, 26, device=device)
            y[:, 0] = 1.0
            imgs = generator(Z, y)                      # [steps, 1, 28, 28]
        else:
            imgs = generator(Z)
    
    imgs_vis = ((imgs.detach().cpu() + 1) / 2).numpy()  # [-1,1] → [0,1]

    fig, axes = plt.subplots(1, steps, figsize=(steps * 1.2, 1.4))
    for i in range(steps):
        ax = axes[i] if steps > 1 else axes
        ax.imshow(imgs_vis[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"{i}", fontsize=8)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, imgs.detach().cpu()


def style_consistency_experiment(conditional_generator, device, n_samples=2, save_path=None):
    """
    Test if conditional GAN maintains style across letters.
    
    TODO:
    1. Fix a latent code z
    2. Generate all 26 letters with same z
    3. Measure style consistency
    """
    G = conditional_generator
    assert getattr(G, "conditional", False), "style_consistency_experiment 需要 conditional 生成器"

    G.eval()
    z_dim = getattr(G, "z_dim", 100)

    # 收集每个字母的多张图（至少 2 张，metrics 内部按同一字母两两比较）
    per_letter = {c: [] for c in range(26)}
    with torch.no_grad():
        for _ in range(n_samples):
            z = torch.randn(1, z_dim, device=device)
            # 用相同 z、不同 one-hot 生成 26 个字母
            labels = torch.eye(26, device=device)  # [26,26]
            Z = z.expand(26, z_dim)                # [26,z_dim]
            imgs = G(Z, labels)                    # [26,1,28,28]
            # 拆回字母字典
            for c in range(26):
                per_letter[c].append(imgs[c].detach().cpu())

    # 计算风格一致性
    consistency = font_consistency_score(per_letter, n_samples=n_samples)

    # 画一个固定 z 的字母网格，便于查看（使用提供的可视化函数更省事）
    # 注意：plot_alphabet_grid 内部会重新采样 z，这里简单再画一张即可
    fig = plot_alphabet_grid(G, device=device, z_dim=z_dim, seed=641)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return consistency

def mode_recovery_experiment(generator_checkpoints, device='cuda', n_samples=1000):
    """
    Analyze how mode collapse progresses and potentially recovers.
    
    TODO:
    1. Load checkpoints from different epochs
    2. Measure mode coverage at each checkpoint
    3. Identify when specific letters disappear/reappear
    """
    labels, coverage, missing = [], [], []

    for item in generator_checkpoints:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            tag, payload = item
        else:
            # 不符合约定，跳过
            continue

        # 拿到一个可评估的生成器实例
        if callable(payload) or (isinstance(payload, (list, tuple)) and len(payload) == 2):
            # 视为 (gen_ctor, ckpt_path)
            if isinstance(payload, (list, tuple)) and len(payload) == 2:
                gen_ctor, ckpt_path = payload
            else:
                # callable 本身当构造器，ckpt_path 置空（不推荐）
                gen_ctor, ckpt_path = payload, None
            G = gen_ctor().to(device)
            if ckpt_path:
                state = torch.load(ckpt_path, map_location=device)
                G.load_state_dict(state)
            G.eval()
        else:
            # 视为已构造好的生成器实例
            G = payload.to(device)
            G.eval()

        # 生成一批样本，计算覆盖度
        z_dim = getattr(G, "z_dim", 100)
        with torch.no_grad(): 
            z = torch.randn(n_samples, z_dim, device=device)
            if getattr(G, "conditional", False):
                # 无标签插值不适用覆盖度，这里给随机标签以增加多样性
                labels_oh = torch.eye(26, device=device)[torch.randint(0, 26, (n_samples,), device=device)]
                imgs = G(z, labels_oh)
            else:
                imgs = G(z)

        stats = mode_coverage_score(imgs.detach().cpu())
        labels.append(tag)
        coverage.append(float(stats["coverage_score"]))
        missing.append(stats["missing_letters"])

    return {"labels": labels, "coverage": coverage, "missing": missing}