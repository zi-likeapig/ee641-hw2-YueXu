"""
Latent space analysis tools for hierarchical VAE.
"""
import os, sys, math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from sklearn.manifold import TSNE
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from provided.visualize import plot_drum_pattern, plot_latent_space_2d
from provided.metrics import mode_coverage_score

def _to_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)

def draw_pattern(ax, patt_16x9, title=None, vmin=0, vmax=1, show_grid=True, border=True):
    p = _to_np(patt_16x9)
    if p.shape[0] == 16 and p.shape[1] == 9:
        p = p.T  # [9,16]
    H, W = p.shape  # H=9, W=16

    ax.imshow(p, aspect='auto', cmap='gray_r',
              vmin=vmin, vmax=vmax, interpolation='nearest')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)

    if show_grid:
        for x in range(W + 1):
            ax.vlines(x - 0.5, -0.5, H - 0.5, colors='0.85', linewidth=0.7)
        for y in range(H + 1):
            ax.hlines(y - 0.5, -0.5, W - 0.5, colors='0.85', linewidth=0.7)

    if border:
        ax.add_patch(Rectangle((-0.5, -0.5), W, H, fill=False, lw=1.2, ec='0.55'))

    if title:
        ax.text(0.0, 1.02, str(title), transform=ax.transAxes, fontsize=8)

def save_grid_with_provided(patterns_Nx16x9, save_path, title=None, cols=5, subtitles=None):
    P = _to_np(patterns_Nx16x9)
    if P.ndim != 3:
        raise ValueError("patterns must be [N,16,9] or [N,9,16]")

    N = P.shape[0]
    rows = math.ceil(N / cols)
    
    fig = plt.figure(figsize=(cols*3, rows*2.5))
    gs = gridspec.GridSpec(rows, cols, figure=fig)
    
    instruments = ['Kick', 'Snare', 'Hi-hat C', 'Hi-hat O', 
                   'Tom L', 'Tom H', 'Crash', 'Ride', 'Clap']
    
    for i in range(N):
        row, col = i // cols, i % cols
        ax = fig.add_subplot(gs[row, col])
        
        pattern = P[i]
        if pattern.shape[0] == 16 and pattern.shape[1] == 9:
            pattern = pattern.T  
            
        for j in range(9):  # instruments
            for k in range(16):  # time steps
                if pattern[j, k] > 0.5:
                    ax.add_patch(plt.Rectangle((k, j), 1, 1, 
                                              facecolor='blue', 
                                              edgecolor='black',
                                              linewidth=0.5))
        
        for k in range(17):
            ax.axvline(k, color='gray', linewidth=0.5, alpha=0.5)
            if k % 4 == 0:
                ax.axvline(k, color='black', linewidth=1.5, alpha=0.7)
        
        for j in range(10):
            ax.axhline(j, color='gray', linewidth=0.5, alpha=0.5)
        
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 9)
        ax.set_xticks(range(0, 16, 4))
        ax.set_xticklabels([str(k+1) for k in range(0, 16, 4)])
        ax.set_yticks(range(9))
        ax.set_yticklabels(instruments, fontsize=8)
        ax.set_xlabel('Time Step')
        if col == 0:
            ax.set_ylabel('Instrument')
        else:
            ax.set_ylabel('')
            ax.set_yticklabels([])
        ax.invert_yaxis()
        
        if subtitles and i < len(subtitles):
            ax.set_title(subtitles[i], fontsize=10)
        else:
            ax.set_title(f"#{i}", fontsize=10)
    
    for i in range(N, rows * cols):
        row, col = i // cols, i % cols
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14, y=0.98)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def save_grid(patterns_Nx16x9, save_path, title=None, cols=5):
    P = _to_np(patterns_Nx16x9)
    if P.ndim != 3:
        raise ValueError("patterns must be [N,16,9] or [N,9,16]")

    N = P.shape[0]
    rows = math.ceil(N / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.6, rows*2.4), dpi=150)
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis('off')
        if i < N:
            draw_pattern(ax, P[i], title=f"#{i}")

    if title:
        fig.suptitle(title, y=0.995, fontsize=11)
    plt.tight_layout(pad=0.6, w_pad=0.5, h_pad=0.7)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def save_styles_10samples_grid(styles_Sx10x16x9, out_dir, cols=5):
    X = _to_np(styles_Sx10x16x9)
    S = X.shape[0]
    for s in range(S):
        save_grid(
            X[s], 
            os.path.join(out_dir, f"style_{s}_10samples_grid.png"),
            title=f"Style {s}: 10 samples",
            cols=cols
        )


def visualize_latent_hierarchy(model, data_loader, device='cuda'):
    """
    Visualize the two-level latent space structure.
    
    TODO:
    1. Encode all data to get z_high and z_low
    2. Use t-SNE to visualize z_high (colored by genre)
    3. For each z_high cluster, show z_low variations
    4. Create hierarchical visualization
    """
    device = torch.device(device if torch.cuda.is_available() and device != 'cpu' else 'cpu')
    model.to(device).eval()

    zs_high, zs_low, ys = [], [], []
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            else:
                x, y = batch, None
            x = x.to(device)
            mu_h, _, mu_l, _ = model.encode_hierarchy(x)
            zs_high.append(mu_h.cpu())
            zs_low.append(mu_l.cpu())
            if y is None:
                ys.append(torch.zeros(mu_h.size(0), dtype=torch.long))
            else:
                ys.append(y.cpu().long())

    z_high = torch.cat(zs_high, dim=0).numpy()  # [N, Dz_high]
    z_low  = torch.cat(zs_low,  dim=0).numpy()  # [N, Dz_low]
    y      = torch.cat(ys,      dim=0).numpy()  # [N]

    # --- t-SNE on z_high 
    fig = plot_latent_space_2d(z_high, labels=y, title='z_high t-SNE by Style')

    # --- Per-style prototypes and diverse decodes ---
    num_styles = int(y.max()) + 1 if y.size > 0 else 5
    z_high_means = []
    for s in range(num_styles):
        mask = (y == s)
        if mask.sum() == 0:
            z_high_means.append(torch.zeros(model.z_high_dim))
        else:
            z_high_means.append(torch.from_numpy(z_high[mask]).float().mean(dim=0))
    z_high_means = torch.stack(z_high_means, dim=0).to(device)  # [S, Dz_high]

    grid = []
    with torch.no_grad():
        for s in range(num_styles):
            row = []
            for _ in range(8):
                z_h = z_high_means[s:s+1]
                logits = model.decode_hierarchy(z_h, z_low=None)  # z_low~p(z_low|z_high)
                patt = torch.sigmoid(logits)
                patt = (patt > 0.5).float().cpu().squeeze(0)  # [16,9]
                row.append(patt)
            grid.append(torch.stack(row, dim=0))  # [8,16,9]
    grid = torch.stack(grid, dim=0)  # [S,8,16,9]

    return {
        'z_high': z_high,
        'z_low': z_low,
        'labels': y,
        'z_high_means': z_high_means.cpu().numpy(),
        'tsne_fig': fig,
        'per_style_grid': grid,
    }

def interpolate_styles(model, pattern1, pattern2, n_steps=10, device='cuda'):
    """
    Interpolate between two drum patterns at both latent levels.
    
    TODO:
    1. Encode both patterns to get latents
    2. Interpolate z_high (style transition)
    3. Interpolate z_low (variation transition)
    4. Decode and visualize both paths
    5. Compare smooth vs abrupt transitions
    """
    device = torch.device(device if torch.cuda.is_available() and device != 'cpu' else 'cpu')
    model.to(device).eval()

    with torch.no_grad():
        x1 = pattern1.unsqueeze(0).to(device).float()  # [1,16,9]
        x2 = pattern2.unsqueeze(0).to(device).float()

        mu_h1, _, mu_l1, _ = model.encode_hierarchy(x1)
        mu_h2, _, mu_l2, _ = model.encode_hierarchy(x2)

        alphas = np.linspace(0, 1, n_steps, dtype=np.float32)

        # keep z_low = 1，z_high vary (use probabilities)
        seq_style = []
        for a in alphas:
            z_high = (1 - a) * mu_h1 + a * mu_h2
            # z_low  = mu_l1
            #logits = model.decode_hierarchy(z_high, z_low=z_low) 
            #patt = torch.sigmoid(logits).cpu().squeeze(0)  # [16,9]
            logits = model.decode_hierarchy(z_high, z_low=None, temperature=0.6)   # 由 p(z_low|z_high) 生成
            probs  = torch.sigmoid(logits).cpu().squeeze(0)
            patt   = (probs > 0.45).float()
            seq_style.append(patt)
        seq_style = torch.stack(seq_style, dim=0)  # [n_steps,16,9]

        # keep z_high = 1，z_low vary (use probabilities)
        seq_detail = []
        for a in alphas:
            z_high = mu_h1
            z_low  = (1 - a) * mu_l1 + a * mu_l2
            logits = model.decode_hierarchy(z_high, z_low=z_low)
            patt = torch.sigmoid(logits).cpu().squeeze(0)
            seq_detail.append(patt)
        seq_detail = torch.stack(seq_detail, dim=0)

    return {
        'style_path': seq_style, 
        'detail_path': seq_detail,
        'alphas': alphas.tolist(),
    }

def measure_disentanglement(model, data_loader, device='cuda'):
    """
    Measure how well the hierarchy disentangles style from variation.
    
    TODO:
    1. Group patterns by genre
    2. Compute z_high variance within vs across genres
    3. Compute z_low variance for same genre
    4. Return disentanglement metrics
    """
    device = torch.device(device if torch.cuda.is_available() and device != 'cpu' else 'cpu')
    model.to(device).eval()

    zs_high = []
    zs_low  = []
    ys = []
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            else:
                x, y = batch, None
            x = x.to(device)
            mu_h, _, mu_l, _ = model.encode_hierarchy(x)
            zs_high.append(mu_h.cpu())
            zs_low.append(mu_l.cpu())
            if y is None:
                ys.append(torch.zeros(mu_h.size(0), dtype=torch.long))
            else:
                ys.append(y.cpu().long())

    z_high = torch.cat(zs_high, dim=0).numpy()
    z_low  = torch.cat(zs_low,  dim=0).numpy()
    y      = torch.cat(ys,      dim=0).numpy()

    num_styles = int(y.max()) + 1 if y.size > 0 else 5

    # z_high
    centers = []
    within_vars = []
    for s in range(num_styles):
        mask = (y == s)
        if mask.sum() == 0:
            continue
        Zs = z_high[mask]                # [Ns, Dz_high]
        centers.append(Zs.mean(axis=0))
        within_vars.append(Zs.var(axis=0).mean())  # scalar: mean var per dim
    centers = np.stack(centers, axis=0)            # [S, Dz_high]
    within_var_high = float(np.mean(within_vars))

    between_var_high = float(centers.var(axis=0).mean())

    within_low = []
    for s in range(num_styles):
        mask = (y == s)
        if mask.sum() == 0:
            continue
        Zl = z_low[mask]
        within_low.append(Zl.var(axis=0).mean())
    within_var_low = float(np.mean(within_low))

    sep_score = between_var_high / (within_var_high + 1e-8)

    return {
        'within_var_high': within_var_high,
        'between_var_high': between_var_high,
        'within_var_low': within_var_low,
        'separation_score_high': float(sep_score),
        'num_styles': num_styles,
    }

def controllable_generation(model, genre_labels, device='cuda'):
    """
    Test controllable generation using the hierarchy.
    
    TODO:
    1. Learn genre embeddings in z_high space
    2. Generate patterns with specified genre
    3. Control complexity via z_low sampling temperature
    4. Evaluate genre classification accuracy
    """
    device = torch.device(device if torch.cuda.is_available() and device != 'cpu' else 'cpu')
    model.to(device).eval()

    assert hasattr(model, '_z_high_means'), \
        "controllable_generation requires model._z_high_means (set it from visualize_latent_hierarchy['z_high_means'])."
    z_high_means = torch.from_numpy(model._z_high_means).to(device)  # [S, Dz_high]
    num_styles = z_high_means.size(0)

    generated = []
    labels = []
    with torch.no_grad():
        for g in genre_labels:
            g = int(g)
            g = max(0, min(g, num_styles - 1))
            z_h = z_high_means[g:g+1]  # [1, Dz_high]
            logits = model.decode_hierarchy(z_h, z_low=None)
            patt = (torch.sigmoid(logits) > 0.5).float().cpu().squeeze(0)  # [16,9]
            generated.append(patt)
            labels.append(g)
    generated = torch.stack(generated, dim=0)  # [N,16,9]
    labels = torch.tensor(labels, dtype=torch.long)

    preds = []
    with torch.no_grad():
        for i in range(generated.size(0)):
            x = generated[i:i+1].to(device)
            mu_h, _, _, _ = model.encode_hierarchy(x)
            
            dists = torch.cdist(mu_h, z_high_means)  # [1,S]
            pred = int(torch.argmin(dists, dim=1).item())
            preds.append(pred)
    preds = torch.tensor(preds, dtype=torch.long)

    try:
        from provided.metrics import accuracy
        acc = float(accuracy(preds.numpy(), labels.numpy()))
    except Exception:
        acc = float((preds == labels).float().mean().item())

    return {
        'generated': generated,  # [N,16,9]
        'labels': labels,
        'preds': preds,
        'nearest_centroid_acc': acc
    }



def main():
    import os, json, torch
    from torch.utils.data import DataLoader
    from dataset import DrumPatternDataset
    from hierarchical_vae import HierarchicalDrumVAE
    from provided.visualize import plot_drum_pattern

    here = os.path.dirname(__file__)
    results_dir = os.path.join(here, "results")
    gen_dir = os.path.join(results_dir, "generated_patterns")
    lat_dir = os.path.join(results_dir, "latent_analysis")
    for d in (results_dir, gen_dir, lat_dir):
        os.makedirs(d, exist_ok=True)

    root = os.path.abspath(os.path.join(here, os.pardir)) 
    data_dir = os.path.join(root, "data", "drums")
    val_set = DrumPatternDataset(data_dir=data_dir, split="val")
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierarchicalDrumVAE()
    ckpt = torch.load(os.path.join(results_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(ckpt)
    model.to(device).eval()

    out = visualize_latent_hierarchy(model, val_loader, device=device)
    try:
        fig = out["tsne_fig"]
        fig.savefig(os.path.join(lat_dir, "z_high_tsne.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass
    model._z_high_means = out["z_high_means"]

    # grid = out["per_style_grid"]  # [S,8,16,9]
    # S = grid.size(0)
    # import torch as _t

    # extra = []
    # with torch.no_grad():
    #     for s in range(S):
    #         row = []
    #         for _ in range(2):
    #             z_h = _t.from_numpy(model._z_high_means)[s:s+1].to(device)
    #             logits = model.decode_hierarchy(z_h, z_low=None)
    #             patt = (torch.sigmoid(logits) > 0.5).float().cpu().squeeze(0)
    #             row.append(patt)
    #         extra.append(_t.stack(row))
    # extras = _t.stack(extra)  # [S,2,16,9]
    # full = _t.cat([grid, extras], dim=1)  # [S,10,16,9]

    # for s in range(S):
    #     sample_subtitles = [f"Sample {i+1}" for i in range(10)]
    #     save_grid_with_provided(
    #         full[s],  # [10, 16, 9]
    #         os.path.join(gen_dir, f"style_{s}_10samples_grid.png"),
    #         title=f"Style {s}: 10 Generated Samples",
    #         cols=5,
    #         subtitles=sample_subtitles
    #     )

    grid = out["per_style_grid"]  # [S,8,16,9]
    S = grid.size(0)
    import torch as _t

    T  = 1.0   # 新增: 低层采样温度
    TH = 0.45  # 新增: 可视化阈值

    extras = []
    with _t.no_grad():
        for s in range(S):
            row = []
            for _ in range(2):
                z_h = _t.from_numpy(model._z_high_means)[s:s+1].to(device).float()  # ← 加 .float()
                logits = model.decode_hierarchy(z_h, z_low=None, temperature=T)     # ← 传 temperature
                probs  = _t.sigmoid(logits).cpu().squeeze(0)                        # ← 先转概率
                patt   = (probs > TH).float()                                       # ← 阈值 0.45
                row.append(patt)
            extras.append(_t.stack(row))
    extras = _t.stack(extras)  # [S,2,16,9]
    full = _t.cat([grid, extras], dim=1)  # [S,10,16,9]

    for s in range(S):
        sample_subtitles = [f"Sample {i+1}" for i in range(10)]
        save_grid_with_provided(
            full[s],  # [10, 16, 9]
            os.path.join(gen_dir, f"style_{s}_10samples_grid.png"),
            title=f"Style {s}: 10 Generated Samples",
            cols=5,
            subtitles=sample_subtitles
        )
        p1, s1, _ = val_set[0]
        p2, s2, _ = val_set[1]
        print(f"style{s1.item()} -> style{s2.item()}")
    
    inter = interpolate_styles(model, p1, p2, n_steps=10, device=device)

    style_path = inter["style_path"] 
    if isinstance(style_path, list):
        style_path = torch.stack([p.detach().cpu() for p in style_path], dim=0)  # [N,16,9]
    
    n_steps = len(style_path)
    alphas = inter.get("alphas", [i/(n_steps-1) for i in range(n_steps)])
    style_subtitles = [f"α={alpha:.2f}" for alpha in alphas]
    
    save_grid_with_provided(
        style_path, 
        os.path.join(gen_dir, "interp_style_grid.png"),
        title=f"Style Interpolation: Style {s1.item()} → Style {s2.item()}",
        cols=5,
        subtitles=style_subtitles
    )

    detail_path = inter["detail_path"] 
    if isinstance(detail_path, list):
        detail_path = torch.stack([p.detach().cpu() for p in detail_path], dim=0)  # [N,16,9]
    
    detail_subtitles = [f"Step {i+1}" for i in range(len(detail_path))]
    
    save_grid_with_provided(
        detail_path, 
        os.path.join(gen_dir, "interp_detail_grid.png"),
        title=f"Detail Interpolation: Same Style {s1.item()}, Different Details",
        cols=5,
        subtitles=detail_subtitles
    )
    # for t, patt in enumerate(inter["detail_path"]):
    #     fig = plot_drum_pattern(patt.numpy(), title=f"interp_detail_{t}")
    #     fig.savefig(os.path.join(gen_dir, f"interp_detail_{t}.png"), dpi=150, bbox_inches="tight")
    #     plt.close(fig)

    x0, s0, _ = val_set[1]
    print(f"original style: {s0.item()}")
    
    with torch.no_grad():
        _, _, mu_l0, _ = model.encode_hierarchy(x0.unsqueeze(0).to(device)) 
        z_low_src = mu_l0  # [1, z_low_dim] 

        z_means = torch.from_numpy(model._z_high_means).to(device)  # [S, z_high_dim]
        seq = []
        
        # for s in range(min(S, 5)):
        #     logits = model.decode_hierarchy(z_means[s:s+1], z_low=z_low_src)
        #     patt = (torch.sigmoid(logits) > 0.5).float().cpu().squeeze(0)  # [16,9]
        #     seq.append(patt)
        for s in range(min(S, 5)):
            z_h = z_means[s:s+1].to(device).float()
            mu_p, logvar_p = model.cond_prior_low(z_h)
            z_low = 0.7 * z_low_src + 0.3 * mu_p   # γ=0.3 可调：更大=风格更明显，过大则“细节”丢
            logits = model.decode_hierarchy(z_h, z_low=z_low, temperature=0.6)
            probs  = torch.sigmoid(logits).cpu().squeeze(0)
            patt   = (probs > 0.45).float()
            seq.append(patt)

    transfer_seq = torch.stack(seq, dim=0)  # [K,16,9]
    K = transfer_seq.shape[0]
    
    transfer_subtitles = [f"→ Style {s}" for s in range(K)]
    cols = 5 if K >= 5 else K
    
    save_grid_with_provided(
        transfer_seq, 
        os.path.join(gen_dir, "style_transfer_grid.png"),
        title=f"Style Transfer: Original Style {s0.item()} → Different Styles",
        cols=cols,
        subtitles=transfer_subtitles
    )
    # x0, _, _ = val_set[0]
    # with torch.no_grad():
    #     mu_h0, _, mu_l0, _ = model.encode_hierarchy(x0.unsqueeze(0).to(device))
    #     z_low_src = mu_l0
    #     z_means = torch.from_numpy(model._z_high_means).to(device)
    #     for s in range(S):
    #         logits = model.decode_hierarchy(z_means[s:s+1], z_low=z_low_src, temperature=0.8)
    #         patt = (torch.sigmoid(logits) > 0.5).float().cpu().squeeze(0)
    #         fig = plot_drum_pattern(patt.numpy(), title=f"style_transfer_to_{s}")
    #         fig.savefig(os.path.join(gen_dir, f"style_transfer_to_{s}.png"), dpi=150, bbox_inches="tight")
    #         plt.close(fig)

    disent = measure_disentanglement(model, val_loader, device=device)
    with open(os.path.join(lat_dir, "disentanglement_metrics.json"), "w") as f:
        json.dump(disent, f, indent=2)

    # （optional）posterior collapse analysis
    try:
        from training_utils import analyze_posterior_collapse
        collapse = analyze_posterior_collapse(model, val_loader, device=device)
        with open(os.path.join(lat_dir, "posterior_collapse.json"), "w") as f:
            json.dump(collapse, f, indent=2)
    except Exception:
        pass

    with open(os.path.join(results_dir, "analysis_manifest.json"), "w") as f:
        json.dump({
            "generated_patterns_dir": "generated_patterns",
            "latent_analysis_dir": "latent_analysis",
            "notes": [
                "10 per-style samples",
                "style/detail interpolations",
                "style transfer examples",
                "z_high t-SNE + disentanglement metrics"
            ]
        }, f, indent=2)

if __name__ == "__main__":
    main()
