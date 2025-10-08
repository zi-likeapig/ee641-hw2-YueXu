# compare_vanilla_fixed.py
import os, sys, torch, matplotlib.pyplot as plt

sys.path.append("..")
from models import Generator
from provided.metrics import mode_coverage_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
z_dim = 100

def load_generator(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        if "generator_state_dict" in ckpt:
            state = ckpt["generator_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            possible = [v for v in ckpt.values() if isinstance(v, dict)]
            state = possible[0] if possible else ckpt
    else:
        state = ckpt

    G = Generator(z_dim=z_dim).to(device)
    missing, unexpected = G.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] missing={missing}, unexpected={unexpected}")
    G.eval()
    return G

@torch.no_grad()
def coverage_of(ckpt_path):
    G = load_generator(ckpt_path, device)
    z = torch.randn(2000, z_dim, device=device)
    imgs = G(z)
    score = mode_coverage_score(imgs)["coverage_score"]
    return float(score)

cov_v = coverage_of("results_vanilla/best_generator.pth")
cov_f = coverage_of("results_fixed/best_generator.pth")

os.makedirs("results/visualizations", exist_ok=True)
plt.figure(figsize=(4, 3))
plt.bar(["vanilla", "fixed"], [cov_v, cov_f], width=0.6, color=["gray", "steelblue"])
plt.ylim(0, 1.05)
plt.ylabel("Mode Coverage")
plt.title("Vanilla vs Fixed GAN")
for i, v in enumerate([cov_v, cov_f]):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
save_path = "results/visualizations/vanilla_vs_fixed_coverage.png"
plt.savefig(save_path, dpi=150)
plt.close()

print(f"Saved comparison figure to {save_path}")
print(f"Vanilla coverage = {cov_v:.3f}, Fixed coverage = {cov_f:.3f}")
