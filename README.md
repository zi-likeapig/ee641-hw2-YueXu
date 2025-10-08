# EE641 Homework 2: Generative Models and Hierarchical VAEs

**Name:** Yue Xu  
**USC Email:** yxu26506@usc.edu  

---

## How to Run

### 1) Problem 1 — GAN (Vanilla vs Fixed)

`problem1` is not switched by command-line arguments — you must **edit `train.py` directly**.  
Inside the file (near the top or before `main()`), there is a configuration dictionary such as `CONFIG = {"vanilla": {...}, "fixed": {...}}`.

1. Open `problem1/train.py` and set:
   - `cfg = CONFIG["vanilla"]` to train **Vanilla GAN**
   - `cfg = CONFIG["fixed"]` to train **Fixed (stabilized) GAN**
2. Run training:
```bash
cd problem1
python train.py
```
3. Visualize or compare results:
```bash
python training_dynamics.py      # loss curves, mode collapse inspection
python compare_vanilla_fixed.py  # mode coverage bar plot
```

**Result directory structure (slightly different from the official template):**  
For clarity, I separated vanilla and fixed results into different folders for side-by-side comparison.
```
problem1/
  results_vanilla/
    ckpts/                 # generator checkpoints at multiple epochs
    visualizations/        # alphabet_*.png, interpolation, histograms, etc.
    best_generator.pth
    training_log.json
  results_fixed/
    ckpts/
    visualizations/
    best_generator.pth
    training_log.json
  results/
    vanilla_vs_fixed_coverage.png   # summary comparison
```


---

### 2) Problem 2 — Hierarchical Variational Autoencoder (VAE)

**Train the model**
```bash
cd problem2
python train.py
```

**Latent analysis and creative experiments**
```bash
python analyze_latent.py   # KL curves, t-SNE, interpolation, style transfer
python experiments.py      # genre blending / complexity control / humanization / style consistency
```

**Results**
All generated figures and models are saved under:
```
problem2/results/
  generated_patterns/     # interpolation grids, transfers, long sequences, etc.
  best_model.pth
  training_log.json
  posterior_collapse.json   # per-dimension KL values
```
Detailed visuals and discussion are included in `report.pdf`.

---

## Notes

- Recommended environment: **PyTorch ≥ 2.0**  
- Problem 1 requires editing the config block in `train.py`; Problem 2 runs directly via provided scripts.  
- All key visualizations (coverage histograms, interpolation grids, style-consistent long sequences, etc.) are shown in `report.pdf` and `experiment.pdf`.  
- Creative experiments include genre blending, complexity control, humanization, and style consistency.

---

**Author:** Yue Xu  
EE 641 – Deep Learning Systems, Fall 2025  
University of Southern California
