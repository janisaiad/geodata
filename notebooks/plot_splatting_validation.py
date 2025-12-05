#!/usr/bin/env python3
"""
Visualizations for qualitative validation of adaptive splatting.
Compares adaptive vs fixed splatting and shows Sinkhorn divergence evolution.
"""

import torch
import torch.nn.functional as F
from geomloss import SamplesLoss
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from dataclasses import dataclass
from typing import List, Optional
from tqdm import tqdm
from pathlib import Path

# Matplotlib configuration for LaTeX
plt.rcParams['figure.figsize'] = [6, 6]
plt.rcParams['font.size'] = 18
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 22
mpl.rcParams['axes.formatter.limits'] = (-6, 6)
mpl.rcParams['axes.formatter.use_mathtext'] = True
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.top'] = True

# Paths
DATA_DIR = Path("/Data/janis.aiad/geodata/data/faces")
OUTPUT_DIR = Path("/Data/janis.aiad/geodata/refs/reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class OTConfig:
    """Configuration for 5D Optimal Transport."""
    resolution: tuple[int] = (48, 48)
    blur: float = 0.05
    scaling: float = 0.9
    reach: Optional[float] = 0.3
    lambda_color: float = 2.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    sigma_start: float = 1.2
    sigma_end: float = 0.5
    sigma_boost: float = 0.5

def get_5d_cloud(img: torch.Tensor, res: int, lambda_c: float):
    """Converts an image (C, H, W) to a 5D point cloud (N, 5)."""
    C, H, W = img.shape
    scale = res / max(H, W)
    new_H, new_W = int(H * scale), int(W * scale)
    if new_H != H or new_W != W:
        img = F.interpolate(img.unsqueeze(0), size=(new_H, new_W), mode="bilinear").squeeze(0)
    y = torch.linspace(0, 1, new_H, device=img.device)
    x = torch.linspace(0, 1, new_W, device=img.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    pos_spatial = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    colors = img.permute(1, 2, 0).reshape(-1, C)
    cloud_5d = torch.cat([pos_spatial, lambda_c * colors], dim=-1)
    N = cloud_5d.shape[0]
    weights = torch.ones(N, device=img.device) / N
    return cloud_5d, weights, colors, new_H, new_W

def vectorized_gaussian_splatting(positions_2d, attributes, weights, H, W, sigma):
    """Vectorized Splatting (Nadaraya-Watson Kernel Regression)."""
    device = positions_2d.device
    N = positions_2d.shape[0]
    pos_pix = positions_2d * torch.tensor([W - 1, H - 1], device=device)
    radius = int(np.ceil(3 * sigma))
    diameter = 2 * radius + 1
    center_pix = torch.round(pos_pix).long()
    d_range = torch.arange(-radius, radius + 1, device=device)
    dy, dx = torch.meshgrid(d_range, d_range, indexing="ij")
    offsets = torch.stack([dx, dy], dim=-1).reshape(-1, 2)
    neighbor_coords = center_pix.unsqueeze(1) + offsets.unsqueeze(0)
    x_neigh = neighbor_coords[:, :, 0]
    y_neigh = neighbor_coords[:, :, 1]
    mask = (x_neigh >= 0) & (x_neigh < W) & (y_neigh >= 0) & (y_neigh < H)
    dist_sq = ((pos_pix.unsqueeze(1) - neighbor_coords.float()) ** 2).sum(dim=-1)
    gauss_weights = torch.exp(-dist_sq / (2 * sigma**2))
    contrib_weights = (gauss_weights * weights.unsqueeze(1)) * mask.float()
    flat_indices = y_neigh * W + x_neigh
    flat_indices = flat_indices.clamp(0, H * W - 1)
    denom = torch.zeros(H * W, device=device)
    denom.scatter_add_(0, flat_indices.view(-1), contrib_weights.view(-1))
    weighted_attribs = attributes.unsqueeze(1) * contrib_weights.unsqueeze(-1)
    C_channels = attributes.shape[1]
    numer = torch.zeros(H * W, C_channels, device=device)
    for c in range(C_channels):
        val_c = weighted_attribs[:, :, c].view(-1)
        numer[:, c].scatter_add_(0, flat_indices.view(-1), val_c)
    denom = denom.clamp(min=1e-6)
    out_img = numer / denom.unsqueeze(1)
    out_img = out_img.reshape(H, W, C_channels).permute(2, 0, 1)
    return out_img

def compute_sinkhorn_evolution(frames, img_target, config):
    """Computes Sinkhorn divergence S_eps(frame_t, target) for each frame."""
    metric_loss = SamplesLoss(
        loss="sinkhorn", p=2, blur=config.blur, reach=config.reach,
        scaling=config.scaling, debias=True
    )
    distances = []
    target_cloud, target_weights, _, _, _ = get_5d_cloud(
        img_target.to(config.device), config.resolution[1], config.lambda_color
    )
    print("Computing Sinkhorn distances...")
    for i, frame in enumerate(frames):
        current_cloud, current_weights, _, _, _ = get_5d_cloud(
            frame.to(config.device), config.resolution[0], config.lambda_color
        )
        loss_val = metric_loss(current_weights, current_cloud, target_weights, target_cloud)
        distances.append(loss_val.item())
        print(f"Frame {i}: Sinkhorn Div = {loss_val.item():.4f}")
    return distances

class OT5DInterpolator:
    def __init__(self, config: OTConfig):
        self.cfg = config
        self.loss_layer = SamplesLoss(
            loss="sinkhorn", p=2, blur=config.blur, reach=config.reach,
            debias=False, potentials=True, scaling=config.scaling, backend="auto"
        )

    def interpolate(self, img_source, img_target, times: List[float], use_adaptive_sigma=True, fixed_sigma=None):
        """Interpolation with adaptive or fixed splatting."""
        img_source_device = img_source.to(self.cfg.device)
        X_a, w_a, colors_a, Ha, Wa = get_5d_cloud(
            img_source_device, self.cfg.resolution[0], self.cfg.lambda_color
        )
        X_b, w_b, colors_b, Hb, Wb = get_5d_cloud(
            img_target.to(self.cfg.device), self.cfg.resolution[1], self.cfg.lambda_color
        )
        F_pot, G_pot = self.loss_layer(w_a, X_a, w_b, X_b)
        F_pot, G_pot = F_pot.flatten(), G_pot.flatten()
        dist_sq = torch.cdist(X_a, X_b, p=2) ** 2
        C_matrix = dist_sq / 2.0
        epsilon = self.cfg.blur**2
        log_pi = (
            (F_pot[:, None] + G_pot[None, :] - C_matrix) / epsilon
            + torch.log(w_a.flatten()[:, None])
            + torch.log(w_b.flatten()[None, :])
        )
        pi = torch.exp(log_pi).squeeze()
        mask = pi > (pi.max() * 1e-4)
        I_idx, J_idx = mask.nonzero(as_tuple=True)
        weights_ij = pi[I_idx, J_idx]
        pos_a_spatial = X_a[I_idx, :2]
        pos_b_spatial = X_b[J_idx, :2]
        col_a_real = colors_a[I_idx]
        col_b_real = colors_b[J_idx]
        results = []
        N_active = weights_ij.shape[0]
        
        # Prepare resized source and target images
        img_source_resized = F.interpolate(
            img_source_device.unsqueeze(0), 
            size=(Ha, Wa), mode='bilinear'
        ).squeeze(0)
        img_target_resized = F.interpolate(
            img_target.unsqueeze(0).to(self.cfg.device), 
            size=(Hb, Wb), mode='bilinear'
        ).squeeze(0)
        
        for t in tqdm(times, desc=f"Interpolation (adaptive={use_adaptive_sigma})", leave=False):
            # At t=0, return source image directly
            if abs(t) < 1e-6:
                results.append(img_source_resized.cpu())
                continue
            
            # At t=1, return target image directly
            if abs(t - 1.0) < 1e-6:
                results.append(img_target_resized.cpu())
                continue
            
            # Intermediate interpolation
            pos_t = (1 - t) * pos_a_spatial + t * pos_b_spatial
            col_t = (1 - t) * col_a_real + t * col_b_real
            Ht = int((1 - t) * Ha + t * Hb)
            Wt = int((1 - t) * Wa + t * Wb)
            
            # Compute sigma
            if use_adaptive_sigma:
                # Adaptive sigma
                sigma_intrinsic = (1 - t) * self.cfg.sigma_start + t * self.cfg.sigma_end
                sigma_expansion = self.cfg.sigma_boost * 4 * t * (1 - t)
                current_spacing = np.sqrt((Ht * Wt) / (N_active + 1e-6))
                min_sigma_t = current_spacing / 2.0
                sigma_t = max(sigma_intrinsic + sigma_expansion, min_sigma_t * 0.8)
            else:
                # Fixed sigma
                if fixed_sigma is None:
                    fixed_sigma = 0.5  # Default value
                current_spacing = np.sqrt((Ht * Wt) / (N_active + 1e-6))
                min_sigma_t = current_spacing / 2.0
                sigma_t = max(fixed_sigma, min_sigma_t * 0.8)
            
            # Splatting
            img_t = vectorized_gaussian_splatting(pos_t, col_t, weights_ij, Ht, Wt, sigma=sigma_t)
            results.append(img_t.cpu())
        
        return results

def load_image(path: Path) -> torch.Tensor:
    """Loads an image and converts it to a normalized tensor (C, H, W)."""
    img_pil = Image.open(path).convert("RGB")
    img = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float() / 255.0
    return img

def plot_sigma_comparison(img_source, img_target, times):
    """Compares adaptive vs fixed splatting."""
    print("\n" + "=" * 80)
    print("ADAPTIVE VS FIXED SPLATTING COMPARISON")
    print("=" * 80)
    
    config = OTConfig(
        resolution=(48, 48),
        blur=0.05,
        reach=0.3,
        lambda_color=2.0,
        sigma_start=1.2,
        sigma_end=0.5,
        sigma_boost=0.5
    )
    
    n_times = len(times)
    fig, axes = plt.subplots(2, n_times, figsize=(n_times * 2.5, 5))
    
    # Adaptive splatting
    print("\nComputing with adaptive splatting...")
    interpolator_adaptive = OT5DInterpolator(config)
    frames_adaptive = interpolator_adaptive.interpolate(img_source, img_target, times, use_adaptive_sigma=True)
    
    # Fixed splatting (sigma = 0.5)
    print("\nComputing with fixed splatting (σ=0.5)...")
    interpolator_fixed = OT5DInterpolator(config)
    frames_fixed = interpolator_fixed.interpolate(img_source, img_target, times, use_adaptive_sigma=False, fixed_sigma=0.5)
    
    for j, (t, frame_adapt, frame_fixed) in enumerate(zip(times, frames_adaptive, frames_fixed)):
        # Row 1: Adaptive
        ax1 = axes[0, j]
        np_img_adapt = frame_adapt.permute(1, 2, 0).clamp(0, 1).numpy()
        ax1.imshow(np_img_adapt)
        if j == 0:
            ax1.set_ylabel("Adaptive $\\sigma(t)$", fontsize=16)
        if j == 0:
            ax1.set_title(f"$t={t:.2f}$", fontsize=16)
        else:
            ax1.set_title(f"$t={t:.2f}$", fontsize=16)
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Row 2: Fixed
        ax2 = axes[1, j]
        np_img_fixed = frame_fixed.permute(1, 2, 0).clamp(0, 1).numpy()
        ax2.imshow(np_img_fixed)
        if j == 0:
            ax2.set_ylabel("Fixed $\\sigma=0.5$", fontsize=16)
        ax2.set_xticks([])
        ax2.set_yticks([])
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "splatting_adaptive_vs_fixed_faces.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {output_path}")

def plot_sinkhorn_evolution(img_source, img_target, times):
    """Generates a plot showing Sinkhorn divergence evolution."""
    print("\n" + "=" * 80)
    print("SINKHORN DIVERGENCE EVOLUTION")
    print("=" * 80)
    
    config = OTConfig(
        resolution=(48, 48),
        blur=0.05,
        reach=0.3,
        lambda_color=2.0,
        sigma_start=1.2,
        sigma_end=0.5,
        sigma_boost=0.5
    )
    
    # Compute with adaptive splatting
    print("\nComputing with adaptive splatting...")
    interpolator_adaptive = OT5DInterpolator(config)
    frames_adaptive = interpolator_adaptive.interpolate(img_source, img_target, times, use_adaptive_sigma=True)
    distances_adaptive = compute_sinkhorn_evolution(frames_adaptive, img_target, config)
    
    # Compute with fixed splatting
    print("\nComputing with fixed splatting (σ=0.5)...")
    interpolator_fixed = OT5DInterpolator(config)
    frames_fixed = interpolator_fixed.interpolate(img_source, img_target, times, use_adaptive_sigma=False, fixed_sigma=0.5)
    distances_fixed = compute_sinkhorn_evolution(frames_fixed, img_target, config)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, distances_adaptive, marker="o", linestyle="-", color="b", 
            linewidth=2, markersize=8, label="Adaptive $\\sigma(t)$")
    ax.plot(times, distances_fixed, marker="s", linestyle="--", color="r", 
            linewidth=2, markersize=8, label="Fixed $\\sigma=0.5$")
    ax.set_xlabel("Interpolation time $t$")
    ax.set_ylabel("Sinkhorn divergence $S_{\\varepsilon}(\\mu_t, \\nu)$")
    ax.set_title("Convergence to Target in 5D Wasserstein Space")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=14)
    plt.tight_layout()
    output_path = OUTPUT_DIR / "sinkhorn_evolution_comparison_faces.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {output_path}")
    
    # Display values in a table
    print("\n" + "=" * 80)
    print("SINKHORN DIVERGENCE VALUES")
    print("=" * 80)
    print(f"{'t':<8} {'Adaptive σ(t)':<15} {'Fixed σ=0.5':<15} {'Difference':<15}")
    print("-" * 60)
    for t, d_adapt, d_fixed in zip(times, distances_adaptive, distances_fixed):
        diff = d_fixed - d_adapt
        print(f"{t:<8.2f} {d_adapt:<15.4f} {d_fixed:<15.4f} {diff:<15.4f}")

def plot_sigma_evolution(times):
    """Generates a plot showing sigma(t) evolution for adaptive splatting."""
    print("\n" + "=" * 80)
    print("σ(t) EVOLUTION FOR ADAPTIVE SPLATTING")
    print("=" * 80)
    
    sigma_start = 1.2
    sigma_end = 0.5
    sigma_boost = 0.5
    
    # Compute sigma(t)
    sigma_adaptive = []
    sigma_fixed = 0.5
    
    for t in times:
        sigma_intrinsic = (1 - t) * sigma_start + t * sigma_end
        sigma_expansion = sigma_boost * 4 * t * (1 - t)
        sigma_t = sigma_intrinsic + sigma_expansion
        sigma_adaptive.append(sigma_t)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, sigma_adaptive, marker="o", linestyle="-", color="b", 
            linewidth=2, markersize=8, label="Adaptive $\\sigma(t)$")
    ax.axhline(y=sigma_fixed, color="r", linestyle="--", linewidth=2, label="Fixed $\\sigma=0.5$")
    ax.set_xlabel("Interpolation time $t$")
    ax.set_ylabel("Kernel width $\\sigma(t)$")
    ax.set_title("Kernel width evolution for adaptive splatting")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=14)
    plt.tight_layout()
    output_path = OUTPUT_DIR / "sigma_evolution_faces.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {output_path}")

def main():
    print("=" * 80)
    print("QUALITATIVE VALIDATION OF ADAPTIVE SPLATTING")
    print("=" * 80)
    
    # Load images
    print("\nLoading images...")
    img_source = load_image(DATA_DIR / "before.jpg")
    img_target = load_image(DATA_DIR / "after.jpg")
    print(f"Source: {img_source.shape}, Target: {img_target.shape}")
    
    # Interpolation times
    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Generate plots
    plot_sigma_comparison(img_source, img_target, times)
    plot_sinkhorn_evolution(img_source, img_target, times)
    plot_sigma_evolution(times)
    
    print("\n" + "=" * 80)
    print("COMPLETE - All visualizations have been created in:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()

