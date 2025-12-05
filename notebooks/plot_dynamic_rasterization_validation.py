#!/usr/bin/env python3
"""
Visualizations for qualitative validation of dynamic rasterization grid.
Compares dynamic vs fixed grid for images with very different resolutions.
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
import gc
import signal
import sys

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

class OT5DInterpolator:
    def __init__(self, config: OTConfig):
        self.cfg = config
        # Use tensorized backend to avoid KeOps compilation issues that can cause segfaults
        try:
            self.loss_layer = SamplesLoss(
                loss="sinkhorn", p=2, blur=config.blur, reach=config.reach,
                debias=False, potentials=True, scaling=config.scaling, backend="tensorized"
            )
        except Exception as e:
            print(f"Warning: tensorized backend failed, trying auto: {e}")
            self.loss_layer = SamplesLoss(
                loss="sinkhorn", p=2, blur=config.blur, reach=config.reach,
                debias=False, potentials=True, scaling=config.scaling, backend="auto"
            )

    def interpolate(self, img_source, img_target, times: List[float], use_dynamic_grid=True):
        """Interpolation with dynamic or fixed grid."""
        img_source_device = img_source.to(self.cfg.device)
        X_a, w_a, colors_a, Ha, Wa = get_5d_cloud(
            img_source_device, self.cfg.resolution[0], self.cfg.lambda_color
        )
        X_b, w_b, colors_b, Hb, Wb = get_5d_cloud(
            img_target.to(self.cfg.device), self.cfg.resolution[1], self.cfg.lambda_color
        )
        
        # Clear GPU cache before Sinkhorn computation
        if self.cfg.device == "cuda":
            torch.cuda.empty_cache()
        
        try:
            F_pot, G_pot = self.loss_layer(w_a, X_a, w_b, X_b)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM error during Sinkhorn computation. Clearing cache and retrying...")
                if self.cfg.device == "cuda":
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
                # Retry once
                F_pot, G_pot = self.loss_layer(w_a, X_a, w_b, X_b)
            else:
                raise
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
        resolutions = []  # To track used resolutions
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
        
        for t in tqdm(times, desc=f"Interpolation (dynamic_grid={use_dynamic_grid})", leave=False):
            # At t=0, return source image directly
            if abs(t) < 1e-6:
                results.append(img_source_resized.cpu())
                resolutions.append((Ha, Wa))
                continue
            
            # At t=1, return target image directly
            if abs(t - 1.0) < 1e-6:
                results.append(img_target_resized.cpu())
                resolutions.append((Hb, Wb))
                continue
            
            # Intermediate interpolation
            pos_t = (1 - t) * pos_a_spatial + t * pos_b_spatial
            col_t = (1 - t) * col_a_real + t * col_b_real
            
            # Compute grid resolution
            if use_dynamic_grid:
                # Dynamic grid: linear interpolation between source and target
                Ht = int((1 - t) * Ha + t * Hb)
                Wt = int((1 - t) * Wa + t * Wb)
            else:
                # Fixed grid: always use target resolution
                Ht = Hb
                Wt = Wb
            
            resolutions.append((Ht, Wt))
            
            # Compute adaptive sigma
            sigma_intrinsic = (1 - t) * self.cfg.sigma_start + t * self.cfg.sigma_end
            sigma_expansion = self.cfg.sigma_boost * 4 * t * (1 - t)
            current_spacing = np.sqrt((Ht * Wt) / (N_active + 1e-6))
            min_sigma_t = current_spacing / 2.0
            sigma_t = max(sigma_intrinsic + sigma_expansion, min_sigma_t * 0.8)
            
            # Splatting
            img_t = vectorized_gaussian_splatting(pos_t, col_t, weights_ij, Ht, Wt, sigma=sigma_t)
            results.append(img_t.cpu())
        
        return results, resolutions

def load_image(path: Path) -> torch.Tensor:
    """Loads an image and converts it to a normalized tensor (C, H, W)."""
    img_pil = Image.open(path).convert("RGB")
    img = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float() / 255.0
    return img

def resize_image_to_resolution(img: torch.Tensor, target_H: int, target_W: int) -> torch.Tensor:
    """Resizes an image to a specific resolution."""
    return F.interpolate(img.unsqueeze(0), size=(target_H, target_W), mode='bilinear').squeeze(0)

def plot_dynamic_vs_fixed_grid(img_source, img_target, times):
    """Compares dynamic vs fixed grid."""
    print("\n" + "=" * 80)
    print("DYNAMIC VS FIXED GRID COMPARISON")
    print("=" * 80)
    
    # Configuration with very different resolutions
    # High resolution source, low resolution target
    # Using 96x64 instead of 128x64 to avoid memory issues
    config = OTConfig(
        resolution=(96, 64),  # Source 96x96, Target 64x64
        blur=0.05,
        reach=0.3,
        lambda_color=2.0,
        sigma_start=1.2,
        sigma_end=0.5,
        sigma_boost=0.5
    )
    
    n_times = len(times)
    fig, axes = plt.subplots(2, n_times, figsize=(n_times * 2.5, 5))
    
    # Dynamic grid
    print("\nComputing with dynamic grid...")
    interpolator_dynamic = OT5DInterpolator(config)
    frames_dynamic, resolutions_dynamic = interpolator_dynamic.interpolate(
        img_source, img_target, times, use_dynamic_grid=True
    )
    
    # Fixed grid
    print("\nComputing with fixed grid (target resolution)...")
    interpolator_fixed = OT5DInterpolator(config)
    frames_fixed, resolutions_fixed = interpolator_fixed.interpolate(
        img_source, img_target, times, use_dynamic_grid=False
    )
    
    for j, (t, frame_dyn, frame_fix, res_dyn, res_fix) in enumerate(
        zip(times, frames_dynamic, frames_fixed, resolutions_dynamic, resolutions_fixed)
    ):
        # Row 1: Dynamic grid
        ax1 = axes[0, j]
        np_img_dyn = frame_dyn.permute(1, 2, 0).clamp(0, 1).numpy()
        ax1.imshow(np_img_dyn)
        if j == 0:
            ax1.set_ylabel("Dynamic Grid", fontsize=16)
        title = f"$t={t:.2f}$\n$H={res_dyn[0]}, W={res_dyn[1]}$"
        ax1.set_title(title, fontsize=14)
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Row 2: Fixed grid
        ax2 = axes[1, j]
        np_img_fix = frame_fix.permute(1, 2, 0).clamp(0, 1).numpy()
        ax2.imshow(np_img_fix)
        if j == 0:
            ax2.set_ylabel("Fixed Grid", fontsize=16)
        title = f"$t={t:.2f}$\n$H={res_fix[0]}, W={res_fix[1]}$"
        ax2.set_title(title, fontsize=14)
        ax2.set_xticks([])
        ax2.set_yticks([])
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "dynamic_rasterization_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {output_path}")

def plot_resolution_evolution(times, Ha, Wa, Hb, Wb):
    """Generates a plot showing the evolution of grid resolution."""
    print("\n" + "=" * 80)
    print("GRID RESOLUTION EVOLUTION")
    print("=" * 80)
    
    # Compute dynamic resolutions
    H_dynamic = [int((1 - t) * Ha + t * Hb) for t in times]
    W_dynamic = [int((1 - t) * Wa + t * Wb) for t in times]
    H_fixed = [Hb] * len(times)
    W_fixed = [Wb] * len(times)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Height
    ax1.plot(times, H_dynamic, marker="o", linestyle="-", color="b", 
             linewidth=2, markersize=8, label="Dynamic Grid $H(t)$")
    ax1.axhline(y=Hb, color="r", linestyle="--", linewidth=2, label="Fixed Grid $H_{target}$")
    ax1.axhline(y=Ha, color="g", linestyle="--", linewidth=2, alpha=0.5, label="$H_{source}$")
    ax1.set_xlabel("Interpolation time $t$")
    ax1.set_ylabel("Height $H(t)$")
    ax1.set_title("Grid height evolution")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=12)
    
    # Width
    ax2.plot(times, W_dynamic, marker="o", linestyle="-", color="b", 
             linewidth=2, markersize=8, label="Dynamic Grid $W(t)$")
    ax2.axhline(y=Wb, color="r", linestyle="--", linewidth=2, label="Fixed Grid $W_{target}$")
    ax2.axhline(y=Wa, color="g", linestyle="--", linewidth=2, alpha=0.5, label="$W_{source}$")
    ax2.set_xlabel("Interpolation time $t$")
    ax2.set_ylabel("Width $W(t)$")
    ax2.set_title("Grid width evolution")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=12)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "resolution_evolution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {output_path}")
    
    # Display values in a table
    print("\n" + "=" * 80)
    print("RESOLUTION VALUES")
    print("=" * 80)
    print(f"{'t':<8} {'H_dynamic':<12} {'W_dynamic':<12} {'H_fixed':<12} {'W_fixed':<12}")
    print("-" * 60)
    for t, Hd, Wd, Hf, Wf in zip(times, H_dynamic, W_dynamic, H_fixed, W_fixed):
        print(f"{t:<8.2f} {Hd:<12} {Wd:<12} {Hf:<12} {Wf:<12}")

def compute_sinkhorn_evolution(frames, img_target, config, target_resolution=None):
    """Computes Sinkhorn divergence S_eps(frame_t, target) for each frame."""
    # Use tensorized backend to avoid KeOps compilation issues
    try:
        metric_loss = SamplesLoss(
            loss="sinkhorn", p=2, blur=config.blur, reach=config.reach,
            scaling=config.scaling, debias=True, backend="tensorized"
        )
    except Exception as e:
        print(f"Warning: tensorized backend failed, trying auto: {e}")
        metric_loss = SamplesLoss(
            loss="sinkhorn", p=2, blur=config.blur, reach=config.reach,
            scaling=config.scaling, debias=True, backend="auto"
        )
    
    distances = []
    # Use target resolution if provided, otherwise use config resolution
    target_res = target_resolution if target_resolution is not None else config.resolution[1]
    
    # Clear cache before computation
    if config.device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    
    try:
        target_cloud, target_weights, _, _, _ = get_5d_cloud(
            img_target.to(config.device), target_res, config.lambda_color
        )
    except Exception as e:
        print(f"Error creating target cloud: {e}")
        if config.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        raise
    
    print("Computing Sinkhorn distances...")
    for i, frame in enumerate(frames):
        try:
            # Get the resolution of the current frame
            _, H_frame, W_frame = frame.shape
            # Use the maximum dimension to determine the resolution for the 5D cloud
            frame_res = max(H_frame, W_frame)
            
            # Clear cache before each computation
            if config.device == "cuda":
                torch.cuda.empty_cache()
            
            current_cloud, current_weights, _, _, _ = get_5d_cloud(
                frame.to(config.device), frame_res, config.lambda_color
            )
            
            # Compute loss with error handling
            try:
                loss_val = metric_loss(current_weights, current_cloud, target_weights, target_cloud)
                distances.append(loss_val.item())
                print(f"Frame {i}: Sinkhorn Div = {loss_val.item():.4f}")
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    print(f"OOM error at frame {i}. Clearing cache and retrying...")
                    if config.device == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
                    # Retry once
                    loss_val = metric_loss(current_weights, current_cloud, target_weights, target_cloud)
                    distances.append(loss_val.item())
                    print(f"Frame {i}: Sinkhorn Div = {loss_val.item():.4f} (after retry)")
                else:
                    print(f"Error computing loss for frame {i}: {e}")
                    # Use a default value or skip
                    distances.append(float('nan'))
                    print(f"Frame {i}: Sinkhorn Div = NaN (error)")
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            import traceback
            traceback.print_exc()
            distances.append(float('nan'))
            if config.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    
    return distances

def plot_sinkhorn_evolution_comparison(img_source, img_target, times):
    """Generates a plot showing Sinkhorn divergence evolution with and without dynamic grid."""
    print("\n" + "=" * 80)
    print("SINKHORN DIVERGENCE EVOLUTION: DYNAMIC VS FIXED GRID")
    print("=" * 80)
    
    config = OTConfig(
        resolution=(96, 64),  # Source 96x96, Target 64x64
        blur=0.05,
        reach=0.3,
        lambda_color=2.0,
        sigma_start=1.2,
        sigma_end=0.5,
        sigma_boost=0.5
    )
    
    # Compute with dynamic grid
    print("\nComputing with dynamic grid...")
    try:
        interpolator_dynamic = OT5DInterpolator(config)
        frames_dynamic, _ = interpolator_dynamic.interpolate(
            img_source, img_target, times, use_dynamic_grid=True
        )
        
        # Clear cache before computing Sinkhorn divergence
        if config.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        # Use a lower resolution for Sinkhorn divergence computation to avoid memory issues
        # Resize frames to a common resolution (48x48) for fair comparison
        print("Resizing frames to 48x48 for Sinkhorn divergence computation...")
        frames_dynamic_resized = []
        for frame in frames_dynamic:
            frame_resized = F.interpolate(frame.unsqueeze(0), size=(48, 48), mode='bilinear').squeeze(0)
            frames_dynamic_resized.append(frame_resized)
        
        img_target_resized = F.interpolate(img_target.unsqueeze(0), size=(48, 48), mode='bilinear').squeeze(0)
        distances_dynamic = compute_sinkhorn_evolution(frames_dynamic_resized, img_target_resized, config, target_resolution=48)
    except Exception as e:
        print(f"Error during dynamic grid computation: {e}")
        import traceback
        traceback.print_exc()
        if config.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return
    
    # Clear cache
    if config.device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    
    # Compute with fixed grid
    print("\nComputing with fixed grid...")
    try:
        interpolator_fixed = OT5DInterpolator(config)
        frames_fixed, _ = interpolator_fixed.interpolate(
            img_source, img_target, times, use_dynamic_grid=False
        )
        
        # Clear cache before computing Sinkhorn divergence
        if config.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        # Resize frames to same resolution for fair comparison
        print("Resizing frames to 48x48 for Sinkhorn divergence computation...")
        frames_fixed_resized = []
        for frame in frames_fixed:
            frame_resized = F.interpolate(frame.unsqueeze(0), size=(48, 48), mode='bilinear').squeeze(0)
            frames_fixed_resized.append(frame_resized)
        
        distances_fixed = compute_sinkhorn_evolution(frames_fixed_resized, img_target_resized, config, target_resolution=48)
    except Exception as e:
        print(f"Error during fixed grid computation: {e}")
        import traceback
        traceback.print_exc()
        if config.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, distances_dynamic, marker="o", linestyle="-", color="b", 
            linewidth=2, markersize=8, label="Dynamic Grid")
    ax.plot(times, distances_fixed, marker="s", linestyle="--", color="r", 
            linewidth=2, markersize=8, label="Fixed Grid")
    ax.set_xlabel("Interpolation time $t$")
    ax.set_ylabel("Sinkhorn Divergence $S_{\\varepsilon}(\\mu_t, \\nu)$")
    ax.set_title("Convergence to Target in 5D Wasserstein Space")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=14)
    
    # Ajouter les hyperparamètres
    param_text = f"res={config.resolution[0]}×{config.resolution[1]}, ε={config.blur:.3f}, ρ={config.reach:.2f}, λ={config.lambda_color:.1f}, σ_start={config.sigma_start:.1f}, σ_end={config.sigma_end:.1f}, γ={config.sigma_boost:.1f}"
    fig.text(0.5, 0.02, f"Hyperparameters: {param_text}", 
             ha='center', fontsize=10, family='monospace')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    output_path = OUTPUT_DIR / "dynamic_grid_sinkhorn_evolution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {output_path}")
    
    # Afficher les valeurs dans un tableau
    print("\n" + "=" * 80)
    print("SINKHORN DIVERGENCE VALUES")
    print("=" * 80)
    print(f"{'t':<8} {'Dynamic Grid':<15} {'Fixed Grid':<15} {'Difference':<15}")
    print("-" * 60)
    for t, d_dyn, d_fix in zip(times, distances_dynamic, distances_fixed):
        diff = d_fix - d_dyn
        print(f"{t:<8.2f} {d_dyn:<15.4f} {d_fix:<15.4f} {diff:<15.4f}")

def plot_dynamic_rasterization_multiple_targets(img_source, img_target, times):
    """Generates a plot showing dynamic grid for multiple final resolutions."""
    print("\n" + "=" * 80)
    print("DYNAMIC GRID FOR MULTIPLE FINAL RESOLUTIONS")
    print("=" * 80)
    
    # High resolution source
    source_res = 96
    # Multiple target resolutions
    target_resolutions = [16, 32, 48, 64, 80]
    
    n_targets = len(target_resolutions)
    n_times = len(times)
    
    fig, axes = plt.subplots(n_targets, n_times, figsize=(n_times * 2.5, n_targets * 2.5))
    if n_targets == 1:
        axes = axes.reshape(1, -1)
    
    for i, target_res in enumerate(target_resolutions):
        print(f"\nComputing for target resolution {target_res}x{target_res}...")
        config = OTConfig(
            resolution=(source_res, target_res),
            blur=0.05,
            reach=0.3,
            lambda_color=2.0,
            sigma_start=1.2,
            sigma_end=0.5,
            sigma_boost=0.5
        )
        
        try:
            interpolator = OT5DInterpolator(config)
            frames, resolutions = interpolator.interpolate(
                img_source, img_target, times, use_dynamic_grid=True
            )
            
            for j, (t, frame, res) in enumerate(zip(times, frames, resolutions)):
                ax = axes[i, j]
                np_img = frame.permute(1, 2, 0).clamp(0, 1).numpy()
                ax.imshow(np_img)
                if i == 0:
                    ax.set_title(f"$t={t:.2f}$", fontsize=14)
                if j == 0:
                    ax.set_ylabel(f"Target: {target_res}×{target_res}\n$H(t), W(t)$", fontsize=12)
                # Show resolution in title for intermediate frames
                if j > 0 and j < n_times - 1:
                    ax.set_title(f"$t={t:.2f}$\n$H={res[0]}, W={res[1]}$", fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
        except Exception as e:
            print(f"Error for target resolution {target_res}: {e}")
            import traceback
            traceback.print_exc()
            # Fill with empty plots
            for j in range(n_times):
                ax = axes[i, j]
                ax.text(0.5, 0.5, f"Error\n{target_res}×{target_res}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Clear cache between different target resolutions
        if config.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    
    # Ajouter les hyperparamètres
    base_config = OTConfig(
        resolution=(source_res, target_resolutions[0]),
        blur=0.05,
        reach=0.3,
        lambda_color=2.0,
        sigma_start=1.2,
        sigma_end=0.5,
        sigma_boost=0.5
    )
    param_text = f"source_res={source_res}×{source_res}, target_res=varied, ε={base_config.blur:.3f}, ρ={base_config.reach:.2f}, λ={base_config.lambda_color:.1f}, σ_start={base_config.sigma_start:.1f}, σ_end={base_config.sigma_end:.1f}, γ={base_config.sigma_boost:.1f}"
    fig.text(0.5, 0.02, f"Hyperparameters (varying target resolution): {param_text}", 
             ha='center', fontsize=10, family='monospace')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    output_path = OUTPUT_DIR / "dynamic_rasterization_multiple_targets.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {output_path}")

def plot_aliasing_comparison(img_source, img_target):
    """Compares aliasing at t=0 for dynamic vs fixed grid."""
    print("\n" + "=" * 80)
    print("ALIASING COMPARISON AT t=0")
    print("=" * 80)
    
    config = OTConfig(
        resolution=(96, 64),  # Source 96x96, Target 64x64
        blur=0.05,
        reach=0.3,
        lambda_color=2.0,
        sigma_start=1.2,
        sigma_end=0.5,
        sigma_boost=0.5
    )
    
    # Get resolutions
    _, _, _, Ha, Wa = get_5d_cloud(
        img_source.to(config.device), config.resolution[0], config.lambda_color
    )
    _, _, _, Hb, Wb = get_5d_cloud(
        img_target.to(config.device), config.resolution[1], config.lambda_color
    )
    
    # Source image resized to its native resolution
    img_source_resized = F.interpolate(
        img_source.unsqueeze(0).to(config.device), 
        size=(Ha, Wa), mode='bilinear'
    ).squeeze(0)
    
    # Source image projected onto fixed grid (target)
    img_source_aliased = F.interpolate(
        img_source.unsqueeze(0).to(config.device), 
        size=(Hb, Wb), mode='bilinear'
    ).squeeze(0)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original source image
    ax1 = axes[0]
    np_img_orig = img_source.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    ax1.imshow(np_img_orig)
    ax1.set_title(f"Original Source\n$H={img_source.shape[1]}, W={img_source.shape[2]}$", fontsize=14)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Dynamic grid (native resolution at t=0)
    ax2 = axes[1]
    np_img_dyn = img_source_resized.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    ax2.imshow(np_img_dyn)
    ax2.set_title(f"Dynamic Grid at $t=0$\n$H={Ha}, W={Wa}$ (native resolution)", fontsize=14)
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Fixed grid (aliasing)
    ax3 = axes[2]
    np_img_fix = img_source_aliased.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    ax3.imshow(np_img_fix)
    ax3.set_title(f"Fixed Grid at $t=0$\n$H={Hb}, W={Wb}$ (severe aliasing)", fontsize=14)
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "aliasing_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {output_path}")

def main():
    print("=" * 80)
    print("QUALITATIVE VALIDATION OF DYNAMIC RASTERIZATION GRID")
    print("=" * 80)
    
    # Load images
    print("\nLoading images...")
    img_source = load_image(DATA_DIR / "before.jpg")
    img_target = load_image(DATA_DIR / "after.jpg")
    print(f"Source: {img_source.shape}, Target: {img_target.shape}")
    
    # Resize to create resolution contrast
    # High resolution source (96x96)
    img_source_hr = resize_image_to_resolution(img_source, 96, 96)
    # Low resolution target (64x64)
    img_target_lr = resize_image_to_resolution(img_target, 64, 64)
    
    print(f"Source HR: {img_source_hr.shape}, Target LR: {img_target_lr.shape}")
    
    # Interpolation times
    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Generate plots
    plot_dynamic_vs_fixed_grid(img_source_hr, img_target_lr, times)
    plot_aliasing_comparison(img_source_hr, img_target_lr)
    plot_sinkhorn_evolution_comparison(img_source_hr, img_target_lr, times)
    plot_dynamic_rasterization_multiple_targets(img_source_hr, img_target_lr, times)
    
    # Compute resolutions for evolution plot
    config = OTConfig(resolution=(96, 64))
    _, _, _, Ha, Wa = get_5d_cloud(
        img_source_hr.to(config.device), config.resolution[0], config.lambda_color
    )
    _, _, _, Hb, Wb = get_5d_cloud(
        img_target_lr.to(config.device), config.resolution[1], config.lambda_color
    )
    plot_resolution_evolution(times, Ha, Wa, Hb, Wb)
    
    print("\n" + "=" * 80)
    print("COMPLETE - All visualizations have been created in:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()

