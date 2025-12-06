#!/usr/bin/env python3
"""
Plot L2 loss between balanced and unbalanced OT reconstructions at t=0.5
as a function of rho, for different epsilon values.
Fixed lambda = 2.5.
"""

import torch
import torch.nn.functional as F
from geomloss import SamplesLoss
from PIL import Image
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = False  # Use matplotlib's built-in math renderer
matplotlib.rcParams['mathtext.default'] = 'regular'  # Use regular math font
import logging
from datetime import datetime
import gc

# Configuration
DATA_DIR = Path("/Data/janis.aiad/geodata/data/pixelart/images")
OUTPUT_DIR = Path("/Data/janis.aiad/geodata/experiments/pixelart")
LOGS_DIR = Path("/Data/janis.aiad/geodata/logs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = LOGS_DIR / f"rho_l2_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class OTConfig:
    """Configuration for 5D Optimal Transport."""
    resolution: tuple[int] = (64, 64)
    blur: float = 0.05
    scaling: float = 0.9
    reach: Optional[float] = None
    lambda_color: float = 2.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    sigma_start: float = 1.2
    sigma_end: float = 0.5
    sigma_boost: float = 0.5
    use_debias: bool = False
    use_adaptive_sigma: bool = True

def get_5d_cloud(img: torch.Tensor, res: int, lambda_c: float):
    """Convert image (C, H, W) to 5D point cloud (N, 5)."""
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

class OT5DInterpolator:
    def __init__(self, config: OTConfig):
        self.cfg = config
        try:
            self.loss_layer = SamplesLoss(
                loss="sinkhorn", p=2, blur=config.blur, reach=config.reach,
                debias=config.use_debias, potentials=True, scaling=config.scaling, backend="tensorized"
            )
        except Exception as e:
            logger.warning(f"tensorized backend failed, trying auto: {e}")
            self.loss_layer = SamplesLoss(
                loss="sinkhorn", p=2, blur=config.blur, reach=config.reach,
                debias=config.use_debias, potentials=True, scaling=config.scaling, backend="auto"
            )

    def compute_transport_plan(self, img_source, img_target):
        """Compute transport plan pi between source and target."""
        if self.cfg.device == "cuda":
            torch.cuda.empty_cache()
        
        X_a, w_a, colors_a, Ha, Wa = get_5d_cloud(
            img_source.to(self.cfg.device), self.cfg.resolution[0], self.cfg.lambda_color
        )
        X_b, w_b, colors_b, Hb, Wb = get_5d_cloud(
            img_target.to(self.cfg.device), self.cfg.resolution[1], self.cfg.lambda_color
        )
        
        logger.info(f"  Cloud sizes: source={X_a.shape[0]}, target={X_b.shape[0]}")
        
        if self.cfg.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        try:
            F_pot, G_pot = self.loss_layer(w_a, X_a, w_b, X_b)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                logger.warning(f"  OOM error during Sinkhorn computation. Clearing cache and retrying...")
                if self.cfg.device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
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
        
        return {
            'pi': pi.cpu(),
            'X_a': X_a.cpu(),
            'X_b': X_b.cpu(),
            'w_a': w_a.cpu(),
            'w_b': w_b.cpu(),
            'colors_a': colors_a.cpu(),
            'colors_b': colors_b.cpu(),
            'Ha': Ha,
            'Wa': Wa,
            'Hb': Hb,
            'Wb': Wb
        }

    def interpolate_at_t(self, transport_data, t: float):
        """Interpolate image at time t using transport plan."""
        pi = transport_data['pi'].to(self.cfg.device)
        X_a = transport_data['X_a'].to(self.cfg.device)
        X_b = transport_data['X_b'].to(self.cfg.device)
        colors_a = transport_data['colors_a'].to(self.cfg.device)
        colors_b = transport_data['colors_b'].to(self.cfg.device)
        Ha, Wa = transport_data['Ha'], transport_data['Wa']
        Hb, Wb = transport_data['Hb'], transport_data['Wb']
        
        # Compute barycentric weights
        weights_ij = pi / (pi.sum(dim=1, keepdim=True) + 1e-10)
        
        # Interpolate positions and colors
        pos_a_spatial = X_a[:, :2]
        pos_b_spatial = X_b[:, :2]
        pos_t = (1 - t) * pos_a_spatial[:, None, :] + t * pos_b_spatial[None, :, :]  # (N_a, N_b, 2)
        col_t = (1 - t) * colors_a[:, None, :] + t * colors_b[None, :, :]  # (N_a, N_b, 3)
        
        # Weighted average over target indices
        pos_t_weighted = (pos_t * weights_ij[:, :, None]).sum(dim=1)  # (N_a, 2)
        col_t_weighted = (col_t * weights_ij[:, :, None]).sum(dim=1)  # (N_a, 3)
        weights_t = pi.sum(dim=1)  # (N_a,)
        
        # Compute target resolution
        Ht = int((1 - t) * Ha + t * Hb)
        Wt = int((1 - t) * Wa + t * Wb)
        
        # Convert normalized positions to pixel coordinates
        pos_pix = pos_t_weighted * torch.tensor([Wt - 1, Ht - 1], device=pos_t_weighted.device)
        
        # Rasterize to image grid
        img_t = torch.zeros(3, Ht, Wt, device=pos_t_weighted.device)
        x_indices = (pos_pix[:, 0] + 0.5).long().clamp(0, Wt - 1)
        y_indices = (pos_pix[:, 1] + 0.5).long().clamp(0, Ht - 1)
        
        # Accumulate weighted colors
        for c in range(3):
            img_t[c].index_put_(
                (y_indices, x_indices),
                col_t_weighted[:, c] * weights_t,
                accumulate=True
            )
        
        # Normalize by accumulated weights
        count = torch.zeros(Ht, Wt, device=pos_t_weighted.device)
        count.index_put_((y_indices, x_indices), weights_t, accumulate=True)
        count = count.clamp(min=1e-6)
        img_t = img_t / count.unsqueeze(0)
        
        return img_t.cpu()

def load_image(path: Path) -> torch.Tensor:
    """Load image and convert to normalized tensor (C, H, W)."""
    img_pil = Image.open(path).convert("RGB")
    img = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float() / 255.0
    return img

def compute_l2_loss(img1: torch.Tensor, img2: torch.Tensor):
    """Compute L2 loss between two images."""
    # Ensure same resolution
    if img1.shape != img2.shape:
        H, W = min(img1.shape[1], img2.shape[1]), min(img1.shape[2], img2.shape[2])
        img1 = F.interpolate(img1.unsqueeze(0), size=(H, W), mode="bilinear").squeeze(0)
        img2 = F.interpolate(img2.unsqueeze(0), size=(H, W), mode="bilinear").squeeze(0)
    return F.mse_loss(img1, img2).item()

def compute_mean_displacement(transport_data):
    """Compute mean displacement from transport plan."""
    pi = transport_data['pi']
    X_a = transport_data['X_a']
    X_b = transport_data['X_b']
    Ha, Wa = transport_data['Ha'], transport_data['Wa']
    
    # Spatial positions
    pos_a_spatial = X_a[:, :2]  # (N_a, 2)
    pos_b_spatial = X_b[:, :2]  # (N_b, 2)
    
    # Compute barycentric target positions for each source point
    pi_row_sums = pi.sum(dim=1, keepdim=True)  # (N_a, 1)
    pi_normalized = pi / (pi_row_sums + 1e-10)  # (N_a, N_b)
    
    # Weighted target positions: (N_a, N_b) @ (N_b, 2) -> (N_a, 2)
    weighted_targets = torch.matmul(pi_normalized, pos_b_spatial)  # (N_a, 2)
    
    # Displacements in normalized coordinates
    displacements_norm = weighted_targets - pos_a_spatial  # (N_a, 2)
    
    # Convert to pixel coordinates
    displacement_pix = displacements_norm * torch.tensor([Wa, Ha], dtype=displacements_norm.dtype)
    
    # Compute magnitudes
    displacement_magnitudes = torch.norm(displacement_pix, dim=1)  # (N_a,)
    
    # Weight by transport mass
    weights = pi_row_sums.squeeze(1)  # (N_a,)
    mean_disp = (displacement_magnitudes * weights).sum() / (weights.sum() + 1e-10)
    
    return mean_disp.item()

def plot_rho_l2_comparison():
    """Plot L2 loss between balanced and unbalanced OT as function of rho."""
    
    logger.info("=" * 80)
    logger.info("L2 LOSS COMPARISON: BALANCED vs UNBALANCED OT")
    logger.info("=" * 80)
    
    # Load images
    logger.info("Loading images...")
    img_source = load_image(DATA_DIR / "salameche.webp")
    img_target = load_image(DATA_DIR / "strawberry.jpg")
    logger.info(f"Source: {img_source.shape}, Target: {img_target.shape}")
    
    # Parameters
    lambda_color = 2.5
    epsilons = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
    rhos = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.0]
    t = 0.5
    
    # Storage for results
    results = {eps: {'rhos': [], 'l2_losses': [], 'mean_displacements': []} for eps in epsilons}
    
    # Compute balanced OT reference for each epsilon
    logger.info("\nComputing balanced OT references...")
    balanced_reconstructions = {}
    
    for eps in epsilons:
        logger.info(f"\nEpsilon = {eps:.3f}")
        
        # Balanced OT (rho = None)
        config_balanced = OTConfig(
            resolution=(64, 64),
            blur=eps,
            reach=None,  # Balanced
            lambda_color=lambda_color,
            use_debias=False
        )
        
        interpolator_balanced = OT5DInterpolator(config_balanced)
        transport_balanced = interpolator_balanced.compute_transport_plan(img_source, img_target)
        balanced_img = interpolator_balanced.interpolate_at_t(transport_balanced, t)
        balanced_reconstructions[eps] = balanced_img
        
        logger.info(f"  Balanced reconstruction shape: {balanced_img.shape}")
        
        # Clean up
        del transport_balanced, interpolator_balanced
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Now compute unbalanced OT for each rho
        logger.info(f"  Computing unbalanced OT for different rho values...")
        
        for rho in tqdm(rhos, desc=f"  eps={eps:.3f}", leave=False):
            config_unbalanced = OTConfig(
                resolution=(64, 64),
                blur=eps,
                reach=rho,
                lambda_color=lambda_color,
                use_debias=False
            )
            
            try:
                interpolator_unbalanced = OT5DInterpolator(config_unbalanced)
                transport_unbalanced = interpolator_unbalanced.compute_transport_plan(img_source, img_target)
                unbalanced_img = interpolator_unbalanced.interpolate_at_t(transport_unbalanced, t)
                
                # Compute L2 loss
                l2_loss = compute_l2_loss(balanced_img, unbalanced_img)
                
                # Compute mean displacement
                mean_disp = compute_mean_displacement(transport_unbalanced)
                
                results[eps]['rhos'].append(rho)
                results[eps]['l2_losses'].append(l2_loss)
                results[eps]['mean_displacements'].append(mean_disp)
                
                logger.info(f"    rho={rho:.3f}: L2 loss = {l2_loss:.6f}, Mean displacement = {mean_disp:.4f} px")
                
                # Clean up
                del transport_unbalanced, interpolator_unbalanced, unbalanced_img
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                logger.error(f"    Error for rho={rho:.3f}: {e}")
                continue
    
    # Plot results
    logger.info("\nPlotting results...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(epsilons)))
    
    # Plot mean displacement
    for eps, color in zip(epsilons, colors):
        if len(results[eps]['rhos']) > 0:
            # Highlight eps = 0.05 and 0.07 as perfect
            linewidth = 3 if eps in [0.05, 0.07] else 2
            linestyle = '-' if eps in [0.05, 0.07] else '-'
            alpha = 1.0 if eps in [0.05, 0.07] else 0.8
            ax.plot(
                results[eps]['rhos'],
                results[eps]['mean_displacements'],
                marker='o',
                label=rf'$\varepsilon = {eps:.3f}$' + (' (optimal)' if eps in [0.05, 0.07] else ''),
                color=color,
                linewidth=linewidth,
                markersize=6,
                linestyle=linestyle,
                alpha=alpha
            )
    
    # Find and plot inflection point for eps = 0.07
    eps_target = 0.07
    if eps_target in results and len(results[eps_target]['rhos']) > 2:
        rhos_eps = np.array(results[eps_target]['rhos'])
        disp_eps = np.array(results[eps_target]['mean_displacements'])
        
        # Sort by rho to ensure proper order
        sort_idx = np.argsort(rhos_eps)
        rhos_eps = rhos_eps[sort_idx]
        disp_eps = disp_eps[sort_idx]
        
        # Compute second derivative to find inflection point
        # Use log scale for x to account for log scale in plot
        log_rhos = np.log(rhos_eps)
        
        # First derivative
        d1 = np.gradient(disp_eps, log_rhos)
        # Second derivative
        d2 = np.gradient(d1, log_rhos)
        
        # Find inflection point: where second derivative crosses zero or is minimum
        sign_changes = np.where(np.diff(np.sign(d2)))[0]
        if len(sign_changes) > 0:
            # Use the first sign change (inflection point)
            inflection_idx = sign_changes[0]
        else:
            # If no sign change, use point of minimum second derivative (maximum curvature change)
            inflection_idx = np.argmin(d2)
        
        rho_inflection = rhos_eps[inflection_idx]
        disp_inflection = disp_eps[inflection_idx]
        
        # Plot vertical line at inflection point
        ax.axvline(x=rho_inflection, color='red', linestyle='--', linewidth=2.5, 
                   label=rf'Inflection point: $\rho = {rho_inflection:.3f}$ ($\varepsilon = {eps_target:.2f}$)',
                   zorder=10)
        
        logger.info(f"Inflection point for eps={eps_target}: rho={rho_inflection:.4f}, displacement={disp_inflection:.4f}")
    
    ax.set_xlabel(r'$\rho$ (Reach parameter)', fontsize=12)
    ax.set_ylabel('Mean Displacement (pixels)', fontsize=12)
    ax.set_title(rf'Mean Displacement at $t=0.5$ ($\lambda = {lambda_color}$)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    
    # Save figure
    output_path = OUTPUT_DIR / f"rho_l2_comparison_lambda{lambda_color:.1f}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"\n✓ Saved figure: {output_path}")
    
    # Save data
    data_path = OUTPUT_DIR / f"rho_l2_comparison_lambda{lambda_color:.1f}.npz"
    np.savez(
        data_path,
        epsilons=epsilons,
        rhos=rhos,
        results={eps: {
            'rhos': np.array(results[eps]['rhos']),
            'l2_losses': np.array(results[eps]['l2_losses']),
            'mean_displacements': np.array(results[eps]['mean_displacements'])
        } for eps in epsilons}
    )
    logger.info(f"✓ Saved data: {data_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("COMPLETED")
    logger.info("=" * 80)

if __name__ == "__main__":
    plot_rho_l2_comparison()

