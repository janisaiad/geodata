#!/usr/bin/env python3
"""
Plot timelines for different sigma_min values with adaptive splatting.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['mathtext.default'] = 'regular'
import logging
from datetime import datetime
import gc
import glob

# Import from ablation_studies
import sys
sys.path.append(str(Path(__file__).parent))
from ablation_studies import (
    OTConfig, get_5d_cloud, compute_sigma_max, compute_sigma_t,
    compute_average_interparticle_distance, load_image
)

# Configuration
DATA_DIR = Path("/Data/janis.aiad/geodata/data/pixelart/images")
EXPERIMENTS_DIR = Path("/Data/janis.aiad/geodata/experiments/pixelart")
OUTPUT_DIR = Path("/Data/janis.aiad/geodata/experiments/pixelart")
LOGS_DIR = Path("/Data/janis.aiad/geodata/logs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = LOGS_DIR / f"sigma_min_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def vectorized_gaussian_splatting(pos_t, col_t, weights_t, Ht, Wt, sigma):
    """Gaussian splatting reconstruction."""
    device = pos_t.device
    img_t = torch.zeros(3, Ht, Wt, device=device)
    count = torch.zeros(Ht, Wt, device=device)
    
    # Convert normalized positions to pixel coordinates
    pos_pix = pos_t * torch.tensor([Wt - 1, Ht - 1], device=device, dtype=pos_t.dtype)
    
    # Create grid
    y_coords = torch.arange(Ht, device=device, dtype=torch.float32)
    x_coords = torch.arange(Wt, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    grid = torch.stack([xx, yy], dim=-1)  # (Ht, Wt, 2)
    
    # For each particle, splat with Gaussian
    for i in range(pos_t.shape[0]):
        center = pos_pix[i]  # (2,)
        weight = weights_t[i]
        color = col_t[i]  # (3,)
        
        # Compute distances
        dists_sq = torch.sum((grid - center) ** 2, dim=-1)  # (Ht, Wt)
        
        # Gaussian kernel
        kernel = torch.exp(-dists_sq / (2 * sigma ** 2))
        
        # Accumulate
        for c in range(3):
            img_t[c] += kernel * color[c] * weight
        count += kernel * weight
    
    # Normalize
    count = count.clamp(min=1e-10)
    img_t = img_t / count.unsqueeze(0)
    img_t = img_t.clamp(0.0, 1.0)
    
    return img_t

def interpolate_at_t(transport_data, t: float, sigma_min: float, use_adaptive: bool = True):
    """Interpolate image at time t using transport plan with optional adaptive splatting."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pi = transport_data['pi'].to(device)
    X_a = transport_data['X_a'].to(device)
    X_b = transport_data['X_b'].to(device)
    colors_a = transport_data['colors_a'].to(device)
    colors_b = transport_data['colors_b'].to(device)
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
    
    # Ensure normalized coordinates stay in [0, 1] range
    pos_t_weighted = pos_t_weighted.clamp(0.0, 1.0)
    
    # Compute target resolution
    Ht = int((1 - t) * Ha + t * Hb)
    Wt = int((1 - t) * Wa + t * Wb)
    
    if use_adaptive:
        # Compute sigma_max
        sigma_max = compute_sigma_max(X_a, X_b, Ha, Wa, Hb, Wb)
        # Compute adaptive sigma
        sigma = compute_sigma_t(t, sigma_min, sigma_max)
        # Use Gaussian splatting
        img_t = vectorized_gaussian_splatting(pos_t_weighted, col_t_weighted, weights_t, Ht, Wt, sigma)
    else:
        # No splatting: direct placement
        pos_pix = pos_t_weighted * torch.tensor([Wt - 1, Ht - 1], device=device, dtype=pos_t_weighted.dtype)
        img_t = torch.zeros(3, Ht, Wt, device=device)
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
        count = torch.zeros(Ht, Wt, device=device)
        count.index_put_((y_indices, x_indices), weights_t, accumulate=True)
        count = count.clamp(min=1e-6)
        img_t = img_t / count.unsqueeze(0)
        img_t = img_t.clamp(0.0, 1.0)
    
    return img_t.cpu()

def plot_sigma_min_ablation():
    """Plot timelines for different sigma_min values across multiple eps and rho."""
    
    logger.info("=" * 80)
    logger.info("SIGMA_MIN ABLATION: TIMELINES FOR MULTIPLE EPS AND RHO")
    logger.info("=" * 80)
    
    # Parameters - multiple eps and rho values
    epsilons = [0.05, 0.07, 0.10]
    rhos = [0.5, 0.7, 1.0]
    lambda_color = 2.5
    sigma_mins = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    times = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, ..., 1.0
    
    # Load images
    logger.info("Loading images...")
    img_source = load_image(DATA_DIR / "salameche.webp")
    img_target = load_image(DATA_DIR / "strawberry.jpg")
    logger.info(f"Source: {img_source.shape}, Target: {img_target.shape}")
    
    # Iterate over all (eps, rho) combinations
    for eps in epsilons:
        for rho in rhos:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing eps={eps:.3f}, rho={rho:.2f}")
            logger.info(f"{'='*80}")
            
            # Find experiment files for this (eps, rho) combination
            logger.info("Loading transport plans...")
            transport_plans = {}
            for sigma_min in sigma_mins:
                sigma_min_str = f"{sigma_min:.2f}"
                rho_str = f"{rho:.2f}"
                pattern = f"*eps{eps:.3f}_rho{rho_str}_lam{lambda_color:.1f}_smin{sigma_min_str}_debiasTrue_adapsigmaTrue.pt"
                files = list(EXPERIMENTS_DIR.glob(pattern))
                if files:
                    transport_plans[sigma_min] = torch.load(files[0], weights_only=False)
                    logger.info(f"  Loaded: sigma_min={sigma_min:.3f}")
                else:
                    logger.warning(f"  Not found: sigma_min={sigma_min:.3f}")
            
            if len(transport_plans) == 0:
                logger.warning(f"No transport plans found for eps={eps:.3f}, rho={rho:.2f}. Skipping...")
                continue
            
            logger.info("\nComputing interpolations...")
            results = {}
            
            for sigma_min in sigma_mins:
                if sigma_min not in transport_plans:
                    continue
                
                transport_data = transport_plans[sigma_min]
                results[sigma_min] = {
                    'times': [],
                    'images_adaptive': []
                }
                
                for t in tqdm(times, desc=f"sigma_min={sigma_min:.3f}", leave=False):
                    # Adaptive splatting
                    img_adaptive = interpolate_at_t(transport_data, t, sigma_min, use_adaptive=True)
                    
                    results[sigma_min]['times'].append(t)
                    results[sigma_min]['images_adaptive'].append(img_adaptive)
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
            
            # Plot timelines for this (eps, rho) combination
            logger.info("\nPlotting timelines...")
            n_sigma = len([s for s in sigma_mins if s in results])
            if n_sigma == 0:
                continue
                
            fig_timeline, axes = plt.subplots(n_sigma, len(times), figsize=(len(times) * 2, n_sigma * 2))
            if n_sigma == 1:
                axes = axes.reshape(1, -1)
            
            for idx, sigma_min in enumerate(sorted([s for s in sigma_mins if s in results])):
                for t_idx, t in enumerate(times):
                    img = results[sigma_min]['images_adaptive'][t_idx]
                    img_np = img.permute(1, 2, 0).numpy()
                    axes[idx, t_idx].imshow(img_np)
                    axes[idx, t_idx].axis('off')
                    if idx == 0:
                        axes[idx, t_idx].set_title(f'$t={t:.1f}$', fontsize=10)
                    if t_idx == 0:
                        axes[idx, t_idx].set_ylabel(rf'$\sigma_{{min}}={sigma_min:.2f}$', fontsize=10)
            
            plt.suptitle(f'Timelines: $\varepsilon={eps:.3f}$, $\\rho={rho:.2f}$', fontsize=14)
            plt.tight_layout()
            rho_str = f"{rho:.2f}"
            timeline_path = OUTPUT_DIR / f"sigma_min_timelines_eps{eps:.3f}_rho{rho_str}.png"
            plt.savefig(timeline_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"✓ Saved timeline: {timeline_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("COMPLETED")
    logger.info("=" * 80)

if __name__ == "__main__":
    plot_sigma_min_ablation()

