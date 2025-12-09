#!/usr/bin/env python3
"""
Plot general transport timelines for different eps and rho values.
Shows the transport interpolation at different times t.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Optional
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['mathtext.default'] = 'regular'
import logging
from datetime import datetime
import gc

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
log_file = LOGS_DIR / f"transport_timelines_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

def interpolate_at_t(transport_data, t: float, sigma_min: float = 0.1):
    """Interpolate image at time t using transport plan with adaptive splatting."""
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
    
    # For old files without adaptive splatting, use simple direct placement
    # Check if we should use adaptive splatting (for new files) or direct placement (for old files)
    # Use a fixed small sigma for old files
    sigma = 0.5  # Fixed sigma for old transport plans
    
    # Use Gaussian splatting
    img_t = vectorized_gaussian_splatting(pos_t_weighted, col_t_weighted, weights_t, Ht, Wt, sigma)
    
    return img_t.cpu()

def plot_transport_timelines():
    """Plot general transport timelines for different eps and rho values."""
    
    logger.info("=" * 80)
    logger.info("TRANSPORT TIMELINES FOR MULTIPLE EPS AND RHO")
    logger.info("=" * 80)
    
    # Parameters
    lambda_color = 2.5
    times = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, ..., 1.0
    
    # Find all transport plan files (old format without smin)
    logger.info("\nFinding transport plan files...")
    pattern = f"*_lam{lambda_color:.1f}_debiasFalse_adapsigmaFalse.pt"
    all_files = list(EXPERIMENTS_DIR.glob(pattern))
    logger.info(f"Found {len(all_files)} transport plan files")
    
    if len(all_files) == 0:
        logger.error("No transport plans found! Run ablation_studies.py first.")
        return
    
    # Process each file
    for file_path in all_files:
        try:
            # Parse filename to extract eps and rho
            filename = file_path.stem
            # Format: exp_XXXX_eps0.XXX_rhoX.XX_lam2.5_debiasFalse_adapsigmaFalse
            parts = filename.split('_')
            eps = None
            rho = None
            for part in parts:
                if part.startswith('eps'):
                    eps = float(part[3:])
                elif part.startswith('rho'):
                    rho = float(part[3:])
            
            if eps is None or rho is None:
                logger.warning(f"Could not parse eps/rho from {filename}, skipping")
                continue
            
            logger.info(f"\nProcessing: eps={eps:.3f}, rho={rho:.2f}")
            
            # Load transport plan
            transport_data = torch.load(file_path, weights_only=False)
            logger.info(f"  Loaded transport plan")
            
            # Compute interpolations
            images = []
            for t in tqdm(times, desc=f"  Computing", leave=False):
                img = interpolate_at_t(transport_data, t)
                images.append(img)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            # Plot timeline
            logger.info(f"  Plotting...")
            n_times = len(times)
            fig, axes = plt.subplots(1, n_times, figsize=(n_times * 1.5, 2))
            if n_times == 1:
                axes = [axes]
            
            for t_idx, t in enumerate(times):
                img_np = images[t_idx].permute(1, 2, 0).numpy()
                axes[t_idx].imshow(img_np)
                axes[t_idx].axis('off')
                axes[t_idx].set_title(f'$t={t:.1f}$', fontsize=10)
            
            # Add main title with eps and rho values
            rho_str = f"{rho:.2f}"
            plt.suptitle(f'Transport Timeline: $\\varepsilon={eps:.3f}$, $\\rho={rho_str}$', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save individual plot
            timeline_path = OUTPUT_DIR / f"transport_timeline_eps{eps:.3f}_rho{rho_str}.png"
            plt.savefig(timeline_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"  ✓ Saved: {timeline_path}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    logger.info("\n" + "=" * 80)
    logger.info("COMPLETED")
    logger.info("=" * 80)

if __name__ == "__main__":
    plot_transport_timelines()

