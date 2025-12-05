#!/usr/bin/env python3
"""
Comparaison du splatting adaptatif vs fixe vs pas de splatting pour montrer la réduction du tearing.
Compare avec les timelines lambda = 3 et 10 pour montrer l'effet du splatting.
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

# Configuration matplotlib pour LaTeX
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

# Chemins
DATA_DIR = Path("/Data/janis.aiad/geodata/data/faces")
OUTPUT_DIR = Path("/Data/janis.aiad/geodata/refs/reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class OTConfig:
    """Configuration pour Transport Optimal 5D."""
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
    """Convertit une image (C, H, W) en nuage de points 5D (N, 5)."""
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
    """Splatting Vectorisé (Nadaraya-Watson Kernel Regression)."""
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
        self.loss_layer = SamplesLoss(
            loss="sinkhorn", p=2, blur=config.blur, reach=config.reach,
            debias=False, potentials=True, scaling=config.scaling, backend="auto"
        )

    def interpolate(self, img_source, img_target, times: List[float], use_splatting=True, use_adaptive_sigma=True, fixed_sigma=None):
        """Interpolation avec ou sans splatting."""
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
        
        # Préparer les images source et target redimensionnées
        img_source_resized = F.interpolate(
            img_source_device.unsqueeze(0), 
            size=(Ha, Wa), mode='bilinear'
        ).squeeze(0)
        img_target_resized = F.interpolate(
            img_target.unsqueeze(0).to(self.cfg.device), 
            size=(Hb, Wb), mode='bilinear'
        ).squeeze(0)
        
        for t in tqdm(times, desc=f"Interpolation (splatting={use_splatting})", leave=False):
            # À t=0, retourner l'image source directement
            if abs(t) < 1e-6:
                results.append(img_source_resized.cpu())
                continue
            
            # À t=1, retourner l'image target directement
            if abs(t - 1.0) < 1e-6:
                results.append(img_target_resized.cpu())
                continue
            
            # Interpolation intermédiaire
            pos_t = (1 - t) * pos_a_spatial + t * pos_b_spatial
            col_t = (1 - t) * col_a_real + t * col_b_real
            Ht = int((1 - t) * Ha + t * Hb)
            Wt = int((1 - t) * Wa + t * Wb)
            
            if use_splatting:
                # Calcul du sigma
                if use_adaptive_sigma:
                    # Sigma adaptatif
                    sigma_intrinsic = (1 - t) * self.cfg.sigma_start + t * self.cfg.sigma_end
                    sigma_expansion = self.cfg.sigma_boost * 4 * t * (1 - t)
                    current_spacing = np.sqrt((Ht * Wt) / (N_active + 1e-6))
                    min_sigma_t = current_spacing / 2.0
                    sigma_t = max(sigma_intrinsic + sigma_expansion, min_sigma_t * 0.8)
                else:
                    # Sigma fixe
                    if fixed_sigma is None:
                        fixed_sigma = 0.5
                    current_spacing = np.sqrt((Ht * Wt) / (N_active + 1e-6))
                    min_sigma_t = current_spacing / 2.0
                    sigma_t = max(fixed_sigma, min_sigma_t * 0.8)
                
                # Splatting
                img_t = vectorized_gaussian_splatting(pos_t, col_t, weights_ij, Ht, Wt, sigma=sigma_t)
            else:
                # Pas de splatting : placement direct (comme dans plot_lambda_ablation)
                pos_pix = pos_t * torch.tensor([Wt - 1, Ht - 1], device=pos_t.device)
                img_t = torch.zeros(3, Ht, Wt, device=pos_t.device)
                x_indices = (pos_pix[:, 0] + 0.5).long().clamp(0, Wt - 1)
                y_indices = (pos_pix[:, 1] + 0.5).long().clamp(0, Ht - 1)
                
                # Accumuler les couleurs pondérées
                for c in range(3):
                    img_t[c].index_put_(
                        (y_indices, x_indices), 
                        col_t[:, c] * weights_ij,
                        accumulate=True
                    )
                
                # Normaliser par les poids accumulés
                count = torch.zeros(Ht, Wt, device=pos_t.device)
                count.index_put_((y_indices, x_indices), weights_ij, accumulate=True)
                count = count.clamp(min=1e-6)
                img_t = img_t / count.unsqueeze(0)
            
            results.append(img_t.cpu())
        
        return results

def load_image(path: Path) -> torch.Tensor:
    """Charge une image et la convertit en tensor (C, H, W) normalisé."""
    img_pil = Image.open(path).convert("RGB")
    img = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float() / 255.0
    return img

def plot_splatting_tearing_comparison(img_source, img_target, times):
    """Compare le splatting adaptatif vs fixe vs pas de splatting pour montrer la réduction du tearing."""
    print("\n" + "=" * 80)
    print("COMPARAISON SPLATTING POUR RÉDUIRE LE TEARING")
    print("=" * 80)
    
    # Configuration de base
    base_config = OTConfig(
        resolution=(48, 48),
        blur=0.05,
        reach=0.3,
        lambda_color=2.0,  # On utilise lambda=2.0 pour la comparaison de base
        sigma_start=1.2,
        sigma_end=0.5,
        sigma_boost=0.5
    )
    
    # Configurations pour lambda = 3 et 10
    config_lambda3 = OTConfig(
        resolution=(48, 48),
        blur=0.05,
        reach=0.3,
        lambda_color=3.0,
        sigma_start=1.2,
        sigma_end=0.5,
        sigma_boost=0.5
    )
    
    config_lambda10 = OTConfig(
        resolution=(48, 48),
        blur=0.05,
        reach=0.3,
        lambda_color=10.0,
        sigma_start=1.2,
        sigma_end=0.5,
        sigma_boost=0.5
    )
    
    n_times = len(times)
    n_rows = 5  # lambda=3 no splatting, lambda=10 no splatting, adaptive, fixed, no splatting
    fig, axes = plt.subplots(n_rows, n_times, figsize=(n_times * 2.5, n_rows * 2.5))
    
    # Ligne 1: Lambda = 3, pas de splatting (pour montrer le tearing)
    print("\nCalcul pour λ=3.0, pas de splatting...")
    interpolator_l3 = OT5DInterpolator(config_lambda3)
    frames_l3 = interpolator_l3.interpolate(img_source, img_target, times, use_splatting=False)
    for j, (t, frame) in enumerate(zip(times, frames_l3)):
        ax = axes[0, j]
        np_img = frame.permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(np_img)
        if j == 0:
            ax.set_ylabel("$\\lambda=3.0$ (no splatting)", fontsize=14)
        if j == 0:
            ax.set_title(f"$t={t:.2f}$", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ligne 2: Lambda = 10, pas de splatting (pour montrer le tearing)
    print("\nCalcul pour λ=10.0, pas de splatting...")
    interpolator_l10 = OT5DInterpolator(config_lambda10)
    frames_l10 = interpolator_l10.interpolate(img_source, img_target, times, use_splatting=False)
    for j, (t, frame) in enumerate(zip(times, frames_l10)):
        ax = axes[1, j]
        np_img = frame.permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(np_img)
        if j == 0:
            ax.set_ylabel("$\\lambda=10.0$ (no splatting)", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ligne 3: Splatting adaptatif (lambda=2.0)
    print("\nCalcul avec splatting adaptatif (λ=2.0)...")
    interpolator_adaptive = OT5DInterpolator(base_config)
    frames_adaptive = interpolator_adaptive.interpolate(
        img_source, img_target, times, use_splatting=True, use_adaptive_sigma=True
    )
    for j, (t, frame) in enumerate(zip(times, frames_adaptive)):
        ax = axes[2, j]
        np_img = frame.permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(np_img)
        if j == 0:
            ax.set_ylabel("Adaptive $\\sigma(t)$", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ligne 4: Splatting fixe (lambda=2.0)
    print("\nCalcul avec splatting fixe σ=0.5 (λ=2.0)...")
    interpolator_fixed = OT5DInterpolator(base_config)
    frames_fixed = interpolator_fixed.interpolate(
        img_source, img_target, times, use_splatting=True, use_adaptive_sigma=False, fixed_sigma=0.5
    )
    for j, (t, frame) in enumerate(zip(times, frames_fixed)):
        ax = axes[3, j]
        np_img = frame.permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(np_img)
        if j == 0:
            ax.set_ylabel("Fixed $\\sigma=0.5$", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ligne 5: Pas de splatting (lambda=2.0)
    print("\nCalcul sans splatting (λ=2.0)...")
    interpolator_none = OT5DInterpolator(base_config)
    frames_none = interpolator_none.interpolate(img_source, img_target, times, use_splatting=False)
    for j, (t, frame) in enumerate(zip(times, frames_none)):
        ax = axes[4, j]
        np_img = frame.permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(np_img)
        if j == 0:
            ax.set_ylabel("No splatting", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "splatting_tearing_comparison_faces.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Sauvegardé: {output_path}")

def main():
    print("=" * 80)
    print("COMPARAISON SPLATTING POUR RÉDUIRE LE TEARING")
    print("=" * 80)
    
    # Chargement des images
    print("\nChargement des images...")
    img_source = load_image(DATA_DIR / "before.jpg")
    img_target = load_image(DATA_DIR / "after.jpg")
    print(f"Source: {img_source.shape}, Target: {img_target.shape}")
    
    # Temps d'interpolation
    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Génération du plot
    plot_splatting_tearing_comparison(img_source, img_target, times)
    
    print("\n" + "=" * 80)
    print("TERMINÉ - La visualisation a été créée dans:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()

