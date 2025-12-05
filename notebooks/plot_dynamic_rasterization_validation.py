#!/usr/bin/env python3
"""
Visualisations pour la validation qualitative de la grille de rasterisation dynamique.
Compare la grille dynamique vs fixe pour des images de résolutions très différentes.
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
DATA_DIR = Path("/Data/janis.aiad/geodata/data/pixelart/images")
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

    def interpolate(self, img_source, img_target, times: List[float], use_dynamic_grid=True):
        """Interpolation avec grille dynamique ou fixe."""
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
        resolutions = []  # Pour tracker les résolutions utilisées
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
        
        for t in tqdm(times, desc=f"Interpolation (dynamic_grid={use_dynamic_grid})", leave=False):
            # À t=0, retourner l'image source directement
            if abs(t) < 1e-6:
                results.append(img_source_resized.cpu())
                resolutions.append((Ha, Wa))
                continue
            
            # À t=1, retourner l'image target directement
            if abs(t - 1.0) < 1e-6:
                results.append(img_target_resized.cpu())
                resolutions.append((Hb, Wb))
                continue
            
            # Interpolation intermédiaire
            pos_t = (1 - t) * pos_a_spatial + t * pos_b_spatial
            col_t = (1 - t) * col_a_real + t * col_b_real
            
            # Calcul de la résolution de la grille
            if use_dynamic_grid:
                # Grille dynamique : interpolation linéaire entre source et target
                Ht = int((1 - t) * Ha + t * Hb)
                Wt = int((1 - t) * Wa + t * Wb)
            else:
                # Grille fixe : toujours utiliser la résolution du target
                Ht = Hb
                Wt = Wb
            
            resolutions.append((Ht, Wt))
            
            # Calcul du sigma adaptatif
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
    """Charge une image et la convertit en tensor (C, H, W) normalisé."""
    img_pil = Image.open(path).convert("RGB")
    img = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float() / 255.0
    return img

def resize_image_to_resolution(img: torch.Tensor, target_H: int, target_W: int) -> torch.Tensor:
    """Redimensionne une image à une résolution spécifique."""
    return F.interpolate(img.unsqueeze(0), size=(target_H, target_W), mode='bilinear').squeeze(0)

def plot_dynamic_vs_fixed_grid(img_source, img_target, times):
    """Compare la grille dynamique vs fixe."""
    print("\n" + "=" * 80)
    print("COMPARAISON GRILLE DYNAMIQUE VS FIXE")
    print("=" * 80)
    
    # Configuration avec résolutions très différentes
    # Source haute résolution, target basse résolution
    config = OTConfig(
        resolution=(128, 16),  # Source 128x128, Target 16x16
        blur=0.05,
        reach=0.3,
        lambda_color=2.0,
        sigma_start=1.2,
        sigma_end=0.5,
        sigma_boost=0.5
    )
    
    n_times = len(times)
    fig, axes = plt.subplots(2, n_times, figsize=(n_times * 2.5, 5))
    
    # Grille dynamique
    print("\nCalcul avec grille dynamique...")
    interpolator_dynamic = OT5DInterpolator(config)
    frames_dynamic, resolutions_dynamic = interpolator_dynamic.interpolate(
        img_source, img_target, times, use_dynamic_grid=True
    )
    
    # Grille fixe
    print("\nCalcul avec grille fixe (résolution target)...")
    interpolator_fixed = OT5DInterpolator(config)
    frames_fixed, resolutions_fixed = interpolator_fixed.interpolate(
        img_source, img_target, times, use_dynamic_grid=False
    )
    
    for j, (t, frame_dyn, frame_fix, res_dyn, res_fix) in enumerate(
        zip(times, frames_dynamic, frames_fixed, resolutions_dynamic, resolutions_fixed)
    ):
        # Ligne 1: Grille dynamique
        ax1 = axes[0, j]
        np_img_dyn = frame_dyn.permute(1, 2, 0).clamp(0, 1).numpy()
        ax1.imshow(np_img_dyn)
        if j == 0:
            ax1.set_ylabel("Dynamic Grid", fontsize=16)
        title = f"$t={t:.2f}$\n$H={res_dyn[0]}, W={res_dyn[1]}$"
        ax1.set_title(title, fontsize=14)
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Ligne 2: Grille fixe
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
    print(f"\n✓ Sauvegardé: {output_path}")

def plot_resolution_evolution(times, Ha, Wa, Hb, Wb):
    """Génère un plot montrant l'évolution de la résolution de la grille."""
    print("\n" + "=" * 80)
    print("ÉVOLUTION DE LA RÉSOLUTION DE LA GRILLE")
    print("=" * 80)
    
    # Calcul des résolutions dynamiques
    H_dynamic = [int((1 - t) * Ha + t * Hb) for t in times]
    W_dynamic = [int((1 - t) * Wa + t * Wb) for t in times]
    H_fixed = [Hb] * len(times)
    W_fixed = [Wb] * len(times)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Hauteur
    ax1.plot(times, H_dynamic, marker="o", linestyle="-", color="b", 
             linewidth=2, markersize=8, label="Dynamic Grid $H(t)$")
    ax1.axhline(y=Hb, color="r", linestyle="--", linewidth=2, label="Fixed Grid $H_{target}$")
    ax1.axhline(y=Ha, color="g", linestyle="--", linewidth=2, alpha=0.5, label="$H_{source}$")
    ax1.set_xlabel("Temps d'interpolation $t$")
    ax1.set_ylabel("Hauteur $H(t)$")
    ax1.set_title("Évolution de la hauteur de la grille")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=12)
    
    # Largeur
    ax2.plot(times, W_dynamic, marker="o", linestyle="-", color="b", 
             linewidth=2, markersize=8, label="Dynamic Grid $W(t)$")
    ax2.axhline(y=Wb, color="r", linestyle="--", linewidth=2, label="Fixed Grid $W_{target}$")
    ax2.axhline(y=Wa, color="g", linestyle="--", linewidth=2, alpha=0.5, label="$W_{source}$")
    ax2.set_xlabel("Temps d'interpolation $t$")
    ax2.set_ylabel("Largeur $W(t)$")
    ax2.set_title("Évolution de la largeur de la grille")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=12)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "resolution_evolution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Sauvegardé: {output_path}")
    
    # Afficher les valeurs dans un tableau
    print("\n" + "=" * 80)
    print("VALEURS DE RÉSOLUTION")
    print("=" * 80)
    print(f"{'t':<8} {'H_dynamic':<12} {'W_dynamic':<12} {'H_fixed':<12} {'W_fixed':<12}")
    print("-" * 60)
    for t, Hd, Wd, Hf, Wf in zip(times, H_dynamic, W_dynamic, H_fixed, W_fixed):
        print(f"{t:<8.2f} {Hd:<12} {Wd:<12} {Hf:<12} {Wf:<12}")

def plot_aliasing_comparison(img_source, img_target):
    """Compare l'aliasing à t=0 pour grille dynamique vs fixe."""
    print("\n" + "=" * 80)
    print("COMPARAISON DE L'ALIASING À t=0")
    print("=" * 80)
    
    config = OTConfig(
        resolution=(128, 16),  # Source 128x128, Target 16x16
        blur=0.05,
        reach=0.3,
        lambda_color=2.0,
        sigma_start=1.2,
        sigma_end=0.5,
        sigma_boost=0.5
    )
    
    # Obtenir les résolutions
    _, _, _, Ha, Wa = get_5d_cloud(
        img_source.to(config.device), config.resolution[0], config.lambda_color
    )
    _, _, _, Hb, Wb = get_5d_cloud(
        img_target.to(config.device), config.resolution[1], config.lambda_color
    )
    
    # Image source redimensionnée à sa résolution native
    img_source_resized = F.interpolate(
        img_source.unsqueeze(0).to(config.device), 
        size=(Ha, Wa), mode='bilinear'
    ).squeeze(0)
    
    # Image source projetée sur la grille fixe (target)
    img_source_aliased = F.interpolate(
        img_source.unsqueeze(0).to(config.device), 
        size=(Hb, Wb), mode='bilinear'
    ).squeeze(0)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Image source originale
    ax1 = axes[0]
    np_img_orig = img_source.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    ax1.imshow(np_img_orig)
    ax1.set_title(f"Source Originale\n$H={img_source.shape[1]}, W={img_source.shape[2]}$", fontsize=14)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Grille dynamique (résolution native à t=0)
    ax2 = axes[1]
    np_img_dyn = img_source_resized.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    ax2.imshow(np_img_dyn)
    ax2.set_title(f"Dynamic Grid à $t=0$\n$H={Ha}, W={Wa}$ (résolution native)", fontsize=14)
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Grille fixe (aliasing)
    ax3 = axes[2]
    np_img_fix = img_source_aliased.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    ax3.imshow(np_img_fix)
    ax3.set_title(f"Fixed Grid à $t=0$\n$H={Hb}, W={Wb}$ (aliasing sévère)", fontsize=14)
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "aliasing_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Sauvegardé: {output_path}")

def main():
    print("=" * 80)
    print("VALIDATION QUALITATIVE DE LA GRILLE DE RASTERISATION DYNAMIQUE")
    print("=" * 80)
    
    # Chargement des images
    print("\nChargement des images...")
    img_source = load_image(DATA_DIR / "salameche.webp")
    img_target = load_image(DATA_DIR / "strawberry.jpg")
    print(f"Source: {img_source.shape}, Target: {img_target.shape}")
    
    # Redimensionner pour créer un contraste de résolution
    # Source haute résolution (128x128)
    img_source_hr = resize_image_to_resolution(img_source, 128, 128)
    # Target basse résolution (16x16)
    img_target_lr = resize_image_to_resolution(img_target, 16, 16)
    
    print(f"Source HR: {img_source_hr.shape}, Target LR: {img_target_lr.shape}")
    
    # Temps d'interpolation
    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Génération des plots
    plot_dynamic_vs_fixed_grid(img_source_hr, img_target_lr, times)
    plot_aliasing_comparison(img_source_hr, img_target_lr)
    
    # Calcul des résolutions pour le plot d'évolution
    config = OTConfig(resolution=(128, 16))
    _, _, _, Ha, Wa = get_5d_cloud(
        img_source_hr.to(config.device), config.resolution[0], config.lambda_color
    )
    _, _, _, Hb, Wb = get_5d_cloud(
        img_target_lr.to(config.device), config.resolution[1], config.lambda_color
    )
    plot_resolution_evolution(times, Ha, Wa, Hb, Wb)
    
    print("\n" + "=" * 80)
    print("TERMINÉ - Toutes les visualisations ont été créées dans:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()

