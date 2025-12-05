#!/usr/bin/env python3
"""
Visualisations pour l'étude d'ablation sur la stratégie 5D Joint Lifting.
Génère des timelines d'interpolation et des champs de déplacement pour différentes valeurs de lambda.
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

def format_hyperparameters(config: OTConfig, varying_param: str = None, varying_value: str = None):
    """Formate les hyperparamètres pour affichage sur les figures."""
    params = []
    params.append(f"res={config.resolution[0]}×{config.resolution[1]}")
    params.append(f"ε={config.blur:.3f}")
    if config.reach is None:
        params.append("ρ=Balanced")
    else:
        params.append(f"ρ={config.reach:.2f}")
    params.append(f"λ={config.lambda_color:.1f}")
    params.append(f"σ_start={config.sigma_start:.1f}")
    params.append(f"σ_end={config.sigma_end:.1f}")
    params.append(f"γ={config.sigma_boost:.1f}")
    param_str = ", ".join(params)
    if varying_param and varying_value:
        param_str = f"{varying_param}={varying_value} | " + param_str
    return param_str

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

    def interpolate(self, img_source, img_target, times: List[float], use_minimal_sigma=False):
        """Interpolation de Wasserstein avec barycentres géodésiques - SANS SPLATTING.
        
        Utilise uniquement le barycentre géodésique pur, sans flou gaussien.
        """
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
        
        for t in tqdm(times, desc=f"Interpolation λ={self.cfg.lambda_color:.1f}", leave=False):
            # À t=0, retourner l'image source directement (sans flou)
            if abs(t) < 1e-6:
                results.append(img_source_resized.cpu())
                continue
            
            # À t=1, retourner l'image target directement (sans flou)
            if abs(t - 1.0) < 1e-6:
                results.append(img_target_resized.cpu())
                continue
            
            # Pour les interpolations intermédiaires - SANS SPLATTING
            # Juste placement direct des pixels aux positions interpolées (barycentre géodésique pur)
            pos_t = (1 - t) * pos_a_spatial + t * pos_b_spatial
            col_t = (1 - t) * col_a_real + t * col_b_real
            Ht = int((1 - t) * Ha + t * Hb)
            Wt = int((1 - t) * Wa + t * Wb)
            
            # Convertir les positions normalisées [0,1] en coordonnées pixels
            pos_pix = pos_t * torch.tensor([Wt - 1, Ht - 1], device=pos_t.device)
            
            # Placement direct sans splatting : scatter les pixels aux positions interpolées
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
        return results, pos_a_spatial.cpu(), pos_b_spatial.cpu(), weights_ij.cpu(), I_idx, J_idx, Ha, Wa, X_a.cpu(), X_b.cpu(), pi.cpu()

def compute_displacement_field(X_a, X_b, pi, Ha, Wa):
    """Calcule le champ de déplacement à partir du plan de transport complet.
    
    Utilise le plan de transport pi pour calculer le déplacement barycentrique
    pour chaque pixel de la grille source.
    """
    # Positions spatiales (2 premières dimensions de X_a et X_b)
    pos_a_spatial = X_a[:, :2]  # (N_a, 2) positions source
    pos_b_spatial = X_b[:, :2]  # (N_b, 2) positions target
    
    # Positions source (grille régulière normalisée [0,1])
    y_coords = torch.linspace(0, 1, Ha)
    x_coords = torch.linspace(0, 1, Wa)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    grid_positions = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # (H*W, 2)
    
    # Initialiser le champ de déplacement
    displacement_field = torch.zeros(Ha * Wa, 2)
    
    # Pour chaque pixel de la grille source, calculer le déplacement barycentrique
    for grid_idx in range(Ha * Wa):
        grid_pos = grid_positions[grid_idx]
        
        # Trouver le pixel source le plus proche dans X_a
        dists = torch.norm(pos_a_spatial - grid_pos.unsqueeze(0), dim=1)
        closest_idx = dists.argmin()
        
        # Calculer le déplacement barycentrique : T(x_i) = sum_j y_j * pi_ij / sum_k pi_ik
        # où pi_ij est le plan de transport de la source i vers la target j
        pi_row = pi[closest_idx, :]  # (N_b,) plan de transport depuis ce pixel source
        
        # Normaliser pour obtenir la distribution conditionnelle P(y|x)
        pi_row_sum = pi_row.sum()
        if pi_row_sum > 1e-6:
            # Déplacement barycentrique : moyenne pondérée des positions target
            weighted_target = (pi_row.unsqueeze(1) * pos_b_spatial).sum(dim=0) / pi_row_sum
            displacement = weighted_target - grid_pos
            displacement_field[grid_idx] = displacement
    
    # Reshape en grille
    displacement_field = displacement_field.reshape(Ha, Wa, 2)
    
    # Calculer la magnitude (en pixels, pas en coordonnées normalisées)
    displacement_magnitude = torch.norm(
        displacement_field * torch.tensor([Wa, Ha], dtype=displacement_field.dtype), 
        dim=2
    )
    
    return displacement_field, displacement_magnitude

def load_image(path: Path) -> torch.Tensor:
    """Charge une image et la convertit en tensor (C, H, W) normalisé."""
    img_pil = Image.open(path).convert("RGB")
    img = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float() / 255.0
    return img

def plot_lambda_timelines(img_source, img_target, lambdas, times):
    """Génère un plot montrant les timelines d'interpolation pour différentes valeurs de lambda."""
    print("\n" + "=" * 80)
    print("GÉNÉRATION DES TIMELINES POUR DIFFÉRENTES VALEURS DE LAMBDA")
    print("=" * 80)
    
    n_lambdas = len(lambdas)
    n_times = len(times)
    
    fig, axes = plt.subplots(n_lambdas, n_times, figsize=(n_times * 2.5, n_lambdas * 2.5))
    if n_lambdas == 1:
        axes = axes.reshape(1, -1)
    
    for i, lambda_val in enumerate(lambdas):
        print(f"\nCalcul pour λ = {lambda_val:.1f}...")
        config = OTConfig(
            resolution=(48, 48),
            blur=0.05,
            reach=0.3,
            lambda_color=lambda_val,
            sigma_start=1.2,
            sigma_end=0.5,
            sigma_boost=0.5
        )
        
        interpolator = OT5DInterpolator(config)
        # Utiliser sigma minimal pour voir l'effet pur de lambda (sans flou du splatting)
        frames, _, _, _, _, _, _, _, _, _, _ = interpolator.interpolate(img_source, img_target, times, use_minimal_sigma=True)
        
        for j, (t, frame) in enumerate(zip(times, frames)):
            ax = axes[i, j]
            np_img = frame.permute(1, 2, 0).clamp(0, 1).numpy()
            ax.imshow(np_img)
            if i == 0:
                ax.set_title(f"$t={t:.2f}$", fontsize=16)
            if j == 0:
                ax.set_ylabel(f"$\\lambda={lambda_val:.1f}$", fontsize=16)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Ajouter les hyperparamètres en bas de la figure
    base_config = OTConfig(
        resolution=(48, 48),
        blur=0.05,
        reach=0.3,
        lambda_color=2.0,
        sigma_start=1.2,
        sigma_end=0.5,
        sigma_boost=0.5
    )
    param_text = format_hyperparameters(base_config, varying_param="λ", varying_value="varied")
    fig.text(0.5, 0.02, f"Hyperparameters (varying λ): {param_text}", 
             ha='center', fontsize=10, family='monospace')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    output_path = OUTPUT_DIR / "lambda_ablation_timelines.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Sauvegardé: {output_path}")

def plot_displacement_fields(img_source, img_target, lambdas):
    """Génère un plot montrant les champs de déplacement et leur magnitude pour lambda = 10."""
    print("\n" + "=" * 80)
    print("GÉNÉRATION DES CHAMPS DE DÉPLACEMENT POUR LAMBDA = 10")
    print("=" * 80)
    
    # Ne garder que lambda = 10
    lambda_val = 10.0
    if lambda_val not in lambdas:
        print(f"Attention: lambda={lambda_val} n'est pas dans la liste. Utilisation de la valeur la plus proche.")
        lambda_val = min(lambdas, key=lambda x: abs(x - 10.0))
        print(f"Utilisation de lambda={lambda_val}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    print(f"\nCalcul pour λ = {lambda_val:.1f}...")
    config = OTConfig(
        resolution=(48, 48),
        blur=0.05,
        reach=0.3,
        lambda_color=lambda_val,
        sigma_start=1.2,
        sigma_end=0.5,
        sigma_boost=0.5
    )
    
    interpolator = OT5DInterpolator(config)
    times = [0.0, 0.5, 1.0]  # On a juste besoin du plan de transport
    _, pos_a, pos_b, weights_ij, I_idx, J_idx, Ha, Wa, X_a, X_b, pi = interpolator.interpolate(
        img_source, img_target, times, use_minimal_sigma=True
    )
    
    # Calcul du champ de déplacement
    displacement_field, displacement_magnitude = compute_displacement_field(
        X_a, X_b, pi, Ha, Wa
    )
    
    # Plot du champ de déplacement (vecteurs)
    ax1 = axes[0]
    # Sous-échantillonnage pour la visualisation
    step = max(1, min(Ha, Wa) // 15)
    y_coords = np.linspace(0, Ha-1, Ha)
    x_coords = np.linspace(0, Wa-1, Wa)
    Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Convertir les déplacements normalisés [0,1] en pixels
    U = displacement_field[::step, ::step, 0].numpy() * Wa
    V = displacement_field[::step, ::step, 1].numpy() * Ha
    X_sub = X[::step, ::step]
    Y_sub = Y[::step, ::step]
    
    # Afficher l'image source en arrière-plan
    img_source_np = img_source.permute(1, 2, 0).numpy()
    if img_source_np.shape[:2] != (Ha, Wa):
        img_source_resized = F.interpolate(
            img_source.unsqueeze(0), size=(Ha, Wa), mode='bilinear'
        ).squeeze(0).permute(1, 2, 0).numpy()
    else:
        img_source_resized = img_source_np
    
    ax1.imshow(img_source_resized, origin='upper', extent=[0, Wa, Ha, 0])
    ax1.quiver(X_sub, Y_sub, U, V, scale=1.0, scale_units='xy', angles='xy', 
               color='cyan', width=0.002, alpha=0.7, headwidth=3, headlength=3)
    ax1.set_title(f"Displacement Field ($\\lambda={lambda_val:.1f}$)", fontsize=16)
    ax1.set_xlabel("$x$ (pixels)")
    ax1.set_ylabel("$y$ (pixels)")
    ax1.set_aspect('equal')
    
    # Plot de la magnitude
    ax2 = axes[1]
    mag_np = displacement_magnitude.numpy()
    im = ax2.imshow(mag_np, cmap='hot', origin='lower')
    ax2.set_title(f"Displacement Magnitude ($\\lambda={lambda_val:.1f}$)", fontsize=16)
    ax2.set_xlabel("$x$")
    ax2.set_ylabel("$y$")
    plt.colorbar(im, ax=ax2, label="Magnitude")
    
    # Ajouter les hyperparamètres
    param_text = format_hyperparameters(config, varying_param="λ", varying_value=f"{lambda_val:.1f}")
    fig.text(0.5, 0.02, f"Hyperparameters: {param_text}", 
             ha='center', fontsize=10, family='monospace')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    output_path = OUTPUT_DIR / "lambda_ablation_displacement_fields.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Sauvegardé: {output_path}")

def main():
    print("=" * 80)
    print("ÉTUDE D'ABLATION: 5D JOINT LIFTING STRATEGY")
    print("=" * 80)
    
    # Chargement des images
    print("\nChargement des images...")
    img_source = load_image(DATA_DIR / "salameche.webp")
    img_target = load_image(DATA_DIR / "strawberry.jpg")
    print(f"Source: {img_source.shape}, Target: {img_target.shape}")
    
    # Valeurs de lambda à tester
    lambdas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Génération des plots
    plot_lambda_timelines(img_source, img_target, lambdas, times)
    plot_displacement_fields(img_source, img_target, lambdas)
    
    print("\n" + "=" * 80)
    print("TERMINÉ - Toutes les visualisations ont été créées dans:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()

