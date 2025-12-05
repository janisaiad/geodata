#!/usr/bin/env python3
"""
Visualisations pour l'étude d'ablation sur le régime Unbalanced (paramètre ρ).
Génère des timelines d'interpolation et des comparaisons pour différentes valeurs de ρ.
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

class OT5DInterpolator:
    def __init__(self, config: OTConfig):
        self.cfg = config
        self.loss_layer = SamplesLoss(
            loss="sinkhorn", p=2, blur=config.blur, reach=config.reach,
            debias=False, potentials=True, scaling=config.scaling, backend="auto"
        )

    def interpolate(self, img_source, img_target, times: List[float], return_transport_plan=False):
        """Interpolation de Wasserstein avec barycentres géodésiques - SANS SPLATTING."""
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
        
        for t in tqdm(times, desc=f"Interpolation ρ={self.cfg.reach}", leave=False):
            # À t=0, retourner l'image source directement (sans flou)
            if abs(t) < 1e-6:
                results.append(img_source_resized.cpu())
                continue
            
            # À t=1, retourner l'image target directement (sans flou)
            if abs(t - 1.0) < 1e-6:
                results.append(img_target_resized.cpu())
                continue
            
            # Pour les interpolations intermédiaires - SANS SPLATTING
            pos_t = (1 - t) * pos_a_spatial + t * pos_b_spatial
            col_t = (1 - t) * col_a_real + t * col_b_real
            Ht = int((1 - t) * Ha + t * Hb)
            Wt = int((1 - t) * Wa + t * Wb)
            
            # Convertir les positions normalisées [0,1] en coordonnées pixels
            pos_pix = pos_t * torch.tensor([Wt - 1, Ht - 1], device=pos_t.device)
            
            # Placement direct sans splatting
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
        
        if return_transport_plan:
            return results, X_a.cpu(), X_b.cpu(), pi.cpu(), Ha, Wa, Hb, Wb
        return results

def load_image(path: Path) -> torch.Tensor:
    """Charge une image et la convertit en tensor (C, H, W) normalisé."""
    img_pil = Image.open(path).convert("RGB")
    img = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float() / 255.0
    return img

def plot_rho_timelines(img_source, img_target, rhos, times):
    """Génère un plot montrant les timelines d'interpolation pour différentes valeurs de ρ."""
    print("\n" + "=" * 80)
    print("GÉNÉRATION DES TIMELINES POUR DIFFÉRENTES VALEURS DE ρ (REACH)")
    print("=" * 80)
    
    n_rhos = len(rhos)
    n_times = len(times)
    
    fig, axes = plt.subplots(n_rhos, n_times, figsize=(n_times * 2.5, n_rhos * 2.5))
    if n_rhos == 1:
        axes = axes.reshape(1, -1)
    
    for i, rho_val in enumerate(rhos):
        rho_str = "Balanced" if rho_val is None else f"{rho_val:.2f}"
        print(f"\nCalcul pour ρ = {rho_str}...")
        config = OTConfig(
            resolution=(48, 48),
            blur=0.05,
            reach=rho_val,
            lambda_color=2.0,
            sigma_start=1.2,
            sigma_end=0.5,
            sigma_boost=0.5
        )
        
        interpolator = OT5DInterpolator(config)
        frames = interpolator.interpolate(img_source, img_target, times)
        
        for j, (t, frame) in enumerate(zip(times, frames)):
            ax = axes[i, j]
            np_img = frame.permute(1, 2, 0).clamp(0, 1).numpy()
            ax.imshow(np_img)
            if i == 0:
                ax.set_title(f"$t={t:.2f}$", fontsize=16)
            if j == 0:
                rho_label = "Balanced" if rho_val is None else f"$\\rho={rho_val:.2f}$"
                ax.set_ylabel(rho_label, fontsize=16)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "rho_ablation_timelines.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Sauvegardé: {output_path}")

def plot_rho_comparison(img_source, img_target, rhos, t=0.5):
    """Génère un plot comparant différentes valeurs de ρ à un temps fixe."""
    print("\n" + "=" * 80)
    print(f"COMPARAISON DES VALEURS DE ρ À t={t:.2f}")
    print("=" * 80)
    
    n_rhos = len(rhos)
    fig, axes = plt.subplots(1, n_rhos + 2, figsize=((n_rhos + 2) * 2.5, 3))
    
    # Afficher l'image source
    img_source_np = img_source.permute(1, 2, 0).numpy()
    axes[0].imshow(img_source_np)
    axes[0].set_title("Source", fontsize=16)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # Afficher les interpolations pour différentes valeurs de ρ
    for i, rho_val in enumerate(rhos):
        rho_str = "Balanced" if rho_val is None else f"{rho_val:.2f}"
        print(f"\nCalcul pour ρ = {rho_str}...")
        config = OTConfig(
            resolution=(48, 48),
            blur=0.05,
            reach=rho_val,
            lambda_color=2.0,
            sigma_start=1.2,
            sigma_end=0.5,
            sigma_boost=0.5
        )
        
        interpolator = OT5DInterpolator(config)
        frames = interpolator.interpolate(img_source, img_target, [t])
        frame = frames[0]
        
        ax = axes[i + 1]
        np_img = frame.permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(np_img)
        rho_label = "Balanced" if rho_val is None else f"$\\rho={rho_val:.2f}$"
        ax.set_title(rho_label, fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Afficher l'image target
    img_target_np = img_target.permute(1, 2, 0).numpy()
    axes[-1].imshow(img_target_np)
    axes[-1].set_title("Target", fontsize=16)
    axes[-1].set_xticks([])
    axes[-1].set_yticks([])
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "rho_ablation_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Sauvegardé: {output_path}")

def plot_foreground_background_effect(img_source, img_target, rhos, t=0.5):
    """Génère un plot montrant l'effet cross-fade du background pour différentes valeurs de ρ."""
    print("\n" + "=" * 80)
    print(f"ANALYSE FOREGROUND/BACKGROUND POUR DIFFÉRENTES VALEURS DE ρ À t={t:.2f}")
    print("=" * 80)
    
    # Sélectionner quelques valeurs de ρ représentatives
    selected_rhos = [None, 0.01, 0.10, 0.30, 0.50] if len(rhos) > 5 else rhos
    n_rhos = len(selected_rhos)
    
    fig, axes = plt.subplots(1, n_rhos, figsize=(n_rhos * 3, 3))
    if n_rhos == 1:
        axes = [axes]
    
    for i, rho_val in enumerate(selected_rhos):
        rho_str = "Balanced" if rho_val is None else f"{rho_val:.2f}"
        print(f"\nCalcul pour ρ = {rho_str}...")
        config = OTConfig(
            resolution=(48, 48),
            blur=0.05,
            reach=rho_val,
            lambda_color=2.0,
            sigma_start=1.2,
            sigma_end=0.5,
            sigma_boost=0.5
        )
        
        interpolator = OT5DInterpolator(config)
        frames = interpolator.interpolate(img_source, img_target, [t])
        frame = frames[0]
        
        ax = axes[i]
        np_img = frame.permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(np_img)
        rho_label = "Balanced" if rho_val is None else f"$\\rho={rho_val:.2f}$"
        ax.set_title(rho_label, fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "rho_foreground_background_effect.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Sauvegardé: {output_path}")

def compute_displacement_field(X_a, X_b, pi, Ha, Wa):
    """
    Calcule le champ de déplacement barycentrique pour chaque pixel de la grille source.
    Utilise le plan de transport complet pi pour calculer les déplacements.
    """
    # Positions spatiales (2 premières dimensions de X_a et X_b)
    pos_a_spatial = X_a[:, :2]  # (N_a, 2) positions source
    pos_b_spatial = X_b[:, :2]  # (N_b, 2) positions target
    
    # Positions source (grille régulière normalisée [0,1])
    y_coords = torch.linspace(0, 1, Ha, device=X_a.device)
    x_coords = torch.linspace(0, 1, Wa, device=X_a.device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    grid_positions = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # (H*W, 2)
    
    # Initialiser le champ de déplacement
    displacement_field = torch.zeros(Ha * Wa, 2, device=X_a.device)
    
    # Pour chaque pixel de la grille source, calculer le déplacement barycentrique
    for grid_idx in range(Ha * Wa):
        grid_pos = grid_positions[grid_idx]
        
        # Trouver le pixel source le plus proche dans X_a
        dists = torch.norm(pos_a_spatial - grid_pos.unsqueeze(0), dim=1)
        closest_idx = dists.argmin()
        
        # Calculer le déplacement barycentrique : T(x_i) = sum_j y_j * pi_ij / sum_k pi_ik
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
        displacement_field * torch.tensor([Wa, Ha], dtype=displacement_field.dtype, device=X_a.device), 
        dim=2
    )
    
    return displacement_field, displacement_magnitude

def plot_rho_displacement_fields(img_source, img_target, rhos):
    """Génère un plot montrant les champs de déplacement pour différentes valeurs de rho."""
    print("\n" + "=" * 80)
    print("CHAMPS DE DÉPLACEMENT POUR DIFFÉRENTES VALEURS DE ρ")
    print("=" * 80)
    
    n_rhos = len(rhos)
    fig, axes = plt.subplots(n_rhos, 2, figsize=(10, n_rhos * 5))
    if n_rhos == 1:
        axes = axes.reshape(1, -1)
    
    for i, rho_val in enumerate(rhos):
        rho_str = "Balanced" if rho_val is None else f"{rho_val:.2f}"
        print(f"\nCalcul du champ de déplacement pour ρ = {rho_str}...")
        config = OTConfig(
            resolution=(48, 48),
            blur=0.05,
            reach=rho_val,
            lambda_color=2.0,
            sigma_start=1.2,
            sigma_end=0.5,
            sigma_boost=0.5
        )
        
        interpolator = OT5DInterpolator(config)
        _, X_a, X_b, pi, Ha, Wa, _, _ = interpolator.interpolate(
            img_source, img_target, [0.0], return_transport_plan=True
        )
        
        # Calculer le champ de déplacement
        displacement_field, displacement_magnitude = compute_displacement_field(
            X_a, X_b, pi, Ha, Wa
        )
        
        # Convertir en numpy pour visualisation
        disp_field_np = displacement_field.numpy()
        disp_mag_np = displacement_magnitude.numpy()
        
        # Image source pour superposition
        img_source_resized = F.interpolate(
            img_source.unsqueeze(0), size=(Ha, Wa), mode='bilinear'
        ).squeeze(0)
        img_source_np = img_source_resized.permute(1, 2, 0).numpy()
        
        # Plot du champ de déplacement (vecteurs)
        ax1 = axes[i, 0]
        ax1.imshow(img_source_np)
        # Sous-échantillonner les vecteurs pour la visualisation
        step = max(1, min(Ha, Wa) // 10)
        y_coords = np.arange(0, Ha, step)
        x_coords = np.arange(0, Wa, step)
        Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')
        U = disp_field_np[Y, X, 0] * Wa  # Convertir en pixels
        V = -disp_field_np[Y, X, 1] * Ha  # Inverser Y pour matplotlib
        ax1.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, 
                   color='red', width=0.003, alpha=0.7)
        rho_label = "Balanced" if rho_val is None else f"$\\rho={rho_val:.2f}$"
        ax1.set_title(f"Displacement Field - {rho_label}", fontsize=14)
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Plot de la magnitude
        ax2 = axes[i, 1]
        im = ax2.imshow(disp_mag_np, cmap='hot', interpolation='nearest')
        ax2.set_title(f"Displacement Magnitude - {rho_label}", fontsize=14)
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.colorbar(im, ax=ax2, label='Magnitude (pixels)')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "rho_displacement_fields.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Sauvegardé: {output_path}")

def plot_rho_displacement_statistics(img_source, img_target, rhos):
    """Génère des courbes montrant les statistiques de déplacement (max, mean) en fonction de rho."""
    print("\n" + "=" * 80)
    print("STATISTIQUES DE DÉPLACEMENT EN FONCTION DE ρ")
    print("=" * 80)
    
    max_displacements = []
    mean_displacements = []
    rho_values = []
    
    for rho_val in rhos:
        rho_str = "Balanced" if rho_val is None else f"{rho_val:.2f}"
        print(f"\nCalcul des statistiques pour ρ = {rho_str}...")
        config = OTConfig(
            resolution=(48, 48),
            blur=0.05,
            reach=rho_val,
            lambda_color=2.0,
            sigma_start=1.2,
            sigma_end=0.5,
            sigma_boost=0.5
        )
        
        interpolator = OT5DInterpolator(config)
        _, X_a, X_b, pi, Ha, Wa, _, _ = interpolator.interpolate(
            img_source, img_target, [0.0], return_transport_plan=True
        )
        
        # Calculer le champ de déplacement
        _, displacement_magnitude = compute_displacement_field(
            X_a, X_b, pi, Ha, Wa
        )
        
        # Calculer les statistiques
        max_disp = displacement_magnitude.max().item()
        mean_disp = displacement_magnitude.mean().item()
        
        max_displacements.append(max_disp)
        mean_displacements.append(mean_disp)
        
        # Pour le plot, utiliser une valeur numérique pour rho (None -> infini)
        if rho_val is None:
            rho_values.append(1.0)  # Balanced = rho très grand, on met 1.0 pour le plot
        else:
            rho_values.append(rho_val)
        
        print(f"  Max displacement: {max_disp:.4f} pixels")
        print(f"  Mean displacement: {mean_disp:.4f} pixels")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Trier par rho pour un plot lisse
    sorted_indices = np.argsort(rho_values)
    rho_sorted = [rho_values[i] for i in sorted_indices]
    max_disp_sorted = [max_displacements[i] for i in sorted_indices]
    mean_disp_sorted = [mean_displacements[i] for i in sorted_indices]
    
    # Plot max displacement
    ax1.plot(rho_sorted, max_disp_sorted, marker="o", linestyle="-", 
             linewidth=2, markersize=8, color="b")
    ax1.set_xlabel("Reach parameter $\\rho$")
    ax1.set_ylabel("Maximum displacement (pixels)")
    ax1.set_title("Maximum displacement vs $\\rho$")
    ax1.grid(True, linestyle="--", alpha=0.6)
    # Marquer le point Balanced
    balanced_idx = [i for i, r in enumerate(rho_values) if r == 1.0]
    if balanced_idx:
        ax1.scatter([1.0], [max_displacements[balanced_idx[0]]], 
                   color="red", s=200, marker="*", zorder=5, label="Balanced")
        ax1.legend()
    
    # Plot mean displacement
    ax2.plot(rho_sorted, mean_disp_sorted, marker="s", linestyle="-", 
             linewidth=2, markersize=8, color="r")
    ax2.set_xlabel("Reach parameter $\\rho$")
    ax2.set_ylabel("Mean displacement (pixels)")
    ax2.set_title("Mean displacement vs $\\rho$")
    ax2.grid(True, linestyle="--", alpha=0.6)
    # Marquer le point Balanced
    if balanced_idx:
        ax2.scatter([1.0], [mean_displacements[balanced_idx[0]]], 
                   color="red", s=200, marker="*", zorder=5, label="Balanced")
        ax2.legend()
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "rho_displacement_statistics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Sauvegardé: {output_path}")
    
    # Afficher un tableau récapitulatif
    print("\n" + "=" * 80)
    print("TABLEAU RÉCAPITULATIF")
    print("=" * 80)
    print(f"{'ρ':<12} {'Max Displacement':<20} {'Mean Displacement':<20}")
    print("-" * 60)
    for rho_val, max_disp, mean_disp in zip(rhos, max_displacements, mean_displacements):
        rho_str = "Balanced" if rho_val is None else f"{rho_val:.2f}"
        print(f"{rho_str:<12} {max_disp:<20.4f} {mean_disp:<20.4f}")

def main():
    print("=" * 80)
    print("ÉTUDE D'ABLATION: UNBALANCED REGIME (PARAMÈTRE ρ)")
    print("=" * 80)
    
    # Chargement des images
    print("\nChargement des images...")
    img_source = load_image(DATA_DIR / "salameche.webp")
    img_target = load_image(DATA_DIR / "strawberry.jpg")
    print(f"Source: {img_source.shape}, Target: {img_target.shape}")
    
    # Valeurs de ρ à tester (None = Balanced, sinon Unbalanced)
    rhos = [None, 0.01, 0.05, 0.10, 0.30, 0.50]
    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Génération des plots
    plot_rho_timelines(img_source, img_target, rhos, times)
    plot_rho_comparison(img_source, img_target, rhos, t=0.5)
    plot_foreground_background_effect(img_source, img_target, rhos, t=0.5)
    plot_rho_displacement_fields(img_source, img_target, rhos)
    plot_rho_displacement_statistics(img_source, img_target, rhos)
    
    print("\n" + "=" * 80)
    print("TERMINÉ - Toutes les visualisations ont été créées dans:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()

