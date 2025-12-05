"""
Module de calcul des métriques pour le transport optimal 5D.

Métriques implémentées:
- PSNR (Peak Signal-to-Noise Ratio)
- ΔE (Delta E color distance, CIE76)
- Tearing % (via analyse du Jacobien)
- Coverage (taux de pixels non-nuls)
- Mass Error (erreur relative de conservation)
- Sharpness (variance du Laplacien)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convertit RGB (normalisé [0,1]) vers LAB (CIE76).
    
    Args:
        rgb: Tensor (3, H, W) ou (H, W, 3) en RGB [0,1]
    
    Returns:
        lab: Tensor de même shape en LAB
    """
    # Assurer le format (H, W, 3)
    if rgb.dim() == 3 and rgb.shape[0] == 3:
        rgb = rgb.permute(1, 2, 0)
    
    # Conversion RGB -> XYZ (sRGB)
    # Matrice de transformation sRGB -> XYZ (D65)
    M = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], device=rgb.device, dtype=rgb.dtype)
    
    # Gamma correction inverse (sRGB)
    mask = rgb > 0.04045
    rgb_linear = torch.where(
        mask,
        ((rgb + 0.055) / 1.055) ** 2.4,
        rgb / 12.92
    )
    
    # Conversion matricielle
    xyz = torch.einsum('ij,hwj->hwi', M, rgb_linear)
    
    # Normalisation D65
    xyz[:, :, 0] /= 0.95047  # X
    xyz[:, :, 2] /= 1.08883  # Z
    
    # XYZ -> LAB
    mask = xyz > 0.008856
    f = torch.where(
        mask,
        xyz ** (1/3),
        (7.787 * xyz + 16/116)
    )
    
    L = 116 * f[:, :, 1] - 16
    a = 500 * (f[:, :, 0] - f[:, :, 1])
    b = 200 * (f[:, :, 1] - f[:, :, 2])
    
    lab = torch.stack([L, a, b], dim=-1)
    return lab


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Calcule le PSNR entre deux images.
    
    Args:
        img1, img2: Tensors (C, H, W) en [0, max_val]
        max_val: Valeur maximale (1.0 pour images normalisées)
    
    Returns:
        PSNR en dB
    """
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr


def compute_delta_e(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Calcule la distance ΔE (CIE76) moyenne entre deux images RGB.
    
    Args:
        img1, img2: Tensors (C, H, W) en RGB [0,1]
    
    Returns:
        ΔE moyen
    """
    lab1 = rgb_to_lab(img1)
    lab2 = rgb_to_lab(img2)
    
    # Distance euclidienne dans LAB
    delta_e = torch.sqrt(torch.sum((lab1 - lab2) ** 2, dim=-1))
    return delta_e.mean().item()


def compute_tearing_percentage(
    interpolator,
    img_source: torch.Tensor,
    img_target: torch.Tensor,
    t: float,
    resolution: Tuple[int, int] = (64, 64)
) -> float:
    """
    Calcule le pourcentage de tearing via l'analyse du Jacobien.
    
    Réutilise la logique de analyze_tearing_condition de runall.py.
    
    Args:
        interpolator: Instance OT5DInterpolator ou similaire avec get_transport_map
        img_source, img_target: Images sources (C, H, W)
        t: Temps d'interpolation
        resolution: Résolution pour le calcul
    
    Returns:
        Pourcentage de pixels avec tearing
    """
    try:
        # Pour le transport 5D, on utilise le canal moyen pour la carte de transport
        # On pourrait aussi utiliser un canal spécifique
        T_map, H, W = interpolator.get_transport_map(img_source, img_target, channel=0)
        
        # Carte au temps t
        y = torch.linspace(0, 1, H, device=T_map.device)
        x = torch.linspace(0, 1, W, device=T_map.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        X_0 = torch.stack([xx, yy], dim=-1)
        
        X_t = (1 - t) * X_0 + t * T_map
        
        # Calcul du Jacobien
        dx = 1.0 / (W - 1)
        dy = 1.0 / (H - 1)
        
        dX_du = (X_t[:, 1:, :] - X_t[:, :-1, :]) / dx
        dX_du = F.pad(dX_du.permute(2, 0, 1), (0, 1, 0, 0)).permute(1, 2, 0)
        
        dX_dv = (X_t[1:, :, :] - X_t[:-1, :, :]) / dy
        dX_dv = F.pad(dX_dv.permute(2, 0, 1), (0, 0, 0, 1)).permute(1, 2, 0)
        
        j11 = dX_du[..., 0]
        j12 = dX_dv[..., 0]
        j21 = dX_du[..., 1]
        j22 = dX_dv[..., 1]
        
        det_J = j11 * j22 - j12 * j21
        abs_det_J = torch.abs(det_J)
        
        delta_grid = 1.0 / min(H, W)
        tearing_mask = abs_det_J > delta_grid
        num_tearing = tearing_mask.sum().item()
        total_pixels = H * W
        percent_tearing = (num_tearing / total_pixels) * 100
        
        return percent_tearing
        
    except Exception as e:
        logger.warning(f"Erreur calcul tearing: {e}")
        return 0.0


def compute_coverage(img: torch.Tensor, threshold: float = 1e-6) -> float:
    """
    Calcule le taux de pixels non-nuls (coverage).
    
    Args:
        img: Tensor (C, H, W)
        threshold: Seuil pour considérer un pixel comme non-nul
    
    Returns:
        Taux de coverage [0, 1]
    """
    # Somme sur les canaux
    img_sum = torch.sum(img, dim=0)
    non_zero = (img_sum > threshold).sum().item()
    total = img_sum.numel()
    return non_zero / total if total > 0 else 0.0


def compute_mass_error(
    interpolated: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    t: float
) -> float:
    """
    Calcule l'erreur relative de conservation de masse.
    
    La masse théorique devrait être: (1-t) * sum(source) + t * sum(target)
    
    Args:
        interpolated: Image interpolée (C, H, W)
        source: Image source (C, H, W)
        target: Image cible (C, H, W)
        t: Temps d'interpolation
    
    Returns:
        Erreur relative |actual - expected| / expected
    """
    mass_interp = torch.sum(interpolated).item()
    mass_source = torch.sum(source).item()
    mass_target = torch.sum(target).item()
    
    mass_expected = (1 - t) * mass_source + t * mass_target
    
    if mass_expected == 0:
        return 0.0
    
    error = abs(mass_interp - mass_expected) / mass_expected
    return error


def compute_sharpness(img: torch.Tensor) -> float:
    """
    Calcule la netteté via la variance du Laplacien.
    
    Args:
        img: Tensor (C, H, W) en [0,1]
    
    Returns:
        Variance du Laplacien (plus élevé = plus net)
    """
    # Convertir en niveaux de gris pour le Laplacien
    if img.shape[0] == 3:
        gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    else:
        gray = img[0]
    
    # Kernel Laplacien
    laplacian_kernel = torch.tensor([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=img.dtype, device=img.device).unsqueeze(0).unsqueeze(0)
    
    # Appliquer le Laplacien
    laplacian = F.conv2d(
        gray.unsqueeze(0).unsqueeze(0),
        laplacian_kernel,
        padding=1
    )
    
    # Variance du Laplacien
    sharpness = torch.var(laplacian).item()
    return sharpness


class MetricsComputer:
    """
    Classe pour calculer toutes les métriques d'une expérience.
    """
    
    def __init__(self, interpolator=None):
        """
        Args:
            interpolator: Instance avec méthode get_transport_map (optionnel)
        """
        self.interpolator = interpolator
    
    def compute_all(
        self,
        interpolated: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        t: float,
        compute_tearing: bool = False
    ) -> dict:
        """
        Calcule toutes les métriques disponibles.
        
        Args:
            interpolated: Image interpolée (C, H, W)
            source: Image source (C, H, W)
            target: Image cible (C, H, W)
            t: Temps d'interpolation
            compute_tearing: Si True, calcule aussi le tearing (nécessite interpolator)
        
        Returns:
            Dictionnaire avec toutes les métriques
        """
        metrics = {}
        
        # Métriques de base
        metrics['psnr'] = compute_psnr(interpolated, target)
        metrics['delta_e'] = compute_delta_e(interpolated, target)
        metrics['coverage'] = compute_coverage(interpolated)
        metrics['mass_error'] = compute_mass_error(interpolated, source, target, t)
        metrics['sharpness'] = compute_sharpness(interpolated)
        
        # Tearing (optionnel, nécessite interpolator)
        if compute_tearing and self.interpolator is not None:
            try:
                metrics['tearing_pct'] = compute_tearing_percentage(
                    self.interpolator, source, target, t
                )
            except Exception as e:
                logger.warning(f"Impossible de calculer tearing: {e}")
                metrics['tearing_pct'] = 0.0
        else:
            metrics['tearing_pct'] = None
        
        return metrics


def compute_displacement_smoothness(displacement_field: torch.Tensor) -> Dict[str, float]:
    """
    Calcule des métriques de 'smoothness' pour un champ de déplacement.
    
    Args:
        displacement_field: Tensor (H, W, 2) représentant le champ de déplacement (T(x) - x).
    
    Returns:
        Dict de métriques de smoothness.
    """
    H, W, _ = displacement_field.shape
    device = displacement_field.device

    # Magnitude du déplacement
    magnitude = torch.norm(displacement_field, dim=-1)
    mean_displacement = magnitude.mean().item()
    max_displacement = magnitude.max().item()
    std_displacement = magnitude.std().item()

    # Calcul des gradients pour divergence, curl, laplacien
    # dx_du, dy_du (gradients selon x)
    # dx_dv, dy_dv (gradients selon y)
    grad_x_field = torch.gradient(displacement_field[..., 0], dim=1, spacing=torch.tensor(1.0/(W-1), device=device))[0]
    grad_y_field = torch.gradient(displacement_field[..., 1], dim=0, spacing=torch.tensor(1.0/(H-1), device=device))[0]

    # Divergence: div(F) = dFx/dx + dFy/dy
    divergence = grad_x_field + grad_y_field
    mean_divergence = divergence.abs().mean().item()

    # Curl (pour un champ 2D, c'est un scalaire): curl(F) = dFy/dx - dFx/dy
    curl = torch.gradient(displacement_field[..., 1], dim=1, spacing=torch.tensor(1.0/(W-1), device=device))[0] - \
           torch.gradient(displacement_field[..., 0], dim=0, spacing=torch.tensor(1.0/(H-1), device=device))[0]
    mean_curl = curl.abs().mean().item()

    # Laplacien (smoothness): sum(d^2F/dx^2 + d^2F/dy^2)
    laplacian_x = torch.gradient(grad_x_field, dim=1, spacing=torch.tensor(1.0/(W-1), device=device))[0]
    laplacian_y = torch.gradient(grad_y_field, dim=0, spacing=torch.tensor(1.0/(H-1), device=device))[0]
    mean_laplacian = (laplacian_x.abs().mean() + laplacian_y.abs().mean()).item()

    # Score de smoothness (heuristique, à affiner)
    # Plus la divergence, curl, laplacien sont faibles, plus c'est lisse
    # Normalisation pour un score entre 0 et 1 (plus haut = plus lisse)
    smoothness_score = 1.0 / (1.0 + mean_divergence + mean_curl + mean_laplacian)

    return {
        'mean_displacement': mean_displacement,
        'max_displacement': max_displacement,
        'std_displacement': std_displacement,
        'mean_divergence': mean_divergence,
        'mean_curl': mean_curl,
        'mean_laplacian': mean_laplacian,
        'smoothness_score': smoothness_score
    }

