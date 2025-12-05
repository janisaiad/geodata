#!/usr/bin/env python3
"""
Large-scale ablation studies for 5D Optimal Transport.
Computes transport plans at resolution 0.005 (200 steps) with various hyperparameters.
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
import logging
from datetime import datetime
import itertools
import gc

# Configuration
DATA_DIR = Path("/Data/janis.aiad/geodata/data/pixelart/images")
EXPERIMENTS_DIR = Path("/Data/janis.aiad/geodata/experiments")
LOGS_DIR = Path("/Data/janis.aiad/geodata/logs")
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = LOGS_DIR / f"ablation_studies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    # Ablation flags
    use_debias: bool = False
    use_dynamic_rasterization: bool = True
    use_adaptive_sigma: bool = True

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
            debias=config.use_debias, potentials=True, scaling=config.scaling, backend="auto"
        )

    def compute_transport_plan(self, img_source, img_target):
        """Calcule le plan de transport pi entre source et target."""
        X_a, w_a, colors_a, Ha, Wa = get_5d_cloud(
            img_source.to(self.cfg.device), self.cfg.resolution[0], self.cfg.lambda_color
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

def load_image(path: Path) -> torch.Tensor:
    """Charge une image et la convertit en tensor (C, H, W) normalisé."""
    img_pil = Image.open(path).convert("RGB")
    img = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float() / 255.0
    return img

def run_ablation_studies():
    """Exécute les études d'ablation à grande échelle."""
    
    logger.info("=" * 80)
    logger.info("DÉMARRAGE DES ÉTUDES D'ABLATION")
    logger.info("=" * 80)
    
    # Chargement des images
    logger.info("Chargement des images...")
    img_source = load_image(DATA_DIR / "salameche.webp")
    img_target = load_image(DATA_DIR / "strawberry.jpg")
    logger.info(f"Source: {img_source.shape}, Target: {img_target.shape}")
    
    # Grille d'hyperparamètres
    epsilons = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
    rhos = [None, 0.01, 0.02, 0.05, 0.10, 0.15, 0.30, 0.50]
    debias_options = [False, True]
    dynamic_rasterization_options = [False, True]
    adaptive_sigma_options = [False, True]
    
    # Résolution et lambda fixes (comme dans le papier)
    resolution = (48, 48)
    lambda_color = 2.0
    
    # Génération de toutes les combinaisons
    total_experiments = len(epsilons) * len(rhos) * len(debias_options) * len(dynamic_rasterization_options) * len(adaptive_sigma_options)
    logger.info(f"Nombre total d'expériences: {total_experiments}")
    
    experiment_id = 0
    successful = 0
    failed = 0
    
    for eps, rho, use_debias, use_dyn_rast, use_adapt_sigma in itertools.product(
        epsilons, rhos, debias_options, dynamic_rasterization_options, adaptive_sigma_options
    ):
        experiment_id += 1
        
        # Configuration
        config = OTConfig(
            resolution=resolution,
            blur=eps,
            reach=rho,
            lambda_color=lambda_color,
            sigma_start=1.2,
            sigma_end=0.5,
            sigma_boost=0.5,
            use_debias=use_debias,
            use_dynamic_rasterization=use_dyn_rast,
            use_adaptive_sigma=use_adapt_sigma
        )
        
        # Nom du fichier
        rho_str = "balanced" if rho is None else f"{rho:.2f}"
        exp_name = f"exp_{experiment_id:04d}_eps{eps:.3f}_rho{rho_str}_debias{use_debias}_dynrast{use_dyn_rast}_adapsigma{use_adapt_sigma}"
        output_path = EXPERIMENTS_DIR / f"{exp_name}.pt"
        
        logger.info(f"\n[{experiment_id}/{total_experiments}] {exp_name}")
        logger.info(f"  eps={eps:.3f}, rho={rho}, debias={use_debias}, dyn_rast={use_dyn_rast}, adapt_sigma={use_adapt_sigma}")
        
        try:
            # Calcul du plan de transport
            interpolator = OT5DInterpolator(config)
            transport_data = interpolator.compute_transport_plan(img_source, img_target)
            
            # Ajout des métadonnées
            transport_data['config'] = {
                'eps': eps,
                'rho': rho,
                'lambda_color': lambda_color,
                'resolution': resolution,
                'use_debias': use_debias,
                'use_dynamic_rasterization': use_dyn_rast,
                'use_adaptive_sigma': use_adapt_sigma,
                'experiment_id': experiment_id,
                'exp_name': exp_name
            }
            
            # Sauvegarde
            torch.save(transport_data, output_path)
            logger.info(f"  ✓ Sauvegardé: {output_path}")
            logger.info(f"  Taille plan: {transport_data['pi'].shape}, Mémoire: {transport_data['pi'].numel() * 4 / 1024**2:.2f} MB")
            
            successful += 1
            
            # Nettoyage mémoire
            del transport_data, interpolator
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
        except Exception as e:
            logger.error(f"  ✗ Erreur: {e}", exc_info=True)
            failed += 1
            continue
    
    logger.info("\n" + "=" * 80)
    logger.info("RÉSUMÉ")
    logger.info("=" * 80)
    logger.info(f"Expériences réussies: {successful}/{total_experiments}")
    logger.info(f"Expériences échouées: {failed}/{total_experiments}")
    logger.info(f"Répertoire de sortie: {EXPERIMENTS_DIR}")
    logger.info(f"Log: {log_file}")
    logger.info("=" * 80)

if __name__ == "__main__":
    run_ablation_studies()

