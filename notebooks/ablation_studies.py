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
import json
import time

# Configuration
DATA_DIR = Path("/Data/janis.aiad/geodata/data/pixelart/images")
EXPERIMENTS_DIR = Path("/Data/janis.aiad/geodata/experiments/pixelart")
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
    sigma_min: float = 0.1  # Minimum sigma value (only hyperparameter to vary)
    # Ablation flags
    use_debias: bool = False
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

def compute_average_interparticle_distance(X: torch.Tensor, H: int, W: int):
    """Compute average interparticle distance in normalized coordinates."""
    # Spatial positions
    pos_spatial = X[:, :2]  # (N, 2)
    N = pos_spatial.shape[0]
    
    # Compute average spacing: area / number of particles
    area = 1.0  # Normalized coordinates [0,1] x [0,1]
    avg_spacing = np.sqrt(area / N)
    
    return avg_spacing

def compute_sigma_max(X_a: torch.Tensor, X_b: torch.Tensor, Ha: int, Wa: int, Hb: int, Wb: int):
    """Compute sigma_max from average interparticle distance for input and output images."""
    # Compute average interparticle distance for source and target
    avg_dist_a = compute_average_interparticle_distance(X_a, Ha, Wa)
    avg_dist_b = compute_average_interparticle_distance(X_b, Hb, Wb)
    
    # Take the maximum and ensure sigma_max > 0.5 * avg_dist
    avg_dist_max = max(avg_dist_a, avg_dist_b)
    sigma_max = max(0.5 * avg_dist_max, avg_dist_max * 0.6)  # At least 0.5 * avg_dist, or 0.6 * avg_dist
    
    return sigma_max

def compute_sigma_t(t: float, sigma_min: float, sigma_max: float):
    """Compute adaptive sigma at time t: sigma(t) = sigma_min + 4*(sigma_max-sigma_min)*t*(1-t)."""
    return sigma_min + 4.0 * (sigma_max - sigma_min) * t * (1.0 - t)

class OT5DInterpolator:
    def __init__(self, config: OTConfig):
        self.cfg = config
        # Use tensorized backend to avoid KeOps compilation issues that can cause segfaults
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
        """Calcule le plan de transport pi entre source et target."""
        # Clear GPU cache before computation
        if self.cfg.device == "cuda":
            torch.cuda.empty_cache()
        
        X_a, w_a, colors_a, Ha, Wa = get_5d_cloud(
            img_source.to(self.cfg.device), self.cfg.resolution[0], self.cfg.lambda_color
        )
        X_b, w_b, colors_b, Hb, Wb = get_5d_cloud(
            img_target.to(self.cfg.device), self.cfg.resolution[1], self.cfg.lambda_color
        )
        
        logger.info(f"  Cloud sizes: source={X_a.shape[0]}, target={X_b.shape[0]}")
        
        # Clear cache before Sinkhorn computation
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

def measure_compute_time(config: OTConfig, img_source, img_target):
    """Measure computation time for transport plan calculation."""
    start_time = time.time()
    
    interpolator = OT5DInterpolator(config)
    transport_data = interpolator.compute_transport_plan(img_source, img_target)
    
    compute_time = time.time() - start_time
    
    # Clean up
    del transport_data, interpolator
    if config.device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    
    return compute_time

def run_compute_time_benchmark():
    """Benchmark computation times for rho=1.0 with various epsilon values."""
    
    logger.info("=" * 80)
    logger.info("COMPUTE TIME BENCHMARK: rho=1.0, various epsilon")
    logger.info("=" * 80)
    
    # Chargement des images
    logger.info("Loading images...")
    img_source = load_image(DATA_DIR / "salameche.webp")
    img_target = load_image(DATA_DIR / "strawberry.jpg")
    logger.info(f"Source: {img_source.shape}, Target: {img_target.shape}")
    
    # Fixed resolution
    resolution = (64, 64)
    
    # Parameters for benchmark
    rho = 1.0
    # Generate 50 epsilon values in log space from 1e-4 to 100
    epsilons = np.logspace(-4, 2, 50).tolist()
    lambda_color = 2.5
    sigma_min = 0.1
    use_debias = True
    use_adaptive_sigma = True
    
    results = {
        'rho': rho,
        'lambda_color': lambda_color,
        'sigma_min': sigma_min,
        'resolution': resolution,
        'use_debias': use_debias,
        'use_adaptive_sigma': use_adaptive_sigma,
        'compute_times': []
    }
    
    logger.info(f"\nTesting {len(epsilons)} epsilon values with rho={rho}")
    
    for eps in epsilons:
        logger.info(f"\nTesting eps={eps:.3f}...")
        
        config = OTConfig(
            resolution=resolution,
            blur=eps,
            reach=rho,
            lambda_color=lambda_color,
            sigma_min=sigma_min,
            use_debias=use_debias,
            use_adaptive_sigma=use_adaptive_sigma
        )
        
        try:
            compute_time = measure_compute_time(config, img_source, img_target)
            results['compute_times'].append({
                'epsilon': eps,
                'compute_time_seconds': compute_time
            })
            logger.info(f"  ✓ eps={eps:.3f}: {compute_time:.2f} seconds")
        except Exception as e:
            logger.error(f"  ✗ eps={eps:.3f} failed: {e}")
            results['compute_times'].append({
                'epsilon': eps,
                'compute_time_seconds': None,
                'error': str(e)
            })
    
    # Save to JSON
    json_path = EXPERIMENTS_DIR / "compute_times_rho1.0.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n✓ Saved compute times to: {json_path}")
    
    return results

def run_ablation_studies():
    """Exécute les études d'ablation à grande échelle."""
    
    logger.info("=" * 80)
    logger.info("DÉMARRAGE DES ÉTUDES D'ABLATION")
    logger.info("=" * 80)
    
    # Chargement des images
    logger.info("Loading images...")
    img_source = load_image(DATA_DIR / "salameche.webp")
    img_target = load_image(DATA_DIR / "strawberry.jpg")
    logger.info(f"Source: {img_source.shape}, Target: {img_target.shape}")
    
    # Fixed resolution for ablation studies
    resolution = (64, 64)
    _, H_source, W_source = img_source.shape
    _, H_target, W_target = img_target.shape
    logger.info(f"Source image: {H_source}x{W_source}, Target image: {H_target}x{W_target}")
    logger.info(f"Using fixed resolution: {resolution}")
    
    # Grille d'hyperparamètres
    epsilons = [0.07]
    rhos = [0.7]
    lambdas = [2.5]
    sigma_mins = [0,0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]  # Vary sigma_min
    debias_options = [True]
    adaptive_sigma_options = [True]
    
    # Génération de toutes les combinaisons
    total_experiments = len(epsilons) * len(rhos) * len(lambdas) * len(sigma_mins) * len(debias_options) * len(adaptive_sigma_options)
    logger.info(f"Nombre total d'expériences: {total_experiments}")
    logger.info(f"Lambda values: {lambdas}")
    logger.info(f"Sigma_min values: {sigma_mins}")
    
    experiment_id = 0
    successful = 0
    failed = 0
    
    for eps, rho, lambda_color, sigma_min, use_debias, use_adapt_sigma in itertools.product(
        epsilons, rhos, lambdas, sigma_mins, debias_options, adaptive_sigma_options
    ):
        experiment_id += 1
        
        # Configuration
        config = OTConfig(
            resolution=resolution,
            blur=eps,
            reach=rho,
            lambda_color=lambda_color,
            sigma_min=sigma_min,
            use_debias=use_debias,
            use_adaptive_sigma=use_adapt_sigma
        )
        
        # Nom du fichier
        rho_str = "balanced" if rho is None else f"{rho:.2f}"
        lambda_str = f"{lambda_color:.1f}" if lambda_color < 10 else f"{lambda_color:.0f}"
        sigma_min_str = f"{sigma_min:.2f}"
        exp_name = f"exp_{experiment_id:04d}_eps{eps:.3f}_rho{rho_str}_lam{lambda_str}_smin{sigma_min_str}_debias{use_debias}_adapsigma{use_adapt_sigma}"
        output_path = EXPERIMENTS_DIR / f"{exp_name}.pt"
        
        logger.info(f"\n[{experiment_id}/{total_experiments}] {exp_name}")
        logger.info(f"  eps={eps:.3f}, rho={rho}, lambda={lambda_color:.1f}, sigma_min={sigma_min:.3f}, debias={use_debias}, adapt_sigma={use_adapt_sigma}")
        
        try:
            # Clear memory before experiment
            if config.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            # Calcul du plan de transport
            interpolator = OT5DInterpolator(config)
            transport_data = interpolator.compute_transport_plan(img_source, img_target)
            
            # Compute sigma_max from interparticle distances
            X_a = transport_data['X_a']
            X_b = transport_data['X_b']
            Ha, Wa = transport_data['Ha'], transport_data['Wa']
            Hb, Wb = transport_data['Hb'], transport_data['Wb']
            sigma_max = compute_sigma_max(X_a, X_b, Ha, Wa, Hb, Wb)
            
            # Ajout des métadonnées
            transport_data['config'] = {
                'eps': eps,
                'rho': rho,
                'lambda_color': lambda_color,
                'sigma_min': sigma_min,
                'sigma_max': sigma_max,
                'resolution': resolution,
                'use_debias': use_debias,
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

