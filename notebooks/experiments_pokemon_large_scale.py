"""
Script d'expérimentation massive pour le transport optimal 5D avec Pokemon.
Spécifiquement conçu pour Charmander (Salameche) -> Strawberry.

Ce script exécute une grille systématique d'expériences à grande échelle:
1. Ablation Lambda (λ) - large range
2. Comparaison 2D vs 5D
3. Impact du Splatting Adaptatif
4. Sensibilité aux paramètres (ε, ρ) - grille dense
5. Scalabilité résolution - résolutions élevées
6. Robustesse du champ de déplacement

Toutes les métriques sont calculées et exportées en CSV.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from pathlib import Path
import json
import csv
import logging
from tqdm import tqdm
import time
import sys
import os
import pandas as pd
from datetime import datetime
import signal
import gc
import traceback

# Ajouter le répertoire parent au path pour les imports
sys.path.append(str(Path(__file__).parent))

# Imports depuis les modules existants
from metrics_5d import MetricsComputer, compute_psnr, compute_delta_e, compute_coverage, compute_mass_error, compute_sharpness, compute_displacement_smoothness

# Imports depuis 5d_transport
from geomloss import SamplesLoss

# Configuration du logging
def setup_logging(output_dir: str):
    """Configure le logging pour écrire dans un fichier et sur stdout."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Nom du fichier avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pokemon_experiments_{timestamp}.log"
    
    # Configuration du logger root
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Supprimer les handlers existants pour éviter les doublons
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Handler pour fichier
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Handler pour console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Ajouter les handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Logger pour ce module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialisé. Fichier: {log_file}")
    return logger, log_file

# Classe pour rediriger tqdm vers le logger
class TqdmToLogger:
    """Wrapper pour rediriger tqdm vers le logger."""
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.buf = ''
    
    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')
    
    def flush(self):
        if self.buf:
            self.logger.log(self.level, self.buf)

# Logger global (sera initialisé dans le main)
logger = logging.getLogger(__name__)

# Global flag pour sauvegarder en cas de crash
_should_save_on_exit = False
_runner_instance = None

def signal_handler(signum, frame):
    """Handler pour les signaux (SIGSEGV, SIGINT, etc.) - sauvegarde les résultats."""
    global _should_save_on_exit, _runner_instance
    logger.critical(f"Signal reçu: {signum}. Sauvegarde des résultats...")
    if _runner_instance is not None:
        try:
            _runner_instance.save_results()
            logger.critical("Résultats sauvegardés avant arrêt.")
        except Exception as e:
            logger.critical(f"Erreur lors de la sauvegarde d'urgence: {e}")
    sys.exit(1)

# Enregistrer les handlers de signaux
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
# Note: SIGSEGV ne peut pas être capturé en Python, mais on peut essayer
try:
    signal.signal(signal.SIGSEGV, signal_handler)
except (ValueError, OSError):
    pass  # SIGSEGV n'est pas toujours disponible


# ============================================================================
# Configuration et Infrastructure
# ============================================================================

@dataclass
class OTConfig:
    """Configuration pour Transport Optimal 5D."""
    resolution: Tuple[int, int] = (48, 48)
    blur: float = 0.01
    scaling: float = 0.9
    reach: Optional[float] = 0.5
    lambda_color: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    sigma_start: float = 1.2
    sigma_end: float = 0.5
    sigma_boost: float = 0.5


@dataclass
class PokemonExperimentConfig:
    """Configuration pour les expériences Pokemon (large scale)."""
    # Grilles de paramètres - Large scale
    resolutions: List[int] = field(default_factory=lambda: [32, 48, 64, 96, 128, 160, 192, 256])  # Résolutions très élevées
    lambdas: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])  # Large range
    blurs: List[float] = field(default_factory=lambda: [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5])  # Grille dense
    reaches: List[Optional[float]] = field(default_factory=lambda: [None, 0.01, 0.02, 0.05, 0.1, 0.15, 0.3, 0.5, 0.7, 1.0])  # Large range
    times: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])  # Beaucoup de points temporels
    
    # Images Pokemon - Salameche (Charmander) -> Strawberry
    source_image_name: str = "salameche.webp"
    target_image_name: str = "strawberry.jpg"
    image_pair_name: str = "salameche_strawberry"
    
    # Répertoires
    data_dir: str = "/Data/janis.aiad/geodata/data/pixelart/images"
    output_dir: str = "/Data/janis.aiad/geodata/refs/reports/results/pokemon_experiments_salameche_strawberry"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Options
    save_images: bool = True
    save_metrics: bool = True
    save_transport_plans: bool = True  # Sauvegarder les plans de transport
    transport_plan_scale: float = 0.01  # Échelle pour sauvegarder les plans (économie mémoire)
    compute_tearing: bool = False
    progressive_resolution: bool = True  # Tester progressivement les résolutions
    max_resolution: Optional[int] = None  # Limite max de résolution (None = pas de limite)


# ============================================================================
# Utilitaires 5D (copiés depuis experiments_5d_massive.py)
# ============================================================================

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
    
    C_channels = attributes.shape[1]
    numer = torch.zeros(H * W, C_channels, device=device)
    
    weighted_attribs = attributes.unsqueeze(1) * contrib_weights.unsqueeze(-1)
    for c in range(C_channels):
        val_c = weighted_attribs[:, :, c].view(-1)
        numer[:, c].scatter_add_(0, flat_indices.view(-1), val_c)
    
    denom = denom.clamp(min=1e-6)
    out_img = numer / denom.unsqueeze(1)
    out_img = out_img.reshape(H, W, C_channels).permute(2, 0, 1)
    
    return out_img


class OT5DInterpolator:
    """Interpolateur 5D avec support pour splatting adaptatif."""
    
    def __init__(self, config: OTConfig):
        self.cfg = config
        # Utiliser "tensorized" au lieu de "auto" pour éviter les problèmes de compilation KeOps
        # "tensorized" est plus stable mais peut être plus lent
        backend = "tensorized" if config.device == "cuda" else "auto"
        try:
            self.loss_layer = SamplesLoss(
                loss="sinkhorn", p=2, blur=config.blur, reach=config.reach,
                debias=False, potentials=True, scaling=config.scaling, backend=backend
            )
        except Exception as e:
            logger.warning(f"Erreur création SamplesLoss avec backend {backend}, essai avec 'auto': {e}")
            # Fallback vers auto si tensorized échoue
            self.loss_layer = SamplesLoss(
                loss="sinkhorn", p=2, blur=config.blur, reach=config.reach,
                debias=False, potentials=True, scaling=config.scaling, backend="auto"
            )
        self.pi = None  # Cache du plan de transport
    
    def interpolate(self, img_source, img_target, times: List[float]):
        """Interpole entre source et target aux temps spécifiés."""
        logger.debug(f"Interpolation 5D: {len(times)} temps")
        
        # 1. Préparation 5D
        X_a, w_a, colors_a, Ha, Wa = get_5d_cloud(
            img_source.to(self.cfg.device),
            self.cfg.resolution[0],
            self.cfg.lambda_color,
        )
        X_b, w_b, colors_b, Hb, Wb = get_5d_cloud(
            img_target.to(self.cfg.device),
            self.cfg.resolution[1],
            self.cfg.lambda_color,
        )
        
        logger.debug(f"Nuages 5D: Source {X_a.shape[0]} points, Target {X_b.shape[0]} points")
        
        # 2. Sinkhorn & Plan pi
        logger.debug("Calcul Sinkhorn...")
        try:
            # Nettoyer le cache GPU avant le calcul
            if self.cfg.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            F_pot, G_pot = self.loss_layer(w_a, X_a, w_b, X_b)
            F_pot, G_pot = F_pot.flatten(), G_pot.flatten()
        except RuntimeError as e:
            logger.error(f"Erreur RuntimeError lors du calcul Sinkhorn: {e}")
            # Nettoyer et réessayer une fois
            if self.cfg.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            logger.info("Réessai après nettoyage mémoire...")
            F_pot, G_pot = self.loss_layer(w_a, X_a, w_b, X_b)
            F_pot, G_pot = F_pot.flatten(), G_pot.flatten()
        except Exception as e:
            logger.error(f"Erreur fatale lors du calcul Sinkhorn: {e}", exc_info=True)
            raise
        
        dist_sq = torch.cdist(X_a, X_b, p=2) ** 2
        C_matrix = dist_sq / 2.0
        epsilon = self.cfg.blur**2
        
        # Reconstruction log-domain stable
        log_pi = (
            (F_pot[:, None] + G_pot[None, :] - C_matrix) / epsilon
            + torch.log(w_a.flatten()[:, None])
            + torch.log(w_b.flatten()[None, :])
        )
        pi = torch.exp(log_pi).squeeze()
        
        # Cache pour réutilisation
        self.pi = pi
        
        # Filtrage Sparse
        mask = pi > (pi.max() * 1e-4)
        I_idx, J_idx = mask.nonzero(as_tuple=True)
        self.pi_sparse_mask = (I_idx, J_idx)
        weights_ij = pi[I_idx, J_idx]
        
        # Pré-chargement des données
        pos_a_spatial = X_a[I_idx, :2]
        pos_b_spatial = X_b[J_idx, :2]
        col_a_real = colors_a[I_idx]
        col_b_real = colors_b[J_idx]
        
        results = []
        
        # Calcul de la densité théorique pour éviter les trous
        N_active = weights_ij.shape[0]
        avg_spacing = np.sqrt((Hb * Wb) / (N_active + 1e-6))
        min_sigma_theoretical = avg_spacing / 2.0
        
        # Rediriger tqdm vers le logger
        tqdm_logger = TqdmToLogger(logger, logging.INFO)
        for t in tqdm(times, desc="Interpolation 5D", file=tqdm_logger):
            # A. Barycentre Géodésique
            pos_t = (1 - t) * pos_a_spatial + t * pos_b_spatial
            col_t = (1 - t) * col_a_real + t * col_b_real
            
            # B. Sigma "Intelligent"
            sigma_intrinsic = (1 - t) * self.cfg.sigma_start + t * self.cfg.sigma_end
            sigma_expansion = self.cfg.sigma_boost * 4 * t * (1 - t)
            
            # Calcul dynamique de la taille de la grille (Canvas)
            Ht = int((1 - t) * Ha + t * Hb)
            Wt = int((1 - t) * Wa + t * Wb)
            
            # Mise à jour du sigma min théorique pour la grille actuelle
            current_spacing = np.sqrt((Ht * Wt) / (N_active + 1e-6))
            min_sigma_t = current_spacing / 2.0
            
            # Sigma final
            sigma_t = max(sigma_intrinsic + sigma_expansion, min_sigma_t * 0.8)
            
            # C. Splatting sur la grille dynamique (Ht, Wt)
            img_t = vectorized_gaussian_splatting(
                pos_t, col_t, weights_ij, Ht, Wt, sigma=sigma_t
            )
            
            results.append(img_t.cpu())
        
        return results
    
    def get_displacement_field(self, img_source, img_target):
        """Calcule le champ de déplacement spatial T(x) - x depuis le plan de transport 5D."""
        # Préparation 5D
        X_a, w_a, colors_a, Ha, Wa = get_5d_cloud(
            img_source.to(self.cfg.device),
            self.cfg.resolution[0],
            self.cfg.lambda_color,
        )
        X_b, w_b, colors_b, Hb, Wb = get_5d_cloud(
            img_target.to(self.cfg.device),
            self.cfg.resolution[1],
            self.cfg.lambda_color,
        )
        
        # Sinkhorn & Plan pi
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
        
        # Projection barycentrique spatiale
        pos_b_spatial = X_b[:, :2]
        pos_a_spatial = X_a[:, :2]
        
        # Normalisation conditionnelle P(y|x)
        log_row_sum = torch.logsumexp(log_pi, dim=1, keepdim=True)
        log_cond_prob = log_pi - log_row_sum
        cond_prob = torch.exp(log_cond_prob)
        
        # Barycentric projection: T = P * Y_spatial
        T_map_flat = torch.mm(cond_prob, pos_b_spatial)
        
        # Champ de déplacement: displacement = T(x) - x
        displacement_flat = T_map_flat - pos_a_spatial
        
        # Reshape en grille (H, W, 2)
        displacement_field = displacement_flat.view(Ha, Wa, 2)
        
        return displacement_field, Ha, Wa


# ============================================================================
# Chargement d'images Pokemon
# ============================================================================

def load_pokemon_image_pair(source_name: str, target_name: str, data_dir: str, target_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Charge les images Pokemon (Salameche et Strawberry) depuis fichiers.
    
    Args:
        source_name: Nom du fichier source (ex: "salameche.jpg")
        target_name: Nom du fichier target (ex: "pikachu.webp")
        data_dir: Répertoire contenant les images
        target_size: Taille cible pour le redimensionnement
    
    Returns:
        Tuple de deux tensors (C, H, W) normalisés entre 0 et 1
    """
    data_path = Path(data_dir)
    
    # Essayer différentes extensions possibles
    source_extensions = [".jpg", ".jpeg", ".png", ".webp"]
    target_extensions = [".webp", ".jpg", ".jpeg", ".png"]
    
    source_path = None
    target_path = None
    
    # Chercher le fichier source
    for ext in source_extensions:
        candidate = data_path / f"{source_name.split('.')[0]}{ext}"
        if candidate.exists():
            source_path = candidate
            break
    
    # Chercher le fichier target
    for ext in target_extensions:
        candidate = data_path / f"{target_name.split('.')[0]}{ext}"
        if candidate.exists():
            target_path = candidate
            break
    
    if source_path is None or target_path is None:
        logger.error(f"Images non trouvées: source={source_name}, target={target_name}")
        logger.error(f"Source cherchée: {source_path}, Target cherchée: {target_path}")
        logger.error(f"Répertoire: {data_path}")
        if data_path.exists():
            logger.error(f"Fichiers disponibles: {list(data_path.glob('*'))}")
        raise FileNotFoundError(f"Images Pokemon non trouvées dans {data_dir}")
    
    logger.info(f"Chargement images Pokemon: {source_path.name} -> {target_path.name}")
    
    try:
        # Charger et convertir en RGB
        img1_pil = Image.open(source_path).convert("RGB")
        img2_pil = Image.open(target_path).convert("RGB")
        
        # Redimensionner
        img1_pil = img1_pil.resize((target_size, target_size), Image.Resampling.LANCZOS)
        img2_pil = img2_pil.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Convertir en tensor
        img1 = torch.from_numpy(np.array(img1_pil)).permute(2, 0, 1).float() / 255.0
        img2 = torch.from_numpy(np.array(img2_pil)).permute(2, 0, 1).float() / 255.0
        
        logger.info(f"Images chargées: {img1.shape} et {img2.shape}")
        return img1, img2
        
    except Exception as e:
        logger.error(f"Erreur chargement images Pokemon: {e}", exc_info=True)
        raise


# ============================================================================
# Structure de sortie
# ============================================================================

def setup_output_dirs(output_dir: str):
    """Crée la structure de répertoires pour les résultats."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "metrics").mkdir(exist_ok=True)
    (output_path / "images").mkdir(exist_ok=True)
    (output_path / "configs").mkdir(exist_ok=True)
    (output_path / "logs").mkdir(exist_ok=True)
    (output_path / "transport_plans").mkdir(exist_ok=True)
    (output_path / "displacement_fields").mkdir(exist_ok=True)
    return output_path

def log_memory_usage(device: str, context: str = ""):
    """Log l'utilisation mémoire GPU ou CPU."""
    try:
        if device == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            max_allocated = torch.cuda.max_memory_allocated(0) / 1e9
            logger.debug(f"Mémoire GPU {context}: {allocated:.2f} GB allouée / {reserved:.2f} GB réservée / "
                        f"{total:.2f} GB total (max: {max_allocated:.2f} GB)")
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'max_allocated_gb': max_allocated
            }
        else:
            try:
                import psutil
                process = psutil.Process()
                mem_info = process.memory_info()
                mem_gb = mem_info.rss / 1e9
                logger.debug(f"Mémoire CPU {context}: {mem_gb:.2f} GB")
                return {'cpu_memory_gb': mem_gb}
            except ImportError:
                pass
    except Exception as e:
        logger.warning(f"Erreur logging mémoire {context}: {e}")
    return {}


# ============================================================================
# Expériences
# ============================================================================

class PokemonExperimentRunner:
    """Gère l'exécution des expériences Pokemon à grande échelle."""
    
    def __init__(self, exp_config: PokemonExperimentConfig):
        global _runner_instance
        _runner_instance = self  # Enregistrer l'instance globale pour la sauvegarde d'urgence
        
        self.exp_config = exp_config
        self.output_dir = setup_output_dirs(exp_config.output_dir)
        # Initialiser le logging
        logger_instance, log_file = setup_logging(exp_config.output_dir)
        self.log_file = log_file
        self.metrics_computer = MetricsComputer()
        self.results = []
        self.experiment_id = 0
        
        # Charger les images Pokemon une seule fois
        logger.info("=" * 80)
        logger.info("CHARGEMENT DES IMAGES POKEMON")
        logger.info("=" * 80)
        logger.info(f"Source: {exp_config.source_image_name}")
        logger.info(f"Target: {exp_config.target_image_name}")
        logger.info(f"Répertoire: {exp_config.data_dir}")
        
        self.img_source, self.img_target = load_pokemon_image_pair(
            exp_config.source_image_name,
            exp_config.target_image_name,
            exp_config.data_dir,
            target_size=256  # Charger en haute résolution, on redimensionnera après
        )
        
        # Logger toutes les informations sur les images
        logger.info("=" * 80)
        logger.info("INFORMATIONS SUR LES IMAGES CHARGÉES")
        logger.info("=" * 80)
        logger.info(f"Image Source:")
        logger.info(f"  Shape: {self.img_source.shape}")
        logger.info(f"  Taille: {self.img_source.shape[1]}×{self.img_source.shape[2]} pixels")
        logger.info(f"  Canaux: {self.img_source.shape[0]}")
        logger.info(f"  Min: {self.img_source.min().item():.4f}, Max: {self.img_source.max().item():.4f}")
        logger.info(f"  Mean: {self.img_source.mean().item():.4f}, Std: {self.img_source.std().item():.4f}")
        logger.info(f"Image Target:")
        logger.info(f"  Shape: {self.img_target.shape}")
        logger.info(f"  Taille: {self.img_target.shape[1]}×{self.img_target.shape[2]} pixels")
        logger.info(f"  Canaux: {self.img_target.shape[0]}")
        logger.info(f"  Min: {self.img_target.min().item():.4f}, Max: {self.img_target.max().item():.4f}")
        logger.info(f"  Mean: {self.img_target.mean().item():.4f}, Std: {self.img_target.std().item():.4f}")
        logger.info("=" * 80)
        
        logger.info(f"ExperimentRunner initialisé. Output dir: {self.output_dir}")
        logger.info(f"Fichier de log: {log_file}")
        
        # Sauvegarder périodiquement
        self.last_save_time = time.time()
        self.save_interval = 300  # Sauvegarder toutes les 5 minutes
    
    def _check_memory_before_experiment(self, resolution: int) -> bool:
        """Vérifie si on a assez de mémoire pour cette résolution."""
        if self.exp_config.device == "cuda" and torch.cuda.is_available():
            # Estimer la mémoire nécessaire (approximatif)
            # Pour une résolution R, on a environ R^2 points, donc R^4 pour la matrice de coût
            estimated_memory_gb = (resolution ** 4) * 4 / 1e9  # 4 bytes par float32
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            allocated = torch.cuda.memory_allocated(0) / 1e9
            available = total_memory - allocated
            
            logger.debug(f"Résolution {resolution}: mémoire estimée {estimated_memory_gb:.2f} GB, "
                        f"disponible {available:.2f} GB")
            
            if estimated_memory_gb > available * 0.8:  # Garder 20% de marge
                logger.warning(f"Résolution {resolution} peut nécessiter trop de mémoire "
                             f"({estimated_memory_gb:.2f} GB estimé, {available:.2f} GB disponible)")
                return False
        return True
    
    def _save_results_incremental(self):
        """Sauvegarde incrémentale des résultats."""
        current_time = time.time()
        if current_time - self.last_save_time > self.save_interval:
            logger.info("Sauvegarde incrémentale des résultats...")
            try:
                self.save_results()
                self.last_save_time = current_time
            except Exception as e:
                logger.warning(f"Erreur sauvegarde incrémentale: {e}")
    
    def run_single_experiment(
        self,
        resolution: int,
        lambda_val: float,
        blur: float,
        reach: Optional[float],
        times: List[float]
    ) -> List[Dict]:
        """Exécute une seule expérience."""
        self.experiment_id += 1
        exp_id = self.experiment_id
        
        logger.info("=" * 80)
        logger.info(f"EXPÉRIENCE {exp_id}")
        logger.info("=" * 80)
        logger.info(f"Paire d'images: {self.exp_config.image_pair_name}")
        logger.info(f"Résolution: {resolution}×{resolution} pixels")
        logger.info(f"Lambda (λ): {lambda_val}")
        logger.info(f"Blur (ε): {blur}")
        logger.info(f"Reach (ρ): {reach if reach is not None else 'balanced'}")
        logger.info(f"Temps d'interpolation: {times}")
        logger.info(f"Nombre de frames: {len(times)}")
        
        # Vérifier la mémoire avant de commencer
        if not self._check_memory_before_experiment(resolution):
            logger.warning(f"Expérience {exp_id} ignorée: mémoire insuffisante pour résolution {resolution}")
            return []
        
        # Sauvegarde incrémentale
        self._save_results_incremental()
        
        # Redimensionner les images source/target à la résolution demandée
        img_source_resized = F.interpolate(
            self.img_source.unsqueeze(0),
            size=(resolution, resolution),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        img_target_resized = F.interpolate(
            self.img_target.unsqueeze(0),
            size=(resolution, resolution),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # Logger toutes les informations sur les images redimensionnées
        logger.info(f"Image source redimensionnée: {img_source_resized.shape}")
        logger.info(f"  Taille: {img_source_resized.shape[1]}×{img_source_resized.shape[2]} pixels")
        logger.info(f"  Canaux: {img_source_resized.shape[0]}")
        logger.info(f"  Min: {img_source_resized.min().item():.4f}, Max: {img_source_resized.max().item():.4f}")
        logger.info(f"  Mean: {img_source_resized.mean().item():.4f}, Std: {img_source_resized.std().item():.4f}")
        logger.info(f"Image target redimensionnée: {img_target_resized.shape}")
        logger.info(f"  Taille: {img_target_resized.shape[1]}×{img_target_resized.shape[2]} pixels")
        logger.info(f"  Canaux: {img_target_resized.shape[0]}")
        logger.info(f"  Min: {img_target_resized.min().item():.4f}, Max: {img_target_resized.max().item():.4f}")
        logger.info(f"  Mean: {img_target_resized.mean().item():.4f}, Std: {img_target_resized.std().item():.4f}")
        logger.info(f"Nombre de pixels par image: {resolution * resolution}")
        logger.info(f"Nombre total de points 5D (source): {resolution * resolution}")
        logger.info(f"Nombre total de points 5D (target): {resolution * resolution}")
        logger.info(f"Taille matrice de coût estimée: {resolution * resolution} × {resolution * resolution}")
        logger.info("=" * 80)
        
        # Log mémoire avant
        mem_before = log_memory_usage(self.exp_config.device, f"avant exp {exp_id}")
        
        # Configuration OT
        ot_config = OTConfig(
            resolution=(resolution, resolution),
            blur=blur,
            reach=reach,
            lambda_color=lambda_val,
            device=self.exp_config.device
        )
        
        # Interpolateur
        interpolator = None
        try:
            interpolator = OT5DInterpolator(ot_config)
        except Exception as e:
            logger.error(f"Erreur création interpolateur exp {exp_id}: {e}", exc_info=True)
            return []
        
        # Interpolation
        start_time = time.time()
        sinkhorn_time = 0.0
        interpolation_time = 0.0
        try:
            sinkhorn_start = time.time()
            frames = interpolator.interpolate(img_source_resized, img_target_resized, times)
            sinkhorn_time = time.time() - sinkhorn_start
            compute_time = time.time() - start_time
            interpolation_time = compute_time - sinkhorn_time
            
            # Log mémoire après
            mem_after = log_memory_usage(self.exp_config.device, f"après exp {exp_id}")
            
            logger.info("=" * 80)
            logger.info(f"EXPÉRIENCE {exp_id} TERMINÉE")
            logger.info("=" * 80)
            logger.info(f"Temps total: {compute_time:.2f}s")
            logger.info(f"  - Sinkhorn: {sinkhorn_time:.2f}s ({sinkhorn_time/compute_time*100:.1f}%)")
            logger.info(f"  - Interpolation: {interpolation_time:.2f}s ({interpolation_time/compute_time*100:.1f}%)")
            logger.info(f"  - Temps par frame: {compute_time/len(times):.3f}s")
            logger.info(f"Nombre de frames générées: {len(frames)}")
            if len(frames) > 0:
                logger.info(f"Shape des frames: {frames[0].shape}")
                logger.info(f"  - Canaux: {frames[0].shape[0]}")
                logger.info(f"  - Hauteur: {frames[0].shape[1]} pixels")
                logger.info(f"  - Largeur: {frames[0].shape[2]} pixels")
            
            if self.exp_config.device == "cuda" and torch.cuda.is_available():
                if 'allocated_gb' in mem_after:
                    logger.info(f"Mémoire GPU:")
                    logger.info(f"  - Allouée: {mem_after['allocated_gb']:.2f} GB")
                    logger.info(f"  - Max allouée: {mem_after.get('max_allocated_gb', 0):.2f} GB")
                    if 'reserved_gb' in mem_after:
                        logger.info(f"  - Réservée: {mem_after['reserved_gb']:.2f} GB")
            logger.info("=" * 80)
            
            # Sauvegarder le plan de transport si demandé
            if self.exp_config.save_transport_plans and hasattr(interpolator, 'pi') and interpolator.pi is not None:
                self._save_transport_plan(exp_id, interpolator, resolution, lambda_val, blur, reach)
        except RuntimeError as e:
            error_msg = str(e)
            logger.error(f"Erreur RuntimeError interpolation exp {exp_id}: {error_msg}")
            if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
                logger.error("Erreur mémoire GPU détectée. Nettoyage...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
            logger.error(f"Traceback complet:", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Erreur interpolation exp {exp_id}: {e}", exc_info=True)
            logger.error(f"Type d'erreur: {type(e).__name__}")
            logger.error(f"Traceback complet:\n{traceback.format_exc()}")
            # Nettoyer la mémoire même en cas d'erreur
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return []
        finally:
            # Nettoyer la mémoire après chaque expérience
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Calcul des métriques pour chaque frame
        experiment_results = []
        logger.info(f"Calcul des métriques pour {len(frames)} frames...")
        for i, (frame, t) in enumerate(zip(frames, times)):
            metrics = self.metrics_computer.compute_all(
                frame, img_source_resized, img_target_resized, t, compute_tearing=False
            )
            
            # Logger les métriques pour chaque frame (détaillé pour t=0.5)
            if abs(t - 0.5) < 0.01:  # Frame au milieu
                logger.info(f"  Frame t={t:.2f} (milieu):")
                logger.info(f"    PSNR: {metrics['psnr']:.2f} dB" if metrics.get('psnr') is not None else "    PSNR: N/A")
                logger.info(f"    ΔE: {metrics['delta_e']:.2f}" if metrics.get('delta_e') is not None else "    ΔE: N/A")
                logger.info(f"    Sharpness: {metrics['sharpness']:.4f}" if metrics.get('sharpness') is not None else "    Sharpness: N/A")
                logger.info(f"    Coverage: {metrics['coverage']:.4f}" if metrics.get('coverage') is not None else "    Coverage: N/A")
                logger.info(f"    Mass Error: {metrics['mass_error']:.6f}" if metrics.get('mass_error') is not None else "    Mass Error: N/A")
                tearing_val = metrics.get('tearing_pct')
                if tearing_val is not None:
                    logger.info(f"    Tearing: {tearing_val:.2f}%")
                else:
                    logger.info("    Tearing: N/A")
            
            result = {
                'experiment_id': exp_id,
                'image_pair': self.exp_config.image_pair_name,
                'resolution': resolution,
                'lambda': lambda_val,
                'blur': blur,
                'reach': reach if reach is not None else 'balanced',
                'splatting': True,
                't': t,
                'psnr': metrics['psnr'],
                'delta_e': metrics['delta_e'],
                'tearing_pct': metrics['tearing_pct'] or 0.0,
                'coverage': metrics['coverage'],
                'mass_error': metrics['mass_error'],
                'sharpness': metrics['sharpness'],
                'compute_time_total': compute_time,
                'compute_time_per_frame': compute_time / len(times),
                'sinkhorn_time': sinkhorn_time,
                'interpolation_time': interpolation_time,
                'memory_allocated_gb': mem_after.get('allocated_gb', 0.0) if 'allocated_gb' in mem_after else 0.0,
                'memory_max_allocated_gb': mem_after.get('max_allocated_gb', 0.0) if 'max_allocated_gb' in mem_after else 0.0,
                'regime': None,
                'mean_displacement': None,
                'max_displacement': None,
                'std_displacement': None,
                'mean_divergence': None,
                'mean_curl': None,
                'mean_laplacian': None,
                'smoothness_score': None
            }
            experiment_results.append(result)
            
            # Sauvegarde image
            if self.exp_config.save_images:
                img_path = self.output_dir / "images" / f"exp{exp_id}_t{t:.3f}.png"
                frame_np = frame.permute(1, 2, 0).clamp(0, 1).numpy()
                plt.imsave(str(img_path), frame_np, dpi=150)
                logger.debug(f"Image sauvegardée: {img_path}")
        
        return experiment_results
    
    def run_experiment_1_lambda_ablation(self):
        """Expérience 1: Ablation Lambda (λ) - Large range."""
        logger.info("=== Expérience 1: Ablation Lambda (Large Scale) ===")
        
        resolution = 64
        blur = 0.03
        reach = 0.1
        times = [0.5]  # Focus sur t=0.5
        
        for lambda_val in self.exp_config.lambdas:
            results = self.run_single_experiment(resolution, lambda_val, blur, reach, times)
            if results:
                self.results.extend(results)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def run_experiment_2_2d_vs_5d(self):
        """Expérience 2: Comparaison 2D vs 5D."""
        logger.info("=== Expérience 2: Comparaison 2D vs 5D ===")
        
        resolution = 64
        blur = 0.03
        reach = 0.1
        
        # 2D (lambda=0.0)
        results = self.run_single_experiment(resolution, 0.0, blur, reach, self.exp_config.times)
        if results:
            self.results.extend(results)
        
        # 5D optimal (lambda=1.0)
        results = self.run_single_experiment(resolution, 1.0, blur, reach, self.exp_config.times)
        if results:
            self.results.extend(results)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def run_experiment_3_splatting_impact(self):
        """Expérience 3: Impact du Splatting Adaptatif."""
        logger.info("=== Expérience 3: Impact Splatting ===")
        
        lambda_val = 1.0
        blur = 0.03
        reach = 0.1
        times = [0.5]
        
        for resolution in [32, 48, 64, 96, 128]:
            if self.exp_config.max_resolution and resolution > self.exp_config.max_resolution:
                continue
            try:
                # Nettoyer avant chaque résolution
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                results = self.run_single_experiment(resolution, lambda_val, blur, reach, times)
                if results:
                    self.results.extend(results)
                    self._save_results_incremental()
            except Exception as e:
                logger.error(f"Erreur expérience 3 résolution {resolution}: {e}", exc_info=True)
                logger.warning(f"Continuation avec la résolution suivante...")
                continue
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def run_experiment_4_parameter_sensitivity(self):
        """Expérience 4: Sensibilité aux paramètres (ε, ρ) - Grille dense."""
        logger.info("=== Expérience 4: Sensibilité Paramètres (Large Scale) ===")
        
        resolution = 64
        lambda_val = 1.0
        times = [0.5]
        
        for blur in self.exp_config.blurs:
            for reach in self.exp_config.reaches:
                results = self.run_single_experiment(resolution, lambda_val, blur, reach, times)
                if results:
                    self.results.extend(results)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def run_experiment_5_scalability(self):
        """Expérience 5: Scalabilité Résolution - Résolutions très élevées."""
        logger.info("=== Expérience 5: Scalabilité (Large Scale) ===")
        
        lambda_val = 1.0
        blur = 0.03
        reach = 0.1
        times = [0.5]
        
        # Résolutions à tester progressivement
        if self.exp_config.progressive_resolution:
            resolutions_to_test = [32, 48, 64, 96, 128]
            logger.info(f"Test progressif: résolutions {resolutions_to_test}")
        else:
            resolutions_to_test = self.exp_config.resolutions
        
        for resolution in resolutions_to_test:
            if self.exp_config.max_resolution and resolution > self.exp_config.max_resolution:
                logger.info(f"Résolution {resolution} ignorée (max={self.exp_config.max_resolution})")
                continue
            
            logger.info(f"Test résolution {resolution}×{resolution}...")
            try:
                # Nettoyer avant chaque nouvelle résolution
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                results = self.run_single_experiment(resolution, lambda_val, blur, reach, times)
                if results:
                    self.results.extend(results)
                    logger.info(f"✓ Résolution {resolution}×{resolution} réussie")
                    
                    # Sauvegarder après chaque résolution réussie
                    self._save_results_incremental()
                    
                    # Si on est en mode progressif et que ça marche, continuer avec les plus grandes
                    if self.exp_config.progressive_resolution and resolution == 128:
                        for next_res in [160, 192, 256]:
                            if self.exp_config.max_resolution and next_res > self.exp_config.max_resolution:
                                continue
                            logger.info(f"Test résolution {next_res}×{next_res}...")
                            try:
                                # Nettoyer avant chaque nouvelle résolution
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                gc.collect()
                                
                                results_next = self.run_single_experiment(next_res, lambda_val, blur, reach, times)
                                if results_next:
                                    self.results.extend(results_next)
                                    logger.info(f"✓ Résolution {next_res}×{next_res} réussie")
                                    self._save_results_incremental()
                            except Exception as e:
                                logger.error(f"Résolution {next_res}×{next_res} échouée: {e}", exc_info=True)
                                logger.warning("Arrêt du test progressif à cause de l'erreur")
                                break
            except Exception as e:
                logger.error(f"Résolution {resolution}×{resolution} échouée: {e}", exc_info=True)
                logger.error(f"Type d'erreur: {type(e).__name__}")
                logger.error(f"Traceback complet:\n{traceback.format_exc()}")
                if self.exp_config.progressive_resolution:
                    logger.info("Arrêt du test progressif à cause de l'erreur")
                    # Sauvegarder avant d'arrêter
                    try:
                        self.save_results()
                    except:
                        pass
                    break
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def run_experiment_6_displacement_robustness(self):
        """Expérience 6: Robustesse du champ de déplacement selon les régimes."""
        logger.info("=== Expérience 6: Robustesse Champ de Déplacement ===")
        
        resolution = 64
        lambda_val = 1.0
        times = [0.5]
        
        regimes = [
            {"name": "Unbalanced_OT", "blur": 0.01, "reach": 0.1, "description": "ρ fini, ε petit"},
            {"name": "Entropy_Regularized", "blur": 0.1, "reach": None, "description": "ε grand, ρ = ∞"},
            {"name": "Unbalanced_Entropy", "blur": 0.03, "reach": 0.3, "description": "ε et ρ intermédiaires"},
        ]
        
        for regime in regimes:
            logger.info(f"Régime: {regime['name']} ({regime['description']})")
            blur = regime['blur']
            reach = regime['reach']
            
            ot_config = OTConfig(
                resolution=(resolution, resolution),
                blur=blur,
                reach=reach,
                lambda_color=lambda_val,
                device=self.exp_config.device
            )
            
            interpolator = OT5DInterpolator(ot_config)
            
            try:
                # Redimensionner les images
                img_source_resized = F.interpolate(
                    self.img_source.unsqueeze(0),
                    size=(resolution, resolution),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                img_target_resized = F.interpolate(
                    self.img_target.unsqueeze(0),
                    size=(resolution, resolution),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                
                # Calculer le champ de déplacement
                displacement_field, H, W = interpolator.get_displacement_field(img_source_resized, img_target_resized)
                
                # Métriques de smoothness
                smoothness_metrics = compute_displacement_smoothness(displacement_field)
                
                # Sauvegarder le champ de déplacement
                disp_dir = self.output_dir / "displacement_fields"
                disp_dir.mkdir(parents=True, exist_ok=True)
                
                # Visualisation
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # 1. Magnitude
                magnitude = torch.sqrt(displacement_field[..., 0]**2 + displacement_field[..., 1]**2)
                im1 = axes[0].imshow(magnitude.cpu().numpy(), cmap='hot')
                axes[0].set_title(f"Magnitude du Déplacement\n{regime['name']}")
                axes[0].axis('off')
                plt.colorbar(im1, ax=axes[0])
                
                # 2. Champ vectoriel
                step = 4
                y_indices = torch.arange(step//2, H, step)
                x_indices = torch.arange(step//2, W, step)
                yy, xx = torch.meshgrid(y_indices, x_indices, indexing="ij")
                
                disp_sampled = displacement_field[yy, xx].cpu()
                axes[1].imshow(img_source_resized.permute(1, 2, 0).cpu() * 0.6)
                axes[1].quiver(xx.numpy(), yy.numpy(), 
                              disp_sampled[..., 0].numpy() * (W-1),
                              disp_sampled[..., 1].numpy() * (H-1),
                              magnitude[yy, xx].cpu().numpy(),
                              cmap='coolwarm', angles='xy', scale_units='xy', scale=1, alpha=0.8)
                axes[1].set_title(f"Champ Vectoriel\n{regime['name']}")
                axes[1].axis('off')
                
                # 3. Divergence
                disp_np = displacement_field.cpu().numpy()
                grad_x = np.gradient(disp_np[..., 0], axis=1)
                grad_y = np.gradient(disp_np[..., 1], axis=0)
                divergence = grad_x + grad_y
                im3 = axes[2].imshow(divergence, cmap='RdBu', vmin=-np.abs(divergence).max(), 
                                    vmax=np.abs(divergence).max())
                axes[2].set_title(f"Divergence\n{regime['name']}")
                axes[2].axis('off')
                plt.colorbar(im3, ax=axes[2])
                
                plt.tight_layout()
                fig_path = disp_dir / f"displacement_field_{regime['name']}.png"
                plt.savefig(fig_path, dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"Champ de déplacement sauvegardé: {fig_path}")
                
                # Ajouter les métriques aux résultats
                self.experiment_id += 1
                exp_id = self.experiment_id
                result = {
                    'experiment_id': exp_id,
                    'image_pair': self.exp_config.image_pair_name,
                    'resolution': resolution,
                    'lambda': lambda_val,
                    'blur': blur,
                    'reach': reach if reach is not None else 'balanced',
                    'splatting': True,
                    't': 0.5,
                    'regime': regime['name'],
                    'mean_displacement': smoothness_metrics['mean_displacement'],
                    'max_displacement': smoothness_metrics['max_displacement'],
                    'std_displacement': smoothness_metrics['std_displacement'],
                    'mean_divergence': smoothness_metrics['mean_divergence'],
                    'mean_curl': smoothness_metrics['mean_curl'],
                    'mean_laplacian': smoothness_metrics['mean_laplacian'],
                    'smoothness_score': smoothness_metrics['smoothness_score'],
                    'psnr': None,
                    'delta_e': None,
                    'tearing_pct': None,
                    'coverage': None,
                    'mass_error': None,
                    'sharpness': None,
                    'compute_time_total': 0.0,
                    'compute_time_per_frame': 0.0,
                    'sinkhorn_time': 0.0,
                    'interpolation_time': 0.0,
                    'memory_allocated_gb': 0.0,
                    'memory_max_allocated_gb': 0.0
                }
                self.results.append(result)
                logger.info(f"Régime {regime['name']}: smoothness_score={smoothness_metrics['smoothness_score']:.4f}")
                
            except Exception as e:
                logger.error(f"Erreur calcul champ de déplacement pour {regime['name']}: {e}", exc_info=True)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _save_transport_plan(self, exp_id: int, interpolator, resolution: int, lambda_val: float, blur: float, reach: Optional[float]):
        """Sauvegarde le plan de transport à échelle réduite."""
        try:
            if interpolator.pi is None:
                return
            
            if interpolator.pi_sparse_mask is not None:
                I_idx, J_idx = interpolator.pi_sparse_mask
                pi_sparse = interpolator.pi[I_idx, J_idx]
            else:
                pi_sparse = interpolator.pi
            
            pi_scaled = (pi_sparse * self.exp_config.transport_plan_scale).cpu()
            
            reach_str = f"{reach:.2f}" if reach is not None else "balanced"
            plan_dir = self.output_dir / "transport_plans"
            plan_dir.mkdir(parents=True, exist_ok=True)
            plan_path = plan_dir / f"plan_exp{exp_id}_res{resolution}_lam{lambda_val:.1f}_eps{blur:.3f}_rho{reach_str}.pt"
            
            save_dict = {
                'pi': pi_scaled,
                'scale': self.exp_config.transport_plan_scale,
                'metadata': {
                    'experiment_id': exp_id,
                    'image_pair': self.exp_config.image_pair_name,
                    'resolution': resolution,
                    'lambda': lambda_val,
                    'blur': blur,
                    'reach': reach,
                }
            }
            
            if interpolator.pi_sparse_mask is not None:
                save_dict['indices'] = {
                    'I_idx': I_idx.cpu(),
                    'J_idx': J_idx.cpu()
                }
            
            torch.save(save_dict, plan_path)
            logger.debug(f"Plan de transport sauvegardé: {plan_path}")
        except Exception as e:
            logger.warning(f"Erreur sauvegarde plan de transport exp {exp_id}: {e}")
    
    def save_results(self):
        """Sauvegarde tous les résultats en CSV."""
        if not self.results:
            logger.warning("Aucun résultat à sauvegarder")
            return
        
        csv_path = self.output_dir / "metrics" / "all_experiments.csv"
        fieldnames = [
            'experiment_id', 'image_pair', 'resolution', 'lambda', 'blur', 'reach',
            'splatting', 't', 'psnr', 'delta_e', 'tearing_pct', 'coverage',
            'mass_error', 'sharpness', 'compute_time_total', 'compute_time_per_frame',
            'sinkhorn_time', 'interpolation_time', 'memory_allocated_gb', 'memory_max_allocated_gb',
            'regime', 'mean_displacement', 'max_displacement', 'std_displacement',
            'mean_divergence', 'mean_curl', 'mean_laplacian', 'smoothness_score'
        ]
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.results:
                csv_row = {}
                for key in fieldnames:
                    value = row.get(key, None)
                    if value is None:
                        csv_row[key] = ''
                    else:
                        csv_row[key] = value
                writer.writerow(csv_row)
        
        logger.info(f"Résultats sauvegardés: {csv_path} ({len(self.results)} lignes)")
        logger.info(f"Résumé: {len(set(r['experiment_id'] for r in self.results))} expériences")
    
    def run_all_experiments(self):
        """Exécute toutes les expériences."""
        logger.info("=" * 80)
        logger.info("DÉMARRAGE DES EXPÉRIENCES POKEMON À GRANDE ÉCHELLE")
        logger.info("=" * 80)
        logger.info(f"Paire d'images: {self.exp_config.image_pair_name}")
        logger.info(f"Source: {self.exp_config.source_image_name}")
        logger.info(f"Target: {self.exp_config.target_image_name}")
        logger.info(f"Device: {self.exp_config.device}")
        logger.info("")
        logger.info("CONFIGURATION DES EXPÉRIENCES:")
        logger.info(f"  Résolutions: {self.exp_config.resolutions}")
        logger.info(f"    Nombre: {len(self.exp_config.resolutions)}")
        logger.info(f"    Min: {min(self.exp_config.resolutions)}, Max: {max(self.exp_config.resolutions)}")
        logger.info(f"  Lambdas: {self.exp_config.lambdas}")
        logger.info(f"    Nombre: {len(self.exp_config.lambdas)}")
        logger.info(f"    Min: {min(self.exp_config.lambdas):.2f}, Max: {max(self.exp_config.lambdas):.2f}")
        logger.info(f"  Blurs: {self.exp_config.blurs}")
        logger.info(f"    Nombre: {len(self.exp_config.blurs)}")
        logger.info(f"    Min: {min(self.exp_config.blurs):.3f}, Max: {max(self.exp_config.blurs):.3f}")
        logger.info(f"  Reaches: {self.exp_config.reaches}")
        logger.info(f"    Nombre: {len(self.exp_config.reaches)}")
        logger.info(f"  Temps d'interpolation: {len(self.exp_config.times)} points")
        logger.info(f"    De t={min(self.exp_config.times):.2f} à t={max(self.exp_config.times):.2f}")
        logger.info("")
        logger.info("OPTIONS:")
        logger.info(f"  Sauvegarder images: {self.exp_config.save_images}")
        logger.info(f"  Sauvegarder métriques: {self.exp_config.save_metrics}")
        logger.info(f"  Sauvegarder plans de transport: {self.exp_config.save_transport_plans}")
        logger.info(f"  Résolution progressive: {self.exp_config.progressive_resolution}")
        if self.exp_config.max_resolution:
            logger.info(f"  Résolution max: {self.exp_config.max_resolution}")
        logger.info("")
        
        log_memory_usage(self.exp_config.device, "début")
        logger.info("=" * 80)
        
        total_start = time.time()
        
        logger.info("\n>>> Expérience 1: Ablation Lambda (Large Scale)")
        self.run_experiment_1_lambda_ablation()
        
        logger.info("\n>>> Expérience 2: Comparaison 2D vs 5D")
        self.run_experiment_2_2d_vs_5d()
        
        logger.info("\n>>> Expérience 3: Impact Splatting")
        self.run_experiment_3_splatting_impact()
        
        logger.info("\n>>> Expérience 4: Sensibilité Paramètres (Large Scale)")
        self.run_experiment_4_parameter_sensitivity()
        
        logger.info("\n>>> Expérience 5: Scalabilité (Large Scale)")
        self.run_experiment_5_scalability()
        
        logger.info("\n>>> Expérience 6: Robustesse Champ de Déplacement")
        self.run_experiment_6_displacement_robustness()
        
        total_time = time.time() - total_start
        
        logger.info("\n" + "=" * 80)
        logger.info("SAUVEGARDE DES RÉSULTATS")
        logger.info("=" * 80)
        self.save_results()
        
        logger.info("\n" + "=" * 80)
        logger.info("TOUTES LES EXPÉRIENCES TERMINÉES")
        logger.info("=" * 80)
        logger.info(f"Total de résultats: {len(self.results)} lignes")
        logger.info(f"Nombre d'expériences uniques: {len(set(r['experiment_id'] for r in self.results))}")
        logger.info(f"Temps total: {total_time:.2f}s ({total_time/60:.2f} minutes, {total_time/3600:.2f} heures)")
        n_unique_exps = len(set(r['experiment_id'] for r in self.results))
        if n_unique_exps > 0:
            logger.info(f"Temps moyen par expérience: {total_time / n_unique_exps:.2f}s")
        
        # Statistiques sur les résultats
        if self.results:
            logger.info("")
            logger.info("STATISTIQUES SUR LES RÉSULTATS:")
            psnr_values = [r['psnr'] for r in self.results if 'psnr' in r and r['psnr'] is not None]
            if psnr_values:
                logger.info(f"  PSNR: Min={min(psnr_values):.2f} dB, Max={max(psnr_values):.2f} dB, Mean={np.mean(psnr_values):.2f} dB")
            time_values = [r['compute_time_total'] for r in self.results if 'compute_time_total' in r and r['compute_time_total'] is not None]
            if time_values:
                logger.info(f"  Temps: Min={min(time_values):.3f}s, Max={max(time_values):.3f}s, Mean={np.mean(time_values):.3f}s")
            resolutions_tested = sorted(set(r['resolution'] for r in self.results if 'resolution' in r))
            if resolutions_tested:
                logger.info(f"  Résolutions testées: {resolutions_tested}")
            lambdas_tested = sorted(set(r['lambda'] for r in self.results if 'lambda' in r))
            if lambdas_tested:
                logger.info(f"  Lambdas testés: {lambdas_tested}")
        
        mem_final = log_memory_usage(self.exp_config.device, "fin")
        if self.exp_config.device == "cuda" and torch.cuda.is_available():
            if 'max_allocated_gb' in mem_final:
                logger.info(f"Pic mémoire GPU: {mem_final['max_allocated_gb']:.2f} GB")
        
        # Sauvegarde finale
        logger.info("Sauvegarde finale des résultats...")
        self.save_results()
        
        logger.info(f"Fichier de log: {self.log_file}")
        logger.info("=" * 80)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    try:
        exp_config = PokemonExperimentConfig()
        
        # Lancer les expériences
        runner = PokemonExperimentRunner(exp_config)
        runner.run_all_experiments()
    except KeyboardInterrupt:
        logger.critical("Interruption utilisateur (Ctrl+C). Sauvegarde des résultats...")
        if '_runner_instance' in globals() and _runner_instance is not None:
            try:
                _runner_instance.save_results()
                logger.critical("Résultats sauvegardés.")
            except Exception as e:
                logger.critical(f"Erreur sauvegarde: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Erreur fatale non gérée: {e}", exc_info=True)
        logger.critical(f"Type: {type(e).__name__}")
        logger.critical(f"Traceback complet:\n{traceback.format_exc()}")
        if '_runner_instance' in globals() and _runner_instance is not None:
            try:
                _runner_instance.save_results()
                logger.critical("Résultats sauvegardés avant arrêt.")
            except Exception as save_err:
                logger.critical(f"Erreur sauvegarde d'urgence: {save_err}")
        sys.exit(1)

