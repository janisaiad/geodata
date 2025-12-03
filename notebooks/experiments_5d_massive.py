"""
Script d'expérimentation massive pour le transport optimal 5D.

Ce script exécute une grille systématique d'expériences pour:
1. Ablation Lambda (λ)
2. Comparaison 2D vs 5D
3. Impact du Splatting Adaptatif
4. Sensibilité aux paramètres (ε, ρ)
5. Scalabilité résolution

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
import torchvision
import torchvision.transforms as transforms

# Ajouter le répertoire parent au path pour les imports
sys.path.append(str(Path(__file__).parent))

# Imports depuis les modules existants
from metrics_5d import MetricsComputer, compute_psnr, compute_delta_e, compute_coverage, compute_mass_error, compute_sharpness

# Imports depuis 5d_transport (on va extraire les classes)
# Pour l'instant, on va les redéfinir ici pour éviter les problèmes d'imports
from geomloss import SamplesLoss

# Configuration du logging
def setup_logging(output_dir: str):
    """Configure le logging pour écrire dans un fichier et sur stdout."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Nom du fichier avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"experiments_{timestamp}.log"
    
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
class ExperimentConfig:
    """Configuration pour les expériences."""
    # Grilles de paramètres - Résolutions augmentées
    resolutions: List[int] = field(default_factory=lambda: [32, 48, 64, 96, 128])  # Résolutions plus élevées
    lambdas: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 1.5, 2.0])
    blurs: List[float] = field(default_factory=lambda: [0.01, 0.03, 0.05, 0.1, 0.2, 0.3])  # Ajout 0.2 et 0.3
    reaches: List[Optional[float]] = field(default_factory=lambda: [None, 0.01, 0.05, 0.1, 0.3, 0.5])  # Ajout 0.01 et 0.05
    # use_splatting supprimé - le notebook utilise toujours le splatting adaptatif
    times: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])  # Plus de points temporels
    
    # Images de test - utiliser des datasets classiques
    image_pairs: List[str] = field(default_factory=lambda: [
        "cifar_car_airplane",  # Voiture -> Avion (classique en OT)
        "cifar_bird_ship",     # Oiseau -> Bateau
        "mnist_1_0",           # Chiffre 1 -> 0 (morphing de forme)
    ])
    
    # Mode dataset (utiliser torchvision au lieu de fichiers)
    use_torchvision_datasets: bool = True  # Utiliser CIFAR-10, MNIST, etc.
    
    # Répertoires
    data_dir: str = "/home/janis/4A/geodata/data/pixelart/images"
    output_dir: str = "/home/janis/4A/geodata/refs/reports/results/5d_experiments"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Options
    save_images: bool = True
    save_metrics: bool = True
    save_transport_plans: bool = True  # Sauvegarder les plans de transport
    transport_plan_scale: float = 0.01  # Échelle pour sauvegarder les plans (économie mémoire)
    compute_tearing: bool = False  # Désactivé par défaut (nécessite get_transport_map)
    progressive_resolution: bool = True  # Tester progressivement les résolutions


# ============================================================================
# Utilitaires 5D (copiés depuis 5d_transport.ipynb)
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
        self.loss_layer = SamplesLoss(
            loss="sinkhorn", p=2, blur=config.blur, reach=config.reach,
            debias=False, potentials=True, scaling=config.scaling, backend="auto"
        )
        self.pi = None  # Cache du plan de transport
    
    def interpolate(self, img_source, img_target, times: List[float]):
        """Interpole entre source et target aux temps spécifiés - Code exact du notebook."""
        logger.debug(f"Interpolation 5D: {len(times)} temps")
        
        # 1. Préparation 5D (exactement comme dans le notebook)
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
        
        # 2. Sinkhorn & Plan pi (exactement comme dans le notebook)
        logger.debug("Calcul Sinkhorn...")
        F_pot, G_pot = self.loss_layer(w_a, X_a, w_b, X_b)
        F_pot, G_pot = F_pot.flatten(), G_pot.flatten()
        
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
        
        # Cache pour réutilisation (sauvegardé pour export)
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
        min_sigma_theoretical = avg_spacing / 2.0  # Critère de Nyquist
        
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
        """
        Calcule le champ de déplacement spatial T(x) - x depuis le plan de transport 5D.
        Retourne le champ de déplacement et des métriques de smoothness.
        """
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
        
        # Projection barycentrique spatiale: T(x_i) = sum_j y_j * pi_{ij} / sum_k pi_{ik}
        # On utilise uniquement la partie spatiale (2D) de X_b
        pos_b_spatial = X_b[:, :2]  # (M, 2)
        pos_a_spatial = X_a[:, :2]  # (N, 2)
        
        # Normalisation conditionnelle P(y|x)
        log_row_sum = torch.logsumexp(log_pi, dim=1, keepdim=True)
        log_cond_prob = log_pi - log_row_sum
        cond_prob = torch.exp(log_cond_prob)  # (N, M)
        
        # Barycentric projection: T = P * Y_spatial
        T_map_flat = torch.mm(cond_prob, pos_b_spatial)  # (N, 2)
        
        # Champ de déplacement: displacement = T(x) - x
        displacement_flat = T_map_flat - pos_a_spatial  # (N, 2)
        
        # Reshape en grille (H, W, 2)
        displacement_field = displacement_flat.view(Ha, Wa, 2)
        
        return displacement_field, Ha, Wa


# ============================================================================
# Chargement d'images
# ============================================================================

def load_image_from_cifar10(class_idx: int, sample_idx: int = 0, target_size: int = 64) -> torch.Tensor:
    """Charge une image depuis CIFAR-10."""
    try:
        transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor()
        ])
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        # Filtrer par classe
        class_indices = [i for i, (_, label) in enumerate(dataset) if label == class_idx]
        if sample_idx >= len(class_indices):
            sample_idx = 0
        idx = class_indices[sample_idx]
        img, _ = dataset[idx]
        return img
    except Exception as e:
        logger.warning(f"Erreur chargement CIFAR-10 classe {class_idx}: {e}")
        return None

def load_image_from_mnist(digit: int, sample_idx: int = 0, target_size: int = 64) -> torch.Tensor:
    """Charge une image depuis MNIST et la convertit en RGB."""
    try:
        transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Grayscale -> RGB
        ])
        dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        # Filtrer par chiffre
        digit_indices = [i for i, (_, label) in enumerate(dataset) if label == digit]
        if sample_idx >= len(digit_indices):
            sample_idx = 0
        idx = digit_indices[sample_idx]
        img, _ = dataset[idx]
        return img
    except Exception as e:
        logger.warning(f"Erreur chargement MNIST chiffre {digit}: {e}")
        return None

def load_classic_ot_image_pair(pair_name: str, target_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Charge des paires d'images classiques pour le transport optimal.
    
    Paires disponibles:
    - "cifar_car_airplane": Voiture CIFAR-10 -> Avion CIFAR-10
    - "cifar_bird_ship": Oiseau CIFAR-10 -> Bateau CIFAR-10
    - "cifar_truck_automobile": Camion CIFAR-10 -> Automobile CIFAR-10
    - "mnist_1_0": Chiffre 1 MNIST -> Chiffre 0 MNIST
    - "mnist_3_8": Chiffre 3 MNIST -> Chiffre 8 MNIST
    """
    logger.info(f"Chargement paire classique: {pair_name}")
    
    if pair_name == "cifar_car_airplane":
        # CIFAR-10: classe 1 (automobile) -> classe 0 (airplane)
        img1 = load_image_from_cifar10(class_idx=1, sample_idx=0, target_size=target_size)
        img2 = load_image_from_cifar10(class_idx=0, sample_idx=0, target_size=target_size)
    elif pair_name == "cifar_bird_ship":
        # CIFAR-10: classe 2 (bird) -> classe 8 (ship)
        img1 = load_image_from_cifar10(class_idx=2, sample_idx=0, target_size=target_size)
        img2 = load_image_from_cifar10(class_idx=8, sample_idx=0, target_size=target_size)
    elif pair_name == "cifar_truck_automobile":
        # CIFAR-10: classe 9 (truck) -> classe 1 (automobile)
        img1 = load_image_from_cifar10(class_idx=9, sample_idx=0, target_size=target_size)
        img2 = load_image_from_cifar10(class_idx=1, sample_idx=1, target_size=target_size)
    elif pair_name == "mnist_1_0":
        # MNIST: 1 -> 0
        img1 = load_image_from_mnist(digit=1, sample_idx=0, target_size=target_size)
        img2 = load_image_from_mnist(digit=0, sample_idx=0, target_size=target_size)
    elif pair_name == "mnist_3_8":
        # MNIST: 3 -> 8
        img1 = load_image_from_mnist(digit=3, sample_idx=0, target_size=target_size)
        img2 = load_image_from_mnist(digit=8, sample_idx=0, target_size=target_size)
    else:
        logger.warning(f"Paire inconnue: {pair_name}, utilisation d'images synthétiques")
        img1 = None
        img2 = None
    
    if img1 is None or img2 is None:
        # Fallback: images synthétiques
        logger.warning("Utilisation d'images synthétiques de fallback")
        img1 = torch.zeros(3, target_size, target_size)
        img1[0, 10:30, 10:30] = 1.0  # Carré rouge
        img2 = torch.zeros(3, target_size, target_size)
        img2[2, 30:50, 30:50] = 1.0  # Carré bleu décalé
    else:
        logger.info(f"Images chargées: {img1.shape} et {img2.shape}")
    
    return img1, img2

def load_image_pair(img1_name: str, img2_name: str, data_dir: str, target_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """Charge une paire d'images depuis fichiers ou datasets."""
    # Si c'est une paire classique (format "dataset_name")
    if isinstance(img1_name, str) and img1_name.startswith(("cifar_", "mnist_")):
        return load_classic_ot_image_pair(img1_name, target_size=target_size)
    
    # Sinon, charger depuis fichiers
    path1 = Path(data_dir) / img1_name
    path2 = Path(data_dir) / img2_name
    
    logger.debug(f"Chargement images: {path1} et {path2}")
    try:
        img1_pil = Image.open(path1).convert("RGB")
        img2_pil = Image.open(path2).convert("RGB")
        img1_pil = img1_pil.resize((target_size, target_size))
        img2_pil = img2_pil.resize((target_size, target_size))
        
        img1 = torch.from_numpy(np.array(img1_pil)).permute(2, 0, 1).float() / 255.0
        img2 = torch.from_numpy(np.array(img2_pil)).permute(2, 0, 1).float() / 255.0
        logger.debug(f"Images chargées: {img1.shape} et {img2.shape}")
        return img1, img2
    except Exception as e:
        logger.warning(f"Erreur chargement {img1_name}/{img2_name}: {e}. Utilisation d'images synthétiques.")
        # Image synthétique de fallback
        img1 = torch.zeros(3, target_size, target_size)
        img1[0, 10:30, 10:30] = 1.0  # Carré rouge
        img2 = torch.zeros(3, target_size, target_size)
        img2[2, 30:50, 30:50] = 1.0  # Carré bleu décalé
        return img1, img2


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
            # Pour CPU, on peut utiliser psutil si disponible
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

class ExperimentRunner:
    """Gère l'exécution des expériences."""
    
    def __init__(self, exp_config: ExperimentConfig):
        self.exp_config = exp_config
        self.output_dir = setup_output_dirs(exp_config.output_dir)
        # Initialiser le logging
        logger_instance, log_file = setup_logging(exp_config.output_dir)
        self.log_file = log_file
        self.metrics_computer = MetricsComputer()
        self.results = []
        self.experiment_id = 0
        logger.info(f"ExperimentRunner initialisé. Output dir: {self.output_dir}")
        logger.info(f"Fichier de log: {log_file}")
    
    def run_single_experiment(
        self,
        img_source: torch.Tensor,
        img_target: torch.Tensor,
        image_pair_name: str,
        resolution: int,
        lambda_val: float,
        blur: float,
        reach: Optional[float],
        times: List[float]
    ) -> Dict:
        """Exécute une seule expérience."""
        self.experiment_id += 1
        exp_id = self.experiment_id
        
        logger.info(f"Expérience {exp_id}: {image_pair_name}, res={resolution}, λ={lambda_val}, "
                   f"ε={blur}, ρ={reach}, {len(times)} temps")
        
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
        interpolator = OT5DInterpolator(ot_config)
        
        # Interpolation
        start_time = time.time()
        sinkhorn_time = 0.0
        interpolation_time = 0.0
        try:
            # Mesurer le temps Sinkhorn séparément
            sinkhorn_start = time.time()
            frames = interpolator.interpolate(img_source, img_target, times)
            sinkhorn_time = time.time() - sinkhorn_start
            compute_time = time.time() - start_time
            interpolation_time = compute_time - sinkhorn_time
            
            # Log mémoire après
            mem_after = log_memory_usage(self.exp_config.device, f"après exp {exp_id}")
            
            logger.info(f"Expérience {exp_id} terminée en {compute_time:.2f}s "
                       f"(Sinkhorn: {sinkhorn_time:.2f}s, Interpolation: {interpolation_time:.2f}s, "
                       f"{compute_time/len(times):.2f}s/frame)")
            
            # Log détaillé mémoire si GPU
            if self.exp_config.device == "cuda" and torch.cuda.is_available():
                if 'allocated_gb' in mem_after:
                    logger.info(f"  Mémoire GPU: {mem_after['allocated_gb']:.2f} GB allouée "
                               f"(max: {mem_after.get('max_allocated_gb', 0):.2f} GB)")
            
            # Sauvegarder le plan de transport si demandé
            if self.exp_config.save_transport_plans and hasattr(interpolator, 'pi') and interpolator.pi is not None:
                self._save_transport_plan(exp_id, interpolator, image_pair_name, resolution, lambda_val, blur, reach)
        except Exception as e:
            logger.error(f"Erreur interpolation exp {exp_id}: {e}", exc_info=True)
            return None
        
        # Calcul des métriques pour chaque frame
        experiment_results = []
        for i, (frame, t) in enumerate(zip(frames, times)):
            # Redimensionner les images sources pour correspondre à la frame interpolée
            frame_h, frame_w = frame.shape[1], frame.shape[2]
            img_source_resized = F.interpolate(
                img_source.unsqueeze(0), 
                size=(frame_h, frame_w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            img_target_resized = F.interpolate(
                img_target.unsqueeze(0), 
                size=(frame_h, frame_w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
            metrics = self.metrics_computer.compute_all(
                frame, img_source_resized, img_target_resized, t, compute_tearing=False
            )
            
            result = {
                'experiment_id': exp_id,
                'image_pair': image_pair_name,
                'resolution': resolution,
                'lambda': lambda_val,
                'blur': blur,
                'reach': reach if reach is not None else 'balanced',
                'splatting': True,  # Toujours activé (code du notebook)
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
                # Métriques de smoothness (None pour les expériences normales)
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
            
            # Sauvegarde image - TOUTES les images maintenant !
            if self.exp_config.save_images:
                # Format: exp{id}_t{t:.3f}.png pour tous les temps
                img_path = self.output_dir / "images" / f"exp{exp_id}_t{t:.3f}.png"
                frame_np = frame.permute(1, 2, 0).clamp(0, 1).numpy()
                plt.imsave(str(img_path), frame_np, dpi=150)
                logger.debug(f"Image sauvegardée: {img_path}")
        
        return experiment_results
    
    def run_experiment_1_lambda_ablation(self):
        """Expérience 1: Ablation Lambda (λ)."""
        logger.info("=== Expérience 1: Ablation Lambda ===")
        
        resolution = 32
        blur = 0.03
        reach = 0.1
        times = [0.5]  # Focus sur t=0.5
        
        for pair_name in self.exp_config.image_pairs[:1]:  # Première paire seulement
            if isinstance(pair_name, tuple):
                # Ancien format (fichiers)
                img1_name, img2_name = pair_name
                img_source, img_target = load_image_pair(img1_name, img2_name, self.exp_config.data_dir)
                image_pair_name = f"{img1_name.split('.')[0]}_{img2_name.split('.')[0]}"
            else:
                # Nouveau format (datasets)
                img_source, img_target = load_classic_ot_image_pair(pair_name, target_size=64)
                image_pair_name = pair_name
            
            for lambda_val in self.exp_config.lambdas:
                results = self.run_single_experiment(
                    img_source, img_target, image_pair_name,
                    resolution, lambda_val, blur, reach, times
                )
                if results:
                    self.results.extend(results)
        
        # Nettoyage GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def run_experiment_2_2d_vs_5d(self):
        """Expérience 2: Comparaison 2D vs 5D."""
        logger.info("=== Expérience 2: Comparaison 2D vs 5D ===")
        
        resolution = 32
        blur = 0.03
        reach = 0.1
        
        for pair_name in self.exp_config.image_pairs[:1]:
            if isinstance(pair_name, tuple):
                img1_name, img2_name = pair_name
                img_source, img_target = load_image_pair(img1_name, img2_name, self.exp_config.data_dir)
                image_pair_name = f"{img1_name.split('.')[0]}_{img2_name.split('.')[0]}"
            else:
                img_source, img_target = load_classic_ot_image_pair(pair_name, target_size=64)
                image_pair_name = pair_name
            
            # 2D marginal (λ=0.0)
            results = self.run_single_experiment(
                img_source, img_target, image_pair_name,
                resolution, 0.0, blur, reach, self.exp_config.times
            )
            if results:
                self.results.extend(results)
            
            # 5D optimal (λ=1.0)
            results = self.run_single_experiment(
                img_source, img_target, image_pair_name,
                resolution, 1.0, blur, reach, self.exp_config.times
            )
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
        
        for pair_name in self.exp_config.image_pairs[:1]:
            if isinstance(pair_name, tuple):
                img1_name, img2_name = pair_name
                img_source, img_target = load_image_pair(img1_name, img2_name, self.exp_config.data_dir)
                image_pair_name = f"{img1_name.split('.')[0]}_{img2_name.split('.')[0]}"
            else:
                img_source, img_target = load_classic_ot_image_pair(pair_name, target_size=64)
                image_pair_name = pair_name
            
            for resolution in [32, 48, 64]:
                results = self.run_single_experiment(
                    img_source, img_target, image_pair_name,
                    resolution, lambda_val, blur, reach, times
                )
                if results:
                    self.results.extend(results)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def run_experiment_4_parameter_sensitivity(self):
        """Expérience 4: Sensibilité aux paramètres (ε, ρ)."""
        logger.info("=== Expérience 4: Sensibilité Paramètres ===")
        
        resolution = 32
        lambda_val = 1.0
        times = [0.5]
        
        for pair_name in self.exp_config.image_pairs[:1]:
            if isinstance(pair_name, tuple):
                img1_name, img2_name = pair_name
                img_source, img_target = load_image_pair(img1_name, img2_name, self.exp_config.data_dir)
                image_pair_name = f"{img1_name.split('.')[0]}_{img2_name.split('.')[0]}"
            else:
                img_source, img_target = load_classic_ot_image_pair(pair_name, target_size=64)
                image_pair_name = pair_name
            
            for blur in [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]:
                for reach in [None, 0.01, 0.05, 0.1, 0.3, 0.5]:
                    results = self.run_single_experiment(
                        img_source, img_target, image_pair_name,
                        resolution, lambda_val, blur, reach, times
                    )
                    if results:
                        self.results.extend(results)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def run_experiment_6_displacement_robustness(self):
        """Expérience 6: Robustesse du champ de déplacement selon les régimes (ε, ρ)."""
        logger.info("=== Expérience 6: Robustesse Champ de Déplacement ===")
        
        resolution = 32
        lambda_val = 1.0
        times = [0.5]  # Focus sur t=0.5
        
        # Trois régimes à tester:
        # 1. Unbalanced OT (ρ fini, ε petit) - overfitting, non-lisse
        # 2. Entropy-regularized OT (ε grand, ρ = None) - lisse mais sensible aux artefacts
        # 3. Unbalanced, entropy-regularized OT (valeurs intermédiaires) - robuste et lisse
        
        regimes = [
            {"name": "Unbalanced_OT", "blur": 0.01, "reach": 0.1, "description": "ρ fini, ε petit"},
            {"name": "Entropy_Regularized", "blur": 0.1, "reach": None, "description": "ε grand, ρ = ∞"},
            {"name": "Unbalanced_Entropy", "blur": 0.03, "reach": 0.3, "description": "ε et ρ intermédiaires"},
        ]
        
        for pair_name in self.exp_config.image_pairs[:1]:
            if isinstance(pair_name, tuple):
                img1_name, img2_name = pair_name
                img_source, img_target = load_image_pair(img1_name, img2_name, self.exp_config.data_dir)
                image_pair_name = f"{img1_name.split('.')[0]}_{img2_name.split('.')[0]}"
            else:
                img_source, img_target = load_classic_ot_image_pair(pair_name, target_size=64)
                image_pair_name = pair_name
            
            for regime in regimes:
                logger.info(f"Régime: {regime['name']} ({regime['description']})")
                blur = regime['blur']
                reach = regime['reach']
                
                # Configuration OT
                ot_config = OTConfig(
                    resolution=(resolution, resolution),
                    blur=blur,
                    reach=reach,
                    lambda_color=lambda_val,
                    device=self.exp_config.device
                )
                
                # Interpolateur
                interpolator = OT5DInterpolator(ot_config)
                
                try:
                    # Calculer le champ de déplacement
                    displacement_field, H, W = interpolator.get_displacement_field(img_source, img_target)
                    
                    # Métriques de smoothness
                    smoothness_metrics = compute_displacement_smoothness(displacement_field)
                    
                    # Sauvegarder le champ de déplacement
                    disp_dir = self.output_dir / "displacement_fields"
                    disp_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Visualisation du champ de déplacement
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # 1. Magnitude du déplacement
                    magnitude = torch.sqrt(displacement_field[..., 0]**2 + displacement_field[..., 1]**2)
                    im1 = axes[0].imshow(magnitude.cpu().numpy(), cmap='hot')
                    axes[0].set_title(f"Magnitude du Déplacement\n{regime['name']}")
                    axes[0].axis('off')
                    plt.colorbar(im1, ax=axes[0])
                    
                    # 2. Champ vectoriel (quiver)
                    step = 4
                    y_indices = torch.arange(step//2, H, step)
                    x_indices = torch.arange(step//2, W, step)
                    yy, xx = torch.meshgrid(y_indices, x_indices, indexing="ij")
                    
                    disp_sampled = displacement_field[yy, xx].cpu()
                    axes[1].imshow(img_source.permute(1, 2, 0).cpu() * 0.6)
                    axes[1].quiver(xx.numpy(), yy.numpy(), 
                                  disp_sampled[..., 0].numpy() * (W-1),
                                  disp_sampled[..., 1].numpy() * (H-1),
                                  magnitude[yy, xx].cpu().numpy(),
                                  cmap='coolwarm', angles='xy', scale_units='xy', scale=1, alpha=0.8)
                    axes[1].set_title(f"Champ Vectoriel\n{regime['name']}")
                    axes[1].axis('off')
                    
                    # 3. Divergence (expansion/contraction)
                    disp_np = displacement_field.cpu().numpy()
                    grad_x = np.gradient(disp_np[..., 0], axis=1)
                    grad_y = np.gradient(disp_np[..., 1], axis=0)
                    divergence = grad_x + grad_y
                    im3 = axes[2].imshow(divergence, cmap='RdBu', vmin=-np.abs(divergence).max(), 
                                        vmax=np.abs(divergence).max())
                    axes[2].set_title(f"Divergence (Expansion/Contraction)\n{regime['name']}")
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
                        'image_pair': image_pair_name,
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
                        'psnr': None,  # Pas calculé pour cette expérience
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
                    logger.info(f"Régime {regime['name']}: smoothness_score={smoothness_metrics['smoothness_score']:.4f}, "
                               f"mean_laplacian={smoothness_metrics['mean_laplacian']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Erreur calcul champ de déplacement pour {regime['name']}: {e}", exc_info=True)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def run_experiment_5_scalability(self):
        """Expérience 5: Scalabilité Résolution (test progressif)."""
        logger.info("=== Expérience 5: Scalabilité ===")
        
        lambda_val = 1.0
        blur = 0.03
        reach = 0.1
        times = [0.5]
        
        # Résolutions à tester progressivement - Augmentées
        if self.exp_config.progressive_resolution:
            resolutions_to_test = [32, 48, 64]  # Commencer par les moyennes
            logger.info(f"Test progressif: résolutions {resolutions_to_test}")
        else:
            resolutions_to_test = self.exp_config.resolutions
        
        for pair_name in self.exp_config.image_pairs[:1]:
            if isinstance(pair_name, tuple):
                img1_name, img2_name = pair_name
                img_source, img_target = load_image_pair(img1_name, img2_name, self.exp_config.data_dir)
                image_pair_name = f"{img1_name.split('.')[0]}_{img2_name.split('.')[0]}"
            else:
                img_source, img_target = load_classic_ot_image_pair(pair_name, target_size=64)
                image_pair_name = pair_name
            
            for resolution in resolutions_to_test:
                logger.info(f"Test résolution {resolution}×{resolution}...")
                try:
                    results = self.run_single_experiment(
                        img_source, img_target, image_pair_name,
                        resolution, lambda_val, blur, reach, times
                    )
                    if results:
                        self.results.extend(results)
                        logger.info(f"✓ Résolution {resolution}×{resolution} réussie")
                        
                        # Si on est en mode progressif et que ça marche, continuer avec les plus grandes
                        if self.exp_config.progressive_resolution and resolution == 64:
                            # Tester 96 si 64 a réussi
                            logger.info("Test résolution 96×96...")
                            try:
                                results_96 = self.run_single_experiment(
                                    img_source, img_target, image_pair_name,
                                    96, lambda_val, blur, reach, times
                                )
                                if results_96:
                                    self.results.extend(results_96)
                                    logger.info("✓ Résolution 96×96 réussie")
                                    
                                    # Tester 128 si 96 a réussi
                                    logger.info("Test résolution 128×128...")
                                    try:
                                        results_128 = self.run_single_experiment(
                                            img_source, img_target, image_pair_name,
                                            128, lambda_val, blur, reach, times
                                        )
                                        if results_128:
                                            self.results.extend(results_128)
                                            logger.info("✓ Résolution 128×128 réussie")
                                    except Exception as e:
                                        logger.warning(f"Résolution 128×128 échouée: {e}")
                            except Exception as e:
                                logger.warning(f"Résolution 96×96 échouée: {e}")
                except Exception as e:
                    logger.error(f"Résolution {resolution}×{resolution} échouée: {e}")
                    if self.exp_config.progressive_resolution:
                        logger.info("Arrêt du test progressif à cause de l'erreur")
                        break
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _save_transport_plan(self, exp_id: int, interpolator, image_pair_name: str, 
                            resolution: int, lambda_val: float, blur: float, reach: Optional[float]):
        """Sauvegarde le plan de transport à échelle réduite."""
        try:
            if interpolator.pi is None:
                return
            
            # Sauvegarder uniquement la partie sparse (masquée)
            if interpolator.pi_sparse_mask is not None:
                I_idx, J_idx = interpolator.pi_sparse_mask
                pi_sparse = interpolator.pi[I_idx, J_idx]
            else:
                # Si pas de masque, sauvegarder tout mais à échelle réduite
                pi_sparse = interpolator.pi
            
            # Appliquer l'échelle pour économiser la mémoire
            pi_scaled = (pi_sparse * self.exp_config.transport_plan_scale).cpu()
            
            # Créer le nom de fichier
            reach_str = f"{reach:.2f}" if reach is not None else "balanced"
            plan_dir = self.output_dir / "transport_plans"
            plan_dir.mkdir(parents=True, exist_ok=True)
            plan_path = plan_dir / f"plan_exp{exp_id}_res{resolution}_lam{lambda_val:.1f}_eps{blur:.3f}_rho{reach_str}.pt"
            
            # Sauvegarder avec métadonnées
            save_dict = {
                'pi': pi_scaled,
                'scale': self.exp_config.transport_plan_scale,  # Pour restaurer plus tard
                'metadata': {
                    'experiment_id': exp_id,
                    'image_pair': image_pair_name,
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
            logger.debug(f"Plan de transport sauvegardé: {plan_path} (shape: {pi_scaled.shape})")
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
            # Remplacer None par des valeurs par défaut pour CSV
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
        logger.info(f"Résumé: {len(set(r['experiment_id'] for r in self.results))} expériences, "
                   f"{len(set(r['image_pair'] for r in self.results))} paires d'images")
        
        # Sauvegarder aussi un résumé des temps de calcul pour le rapport
        self._save_timing_summary()
    
    def _save_timing_summary(self):
        """Sauvegarde un résumé des temps de calcul pour le rapport LaTeX."""
        import pandas as pd
        df = pd.DataFrame(self.results)
        
        if df.empty:
            return
        
        # Résumé par résolution
        timing_summary = []
        for res in sorted(df['resolution'].unique()):
            df_res = df[df['resolution'] == res]
            timing_summary.append({
                'resolution': res,
                'mean_total_time': df_res['compute_time_total'].mean(),
                'std_total_time': df_res['compute_time_total'].std(),
                'mean_sinkhorn_time': df_res['sinkhorn_time'].mean(),
                'mean_interpolation_time': df_res['interpolation_time'].mean(),
                'n_experiments': len(df_res['experiment_id'].unique())
            })
        
        # Sauvegarder en CSV et LaTeX
        timing_df = pd.DataFrame(timing_summary)
        timing_csv = self.output_dir / "metrics" / "timing_summary.csv"
        timing_df.to_csv(timing_csv, index=False)
        
        # Format LaTeX pour le rapport
        timing_tex = self.output_dir / "metrics" / "timing_summary.tex"
        with open(timing_tex, 'w') as f:
            f.write("% Résumé des temps de calcul pour le rapport\n")
            f.write("% Généré automatiquement par experiments_5d_massive.py\n\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Temps de calcul moyen par résolution}\n")
            f.write("\\label{tab:timing}\n")
            f.write("\\begin{tabular}{lccc}\n")
            f.write("\\toprule\n")
            f.write("Résolution & Temps Total (s) & Sinkhorn (s) & Interpolation (s) \\\\\n")
            f.write("\\midrule\n")
            for _, row in timing_df.iterrows():
                f.write(f"{int(row['resolution'])}×{int(row['resolution'])} & "
                       f"{row['mean_total_time']:.2f} & "
                       f"{row['mean_sinkhorn_time']:.2f} & "
                       f"{row['mean_interpolation_time']:.2f} \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        logger.info(f"Résumé des temps sauvegardé: {timing_csv} et {timing_tex}")
    
    def generate_visualizations(self):
        """Génère toutes les visualisations pour le rapport."""
        if not self.results:
            logger.warning("Aucun résultat pour générer les visualisations")
            return
        
        import pandas as pd
        df = pd.DataFrame(self.results)
        
        # Figure 1: Comparaison 2D vs 5D (séquence temporelle)
        self._plot_2d_vs_5d_comparison(df)
        
        # Figure 2: Ablation Lambda
        self._plot_lambda_ablation(df)
        
        # Figure 3: Impact Splatting
        self._plot_splatting_impact(df)
        
        # Figure 4: Courbes métriques
        self._plot_metric_curves(df)
        
        # Figure 5: Heatmaps paramètres
        self._plot_parameter_heatmaps(df)
        
        # Figure 6: Comparaison robustesse des régimes
        self._plot_displacement_robustness_comparison(df)
        
        logger.info("Visualisations générées")
    
    def _plot_2d_vs_5d_comparison(self, df):
        """Figure principale: Comparaison 2D vs 5D."""
        # Filtrer les données pertinentes
        subset = df[(df['resolution'] == 32) & (df['blur'] == 0.03) & 
                    (df['reach'] == 0.1)]
        
        times = sorted(subset['t'].unique())
        fig, axes = plt.subplots(2, len(times), figsize=(4*len(times), 8))
        
        for i, t in enumerate(times):
            # 2D (lambda=0.0)
            row_2d = subset[(subset['lambda'] == 0.0) & (subset['t'] == t)]
            if not row_2d.empty:
                img_path = self.output_dir / "images" / f"exp{int(row_2d.iloc[0]['experiment_id'])}_t{t:.1f}.png"
                if img_path.exists():
                    img = plt.imread(str(img_path))
                    axes[0, i].imshow(img)
            axes[0, i].set_title(f"2D (λ=0.0)\nt={t:.1f}")
            axes[0, i].axis('off')
            
            # 5D (lambda=1.0)
            row_5d = subset[(subset['lambda'] == 1.0) & (subset['t'] == t)]
            if not row_5d.empty:
                img_path = self.output_dir / "images" / f"exp{int(row_5d.iloc[0]['experiment_id'])}_t{t:.1f}.png"
                if img_path.exists():
                    img = plt.imread(str(img_path))
                    axes[1, i].imshow(img)
            axes[1, i].set_title(f"5D (λ=1.0)\nt={t:.1f}")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "images" / "comparison_2d_vs_5d.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_lambda_ablation(self, df):
        """Ablation Lambda à t=0.5."""
        subset = df[(df['t'] == 0.5) & (df['resolution'] == 32) & 
                    (df['blur'] == 0.03) & (df['reach'] == 0.1)]
        
        lambdas = sorted(subset['lambda'].unique())
        fig, axes = plt.subplots(1, len(lambdas), figsize=(4*len(lambdas), 4))
        
        for i, lam in enumerate(lambdas):
            row = subset[subset['lambda'] == lam]
            if not row.empty:
                img_path = self.output_dir / "images" / f"exp{int(row.iloc[0]['experiment_id'])}_t0.5.png"
                if img_path.exists():
                    img = plt.imread(str(img_path))
                    axes[i].imshow(img)
            axes[i].set_title(f"λ={lam:.1f}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "images" / "ablation_lambda.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_splatting_impact(self, df):
        """Impact de la résolution sur le splatting adaptatif."""
        subset = df[(df['t'] == 0.5) & (df['lambda'] == 1.0) & 
                    (df['blur'] == 0.03) & (df['reach'] == 0.1)]
        
        resolutions = sorted(subset['resolution'].unique())
        if len(resolutions) == 0:
            logger.warning("Aucune donnée pour la visualisation du splatting")
            return
        
        fig, axes = plt.subplots(1, len(resolutions), figsize=(4*len(resolutions), 4))
        if len(resolutions) == 1:
            axes = [axes]
        
        for i, res in enumerate(resolutions):
            row = subset[subset['resolution'] == res]
            if not row.empty:
                exp_id = int(row.iloc[0]['experiment_id'])
                # Chercher l'image avec le bon format de temps
                img_path = self.output_dir / "images" / f"exp{exp_id}_t0.500.png"
                if not img_path.exists():
                    # Essayer avec format .1f
                    img_path = self.output_dir / "images" / f"exp{exp_id}_t0.5.png"
                if img_path.exists():
                    img = plt.imread(str(img_path))
                    axes[i].imshow(img)
            axes[i].set_title(f"Résolution {res}×{res}\nSplatting Adaptatif")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "images" / "splatting_impact.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_metric_curves(self, df):
        """Courbes de métriques en fonction du temps."""
        subset = df[(df['resolution'] == 32) & (df['blur'] == 0.03) & 
                    (df['reach'] == 0.1)]
        
        times = sorted(subset['t'].unique())
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # PSNR
        for lam in [0.0, 1.0]:
            data = subset[subset['lambda'] == lam].sort_values('t')
            axes[0, 0].plot(data['t'], data['psnr'], marker='o', label=f"λ={lam:.1f}")
        axes[0, 0].set_xlabel('Temps t')
        axes[0, 0].set_ylabel('PSNR (dB)')
        axes[0, 0].set_title('PSNR vs Temps')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # ΔE
        for lam in [0.0, 1.0]:
            data = subset[subset['lambda'] == lam].sort_values('t')
            axes[0, 1].plot(data['t'], data['delta_e'], marker='o', label=f"λ={lam:.1f}")
        axes[0, 1].set_xlabel('Temps t')
        axes[0, 1].set_ylabel('ΔE (CIE76)')
        axes[0, 1].set_title('ΔE vs Temps')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Coverage
        for lam in [0.0, 1.0]:
            data = subset[subset['lambda'] == lam].sort_values('t')
            axes[1, 0].plot(data['t'], data['coverage'], marker='o', label=f"λ={lam:.1f}")
        axes[1, 0].set_xlabel('Temps t')
        axes[1, 0].set_ylabel('Coverage')
        axes[1, 0].set_title('Coverage vs Temps')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Sharpness
        for lam in [0.0, 1.0]:
            data = subset[subset['lambda'] == lam].sort_values('t')
            axes[1, 1].plot(data['t'], data['sharpness'], marker='o', label=f"λ={lam:.1f}")
        axes[1, 1].set_xlabel('Temps t')
        axes[1, 1].set_ylabel('Sharpness')
        axes[1, 1].set_title('Sharpness vs Temps')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "images" / "metric_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_heatmaps(self, df):
        """Heatmaps de sensibilité aux paramètres."""
        subset = df[(df['t'] == 0.5) & (df['resolution'] == 32) & 
                    (df['lambda'] == 1.0)]
        
        blurs = sorted(subset['blur'].unique())
        reaches = sorted([r for r in subset['reach'].unique() if r != 'balanced'], key=lambda x: float(x) if isinstance(x, str) else x)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Heatmap PSNR
        psnr_matrix = np.zeros((len(reaches), len(blurs)))
        for i, r in enumerate(reaches):
            for j, b in enumerate(blurs):
                row = subset[(subset['blur'] == b) & (subset['reach'] == r)]
                if not row.empty:
                    psnr_matrix[i, j] = row.iloc[0]['psnr']
        
        im1 = axes[0].imshow(psnr_matrix, aspect='auto', cmap='viridis')
        axes[0].set_xticks(range(len(blurs)))
        axes[0].set_xticklabels([f"{b:.2f}" for b in blurs])
        axes[0].set_yticks(range(len(reaches)))
        axes[0].set_yticklabels([f"{r:.1f}" for r in reaches])
        axes[0].set_xlabel('Blur (ε)')
        axes[0].set_ylabel('Reach (ρ)')
        axes[0].set_title('PSNR (dB)')
        plt.colorbar(im1, ax=axes[0])
        
        # Heatmap ΔE
        de_matrix = np.zeros((len(reaches), len(blurs)))
        for i, r in enumerate(reaches):
            for j, b in enumerate(blurs):
                row = subset[(subset['blur'] == b) & (subset['reach'] == r)]
                if not row.empty:
                    de_matrix[i, j] = row.iloc[0]['delta_e']
        
        im2 = axes[1].imshow(de_matrix, aspect='auto', cmap='plasma_r')
        axes[1].set_xticks(range(len(blurs)))
        axes[1].set_xticklabels([f"{b:.2f}" for b in blurs])
        axes[1].set_yticks(range(len(reaches)))
        axes[1].set_yticklabels([f"{r:.1f}" for r in reaches])
        axes[1].set_xlabel('Blur (ε)')
        axes[1].set_ylabel('Reach (ρ)')
        axes[1].set_title('ΔE (CIE76)')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "images" / "parameter_heatmaps.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_displacement_robustness_comparison(self, df):
        """Comparaison de la robustesse des différents régimes."""
        # Filtrer les résultats de l'expérience 6
        subset = df[df['regime'].notna()]
        
        if subset.empty:
            logger.warning("Aucune donnée pour la comparaison de robustesse")
            return
        
        regimes = subset['regime'].unique()
        if len(regimes) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Smoothness Score
        smoothness_data = [subset[subset['regime'] == r]['smoothness_score'].values[0] 
                          if len(subset[subset['regime'] == r]) > 0 else 0 
                          for r in regimes]
        axes[0, 0].bar(regimes, smoothness_data, color=['red', 'blue', 'green'])
        axes[0, 0].set_ylabel('Smoothness Score')
        axes[0, 0].set_title('Smoothness du Champ de Déplacement')
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Mean Laplacian (plus petit = plus lisse)
        laplacian_data = [subset[subset['regime'] == r]['mean_laplacian'].values[0] 
                         if len(subset[subset['regime'] == r]) > 0 else 0 
                         for r in regimes]
        axes[0, 1].bar(regimes, laplacian_data, color=['red', 'blue', 'green'])
        axes[0, 1].set_ylabel('Mean Laplacian')
        axes[0, 1].set_title('Rugosité du Champ (Laplacien)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Mean Displacement
        disp_data = [subset[subset['regime'] == r]['mean_displacement'].values[0] 
                    if len(subset[subset['regime'] == r]) > 0 else 0 
                    for r in regimes]
        axes[1, 0].bar(regimes, disp_data, color=['red', 'blue', 'green'])
        axes[1, 0].set_ylabel('Mean Displacement')
        axes[1, 0].set_title('Amplitude Moyenne du Déplacement')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Mean Divergence
        div_data = [subset[subset['regime'] == r]['mean_divergence'].values[0] 
                   if len(subset[subset['regime'] == r]) > 0 else 0 
                   for r in regimes]
        axes[1, 1].bar(regimes, div_data, color=['red', 'blue', 'green'])
        axes[1, 1].set_ylabel('Mean |Divergence|')
        axes[1, 1].set_title('Expansion/Contraction Moyenne')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "images" / "displacement_robustness_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def run_all_experiments(self):
        """Exécute toutes les expériences."""
        logger.info("=" * 80)
        logger.info("DÉMARRAGE DES EXPÉRIENCES MASSIVES 5D")
        logger.info("=" * 80)
        logger.info(f"Device: {self.exp_config.device}")
        logger.info(f"Paires d'images: {len(self.exp_config.image_pairs)}")
        logger.info(f"Résolutions: {self.exp_config.resolutions}")
        logger.info(f"Lambdas: {self.exp_config.lambdas}")
        logger.info(f"Blurs: {self.exp_config.blurs}")
        logger.info(f"Reaches: {self.exp_config.reaches}")
        
        # Log mémoire initiale
        log_memory_usage(self.exp_config.device, "début")
        logger.info("=" * 80)
        
        total_start = time.time()
        
        logger.info("\n>>> Expérience 1: Ablation Lambda")
        self.run_experiment_1_lambda_ablation()
        
        logger.info("\n>>> Expérience 2: Comparaison 2D vs 5D")
        self.run_experiment_2_2d_vs_5d()
        
        logger.info("\n>>> Expérience 3: Impact Splatting")
        self.run_experiment_3_splatting_impact()
        
        logger.info("\n>>> Expérience 4: Sensibilité Paramètres")
        self.run_experiment_4_parameter_sensitivity()
        
        logger.info("\n>>> Expérience 5: Scalabilité")
        self.run_experiment_5_scalability()
        
        logger.info("\n>>> Expérience 6: Robustesse Champ de Déplacement")
        self.run_experiment_6_displacement_robustness()
        
        total_time = time.time() - total_start
        
        logger.info("\n" + "=" * 80)
        logger.info("SAUVEGARDE DES RÉSULTATS")
        logger.info("=" * 80)
        self.save_results()
        
        logger.info("\n" + "=" * 80)
        logger.info("GÉNÉRATION DES VISUALISATIONS")
        logger.info("=" * 80)
        self.generate_visualizations()
        
        logger.info("\n" + "=" * 80)
        logger.info("TOUTES LES EXPÉRIENCES TERMINÉES")
        logger.info("=" * 80)
        logger.info(f"Total: {len(self.results)} résultats")
        logger.info(f"Temps total: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        
        # Log mémoire finale
        mem_final = log_memory_usage(self.exp_config.device, "fin")
        if self.exp_config.device == "cuda" and torch.cuda.is_available():
            if 'max_allocated_gb' in mem_final:
                logger.info(f"Pic mémoire GPU: {mem_final['max_allocated_gb']:.2f} GB")
        
        logger.info(f"Fichier de log: {self.log_file}")
        logger.info("=" * 80)


# ============================================================================
# Tests de validation
# ============================================================================

def run_validation_tests(exp_config: ExperimentConfig) -> bool:
    """
    Exécute des tests de validation pour vérifier que toutes les fonctions fonctionnent.
    Retourne True si tous les tests passent, False sinon.
    """
    logger.info("=" * 80)
    logger.info("TESTS DE VALIDATION")
    logger.info("=" * 80)
    
    all_passed = True
    
    # Test 1: Chargement d'images
    logger.info("Test 1: Chargement d'images...")
    try:
        if exp_config.image_pairs:
            pair_name = exp_config.image_pairs[0]
            if isinstance(pair_name, tuple):
                img1_name, img2_name = pair_name
                img1, img2 = load_image_pair(img1_name, img2_name, exp_config.data_dir, target_size=32)
            else:
                img1, img2 = load_classic_ot_image_pair(pair_name, target_size=32)
            assert img1.shape[0] == 3, "Image doit avoir 3 canaux"
            assert img2.shape[0] == 3, "Image doit avoir 3 canaux"
            logger.info("✓ Test 1 réussi: Images chargées correctement")
        else:
            logger.warning("⚠ Test 1 ignoré: Aucune paire d'images configurée")
    except Exception as e:
        logger.error(f"✗ Test 1 échoué: {e}", exc_info=True)
        all_passed = False
    
    # Test 2: Création nuage 5D
    logger.info("Test 2: Création nuage 5D...")
    try:
        if exp_config.image_pairs:
            pair_name = exp_config.image_pairs[0]
            if isinstance(pair_name, tuple):
                img1_name, img2_name = pair_name
                img1, img2 = load_image_pair(img1_name, img2_name, exp_config.data_dir, target_size=32)
            else:
                img1, img2 = load_classic_ot_image_pair(pair_name, target_size=32)
            cloud, weights, colors, H, W = get_5d_cloud(img1.to(exp_config.device), 16, 1.0)
            assert cloud.shape[1] == 5, "Nuage doit être 5D"
            assert weights.shape[0] == cloud.shape[0], "Poids doivent correspondre au nuage"
            logger.info(f"✓ Test 2 réussi: Nuage 5D créé ({cloud.shape[0]} points)")
        else:
            logger.warning("⚠ Test 2 ignoré: Aucune paire d'images configurée")
    except Exception as e:
        logger.error(f"✗ Test 2 échoué: {e}", exc_info=True)
        all_passed = False
    
    # Test 3: Configuration OT et Interpolateur
    logger.info("Test 3: Configuration OT et Interpolateur...")
    try:
        ot_config = OTConfig(
            resolution=(16, 16),
            blur=0.03,
            reach=0.1,
            lambda_color=1.0,
            device=exp_config.device
        )
        interpolator = OT5DInterpolator(ot_config)
        logger.info("✓ Test 3 réussi: Interpolateur créé")
    except Exception as e:
        logger.error(f"✗ Test 3 échoué: {e}", exc_info=True)
        all_passed = False
    
    # Test 4: Interpolation basique
    logger.info("Test 4: Interpolation basique...")
    try:
        if exp_config.image_pairs:
            pair_name = exp_config.image_pairs[0]
            if isinstance(pair_name, tuple):
                img1_name, img2_name = pair_name
                img1, img2 = load_image_pair(img1_name, img2_name, exp_config.data_dir, target_size=32)
            else:
                img1, img2 = load_classic_ot_image_pair(pair_name, target_size=32)
            ot_config = OTConfig(
                resolution=(16, 16),
                blur=0.03,
                reach=0.1,
                lambda_color=1.0,
                device=exp_config.device
            )
            interpolator = OT5DInterpolator(ot_config)
            frames = interpolator.interpolate(img1, img2, [0.0, 0.5, 1.0])
            assert len(frames) == 3, "Doit générer 3 frames"
            assert frames[0].shape[0] == 3, "Frame doit avoir 3 canaux"
            logger.info(f"✓ Test 4 réussi: Interpolation fonctionne ({len(frames)} frames générées)")
        else:
            logger.warning("⚠ Test 4 ignoré: Aucune paire d'images configurée")
    except Exception as e:
        logger.error(f"✗ Test 4 échoué: {e}", exc_info=True)
        all_passed = False
    
    # Test 5: Calcul des métriques
    logger.info("Test 5: Calcul des métriques...")
    try:
        if exp_config.image_pairs:
            pair_name = exp_config.image_pairs[0]
            if isinstance(pair_name, tuple):
                img1_name, img2_name = pair_name
                img1, img2 = load_image_pair(img1_name, img2_name, exp_config.data_dir, target_size=32)
            else:
                img1, img2 = load_classic_ot_image_pair(pair_name, target_size=32)
            ot_config = OTConfig(
                resolution=(16, 16),
                blur=0.03,
                reach=0.1,
                lambda_color=1.0,
                device=exp_config.device
            )
            interpolator = OT5DInterpolator(ot_config)
            frames = interpolator.interpolate(img1, img2, [0.5])
            
            # Redimensionner les images sources pour correspondre à la frame interpolée
            frame = frames[0]
            img1_resized = F.interpolate(img1.unsqueeze(0), size=(frame.shape[1], frame.shape[2]), mode='bilinear').squeeze(0)
            img2_resized = F.interpolate(img2.unsqueeze(0), size=(frame.shape[1], frame.shape[2]), mode='bilinear').squeeze(0)
            
            metrics_computer = MetricsComputer()
            metrics = metrics_computer.compute_all(frame, img1_resized, img2_resized, 0.5, compute_tearing=False)
            
            assert 'psnr' in metrics, "PSNR doit être calculé"
            assert 'delta_e' in metrics, "Delta E doit être calculé"
            assert 'coverage' in metrics, "Coverage doit être calculé"
            assert 'mass_error' in metrics, "Mass error doit être calculé"
            assert 'sharpness' in metrics, "Sharpness doit être calculé"
            logger.info(f"✓ Test 5 réussi: Métriques calculées (PSNR={metrics['psnr']:.2f}dB, ΔE={metrics['delta_e']:.2f})")
        else:
            logger.warning("⚠ Test 5 ignoré: Aucune paire d'images configurée")
    except Exception as e:
        logger.error(f"✗ Test 5 échoué: {e}", exc_info=True)
        all_passed = False
    
    # Test 6: Sauvegarde de fichiers
    logger.info("Test 6: Sauvegarde de fichiers...")
    try:
        output_dir = setup_output_dirs(exp_config.output_dir)
        test_file = output_dir / "metrics" / "test.csv"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        with open(test_file, 'w') as f:
            f.write("test\n")
        assert test_file.exists(), "Fichier de test doit être créé"
        test_file.unlink()  # Supprimer le fichier de test
        logger.info("✓ Test 6 réussi: Sauvegarde de fichiers fonctionne")
    except Exception as e:
        logger.error(f"✗ Test 6 échoué: {e}", exc_info=True)
        all_passed = False
    
    # Test 7: GPU disponible (si CUDA)
    logger.info("Test 7: Vérification GPU...")
    try:
        if exp_config.device == "cuda":
            if torch.cuda.is_available():
                logger.info(f"✓ Test 7 réussi: GPU disponible ({torch.cuda.get_device_name(0)})")
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                allocated = torch.cuda.memory_allocated(0) / 1e9
                reserved = torch.cuda.memory_reserved(0) / 1e9
                logger.info(f"  Mémoire GPU: {total_mem:.2f} GB total, {allocated:.2f} GB allouée, {reserved:.2f} GB réservée")
            else:
                logger.warning("⚠ Test 7: CUDA demandé mais non disponible, utilisation CPU")
                all_passed = False
        else:
            logger.info("✓ Test 7: Mode CPU")
    except Exception as e:
        logger.error(f"✗ Test 7 échoué: {e}", exc_info=True)
        all_passed = False
    
    # Test 8: Imports
    logger.info("Test 8: Vérification des imports...")
    try:
        import geomloss
        import pandas as pd
        logger.info("✓ Test 8 réussi: Tous les imports fonctionnent")
    except ImportError as e:
        logger.error(f"✗ Test 8 échoué: Import manquant - {e}", exc_info=True)
        all_passed = False
    
    logger.info("=" * 80)
    if all_passed:
        logger.info("✓ TOUS LES TESTS SONT PASSÉS")
    else:
        logger.error("✗ CERTAINS TESTS ONT ÉCHOUÉ - VÉRIFIEZ LES ERREURS CI-DESSUS")
    logger.info("=" * 80)
    
    return all_passed


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    exp_config = ExperimentConfig()
    
    # Exécuter les tests de validation
    if not run_validation_tests(exp_config):
        logger.error("Les tests de validation ont échoué. Arrêt du script.")
        logger.error("Veuillez corriger les erreurs avant de continuer.")
        sys.exit(1)
    
    # Lancer les expériences
    runner = ExperimentRunner(exp_config)
    runner.run_all_experiments()

