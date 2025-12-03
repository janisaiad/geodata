import torch
import torch.nn.functional as F
from geomloss import SamplesLoss
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional
import os
import logging

# ==========================================
# LOGGING SETUP
# ==========================================
os.makedirs("logs", exist_ok=True)
# Reset logging handlers to avoid duplicates if re-run
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename='logs/tearing_analysis.log',
    filemode='w', # Overwrite each run
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Add console handler
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# ==========================================
# CONFIGURATION & UTILS
# ==========================================

@dataclass
class OTConfig:
    resolution: int = 64
    blur: float = 0.03
    scaling: float = 0.9
    reach: Optional[float] = None  # None = Balanced
    p: int = 2
    device: str = "cuda" # Force CPU pour stabilité (évite CUDA error unknown)
    sigma: float = None  # Pour splatting

def get_measures_from_image(img: torch.Tensor, res: int):
    """Convertit une image en mesures de probabilité sur grille [0,1]^2."""
    C, H, W = img.shape
    scale = res / max(H, W)
    new_H, new_W = int(H * scale), int(W * scale)
    
    if new_H != H or new_W != W:
        img = F.interpolate(img.unsqueeze(0), size=(new_H, new_W), mode="bilinear", align_corners=False).squeeze(0)
    
    y = torch.linspace(0, 1, new_H, device=img.device)
    x = torch.linspace(0, 1, new_W, device=img.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    
    positions = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    weights = img.reshape(C, -1).clamp(min=1e-7)
    total_mass = weights.sum(dim=1, keepdim=True)
    weights_normalized = weights / total_mass
    
    return positions, weights_normalized, new_H, new_W, total_mass

def gaussian_rasterize(positions, weights, H, W, sigma=0.7):
    """Rasterization par Splatting Gaussien."""
    device = positions.device
    # Mapping [0,1] -> [0, W-1] correct (vs W dans le notebook original)
    pos_pixel = positions * torch.tensor([W - 1, H - 1], device=device)
    radius = int(3 * sigma + 1)
    img = torch.zeros((H, W), device=device)
    
    x_c = torch.round(pos_pixel[:, 0]).long()
    y_c = torch.round(pos_pixel[:, 1]).long()
    
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            x_k = x_c + dx
            y_k = y_c + dy
            valid_mask = (x_k >= 0) & (x_k < W) & (y_k >= 0) & (y_k < H)
            
            if not valid_mask.any(): continue
            
            idx_valid = torch.where(valid_mask)[0]
            dist_sq = (pos_pixel[idx_valid, 0] - x_k[idx_valid].float()) ** 2 + \
                      (pos_pixel[idx_valid, 1] - y_k[idx_valid].float()) ** 2
            
            gauss_weight = torch.exp(-dist_sq / (2 * sigma**2))
            val_to_add = weights[idx_valid] * gauss_weight
            
            # Normalisation physique correcte (conservation masse)
            # On renormalise par l'intégrale de la gaussienne discrète locale
            # Simplification: facteur constant ici, mais renormalisation globale à la fin
            
            linear_idx = y_k[idx_valid] * W + x_k[idx_valid]
            img.view(-1).index_add_(0, linear_idx, val_to_add)
            
    # Renormalisation globale pour conserver la masse exacte
    current_mass = img.sum()
    target_mass = weights.sum()
    if current_mass > 1e-10:
        img = img * (target_mass / current_mass)
        
    return img

def bilinear_rasterize(positions, weights, H, W):
    device = positions.device
    pos_pixel = positions * torch.tensor([W - 1, H - 1], device=device)
    x0 = torch.floor(pos_pixel[:, 0]).long().clamp(0, W - 2)
    y0 = torch.floor(pos_pixel[:, 1]).long().clamp(0, H - 2)
    x1 = x0 + 1
    y1 = y0 + 1
    
    wa = (x1.float() - pos_pixel[:, 0]) * (y1.float() - pos_pixel[:, 1])
    wb = (x1.float() - pos_pixel[:, 0]) * (pos_pixel[:, 1] - y0.float())
    wc = (pos_pixel[:, 0] - x0.float()) * (y1.float() - pos_pixel[:, 1])
    wd = (pos_pixel[:, 0] - x0.float()) * (pos_pixel[:, 1] - y0.float())
    
    img = torch.zeros((H, W), device=device)
    idx00 = y0 * W + x0
    idx01 = y0 * W + x1
    idx10 = y1 * W + x0
    idx11 = y1 * W + x1
    
    img.view(-1).index_add_(0, idx00, weights * wa)
    img.view(-1).index_add_(0, idx01, weights * wb)
    img.view(-1).index_add_(0, idx10, weights * wc)
    img.view(-1).index_add_(0, idx11, weights * wd)
    
    return img

class WassersteinInterpolator:
    def __init__(self, config: OTConfig):
        self.cfg = config
        self.epsilon = config.blur**config.p
        self.loss_layer = SamplesLoss(
            loss="sinkhorn", p=config.p, blur=config.blur,
            reach=config.reach, debias=False, potentials=True,  # debias=False CRITIQUE
            scaling=config.scaling, backend="tensorized"
        )

    def get_transport_map(self, img1, img2, channel=0):
        """
        Calcule la carte de transport barycentrique T(x) pour un canal donné.
        Retourne:
            T_x: Tensor (H, W, 2) positions cibles pour chaque pixel source
        """
        device = self.cfg.device
        pos_a, w_a, Ha, Wa, _ = get_measures_from_image(img1.to(device), self.cfg.resolution)
        pos_b, w_b, Hb, Wb, _ = get_measures_from_image(img2.to(device), self.cfg.resolution)
        
        # Sélection du canal spécifique
        wa_c = w_a[channel:channel+1]
        wb_c = w_b[channel:channel+1]
        
        # Format batch pour GeomLoss
        pos_a_batch = pos_a.unsqueeze(0).contiguous()
        pos_b_batch = pos_b.unsqueeze(0).contiguous()
        
        F_pot, G_pot = self.loss_layer(wa_c, pos_a_batch, wb_c, pos_b_batch)
        
        f, g = F_pot[0], G_pot[0]
        wa_vec, wb_vec = wa_c[0], wb_c[0]
        
        # Calcul du plan de transport log_pi
        # C_matrix: (N, M)
        dist = torch.cdist(pos_a, pos_b, p=2)
        C_matrix = (dist**2) / 2
        
        log_pi = (f[:, None] + g[None, :] - C_matrix) / self.epsilon + \
                 torch.log(wa_vec[:, None]) + torch.log(wb_vec[None, :])
        
        # Pour la projection barycentrique : T(x_i) = sum_j y_j * pi_{ij} / sum_k pi_{ik}
        # Attention numerique : log-sum-exp
        # Mais on veut juste la moyenne pondérée. 
        # P_ij = exp(log_pi_ij).
        # Row sums: sum_j P_ij = wa_vec (approximativement, sauf Unbalanced)
        
        # Normalisation conditionnelle P(y|x)
        # log P(y_j | x_i) = log_pi_{ij} - log(sum_k exp(log_pi_{ik}))
        log_row_sum = torch.logsumexp(log_pi, dim=1, keepdim=True)
        log_cond_prob = log_pi - log_row_sum
        cond_prob = torch.exp(log_cond_prob) # (N, M)
        
        # Barycentric projection: T = P * Y
        # (N, M) @ (M, 2) -> (N, 2)
        T_map_flat = torch.mm(cond_prob, pos_b)
        
        # Reshape en grille (H, W, 2)
        T_map = T_map_flat.view(Ha, Wa, 2)
        return T_map, Ha, Wa

    def interpolate(self, img1, img2, t):
        pos_a, w_a, Ha, Wa, mass_a = get_measures_from_image(img1.to(self.cfg.device), self.cfg.resolution)
        pos_b, w_b, Hb, Wb, mass_b = get_measures_from_image(img2.to(self.cfg.device), self.cfg.resolution)
        
        C = w_a.shape[0]
        pos_a_batch = pos_a.unsqueeze(0).expand(C, -1, -1).contiguous()
        pos_b_batch = pos_b.unsqueeze(0).expand(C, -1, -1).contiguous()
        
        F_pot, G_pot = self.loss_layer(w_a, pos_a_batch, w_b, pos_b_batch)
        
        # Interpolation masse (approx linéaire pour visualisation)
        current_total_mass = (1 - t) * mass_a + t * mass_b
        reconstructed_channels = []
        
        for c in range(C):
            f, g = F_pot[c], G_pot[c]
            wa_c, wb_c = w_a[c], w_b[c]
            xa, xb = pos_a, pos_b
            
            dist = torch.cdist(xa, xb, p=2)
            C_matrix = (dist**2) / 2
            
            log_pi = (f[:, None] + g[None, :] - C_matrix) / self.epsilon + \
                     torch.log(wa_c[:, None]) + torch.log(wb_c[None, :])
            pi = torch.exp(log_pi)
            
            # Displacement interpolation
            pos_t = (1 - t) * xa[:, None, :] + t * xb[None, :, :]
            
            flat_weights = pi.view(-1)
            flat_pos = pos_t.view(-1, 2)
            
            # Sigma adaptatif
            sigma_t = self.cfg.sigma
            if sigma_t is None: # Heuristique adaptive par défaut
                 # Estimation basique expansion (ratio dists médianes)
                 sigma_t = 0.3 * (1 + 4*t*(1-t)) # Simple boost temporel pour la démo
            
            if sigma_t > 0:
                channel_img = gaussian_rasterize(flat_pos, flat_weights, Hb, Wb, sigma=sigma_t)
            else:
                channel_img = bilinear_rasterize(flat_pos, flat_weights, Hb, Wb)
                
            reconstructed_channels.append(channel_img)
            
        img_t = torch.stack(reconstructed_channels, dim=0)
        # Retouche finale luminosité physique
        for c in range(C):
            if img_t[c].sum() > 0:
                img_t[c] *= current_total_mass[c] / img_t[c].sum()
                
        return img_t.cpu()

# ==========================================
# EXPERIMENTATIONS
# ==========================================

def analyze_tearing_condition(interp, img1, img2, t, name=""):
    """
    Vérifie la condition de tearing: |det(nabla Xt)| > Delta_grid
    """
    logging.info(f"--- Analyse Tearing: {name} (t={t}) ---")
    
    # 1. Calculer la carte de transport T(x)
    # On utilise le canal moyen ou le premier canal pertinent
    # Pour CIFAR Car -> Noise, utilisons le canal 0 (rouge) ou mean
    # Ici on prend canal 0
    T_map, H, W = interp.get_transport_map(img1, img2, channel=0)
    
    # 2. Calculer la carte au temps t : X_t(x) = (1-t)x + tT(x)
    # Grille initiale
    y = torch.linspace(0, 1, H, device=T_map.device)
    x = torch.linspace(0, 1, W, device=T_map.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    X_0 = torch.stack([xx, yy], dim=-1) # (H, W, 2)
    
    X_t = (1 - t) * X_0 + t * T_map
    
    # 3. Calculer le Jacobien par différences finies
    # Delta grid
    dx = 1.0 / (W - 1) # ou 1/W
    dy = 1.0 / (H - 1)
    
    # Gradients
    # dX_t/du (variation selon x)
    # (X_t[:, 1:] - X_t[:, :-1]) / dx
    dX_du = (X_t[:, 1:, :] - X_t[:, :-1, :]) / dx
    # Pad pour garder la taille
    dX_du = F.pad(dX_du.permute(2,0,1), (0,1,0,0)).permute(1,2,0) # (H, W, 2)
    
    # dX_t/dv (variation selon y)
    dX_dv = (X_t[1:, :, :] - X_t[:-1, :, :]) / dy
    dX_dv = F.pad(dX_dv.permute(2,0,1), (0,0,0,1)).permute(1,2,0) # (H, W, 2)
    
    # Matrice Jacobienne J = [[dx_du, dx_dv], [dy_du, dy_dv]]
    # dX_du contains [dx/du, dy/du]
    # dX_dv contains [dx/dv, dy/dv]
    
    j11 = dX_du[..., 0] # dx/du
    j12 = dX_dv[..., 0] # dx/dv
    j21 = dX_du[..., 1] # dy/du
    j22 = dX_dv[..., 1] # dy/dv
    
    # Determinant
    det_J = j11 * j22 - j12 * j21
    abs_det_J = torch.abs(det_J)
    
    # 4. Vérifier la condition |det| > Delta_grid
    # Delta_grid est la taille caractéristique de la grille.
    # Ici det_J est le facteur d'expansion de l'aire.
    # La condition utilisateur est |det(Grad Xt)| > Delta_grid.
    # Si Delta_grid est genre 0.03 (blur), ou 1/W (~0.015).
    # Interprétons Delta_grid comme 1/min(H,W).
    delta_grid = 1.0 / min(H, W)
    
    tearing_mask = abs_det_J > delta_grid
    num_tearing = tearing_mask.sum().item()
    total_pixels = H * W
    percent_tearing = (num_tearing / total_pixels) * 100
    
    logging.info(f"Grid Size: {H}x{W}, Delta_grid: {delta_grid:.4f}")
    logging.info(f"Jacobian Det Stats: Min={abs_det_J.min():.4f}, Max={abs_det_J.max():.4f}, Mean={abs_det_J.mean():.4f}")
    logging.info(f"Tearing Condition (|det| > {delta_grid:.4f}): {num_tearing}/{total_pixels} pixels ({percent_tearing:.2f}%)")
    
    if percent_tearing > 50:
        logging.warning("!!! TEARING MAJEUR DÉTECTÉ !!!")
    elif percent_tearing > 10:
        logging.warning("! Tearing Significatif !")
    else:
        logging.info("Tearing mineur ou absent.")
        
    return abs_det_J.cpu()

def plot_particle_tracking(interp, img1, img2, name="tracking"):
    """
    Visualise le déplacement de particules (pixels) via la carte de transport.
    """
    # Calculer T(x) pour un canal (ex: Canal Bleu pour voir son comportement, ou moyenne)
    # T_map est (H, W, 2) coordonnées dans [0,1]
    # On utilise le canal 2 (Bleu) car l'utilisateur s'intéresse à la destruction du bleu
    try:
        T_map, H, W = interp.get_transport_map(img1, img2, channel=2)
    except Exception as e:
        logging.warning(f"Erreur tracking particules: {e}")
        return

    # Sélectionner une grille de points de départ
    step = 4 # Espacement plus fin
    y_indices = torch.arange(step//2, H, step)
    x_indices = torch.arange(step//2, W, step)
    yy, xx = torch.meshgrid(y_indices, x_indices, indexing="ij")
    
    # Coordonnées de départ (normalisées [0,1])
    start_pts = torch.stack([xx, yy], dim=-1).float() 
    start_pts[..., 0] /= (W-1)
    start_pts[..., 1] /= (H-1)
    
    # Coordonnées d'arrivée
    # On sample T_map aux indices choisis
    end_pts = T_map[yy, xx].cpu() # (Ny, Nx, 2)
    start_pts = start_pts.cpu()
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # Afficher l'image source assombrie pour mieux voir les flèches
    ax.imshow(img1.permute(1,2,0).cpu() * 0.6)
    
    # Dessiner les flèches
    sx = start_pts[..., 0] * (W-1)
    sy = start_pts[..., 1] * (H-1)
    ex = end_pts[..., 0] * (W-1)
    ey = end_pts[..., 1] * (H-1)
    
    # Colorer les flèches selon la longueur du déplacement
    dist = np.sqrt((ex-sx)**2 + (ey-sy)**2)
    
    ax.quiver(sx.numpy(), sy.numpy(), (ex-sx).numpy(), (ey-sy).numpy(), dist,
              angles='xy', scale_units='xy', scale=1, cmap='coolwarm', alpha=0.9)
    
    ax.set_title(f"Suivi de Particules (Canal Bleu)\n{name}", fontsize=15)
    ax.axis('off')
    
    fname = f"/home/janis/4A/geodata/refs/reports/results/image_particle_tracking_{name}.png"
    plt.savefig(fname, bbox_inches='tight')
    print(f"Sauvegardé: {fname}")
    plt.close(fig)

def load_images():
    # Chemins à adapter
    path1 = "/home/janis/4A/geodata/data/pixelart/images/salameche.jpg" # Salamèche (Cible)
    path2 = "/home/janis/4A/geodata/data/pixelart/images/pikachu.webp" # Pikachu (Source)
    
    try:
        img1_pil = Image.open(path1).convert("RGB")
        img2_pil = Image.open(path2).convert("RGB")
        
        # Redimensionner pour vitesse
        img1_pil = img1_pil.resize((64, 64))
        img2_pil = img2_pil.resize((64, 64))
        
        img1 = torch.from_numpy(np.array(img1_pil)).permute(2, 0, 1).float() / 255.0
        img2 = torch.from_numpy(np.array(img2_pil)).permute(2, 0, 1).float() / 255.0
        return img1, img2
    except Exception as e:
        print(f"Erreur chargement images: {e}")
        print("Génération synthétique...")
        img1 = torch.zeros(3, 64, 64)
        img1[0, 20:44, 20:44] = 1.0 # Carré Rouge
        img2 = torch.zeros(3, 64, 64)
        img2[2, 20:44, 20:44] = 1.0 # Carré Bleu (même position pour test couleur pure)
        # Décalage pour test transport
        img2 = torch.zeros(3, 64, 64)
        img2[2, 30:54, 30:54] = 1.0 
        return img1, img2

def load_cifar_car():
    path = "/home/janis/4A/geodata/data/cifar10/automobile/0000.jpg"
    try:
        img = Image.open(path).convert("RGB")
        # Resize pour matching dimension avec pixelart si besoin, ou garder 32x32 original
        img = img.resize((32, 32)) 
        t_img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return t_img
    except Exception as e:
        print(f"Erreur chargement CIFAR: {e}")
        return torch.rand(3, 32, 32)

def run_experiments():
    print("Chargement des images...")
    img1, img2 = load_images() # img1=Fraise, img2=Salamèche
    source = img2 # Salamèche
    target = img1 # Fraise
    
    car_img = load_cifar_car() # CIFAR Car
    # Pour CIFAR, on transporte vers un bruit aléatoire ou une autre image ?
    # Dans le notebook original, c'était vers un tenseur aléatoire.
    # Faisons Car -> Noise pour montrer la destruction de structure
    noise_img = torch.rand_like(car_img)
    
    # 1. Comparaison Balanced vs Unbalanced (Figure 1) - GRANDE TAILLE
    print("Génération Figure 1: Balanced vs Unbalanced...")
    
    # Balanced
    cfg_bal = OTConfig(reach=None, blur=0.05, sigma=0.5)
    interp_bal = WassersteinInterpolator(cfg_bal)
    res_bal_mid = interp_bal.interpolate(source, target, 0.5)
    res_bal_end = interp_bal.interpolate(source, target, 1.0)
    
    # Unbalanced
    cfg_unbal = OTConfig(reach=0.1, blur=0.05, sigma=0.5)
    interp_unbal = WassersteinInterpolator(cfg_unbal)
    res_unbal_mid = interp_unbal.interpolate(source, target, 0.5)
    res_unbal_end = interp_unbal.interpolate(source, target, 1.0)
    
    fig1, ax = plt.subplots(2, 3, figsize=(18, 12)) # Plus grand
    # Row 1: t=0.5
    ax[0,0].imshow(source.permute(1,2,0).clamp(0,1))
    ax[0,0].set_title("Source (t=0)")
    ax[0,1].imshow(res_unbal_mid.permute(1,2,0).clamp(0,1))
    ax[0,1].set_title("Unbalanced (t=0.5)")
    ax[0,2].imshow(res_bal_mid.permute(1,2,0).clamp(0,1))
    ax[0,2].set_title("Balanced (t=0.5)")
    
    # Row 2: t=1.0
    ax[1,0].imshow(target.permute(1,2,0).clamp(0,1))
    ax[1,0].set_title("Cible (Reference)")
    ax[1,1].imshow(res_unbal_end.permute(1,2,0).clamp(0,1))
    ax[1,1].set_title("Unbalanced (t=1.0)\nReconstruction Partielle")
    ax[1,2].imshow(res_bal_end.permute(1,2,0).clamp(0,1))
    ax[1,2].set_title("Balanced (t=1.0)\nReconstruction Totale")
    
    for a in ax.flat: a.axis('off')
    plt.savefig("/home/janis/4A/geodata/refs/reports/results/image_5838f6.png", bbox_inches='tight')
    print("Sauvegardé: results/image_5838f6.png")
    
    # 2. Analyse du Tearing (Reach x Blur) - CIFAR EXPERIMENT
    print("Génération Figure 2: Analyse du Tearing (CIFAR Car)...")
    
    reaches = [0.01, 0.1, 0.5]
    blurs = [0.01, 0.05, 0.1]
    
    # Logging de la condition de tearing sur un cas critique
    # Cas critique: Reach élevé (transport forcé), Blur faible
    print("Vérification Condition Tearing (Log)...")
    cfg_critique = OTConfig(reach=0.5, blur=0.01, sigma=0.0)
    interp_critique = WassersteinInterpolator(cfg_critique)
    analyze_tearing_condition(interp_critique, car_img, noise_img, 0.5, name="CIFAR_Reach0.5_Blur0.01")

    fig2, axes = plt.subplots(len(blurs), len(reaches), figsize=(18, 12)) # Plus grand
    fig2.suptitle("Impact Reach/Blur sur Tearing (CIFAR Car -> Noise, t=0.5)", fontsize=20)
    
    for i, blur in enumerate(blurs):
        for j, reach in enumerate(reaches):
            # Sigma=0 pour montrer le tearing brut (interpolation bilinéaire)
            cfg = OTConfig(reach=reach, blur=blur, sigma=0.0) 
            interp = WassersteinInterpolator(cfg)
            res = interp.interpolate(car_img, noise_img, 0.5)
            
            ax = axes[i, j]
            ax.imshow(res.permute(1,2,0).clamp(0,1))
            if i == 0: ax.set_title(f"Reach = {reach}", fontsize=14)
            if j == 0: ax.set_ylabel(f"Blur = {blur}", fontsize=14)
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig("/home/janis/4A/geodata/refs/reports/results/image_56e3bb.png", bbox_inches='tight')
    print("Sauvegardé: results/image_56e3bb.png (CIFAR Tearing)")

    # --- NOUVELLE SECTION: ABLATION STUDY SIGMA ---
    print("Génération Figure 5: Ablation Study - Stratégies Sigma (t=0.2 et t=0.8)...")
    # On compare: Bilinéaire (Sigma=0), Sigma Fixe (0.3), Sigma Adaptatif (None -> calculé)
    # Sur le cas critique identifié plus haut (Reach 0.5, Blur 0.01)
    
    strategies = [
        ("Bilinéaire (sigma=0)", 0.0),
        ("Sigma Fixe (sigma=0.2)", 0.2),
        ("Sigma Fixe (sigma=1.0)", 1.0),
        ("Adaptatif (Heuristique)", None)
    ]
    
    cfg_base = OTConfig(reach=0.5, blur=0.01) # Cas sujet au tearing

    for t_val in [0.2, 0.8]:
        fig5, ax = plt.subplots(1, 4, figsize=(24, 6))
    
    for i, (name, sig) in enumerate(strategies):
        cfg_ablation = OTConfig(reach=0.5, blur=0.01, sigma=sig)
        interp_ab = WassersteinInterpolator(cfg_ablation)
            res_ab = interp_ab.interpolate(car_img, noise_img, t_val)
        
        ax[i].imshow(res_ab.permute(1,2,0).clamp(0,1))
            ax[i].set_title(f"{name} (t={t_val})", fontsize=14)
        ax[i].axis('off')
        
        filename = f"/home/janis/4A/geodata/refs/reports/results/image_sigma_ablation_t{int(t_val*10)}.png"
        plt.savefig(filename, bbox_inches='tight')
        print(f"Sauvegardé: {filename}")
        plt.close(fig5)
    # ----------------------------------------------

    # 3. Effet du Reach (Nouvelle Figure) - Plus grand
    print("Génération Figure 3: Effet du paramètre Reach (rho)...")
    reaches = [0.01, 0.1, 0.5, None]
    titles = ["Reach=0.01 (Local)", "Reach=0.1 (Optimal)", "Reach=0.5 (Large)", "Reach=None (Balanced)"]
    
    fig3, ax = plt.subplots(1, 4, figsize=(24, 6)) # Plus grand
    
    for i, r in enumerate(reaches):
        cfg = OTConfig(reach=r, blur=0.05, sigma=0.5)
        interp = WassersteinInterpolator(cfg)
        res = interp.interpolate(source, target, 0.5)
        ax[i].imshow(res.permute(1,2,0).clamp(0,1))
        ax[i].set_title(titles[i], fontsize=14)
        ax[i].axis('off')
        
    plt.savefig("/home/janis/4A/geodata/refs/reports/results/image_reach_effect.png", bbox_inches='tight')
    print("Sauvegardé: results/image_reach_effect.png")

    # 4. Inversion Source/Cible (Pikachu/Salamèche) - Plus grand
    print("Génération Figure 4: Inversion Source/Cible (Séquence t=0.1, 0.2, 0.8, 0.9)...")
    
    # Re-vérifions load_images
    source_pikachu = img2 
    target_salameche = img1
    
    cfg_bal = OTConfig(reach=None, blur=0.05, sigma=0.5)
    cfg_unbal = OTConfig(reach=0.1, blur=0.05, sigma=0.5)
    
    interp_bal = WassersteinInterpolator(cfg_bal)
    interp_unbal = WassersteinInterpolator(cfg_unbal)
    
    time_steps = [0.1, 0.2, 0.8, 0.9]
    
    for t in time_steps:
        res_A_bal = interp_bal.interpolate(source_pikachu, target_salameche, t)
        res_A_unbal = interp_unbal.interpolate(source_pikachu, target_salameche, t)
    
        res_B_bal = interp_bal.interpolate(target_salameche, source_pikachu, t)
        res_B_unbal = interp_unbal.interpolate(target_salameche, source_pikachu, t)
    
    fig4, ax = plt.subplots(2, 2, figsize=(16, 16)) # Très grand
    
    ax[0,0].imshow(res_A_bal.permute(1,2,0).clamp(0,1))
        ax[0,0].set_title(f"Pikachu -> Salamèche (Balanced, t={t})", fontsize=14)
    ax[0,1].imshow(res_A_unbal.permute(1,2,0).clamp(0,1))
        ax[0,1].set_title(f"Pikachu -> Salamèche (Unbalanced, t={t})\nDestruction/Création", fontsize=14)
    
    ax[1,0].imshow(res_B_bal.permute(1,2,0).clamp(0,1))
        ax[1,0].set_title(f"Salamèche -> Pikachu (Balanced, t={t})", fontsize=14)
    ax[1,1].imshow(res_B_unbal.permute(1,2,0).clamp(0,1))
        ax[1,1].set_title(f"Salamèche -> Pikachu (Unbalanced, t={t})", fontsize=14)
    
    for a in ax.flat: a.axis('off')
        
        fname = f"/home/janis/4A/geodata/refs/reports/results/image_swap_source_target_t{int(t*10)}.png"
        plt.savefig(fname, bbox_inches='tight')
        print(f"Sauvegardé: {fname}")
        plt.close(fig4)

    # 5. TRACKING PARTICULES (Nouveau)
    print("Génération Figure Tracking Particules...")
    # On regarde le transport de Salamèche vers Fraise (Bleu à détruire/déplacer)
    # Salamèche n'a pas bcp de bleu, mais Fraise en a un peu dans le fond ? Non pixel art c'est Pikachu/Salamèche.
    # Pikachu (jaune) -> Salamèche (Orange). Le bleu est dans le fond ?
    # Testons sur source -> target
    plot_particle_tracking(interp_bal, source, target, name="Balanced")
    plot_particle_tracking(interp_unbal, source, target, name="Unbalanced")

    print("\n--- TERMINÉ ---")

if __name__ == "__main__":
    run_experiments()
