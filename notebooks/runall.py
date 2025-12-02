
import torch
import torch.nn.functional as F
from geomloss import SamplesLoss
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional
import os

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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
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

def run_experiments():
    print("Chargement des images...")
    img1, img2 = load_images() # img1=Fraise(rouge), img2=Salamèche(bleu/orange) -> Inversons pour Salamèche->Fraise
    # Salamèche est img2, Fraise est img1 dans les paths ci-dessus
    source = img2 # Salamèche
    target = img1 # Fraise
    
    # 1. Comparaison Balanced vs Unbalanced (Figure 1)
    print("Génération Figure 1: Balanced vs Unbalanced...")
    
    # Balanced
    cfg_bal = OTConfig(reach=None, blur=0.05, sigma=0.5) # Sigma fixe pour isoler effet transport
    interp_bal = WassersteinInterpolator(cfg_bal)
    res_bal = interp_bal.interpolate(source, target, 0.5)
    
    # Unbalanced
    cfg_unbal = OTConfig(reach=0.1, blur=0.05, sigma=0.5)
    interp_unbal = WassersteinInterpolator(cfg_unbal)
    res_unbal = interp_unbal.interpolate(source, target, 0.5)
    
    fig1, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(source.permute(1,2,0).clamp(0,1))
    ax[0].set_title("Source (Salamèche)")
    ax[1].imshow(res_unbal.permute(1,2,0).clamp(0,1))
    ax[1].set_title("Unbalanced (Reach=0.1)\nFade + Transport")
    ax[2].imshow(res_bal.permute(1,2,0).clamp(0,1))
    ax[2].set_title("Balanced\nTransport Forcé (Fantôme)")
    for a in ax: a.axis('off')
    plt.savefig("/home/janis/4A/geodata/refs/reports/image_5838f6.png", bbox_inches='tight')
    print("Sauvegardé: image_5838f6.png")
    
    # 2. Analyse du Tearing (Figure 2)
    print("Génération Figure 2: Analyse du Tearing (Reach x Blur)...")
    
    # On utilise l'interpolation naïve pour bien visualiser le tearing
    reaches = [0.01, 0.1, 0.5]
    blurs = [0.01, 0.05]
    
    fig2, axes = plt.subplots(len(blurs), len(reaches), figsize=(15, 8))
    fig2.suptitle("Impact de Reach et Blur sur le Tearing (Interpolation Naïve)", fontsize=16)
    
    for i, blur in enumerate(blurs):
        for j, reach in enumerate(reaches):
            cfg = OTConfig(reach=reach, blur=blur, sigma=0.0) # Sigma=0 -> Bilinéaire -> Tearing visible
            interp = WassersteinInterpolator(cfg)
            res = interp.interpolate(source, target, 0.5)
            
            ax = axes[i, j]
            ax.imshow(res.permute(1,2,0).clamp(0,1))
            if i == 0: ax.set_title(f"Reach = {reach}")
            if j == 0: ax.set_ylabel(f"Blur = {blur}")
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig("/home/janis/4A/geodata/refs/reports/image_56e3bb.png", bbox_inches='tight')
    print("Sauvegardé: image_56e3bb.png (Grille Tearing)")

    # 3. Effet du Reach (Nouvelle Figure)
    print("Génération Figure 3: Effet du paramètre Reach (rho)...")
    reaches = [0.01, 0.1, 0.5, None]
    titles = ["Reach=0.01 (Très local)", "Reach=0.1 (Optimal)", "Reach=0.5 (Large)", "Reach=None (Balanced)"]
    
    fig3, ax = plt.subplots(1, 4, figsize=(20, 5))
    
    for i, r in enumerate(reaches):
        cfg = OTConfig(reach=r, blur=0.05, sigma=0.5)
        interp = WassersteinInterpolator(cfg)
        res = interp.interpolate(source, target, 0.5)
        ax[i].imshow(res.permute(1,2,0).clamp(0,1))
        ax[i].set_title(titles[i])
        ax[i].axis('off')
        
    plt.savefig("/home/janis/4A/geodata/refs/reports/image_reach_effect.png", bbox_inches='tight')
    print("Sauvegardé: image_reach_effect.png")

    print("\n--- TERMINÉ ---")

if __name__ == "__main__":
    run_experiments()

