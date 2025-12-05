#!/usr/bin/env python3
"""
Analyse des Expériences Pokemon - Transport Optimal 5D

Ce script analyse en détail les résultats des expériences de transport optimal 5D 
pour l'interpolation entre Charmander (Salameche) et Strawberry.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path
import warnings
import json
import sys

# Imports optionnels pour torch/torchvision (utilisés seulement dans les fonctions utilitaires)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import torchvision
    TORCHVISION_AVAILABLE = True
except (ImportError, RuntimeError):
    TORCHVISION_AVAILABLE = False
    torchvision = None

warnings.filterwarnings('ignore')

# Configuration des graphiques (style publication)
plt.rcParams['figure.figsize'] = [6, 6]
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'normal'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 22
mpl.rcParams['axes.formatter.limits'] = (-6, 6)
mpl.rcParams['axes.formatter.use_mathtext'] = True
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.top'] = True

# Fonctions utilitaires pour l'affichage d'images
def imshow(images, mean=0.5, std=0.5):
    """Affiche une grille d'images avec normalisation."""
    if TORCH_AVAILABLE and TORCHVISION_AVAILABLE and isinstance(images, torch.Tensor):
        img = torchvision.utils.make_grid(images)
        # Unnormalize the image
        if images.shape[1] > 1:  # Multi channels
            if isinstance(mean, (int, float)):
                mean = [mean] * images.shape[1]
            if isinstance(std, (int, float)):
                std = [std] * images.shape[1]
            for t, m, s in zip(img, mean, std):
                t.mul_(s).add_(m)
        else:
            img = img * std[0] + mean[0]  # Single channel
        # Plot it
        fig, ax = plt.subplots()
        ax.imshow(np.transpose(img.numpy(), (1, 2, 0)))
        ax.set_axis_off()
        return fig
    else:
        # Si c'est déjà un numpy array
        fig, ax = plt.subplots()
        ax.imshow(images)
        ax.set_axis_off()
        return fig

def cvtImg(img):
    """Convertit un tensor d'image en numpy array normalisé."""
    if TORCH_AVAILABLE and isinstance(img, torch.Tensor):
        # Unnormalize the image 
        img = img.permute([0, 2, 3, 1])
        img = img - img.min()
        img = (img / img.max())
        # Return it as a numpy array
        return img.numpy().astype(np.float32)
    else:
        return img

def show_examples(x, n_examples=25, n_cols=5):
    """Affiche une grille d'exemples d'images."""
    if isinstance(x, torch.Tensor):
        imgs = cvtImg(x)  # Unnormalize images
    else:
        imgs = x
    n_rows = int(np.ceil(n_examples / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    for i in range(min(n_examples, len(imgs))):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].imshow(imgs[i])
        axes[row, col].axis('off')
    # Cacher les axes inutilisés
    for i in range(n_examples, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    return fig

# Compute nearest neighbors of an image in a dataset 
def compute_knn(dataset, x, k=3):
    """Calcule les k plus proches voisins d'une image dans un dataset."""
    if TORCH_AVAILABLE and isinstance(dataset, torch.Tensor) and isinstance(x, torch.Tensor):
        dist = torch.norm(dataset - x, dim=(1, 2, 3), p=None)
        knn = dist.topk(k, largest=False)
        return knn
    else:
        # Version numpy
        dist = np.linalg.norm(dataset - x, axis=(1, 2, 3))
        knn_indices = np.argsort(dist)[:k]
        return knn_indices

# Chemins
RESULTS_DIR = Path("/Data/janis.aiad/geodata/refs/reports/results/pokemon_experiments_salameche_strawberry")
CSV_PATH = RESULTS_DIR / "metrics" / "all_experiments.csv"
IMAGES_DIR = RESULTS_DIR / "images"

# Initialiser les variables expérimentales comme DataFrames vides
exp1 = pd.DataFrame()
exp2 = pd.DataFrame()
exp3 = pd.DataFrame()
exp4 = pd.DataFrame()
exp5 = pd.DataFrame()
exp6 = pd.DataFrame()

# Vérifier que les répertoires existent
if not RESULTS_DIR.exists():
    print(f"ERREUR: Le répertoire {RESULTS_DIR} n'existe pas!")
    sys.exit(1)

if not CSV_PATH.exists():
    print(f"ERREUR: Le fichier CSV {CSV_PATH} n'existe pas!")
    sys.exit(1)

# Créer le répertoire images s'il n'existe pas
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Charger les données
print("=" * 80)
print("CHARGEMENT DES DONNÉES")
print("=" * 80)
print(f"Fichier CSV: {CSV_PATH}")
print(f"Répertoire des résultats: {RESULTS_DIR}")
print(f"Répertoire des images: {IMAGES_DIR}")
print("")

try:
    df = pd.read_csv(CSV_PATH)
    print(f"✓ CSV chargé avec succès")
    print(f"  Nombre total de lignes: {len(df)}")
    if len(df) == 0:
        print("ATTENTION: Le fichier CSV est vide!")
        sys.exit(1)
    
    if 'experiment_id' in df.columns:
        n_unique = df['experiment_id'].nunique()
        print(f"  Nombre d'expériences uniques: {n_unique}")
        print(f"  Nombre moyen de frames par expérience: {len(df) / n_unique:.1f}")
    
    print(f"\nColonnes disponibles ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        non_null = df[col].notna().sum()
        print(f"  {i:2d}. {col:30s} ({non_null}/{len(df)} valeurs non-null)")
    
    print("\nPremières lignes:")
    print(df.head())
    
    # Statistiques sur les données
    print("\n" + "=" * 80)
    print("STATISTIQUES SUR LES DONNÉES")
    print("=" * 80)
    if 'resolution' in df.columns:
        resolutions = sorted(df['resolution'].unique())
        print(f"Résolutions: {resolutions}")
        print(f"  Nombre: {len(resolutions)}, Min: {min(resolutions)}, Max: {max(resolutions)}")
    if 'lambda' in df.columns:
        lambdas = sorted(df['lambda'].unique())
        print(f"Lambdas: {lambdas}")
        print(f"  Nombre: {len(lambdas)}, Min: {min(lambdas):.2f}, Max: {max(lambdas):.2f}")
    if 'blur' in df.columns:
        blurs = sorted(df['blur'].unique())
        print(f"Blurs: {blurs}")
        print(f"  Nombre: {len(blurs)}, Min: {min(blurs):.3f}, Max: {max(blurs):.3f}")
    if 'reach' in df.columns:
        reaches = sorted([r for r in df['reach'].unique() if pd.notna(r)])
        print(f"Reaches: {reaches}")
        print(f"  Nombre: {len(reaches)}")
    if 't' in df.columns:
        times = sorted(df['t'].unique())
        print(f"Temps: {times}")
        print(f"  Nombre: {len(times)}, Min: {min(times):.2f}, Max: {max(times):.2f}")
    if 'psnr' in df.columns:
        psnr_data = df['psnr'].dropna()
        if len(psnr_data) > 0:
            print(f"PSNR: Min={psnr_data.min():.2f} dB, Max={psnr_data.max():.2f} dB, Mean={psnr_data.mean():.2f} dB")
    if 'compute_time_total' in df.columns:
        time_data = df['compute_time_total'].dropna()
        if len(time_data) > 0:
            print(f"Temps de calcul: Min={time_data.min():.3f}s, Max={time_data.max():.3f}s, Mean={time_data.mean():.3f}s")
    print("=" * 80)
    print("")
    
except Exception as e:
    print(f"ERREUR lors du chargement du CSV: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 2. Vue d'Ensemble des Données
# ============================================================================

print("\n" + "=" * 80)
print("2. VUE D'ENSEMBLE DES DONNÉES")
print("=" * 80)

# Statistiques descriptives
print("\n=== STATISTIQUES DESCRIPTIVES ===")
if 'resolution' in df.columns:
    print(f"\nRésolutions testées: {sorted(df['resolution'].unique())}")
if 'lambda' in df.columns:
    print(f"Lambdas testés: {sorted(df['lambda'].unique())}")
if 'blur' in df.columns:
    print(f"Blurs testés: {sorted(df['blur'].unique())}")
if 'reach' in df.columns:
    print(f"Reaches testés: {sorted([r for r in df['reach'].unique() if pd.notna(r)])}")

# Distribution des expériences
print("\nGénération des graphiques de distribution...")
try:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Résolutions
    if 'resolution' in df.columns:
        axes[0, 0].hist(df['resolution'].unique(), bins=20, edgecolor='black')
        axes[0, 0].set_xlabel('Résolution')
        axes[0, 0].set_ylabel('Nombre d\'expériences')
        axes[0, 0].set_title('Distribution des Résolutions')
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'Colonne "resolution" non disponible', ha='center', va='center')
        axes[0, 0].set_title('Distribution des Résolutions')
    
    # Lambdas
    if 'lambda' in df.columns:
        axes[0, 1].hist(df['lambda'].unique(), bins=20, edgecolor='black')
        axes[0, 1].set_xlabel('Lambda (λ)')
        axes[0, 1].set_ylabel('Nombre d\'expériences')
        axes[0, 1].set_title('Distribution des Lambdas')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'Colonne "lambda" non disponible', ha='center', va='center')
        axes[0, 1].set_title('Distribution des Lambdas')
    
    # Blurs
    if 'blur' in df.columns:
        axes[1, 0].hist(df['blur'].unique(), bins=20, edgecolor='black')
        axes[1, 0].set_xlabel('Blur (ε)')
        axes[1, 0].set_ylabel('Nombre d\'expériences')
        axes[1, 0].set_title('Distribution des Blurs')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Colonne "blur" non disponible', ha='center', va='center')
        axes[1, 0].set_title('Distribution des Blurs')
    
    # Temps de calcul
    if 'compute_time_total' in df.columns:
        time_data = df['compute_time_total'].dropna()
        if len(time_data) > 0:
            axes[1, 1].hist(time_data, bins=50, edgecolor='black')
            axes[1, 1].set_xlabel('Temps de calcul (s)')
            axes[1, 1].set_ylabel('Fréquence')
            axes[1, 1].set_title('Distribution des Temps de Calcul')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Aucune donnée de temps disponible', ha='center', va='center')
            axes[1, 1].set_title('Distribution des Temps de Calcul')
    else:
        axes[1, 1].text(0.5, 0.5, 'Colonne "compute_time_total" non disponible', ha='center', va='center')
        axes[1, 1].set_title('Distribution des Temps de Calcul')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "overview_distributions.png", dpi=300, bbox_inches='tight')
    print(f"\nGraphique sauvegardé: {RESULTS_DIR / 'overview_distributions.png'}")
    plt.close()
except Exception as e:
    print(f"ERREUR lors de la génération du graphique de vue d'ensemble: {e}")
    plt.close('all')

# ============================================================================
# 3. Expérience 1: Ablation Lambda (λ)
# ============================================================================

print("\n" + "=" * 80)
print("3. EXPÉRIENCE 1: ABLATION LAMBDA (λ)")
print("=" * 80)
print("Objectif: Déterminer la valeur optimale de λ pour l'interpolation.")

# Filtrer les données de l'expérience 1 (t=0.5, res=64, blur=0.03, reach=0.1)
required_cols_exp1 = ['t', 'resolution', 'blur', 'reach']
if all(col in df.columns for col in required_cols_exp1):
    exp1 = df[(df['t'] == 0.5) & (df['resolution'] == 64) & 
              (df['blur'] == 0.03) & (df['reach'] == 0.1)].copy()
else:
    print(f"ATTENTION: Colonnes manquantes pour exp1: {[c for c in required_cols_exp1 if c not in df.columns]}")
    exp1 = pd.DataFrame()

if len(exp1) > 0 and 'lambda' in exp1.columns:
    try:
        exp1 = exp1.sort_values('lambda')
        print(f"  Génération de 6 graphiques pour l'expérience 1...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # PSNR vs Lambda
        if 'psnr' in exp1.columns:
            axes[0, 0].plot(exp1['lambda'], exp1['psnr'], marker='o', linewidth=2, markersize=8)
            axes[0, 0].set_xlabel('Lambda (λ)')
            axes[0, 0].set_ylabel('PSNR (dB)')
            axes[0, 0].set_title('PSNR vs Lambda')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='λ=1.0 (5D optimal)')
            axes[0, 0].legend()
        else:
            axes[0, 0].text(0.5, 0.5, 'PSNR non disponible', ha='center', va='center')
            axes[0, 0].set_title('PSNR vs Lambda')
        
        # Delta E vs Lambda
        if 'delta_e' in exp1.columns:
            axes[0, 1].plot(exp1['lambda'], exp1['delta_e'], marker='o', linewidth=2, markersize=8, color='orange')
            axes[0, 1].set_xlabel('Lambda (λ)')
            axes[0, 1].set_ylabel('ΔE (CIE76)')
            axes[0, 1].set_title('Delta E vs Lambda (plus bas = mieux)')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].axvline(x=1.0, color='r', linestyle='--', alpha=0.5)
        else:
            axes[0, 1].text(0.5, 0.5, 'Delta E non disponible', ha='center', va='center')
            axes[0, 1].set_title('Delta E vs Lambda')
        
        # Sharpness vs Lambda
        if 'sharpness' in exp1.columns:
            axes[0, 2].plot(exp1['lambda'], exp1['sharpness'], marker='o', linewidth=2, markersize=8, color='green')
            axes[0, 2].set_xlabel('Lambda (λ)')
            axes[0, 2].set_ylabel('Sharpness')
            axes[0, 2].set_title('Sharpness vs Lambda')
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].axvline(x=1.0, color='r', linestyle='--', alpha=0.5)
        else:
            axes[0, 2].text(0.5, 0.5, 'Sharpness non disponible', ha='center', va='center')
            axes[0, 2].set_title('Sharpness vs Lambda')
        
        # Mass Error vs Lambda
        if 'mass_error' in exp1.columns:
            axes[1, 0].plot(exp1['lambda'], exp1['mass_error'], marker='o', linewidth=2, markersize=8, color='purple')
            axes[1, 0].set_xlabel('Lambda (λ)')
            axes[1, 0].set_ylabel('Mass Error')
            axes[1, 0].set_title('Mass Error vs Lambda (plus bas = mieux)')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].axvline(x=1.0, color='r', linestyle='--', alpha=0.5)
        else:
            axes[1, 0].text(0.5, 0.5, 'Mass Error non disponible', ha='center', va='center')
            axes[1, 0].set_title('Mass Error vs Lambda')
        
        # Coverage vs Lambda
        if 'coverage' in exp1.columns:
            axes[1, 1].plot(exp1['lambda'], exp1['coverage'], marker='o', linewidth=2, markersize=8, color='brown')
            axes[1, 1].set_xlabel('Lambda (λ)')
            axes[1, 1].set_ylabel('Coverage')
            axes[1, 1].set_title('Coverage vs Lambda')
            axes[1, 1].set_ylim([0.95, 1.01])
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axvline(x=1.0, color='r', linestyle='--', alpha=0.5)
        else:
            axes[1, 1].text(0.5, 0.5, 'Coverage non disponible', ha='center', va='center')
            axes[1, 1].set_title('Coverage vs Lambda')
        
        # Temps de calcul vs Lambda
        if 'compute_time_total' in exp1.columns:
            axes[1, 2].plot(exp1['lambda'], exp1['compute_time_total'], marker='o', linewidth=2, markersize=8, color='red')
            axes[1, 2].set_xlabel('Lambda (λ)')
            axes[1, 2].set_ylabel('Temps de calcul (s)')
            axes[1, 2].set_title('Temps de Calcul vs Lambda')
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].axvline(x=1.0, color='r', linestyle='--', alpha=0.5)
        else:
            axes[1, 2].text(0.5, 0.5, 'Temps de calcul non disponible', ha='center', va='center')
            axes[1, 2].set_title('Temps de Calcul vs Lambda')
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "exp1_lambda_ablation.png", dpi=300, bbox_inches='tight')
        output_path = RESULTS_DIR / "exp1_lambda_ablation.png"
        print(f"✓ Graphique sauvegardé: {output_path}")
        if output_path.exists():
            print(f"  Taille: {output_path.stat().st_size / 1024:.1f} KB")
        plt.close()
        
        # Trouver le lambda optimal (max PSNR)
        if 'psnr' in exp1.columns and exp1['psnr'].notna().any():
            best_lambda_psnr = exp1.loc[exp1['psnr'].idxmax(), 'lambda']
            print(f"\n=== RÉSULTATS EXPÉRIENCE 1 ===")
            print(f"Lambda optimal (PSNR max): λ = {best_lambda_psnr:.2f} (PSNR = {exp1.loc[exp1['psnr'].idxmax(), 'psnr']:.2f} dB)")
        if 'delta_e' in exp1.columns and exp1['delta_e'].notna().any():
            best_lambda_de = exp1.loc[exp1['delta_e'].idxmin(), 'lambda']
            print(f"Lambda optimal (ΔE min): λ = {best_lambda_de:.2f} (ΔE = {exp1.loc[exp1['delta_e'].idxmin(), 'delta_e']:.2f})")
        
        print(f"\nComparaison λ=0.0 (2D) vs λ=1.0 (5D):")
        if len(exp1[exp1['lambda'] == 0.0]) > 0 and len(exp1[exp1['lambda'] == 1.0]) > 0:
            lambda_0 = exp1[exp1['lambda'] == 0.0].iloc[0]
            lambda_1 = exp1[exp1['lambda'] == 1.0].iloc[0]
            if 'psnr' in lambda_0.index and 'psnr' in lambda_1.index:
                print(f"  PSNR: {lambda_0['psnr']:.2f} dB (2D) vs {lambda_1['psnr']:.2f} dB (5D) → {'5D meilleur' if lambda_1['psnr'] > lambda_0['psnr'] else '2D meilleur'}")
            if 'delta_e' in lambda_0.index and 'delta_e' in lambda_1.index:
                print(f"  ΔE: {lambda_0['delta_e']:.2f} (2D) vs {lambda_1['delta_e']:.2f} (5D) → {'5D meilleur' if lambda_1['delta_e'] < lambda_0['delta_e'] else '2D meilleur'}")
    except Exception as e:
        print(f"ERREUR lors du traitement de l'expérience 1: {e}")
        plt.close('all')
else:
    print("Aucune donnée pour l'expérience 1")
    exp1 = pd.DataFrame()  # Initialize empty dataframe

# ============================================================================
# 4. Expérience 2: Comparaison 2D vs 5D (Séquence Temporelle)
# ============================================================================

print("\n" + "=" * 80)
print("4. EXPÉRIENCE 2: COMPARAISON 2D vs 5D (SÉQUENCE TEMPORELLE)")
print("=" * 80)
print("Objectif: Comparer le transport 2D (spatial) et 5D (spatial+couleur) sur toute la séquence.")

# Filtrer les données de l'expérience 2 (res=64, blur=0.03, reach=0.1)
required_cols_exp2 = ['resolution', 'blur', 'reach', 'lambda']
if all(col in df.columns for col in required_cols_exp2):
    exp2 = df[(df['resolution'] == 64) & (df['blur'] == 0.03) & 
              (df['reach'] == 0.1) & (df['lambda'].isin([0.0, 1.0]))].copy()
else:
    print(f"ATTENTION: Colonnes manquantes pour exp2: {[c for c in required_cols_exp2 if c not in df.columns]}")
    exp2 = pd.DataFrame()

if len(exp2) > 0 and 't' in exp2.columns:
    try:
        exp2_2d = exp2[exp2['lambda'] == 0.0].sort_values('t') if len(exp2[exp2['lambda'] == 0.0]) > 0 else pd.DataFrame()
        exp2_5d = exp2[exp2['lambda'] == 1.0].sort_values('t') if len(exp2[exp2['lambda'] == 1.0]) > 0 else pd.DataFrame()
        
        if len(exp2_2d) > 0 or len(exp2_5d) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            
            # PSNR vs Temps
            if 'psnr' in exp2.columns:
                if len(exp2_2d) > 0 and 'psnr' in exp2_2d.columns:
                    axes[0, 0].plot(exp2_2d['t'], exp2_2d['psnr'], marker='o', label='2D (λ=0.0)', linewidth=2)
                if len(exp2_5d) > 0 and 'psnr' in exp2_5d.columns:
                    axes[0, 0].plot(exp2_5d['t'], exp2_5d['psnr'], marker='s', label='5D (λ=1.0)', linewidth=2)
                axes[0, 0].set_xlabel('Temps t')
                axes[0, 0].set_ylabel('PSNR (dB)')
                axes[0, 0].set_title('PSNR vs Temps: 2D vs 5D')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            else:
                axes[0, 0].text(0.5, 0.5, 'PSNR non disponible', ha='center', va='center')
                axes[0, 0].set_title('PSNR vs Temps: 2D vs 5D')
            
            # Delta E vs Temps
            if 'delta_e' in exp2.columns:
                if len(exp2_2d) > 0 and 'delta_e' in exp2_2d.columns:
                    axes[0, 1].plot(exp2_2d['t'], exp2_2d['delta_e'], marker='o', label='2D (λ=0.0)', linewidth=2)
                if len(exp2_5d) > 0 and 'delta_e' in exp2_5d.columns:
                    axes[0, 1].plot(exp2_5d['t'], exp2_5d['delta_e'], marker='s', label='5D (λ=1.0)', linewidth=2)
                axes[0, 1].set_xlabel('Temps t')
                axes[0, 1].set_ylabel('ΔE (CIE76)')
                axes[0, 1].set_title('Delta E vs Temps: 2D vs 5D')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            else:
                axes[0, 1].text(0.5, 0.5, 'Delta E non disponible', ha='center', va='center')
                axes[0, 1].set_title('Delta E vs Temps: 2D vs 5D')
            
            # Sharpness vs Temps
            if 'sharpness' in exp2.columns:
                if len(exp2_2d) > 0 and 'sharpness' in exp2_2d.columns:
                    axes[1, 0].plot(exp2_2d['t'], exp2_2d['sharpness'], marker='o', label='2D (λ=0.0)', linewidth=2)
                if len(exp2_5d) > 0 and 'sharpness' in exp2_5d.columns:
                    axes[1, 0].plot(exp2_5d['t'], exp2_5d['sharpness'], marker='s', label='5D (λ=1.0)', linewidth=2)
                axes[1, 0].set_xlabel('Temps t')
                axes[1, 0].set_ylabel('Sharpness')
                axes[1, 0].set_title('Sharpness vs Temps: 2D vs 5D')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'Sharpness non disponible', ha='center', va='center')
                axes[1, 0].set_title('Sharpness vs Temps: 2D vs 5D')
            
            # Mass Error vs Temps
            if 'mass_error' in exp2.columns:
                if len(exp2_2d) > 0 and 'mass_error' in exp2_2d.columns:
                    axes[1, 1].plot(exp2_2d['t'], exp2_2d['mass_error'], marker='o', label='2D (λ=0.0)', linewidth=2)
                if len(exp2_5d) > 0 and 'mass_error' in exp2_5d.columns:
                    axes[1, 1].plot(exp2_5d['t'], exp2_5d['mass_error'], marker='s', label='5D (λ=1.0)', linewidth=2)
                axes[1, 1].set_xlabel('Temps t')
                axes[1, 1].set_ylabel('Mass Error')
                axes[1, 1].set_title('Mass Error vs Temps: 2D vs 5D')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'Mass Error non disponible', ha='center', va='center')
                axes[1, 1].set_title('Mass Error vs Temps: 2D vs 5D')
            
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / "exp2_2d_vs_5d.png", dpi=300, bbox_inches='tight')
            output_path = RESULTS_DIR / "exp2_2d_vs_5d.png"
            print(f"✓ Graphique sauvegardé: {output_path}")
            if output_path.exists():
                print(f"  Taille: {output_path.stat().st_size / 1024:.1f} KB")
            plt.close()
            
            # Statistiques comparatives
            print("\n=== RÉSULTATS EXPÉRIENCE 2 ===")
            if len(exp2_2d) > 0 and 'psnr' in exp2_2d.columns:
                print(f"PSNR moyen - 2D: {exp2_2d['psnr'].mean():.2f} dB", end="")
            if len(exp2_5d) > 0 and 'psnr' in exp2_5d.columns:
                print(f", 5D: {exp2_5d['psnr'].mean():.2f} dB")
            if 'delta_e' in exp2.columns:
                if len(exp2_2d) > 0 and 'delta_e' in exp2_2d.columns:
                    print(f"ΔE moyen - 2D: {exp2_2d['delta_e'].mean():.2f}", end="")
                if len(exp2_5d) > 0 and 'delta_e' in exp2_5d.columns:
                    print(f", 5D: {exp2_5d['delta_e'].mean():.2f}")
            if 'sharpness' in exp2.columns:
                if len(exp2_2d) > 0 and 'sharpness' in exp2_2d.columns:
                    print(f"Sharpness moyen - 2D: {exp2_2d['sharpness'].mean():.4f}", end="")
                if len(exp2_5d) > 0 and 'sharpness' in exp2_5d.columns:
                    print(f", 5D: {exp2_5d['sharpness'].mean():.4f}")
            if 'mass_error' in exp2.columns:
                if len(exp2_2d) > 0 and 'mass_error' in exp2_2d.columns:
                    print(f"Mass Error moyen - 2D: {exp2_2d['mass_error'].mean():.6f}", end="")
                if len(exp2_5d) > 0 and 'mass_error' in exp2_5d.columns:
                    print(f", 5D: {exp2_5d['mass_error'].mean():.6f}")
    except Exception as e:
        print(f"ERREUR lors du traitement de l'expérience 2: {e}")
        plt.close('all')
else:
    print("Aucune donnée pour l'expérience 2")
    exp2 = pd.DataFrame()  # Initialize empty dataframe

# ============================================================================
# 5. Expérience 3: Impact du Splatting Adaptatif (Résolution)
# ============================================================================

print("\n" + "=" * 80)
print("5. EXPÉRIENCE 3: IMPACT DU SPLATTING ADAPTATIF (RÉSOLUTION)")
print("=" * 80)
print("Objectif: Évaluer l'effet de la résolution sur la qualité.")

# Filtrer les données de l'expérience 3 (t=0.5, lambda=1.0, blur=0.03, reach=0.1)
required_cols_exp3 = ['t', 'lambda', 'blur', 'reach']
if all(col in df.columns for col in required_cols_exp3):
    exp3 = df[(df['t'] == 0.5) & (df['lambda'] == 1.0) & 
              (df['blur'] == 0.03) & (df['reach'] == 0.1)].copy()
else:
    print(f"ATTENTION: Colonnes manquantes pour exp3: {[c for c in required_cols_exp3 if c not in df.columns]}")
    exp3 = pd.DataFrame()

if len(exp3) > 0 and 'resolution' in exp3.columns:
    try:
        exp3 = exp3.sort_values('resolution')
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # PSNR vs Résolution
        if 'psnr' in exp3.columns:
            axes[0, 0].plot(exp3['resolution'], exp3['psnr'], marker='o', linewidth=2, markersize=8)
            axes[0, 0].set_xlabel('Résolution')
            axes[0, 0].set_ylabel('PSNR (dB)')
            axes[0, 0].set_title('PSNR vs Résolution')
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'PSNR non disponible', ha='center', va='center')
            axes[0, 0].set_title('PSNR vs Résolution')
        
        # Delta E vs Résolution
        if 'delta_e' in exp3.columns:
            axes[0, 1].plot(exp3['resolution'], exp3['delta_e'], marker='o', linewidth=2, markersize=8, color='orange')
            axes[0, 1].set_xlabel('Résolution')
            axes[0, 1].set_ylabel('ΔE (CIE76)')
            axes[0, 1].set_title('Delta E vs Résolution')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'Delta E non disponible', ha='center', va='center')
            axes[0, 1].set_title('Delta E vs Résolution')
        
        # Sharpness vs Résolution
        if 'sharpness' in exp3.columns:
            axes[0, 2].plot(exp3['resolution'], exp3['sharpness'], marker='o', linewidth=2, markersize=8, color='green')
            axes[0, 2].set_xlabel('Résolution')
            axes[0, 2].set_ylabel('Sharpness')
            axes[0, 2].set_title('Sharpness vs Résolution')
            axes[0, 2].grid(True, alpha=0.3)
        else:
            axes[0, 2].text(0.5, 0.5, 'Sharpness non disponible', ha='center', va='center')
            axes[0, 2].set_title('Sharpness vs Résolution')
        
        # Temps de calcul vs Résolution (échelle log)
        if 'compute_time_total' in exp3.columns:
            axes[1, 0].plot(exp3['resolution'], exp3['compute_time_total'], marker='o', linewidth=2, markersize=8, color='red')
            axes[1, 0].set_xlabel('Résolution')
            axes[1, 0].set_ylabel('Temps de calcul (s)')
            axes[1, 0].set_title('Temps de Calcul vs Résolution')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Temps de calcul non disponible', ha='center', va='center')
            axes[1, 0].set_title('Temps de Calcul vs Résolution')
        
        # Mémoire GPU vs Résolution
        if 'memory_max_allocated_gb' in exp3.columns and exp3['memory_max_allocated_gb'].notna().any():
            axes[1, 1].plot(exp3['resolution'], exp3['memory_max_allocated_gb'], marker='o', linewidth=2, markersize=8, color='purple')
            axes[1, 1].set_xlabel('Résolution')
            axes[1, 1].set_ylabel('Mémoire GPU (GB)')
            axes[1, 1].set_title('Mémoire GPU vs Résolution')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Mémoire GPU non disponible', ha='center', va='center')
            axes[1, 1].set_title('Mémoire GPU vs Résolution')
        
        # Temps Sinkhorn vs Résolution
        if 'sinkhorn_time' in exp3.columns:
            axes[1, 2].plot(exp3['resolution'], exp3['sinkhorn_time'], marker='o', linewidth=2, markersize=8, color='brown')
            axes[1, 2].set_xlabel('Résolution')
            axes[1, 2].set_ylabel('Temps Sinkhorn (s)')
            axes[1, 2].set_title('Temps Sinkhorn vs Résolution')
            axes[1, 2].set_yscale('log')
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'Temps Sinkhorn non disponible', ha='center', va='center')
            axes[1, 2].set_title('Temps Sinkhorn vs Résolution')
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "exp3_resolution_impact.png", dpi=300, bbox_inches='tight')
        output_path = RESULTS_DIR / "exp3_resolution_impact.png"
        print(f"✓ Graphique sauvegardé: {output_path}")
        if output_path.exists():
            print(f"  Taille: {output_path.stat().st_size / 1024:.1f} KB")
        plt.close()
        
        # Analyse de scalabilité
        print("\n=== RÉSULTATS EXPÉRIENCE 3 ===")
        print("Scalabilité:")
        for res in sorted(exp3['resolution'].unique()):
            res_data = exp3[exp3['resolution'] == res].iloc[0]
            print(f"  Résolution {res}×{res}:")
            if 'psnr' in res_data.index:
                print(f"    PSNR: {res_data['psnr']:.2f} dB")
            if 'compute_time_total' in res_data.index:
                print(f"    Temps: {res_data['compute_time_total']:.3f} s")
            if 'memory_max_allocated_gb' in res_data.index and pd.notna(res_data.get('memory_max_allocated_gb')):
                print(f"    Mémoire: {res_data['memory_max_allocated_gb']:.2f} GB")
    except Exception as e:
        print(f"ERREUR lors du traitement de l'expérience 3: {e}")
        plt.close('all')
else:
    print("Aucune donnée pour l'expérience 3")
    exp3 = pd.DataFrame()  # Initialize empty dataframe

# ============================================================================
# 6. Expérience 4: Sensibilité aux Paramètres (ε, ρ)
# ============================================================================

print("\n" + "=" * 80)
print("6. EXPÉRIENCE 4: SENSIBILITÉ AUX PARAMÈTRES (ε, ρ)")
print("=" * 80)
print("Objectif: Cartographier l'impact des paramètres de régularisation.")

# Filtrer les données de l'expérience 4 (t=0.5, res=64, lambda=1.0)
required_cols_exp4 = ['t', 'resolution', 'lambda']
if all(col in df.columns for col in required_cols_exp4):
    exp4 = df[(df['t'] == 0.5) & (df['resolution'] == 64) & 
              (df['lambda'] == 1.0)].copy()
else:
    print(f"ATTENTION: Colonnes manquantes pour exp4: {[c for c in required_cols_exp4 if c not in df.columns]}")
    exp4 = pd.DataFrame()

if len(exp4) > 0:
    try:
        # Préparer les données pour les heatmaps
        if 'blur' not in exp4.columns or 'reach' not in exp4.columns:
            print("ATTENTION: Colonnes 'blur' ou 'reach' manquantes pour exp4")
            raise ValueError("Colonnes manquantes")
        
        blurs = sorted(exp4['blur'].unique())
        reaches = sorted([r for r in exp4['reach'].unique() if pd.notna(r)])
        
        if len(blurs) == 0 or len(reaches) == 0:
            print("ATTENTION: Pas de données blur ou reach pour exp4")
            raise ValueError("Données insuffisantes")
        
        # Créer les matrices pour les heatmaps
        psnr_matrix = np.zeros((len(reaches), len(blurs)))
        de_matrix = np.zeros((len(reaches), len(blurs)))
        time_matrix = np.zeros((len(reaches), len(blurs)))
        
        for i, r in enumerate(reaches):
            for j, b in enumerate(blurs):
                subset = exp4[(exp4['blur'] == b) & (exp4['reach'] == r)]
                if len(subset) > 0:
                    if 'psnr' in subset.columns:
                        psnr_matrix[i, j] = subset['psnr'].mean()
                    if 'delta_e' in subset.columns:
                        de_matrix[i, j] = subset['delta_e'].mean()
                    if 'compute_time_total' in subset.columns:
                        time_matrix[i, j] = subset['compute_time_total'].mean()
                else:
                    psnr_matrix[i, j] = np.nan
                    de_matrix[i, j] = np.nan
                    time_matrix[i, j] = np.nan
        
        # Heatmaps
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # PSNR Heatmap
        im1 = axes[0].imshow(psnr_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
        axes[0].set_xticks(range(len(blurs)))
        axes[0].set_xticklabels([f"{b:.2f}" for b in blurs], rotation=45)
        axes[0].set_yticks(range(len(reaches)))
        axes[0].set_yticklabels([str(r) if isinstance(r, str) else f"{r:.2f}" for r in reaches])
        axes[0].set_xlabel('Blur (ε)')
        axes[0].set_ylabel('Reach (ρ)')
        axes[0].set_title('PSNR (dB) - Plus haut = mieux')
        plt.colorbar(im1, ax=axes[0])
        
        # Delta E Heatmap
        im2 = axes[1].imshow(de_matrix, aspect='auto', cmap='plasma_r', interpolation='nearest')
        axes[1].set_xticks(range(len(blurs)))
        axes[1].set_xticklabels([f"{b:.2f}" for b in blurs], rotation=45)
        axes[1].set_yticks(range(len(reaches)))
        axes[1].set_yticklabels([str(r) if isinstance(r, str) else f"{r:.2f}" for r in reaches])
        axes[1].set_xlabel('Blur (ε)')
        axes[1].set_ylabel('Reach (ρ)')
        axes[1].set_title('Delta E - Plus bas = mieux')
        plt.colorbar(im2, ax=axes[1])
        
        # Temps de calcul Heatmap
        im3 = axes[2].imshow(time_matrix, aspect='auto', cmap='hot', interpolation='nearest')
        axes[2].set_xticks(range(len(blurs)))
        axes[2].set_xticklabels([f"{b:.2f}" for b in blurs], rotation=45)
        axes[2].set_yticks(range(len(reaches)))
        axes[2].set_yticklabels([str(r) if isinstance(r, str) else f"{r:.2f}" for r in reaches])
        axes[2].set_xlabel('Blur (ε)')
        axes[2].set_ylabel('Reach (ρ)')
        axes[2].set_title('Temps de Calcul (s)')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "exp4_parameter_sensitivity.png", dpi=300, bbox_inches='tight')
        output_path = RESULTS_DIR / "exp4_parameter_sensitivity.png"
        print(f"✓ Graphique sauvegardé: {output_path}")
        if output_path.exists():
            print(f"  Taille: {output_path.stat().st_size / 1024:.1f} KB")
        plt.close()
        
        # Trouver les paramètres optimaux
        best_idx = np.unravel_index(np.nanargmax(psnr_matrix), psnr_matrix.shape)
        best_blur = blurs[best_idx[1]]
        best_reach = reaches[best_idx[0]]
        best_psnr = psnr_matrix[best_idx]
    
        print("\n=== RÉSULTATS EXPÉRIENCE 4 ===")
        print(f"Paramètres optimaux (PSNR max):")
        best_reach_str = str(best_reach) if isinstance(best_reach, str) else f"{best_reach:.2f}"
        print(f"  ε = {best_blur:.3f}, ρ = {best_reach_str}")
        print(f"  PSNR = {best_psnr:.2f} dB")
        
        # Analyse de la sensibilité
        print(f"\nSensibilité au blur (ε):")
        for b in blurs[:5]:  # Premiers 5 blurs
            subset = exp4[exp4['blur'] == b]
            if len(subset) > 0 and 'psnr' in subset.columns:
                print(f"  ε = {b:.3f}: PSNR moyen = {subset['psnr'].mean():.2f} dB")
    
        print(f"\nSensibilité au reach (ρ):")
        for r in reaches[:5]:  # Premiers 5 reaches
            subset = exp4[exp4['reach'] == r]
            if len(subset) > 0 and 'psnr' in subset.columns:
                r_str = str(r) if isinstance(r, str) else f"{r:.2f}"
                print(f"  ρ = {r_str}: PSNR moyen = {subset['psnr'].mean():.2f} dB")
    except Exception as e:
        print(f"ERREUR lors du traitement de l'expérience 4: {e}")
        plt.close('all')
else:
    print("Aucune donnée pour l'expérience 4")
    exp4 = pd.DataFrame()  # Initialize empty dataframe

# ============================================================================
# 7. Expérience 5: Scalabilité Résolution (Détail)
# ============================================================================

print("\n" + "=" * 80)
print("7. EXPÉRIENCE 5: SCALABILITÉ RÉSOLUTION (DÉTAIL)")
print("=" * 80)
print("Objectif: Analyser en détail la scalabilité.")

# Filtrer les données de scalabilité (t=0.5, lambda=1.0, blur=0.03, reach=0.1)
exp5 = df[(df['t'] == 0.5) & (df['lambda'] == 1.0) & 
          (df['blur'] == 0.03) & (df['reach'] == 0.1)].copy()

if len(exp5) > 0:
    exp5 = exp5.sort_values('resolution')
    
    # Analyse de complexité
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Temps vs Résolution (échelle log-log pour vérifier la complexité)
    axes[0, 0].loglog(exp5['resolution'], exp5['compute_time_total'], marker='o', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Résolution (log)')
    axes[0, 0].set_ylabel('Temps de calcul (s, log)')
    axes[0, 0].set_title('Complexité Temporelle (log-log)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Ajuster une loi de puissance
    if len(exp5) > 2:
        log_res = np.log(exp5['resolution'].values)
        log_time = np.log(exp5['compute_time_total'].values)
        coeffs = np.polyfit(log_res, log_time, 1)
        fitted_time = np.exp(coeffs[1]) * exp5['resolution'] ** coeffs[0]
        axes[0, 0].loglog(exp5['resolution'], fitted_time, 'r--', 
                         label=f'Fit: O(n^{coeffs[0]:.2f})', alpha=0.7)
        axes[0, 0].legend()
        print(f"Complexité estimée: O(n^{coeffs[0]:.2f})")
    
    # Temps Sinkhorn vs Temps Interpolation
    axes[0, 1].plot(exp5['resolution'], exp5['sinkhorn_time'], marker='o', label='Sinkhorn', linewidth=2)
    axes[0, 1].plot(exp5['resolution'], exp5['interpolation_time'], marker='s', label='Interpolation', linewidth=2)
    axes[0, 1].set_xlabel('Résolution')
    axes[0, 1].set_ylabel('Temps (s)')
    axes[0, 1].set_title('Décomposition Temps Sinkhorn vs Interpolation')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Qualité vs Résolution
    axes[1, 0].plot(exp5['resolution'], exp5['psnr'], marker='o', label='PSNR', linewidth=2, color='blue')
    ax2 = axes[1, 0].twinx()
    ax2.plot(exp5['resolution'], exp5['delta_e'], marker='s', label='ΔE', linewidth=2, color='orange')
    axes[1, 0].set_xlabel('Résolution')
    axes[1, 0].set_ylabel('PSNR (dB)', color='blue')
    ax2.set_ylabel('ΔE (CIE76)', color='orange')
    axes[1, 0].set_title('Qualité vs Résolution')
    axes[1, 0].tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='orange')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Efficacité (Qualité / Temps)
    efficiency = exp5['psnr'] / exp5['compute_time_total']
    axes[1, 1].plot(exp5['resolution'], efficiency, marker='o', linewidth=2, markersize=8, color='green')
    axes[1, 1].set_xlabel('Résolution')
    axes[1, 1].set_ylabel('Efficacité (PSNR / Temps)')
    axes[1, 1].set_title('Efficacité vs Résolution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp5_scalability.png", dpi=300, bbox_inches='tight')
    output_path = RESULTS_DIR / "exp5_scalability.png"
    print(f"✓ Graphique sauvegardé: {output_path}")
    if output_path.exists():
        print(f"  Taille: {output_path.stat().st_size / 1024:.1f} KB")
    plt.close()
    
    print("\n=== RÉSULTATS EXPÉRIENCE 5 ===")
    print("Analyse de scalabilité:")
    for res in sorted(exp5['resolution'].unique()):
        res_data = exp5[exp5['resolution'] == res].iloc[0]
        efficiency = res_data['psnr'] / res_data['compute_time_total']
        print(f"  {res}×{res}: PSNR={res_data['psnr']:.2f} dB, "
              f"Temps={res_data['compute_time_total']:.3f} s, "
              f"Efficacité={efficiency:.2f} dB/s")
else:
    print("Aucune donnée pour l'expérience 5")
    exp5 = pd.DataFrame()  # Initialize empty dataframe

# ============================================================================
# 8. Expérience 6: Robustesse du Champ de Déplacement
# ============================================================================

print("\n" + "=" * 80)
print("8. EXPÉRIENCE 6: ROBUSTESSE DU CHAMP DE DÉPLACEMENT")
print("=" * 80)
print("Objectif: Analyser la régularité du champ de déplacement selon les régimes.")

# Filtrer les données de l'expérience 6 (avec métriques de smoothness)
exp6 = df[df['regime'].notna()].copy()

if len(exp6) > 0:
    regimes = exp6['regime'].unique()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Smoothness Score
    smoothness_data = [exp6[exp6['regime'] == r]['smoothness_score'].values[0] 
                      if len(exp6[exp6['regime'] == r]) > 0 else 0 
                      for r in regimes]
    axes[0, 0].bar(regimes, smoothness_data, color=['red', 'blue', 'green'])
    axes[0, 0].set_ylabel('Smoothness Score')
    axes[0, 0].set_title('Smoothness du Champ de Déplacement')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Mean Laplacian (plus petit = plus lisse)
    laplacian_data = [exp6[exp6['regime'] == r]['mean_laplacian'].values[0] 
                     if len(exp6[exp6['regime'] == r]) > 0 else 0 
                     for r in regimes]
    axes[0, 1].bar(regimes, laplacian_data, color=['red', 'blue', 'green'])
    axes[0, 1].set_ylabel('Mean Laplacian')
    axes[0, 1].set_title('Rugosité du Champ (Laplacien)')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Mean Displacement
    disp_data = [exp6[exp6['regime'] == r]['mean_displacement'].values[0] 
                if len(exp6[exp6['regime'] == r]) > 0 else 0 
                for r in regimes]
    axes[1, 0].bar(regimes, disp_data, color=['red', 'blue', 'green'])
    axes[1, 0].set_ylabel('Mean Displacement')
    axes[1, 0].set_title('Amplitude Moyenne du Déplacement')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Mean Divergence
    div_data = [exp6[exp6['regime'] == r]['mean_divergence'].values[0] 
               if len(exp6[exp6['regime'] == r]) > 0 else 0 
               for r in regimes]
    axes[1, 1].bar(regimes, div_data, color=['red', 'blue', 'green'])
    axes[1, 1].set_ylabel('Mean |Divergence|')
    axes[1, 1].set_title('Expansion/Contraction Moyenne')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp6_displacement_robustness.png", dpi=300, bbox_inches='tight')
    output_path = RESULTS_DIR / "exp6_displacement_robustness.png"
    print(f"✓ Graphique sauvegardé: {output_path}")
    if output_path.exists():
        print(f"  Taille: {output_path.stat().st_size / 1024:.1f} KB")
    plt.close()
    
    print("\n=== RÉSULTATS EXPÉRIENCE 6 ===")
    for regime in regimes:
        regime_data = exp6[exp6['regime'] == regime].iloc[0]
        print(f"\nRégime: {regime}")
        print(f"  Smoothness Score: {regime_data['smoothness_score']:.4f}")
        print(f"  Mean Laplacian: {regime_data['mean_laplacian']:.4f}")
        print(f"  Mean Displacement: {regime_data['mean_displacement']:.4f}")
        print(f"  Mean Divergence: {regime_data['mean_divergence']:.4f}")
        print(f"  Mean Curl: {regime_data['mean_curl']:.4f}")
    
    # Trouver le régime le plus lisse
    best_regime = exp6.loc[exp6['smoothness_score'].idxmax(), 'regime']
    print(f"\nRégime le plus lisse: {best_regime} (score = {exp6['smoothness_score'].max():.4f})")
else:
    print("Aucune donnée pour l'expérience 6")
    exp6 = pd.DataFrame()  # Initialize empty dataframe

# ============================================================================
# 9. Analyse Corrélations et Relations
# ============================================================================

print("\n" + "=" * 80)
print("9. ANALYSE CORRÉLATIONS ET RELATIONS")
print("=" * 80)
print("Objectif: Identifier les corrélations entre métriques et paramètres.")

# Sélectionner les colonnes numériques pertinentes
numeric_cols = ['resolution', 'lambda', 'blur', 'reach', 't', 'psnr', 'delta_e', 
                'sharpness', 'mass_error', 'coverage', 'compute_time_total', 
                'sinkhorn_time', 'interpolation_time']

# Filtrer les colonnes qui existent
numeric_cols = [col for col in numeric_cols if col in df.columns]

# Matrice de corrélation
try:
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        
        # Heatmap de corrélation
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Matrice de Corrélation entre Métriques et Paramètres')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "correlation_matrix.png", dpi=300, bbox_inches='tight')
        output_path = RESULTS_DIR / "correlation_matrix.png"
        print(f"✓ Graphique sauvegardé: {output_path}")
        if output_path.exists():
            print(f"  Taille: {output_path.stat().st_size / 1024:.1f} KB")
        plt.close()
        
        # Analyse des corrélations fortes
        print("\n=== CORRÉLATIONS FORTES (|r| > 0.7) ===")
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if not np.isnan(corr_val) and abs(corr_val) > 0.7:
                    print(f"{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_val:.3f}")
    else:
        print("Pas assez de colonnes numériques pour l'analyse de corrélation")
except Exception as e:
    print(f"ERREUR lors de l'analyse de corrélation: {e}")
    plt.close('all')

# ============================================================================
# 10. Visualisation des Images Résultantes
# ============================================================================

print("\n" + "=" * 80)
print("10. VISUALISATION DES IMAGES RÉSULTANTES")
print("=" * 80)
print("Objectif: Visualiser quelques exemples d'interpolations.")

# Sélectionner quelques expériences intéressantes à visualiser
interesting_exps = []

# Meilleur PSNR
if 'psnr' in df.columns:
    best_psnr_exp = df.loc[df['psnr'].idxmax(), 'experiment_id']
    interesting_exps.append(('Meilleur PSNR', best_psnr_exp))

# Lambda optimal
if len(exp1) > 0:
    best_lambda_exp = exp1.loc[exp1['psnr'].idxmax(), 'experiment_id']
    interesting_exps.append(('Lambda optimal', best_lambda_exp))

# 2D vs 5D à t=0.5
if len(exp2) > 0:
    exp2_t05 = exp2[exp2['t'] == 0.5]
    if len(exp2_t05[exp2_t05['lambda'] == 0.0]) > 0:
        exp_2d = exp2_t05[exp2_t05['lambda'] == 0.0].iloc[0]['experiment_id']
        interesting_exps.append(('2D (λ=0.0)', exp_2d))
    if len(exp2_t05[exp2_t05['lambda'] == 1.0]) > 0:
        exp_5d = exp2_t05[exp2_t05['lambda'] == 1.0].iloc[0]['experiment_id']
        interesting_exps.append(('5D (λ=1.0)', exp_5d))

# Visualiser les images
try:
    if len(interesting_exps) > 0:
        n_exps = len(interesting_exps)
        fig, axes = plt.subplots(1, n_exps, figsize=(5*n_exps, 5))
        if n_exps == 1:
            axes = [axes]
        
        for idx, (name, exp_id) in enumerate(interesting_exps):
            try:
                # Chercher l'image à t=0.5
                img_path = IMAGES_DIR / f"exp{int(exp_id)}_t0.500.png"
                if not img_path.exists():
                    img_path = IMAGES_DIR / f"exp{int(exp_id)}_t0.5.png"
                
                if img_path.exists():
                    img = plt.imread(str(img_path))
                    axes[idx].imshow(img)
                    axes[idx].set_title(f"{name}\nExp {int(exp_id)}")
                    axes[idx].axis('off')
                else:
                    axes[idx].text(0.5, 0.5, f"Image non trouvée\nExp {int(exp_id)}", 
                                  ha='center', va='center', transform=axes[idx].transAxes)
                    axes[idx].axis('off')
            except Exception as e:
                print(f"ERREUR lors du chargement de l'image pour exp {exp_id}: {e}")
                axes[idx].text(0.5, 0.5, f"Erreur\nExp {int(exp_id)}", 
                              ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "sample_images.png", dpi=300, bbox_inches='tight')
        output_path = RESULTS_DIR / "sample_images.png"
        print(f"✓ Graphique sauvegardé: {output_path}")
        if output_path.exists():
            print(f"  Taille: {output_path.stat().st_size / 1024:.1f} KB")
        plt.close()
    else:
        print("Aucune expérience intéressante à visualiser")
except Exception as e:
    print(f"ERREUR lors de la visualisation des images: {e}")
    plt.close('all')

# ============================================================================
# 11. Conclusions et Recommandations
# ============================================================================

print("\n" + "=" * 80)
print("11. CONCLUSIONS ET RECOMMANDATIONS")
print("=" * 80)

# 1. Lambda optimal
if len(exp1) > 0:
    best_lambda = exp1.loc[exp1['psnr'].idxmax(), 'lambda']
    print(f"\n1. LAMBDA OPTIMAL:")
    print(f"   Valeur recommandée: λ = {best_lambda:.2f}")
    print(f"   PSNR obtenu: {exp1.loc[exp1['psnr'].idxmax(), 'psnr']:.2f} dB")

# 2. Comparaison 2D vs 5D
if len(exp2) > 0:
    exp2_t05 = exp2[exp2['t'] == 0.5]
    if len(exp2_t05[exp2_t05['lambda'] == 0.0]) > 0 and len(exp2_t05[exp2_t05['lambda'] == 1.0]) > 0:
        psnr_2d = exp2_t05[exp2_t05['lambda'] == 0.0].iloc[0]['psnr']
        psnr_5d = exp2_t05[exp2_t05['lambda'] == 1.0].iloc[0]['psnr']
        print(f"\n2. COMPARAISON 2D vs 5D:")
        print(f"   2D (λ=0.0): PSNR = {psnr_2d:.2f} dB")
        print(f"   5D (λ=1.0): PSNR = {psnr_5d:.2f} dB")
        print(f"   Recommandation: {'5D' if psnr_5d > psnr_2d else '2D'} est meilleur")

# 3. Résolution optimale
if len(exp3) > 0:
    # Trouver le meilleur compromis qualité/temps
    exp3['efficiency'] = exp3['psnr'] / exp3['compute_time_total']
    best_res = exp3.loc[exp3['efficiency'].idxmax(), 'resolution']
    print(f"\n3. RÉSOLUTION OPTIMALE:")
    print(f"   Meilleur compromis qualité/temps: {int(best_res)}×{int(best_res)}")
    print(f"   Efficacité: {exp3.loc[exp3['efficiency'].idxmax(), 'efficiency']:.2f} dB/s")

# 4. Paramètres optimaux
if len(exp4) > 0:
    best_exp4 = exp4.loc[exp4['psnr'].idxmax()]
    print(f"\n4. PARAMÈTRES OPTIMAUX:")
    print(f"   ε (blur) = {best_exp4['blur']:.3f}")
    print(f"   ρ (reach) = {best_exp4['reach']:.2f if pd.notna(best_exp4['reach']) else 'balanced'}")
    print(f"   PSNR obtenu: {best_exp4['psnr']:.2f} dB")

# 5. Régime le plus lisse
if len(exp6) > 0:
    best_regime = exp6.loc[exp6['smoothness_score'].idxmax(), 'regime']
    print(f"\n5. RÉGIME LE PLUS LISSE:")
    print(f"   {best_regime}")
    print(f"   Smoothness Score: {exp6['smoothness_score'].max():.4f}")

print("\n" + "=" * 80)

# ============================================================================
# 12. Export des Résultats d'Analyse
# ============================================================================

print("\n" + "=" * 80)
print("12. EXPORT DES RÉSULTATS D'ANALYSE")
print("=" * 80)

# Créer un résumé des résultats
try:
    summary = {
        'total_experiments': len(df),
    }
    
    if 'experiment_id' in df.columns:
        summary['unique_experiments'] = df['experiment_id'].nunique()
    if 'resolution' in df.columns:
        summary['resolutions_tested'] = sorted(df['resolution'].unique()).tolist()
    if 'lambda' in df.columns:
        summary['lambdas_tested'] = sorted(df['lambda'].unique()).tolist()
    
    if len(exp1) > 0 and 'psnr' in exp1.columns and exp1['psnr'].notna().any():
        summary['best_lambda'] = float(exp1.loc[exp1['psnr'].idxmax(), 'lambda'])
        summary['best_psnr'] = float(exp1.loc[exp1['psnr'].idxmax(), 'psnr'])
    
    if len(exp4) > 0 and 'psnr' in exp4.columns and exp4['psnr'].notna().any():
        best_exp4 = exp4.loc[exp4['psnr'].idxmax()]
        if 'blur' in best_exp4.index:
            summary['best_blur'] = float(best_exp4['blur'])
        if 'reach' in best_exp4.index:
            summary['best_reach'] = float(best_exp4['reach']) if pd.notna(best_exp4['reach']) else None
    
    # Sauvegarder le résumé
    summary_path = RESULTS_DIR / "analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Résumé sauvegardé dans: {summary_path}")
    print("\nRésumé:")
    print(json.dumps(summary, indent=2))
except Exception as e:
    print(f"ERREUR lors de l'export du résumé: {e}")

print("\n" + "=" * 80)
print("ANALYSE TERMINÉE")
print("=" * 80)

# ============================================================================
# Main function wrapper for loop execution
# ============================================================================

def main():
    """Main function to run the analysis."""
    try:
        # All the code above is executed when script is run directly
        pass
    except KeyboardInterrupt:
        print("\n\nInterruption par l'utilisateur (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERREUR FATALE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
