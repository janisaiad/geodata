#!/usr/bin/env python3
"""
Création de GIFs et Visualisations A Posteriori

Ce script génère des GIFs animés et des visualisations à partir des résultats
des expériences de transport optimal 5D.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import imageio
from PIL import Image
import torch
import torchvision
from typing import List, Dict, Optional, Tuple
import warnings
import re
from collections import defaultdict

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

# Chemins
RESULTS_DIR = Path("/Data/janis.aiad/geodata/refs/reports/results/pokemon_experiments_salameche_strawberry")
IMAGES_DIR = RESULTS_DIR / "images"
GIFS_DIR = RESULTS_DIR / "gifs"
VIZ_DIR = RESULTS_DIR / "visualizations"
CSV_PATH = RESULTS_DIR / "metrics" / "all_experiments.csv"

# Créer les répertoires de sortie
GIFS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("CRÉATION DE GIFS ET VISUALISATIONS")
print("=" * 80)
print(f"Répertoire des images: {IMAGES_DIR}")
print(f"Répertoire de sortie GIFs: {GIFS_DIR}")
print(f"Répertoire de sortie visualisations: {VIZ_DIR}")
print("")

def load_experiment_images(exp_id: int) -> List[Tuple[float, Path]]:
    """Charge toutes les images d'une expérience, triées par temps t."""
    pattern = f"exp{exp_id}_t*.png"
    image_files = sorted(IMAGES_DIR.glob(pattern))
    
    images_with_t = []
    for img_path in image_files:
        # Extraire le temps t du nom de fichier (exp123_t0.500.png -> 0.500)
        match = re.search(r'_t([\d.]+)\.png', img_path.name)
        if match:
            t = float(match.group(1))
            images_with_t.append((t, img_path))
    
    # Trier par temps
    images_with_t.sort(key=lambda x: x[0])
    return images_with_t

def create_gif(exp_id: int, images: List[Tuple[float, Path]], duration: float = 0.2, loop: int = 0) -> Path:
    """Crée un GIF à partir d'une liste d'images."""
    if not images:
        print(f"  Aucune image trouvée pour l'expérience {exp_id}")
        return None
    
    gif_path = GIFS_DIR / f"exp{exp_id}_interpolation.gif"
    
    # Charger les images
    frames = []
    for t, img_path in images:
        try:
            img = Image.open(img_path)
            frames.append(img)
        except Exception as e:
            print(f"  Erreur chargement image {img_path.name}: {e}")
            continue
    
    if not frames:
        print(f"  Aucune image valide pour l'expérience {exp_id}")
        return None
    
    # Créer le GIF
    try:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration * 1000,  # en millisecondes
            loop=loop
        )
        print(f"  ✓ GIF créé: {gif_path.name} ({len(frames)} frames, {duration}s/frame)")
        return gif_path
    except Exception as e:
        print(f"  Erreur création GIF pour exp {exp_id}: {e}")
        return None

def create_comparison_grid(exp_ids: List[int], labels: List[str], title: str, output_path: Path, t: float = 0.5):
    """Crée une grille comparant plusieurs expériences à un temps t donné."""
    n_exps = len(exp_ids)
    if n_exps == 0:
        return
    
    # Calculer la grille
    n_cols = min(4, n_exps)
    n_rows = int(np.ceil(n_exps / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1:
        axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (exp_id, label) in enumerate(zip(exp_ids, labels)):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Chercher l'image la plus proche de t
        images = load_experiment_images(exp_id)
        if not images:
            ax.text(0.5, 0.5, f"Exp {exp_id}\nNon trouvée", ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue
        
        # Trouver l'image la plus proche de t
        closest_img = min(images, key=lambda x: abs(x[0] - t))
        img_path = closest_img[1]
        
        try:
            img = plt.imread(str(img_path))
            ax.imshow(img)
            ax.set_title(f"{label}\nExp {exp_id}, t={closest_img[0]:.2f}", fontsize=14)
            ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, f"Erreur\n{str(e)[:30]}", ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    # Cacher les axes inutilisés
    for idx in range(n_exps, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.axis('off')
    
    plt.suptitle(title, fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Grille sauvegardée: {output_path.name}")
    plt.close()

def create_lambda_comparison(df: pd.DataFrame):
    """Crée une visualisation comparant différentes valeurs de lambda."""
    print("\n" + "=" * 80)
    print("CRÉATION DE LA COMPARAISON LAMBDA")
    print("=" * 80)
    
    # Filtrer les données à t=0.5, res=64, blur=0.03, reach=0.1
    exp_data = df[(df['t'] == 0.5) & (df['resolution'] == 64) & 
                  (df['blur'] == 0.03) & (df['reach'] == 0.1)].copy()
    
    if len(exp_data) == 0:
        print("  Aucune donnée disponible pour la comparaison lambda")
        return
    
    # Grouper par lambda
    lambdas = sorted(exp_data['lambda'].unique())
    exp_ids = []
    labels = []
    
    for lam in lambdas:
        exp_subset = exp_data[exp_data['lambda'] == lam]
        if len(exp_subset) > 0:
            exp_id = exp_subset.iloc[0]['experiment_id']
            exp_ids.append(int(exp_id))
            labels.append(f"λ={lam:.1f}")
    
    if exp_ids:
        output_path = VIZ_DIR / "lambda_comparison.png"
        create_comparison_grid(exp_ids, labels, "Comparaison Lambda (λ) - t=0.5", output_path, t=0.5)
        
        # Créer aussi un GIF pour chaque lambda
        print("\n  Création de GIFs individuels pour chaque lambda:")
        for exp_id, label in zip(exp_ids, labels):
            images = load_experiment_images(exp_id)
            if images:
                create_gif(exp_id, images, duration=0.15)

def create_resolution_comparison(df: pd.DataFrame):
    """Crée une visualisation comparant différentes résolutions."""
    print("\n" + "=" * 80)
    print("CRÉATION DE LA COMPARAISON RÉSOLUTION")
    print("=" * 80)
    
    # Filtrer les données à t=0.5, lambda=1.0, blur=0.03, reach=0.1
    exp_data = df[(df['t'] == 0.5) & (df['lambda'] == 1.0) & 
                  (df['blur'] == 0.03) & (df['reach'] == 0.1)].copy()
    
    if len(exp_data) == 0:
        print("  Aucune donnée disponible pour la comparaison résolution")
        return
    
    # Grouper par résolution
    resolutions = sorted(exp_data['resolution'].unique())
    exp_ids = []
    labels = []
    
    for res in resolutions:
        exp_subset = exp_data[exp_data['resolution'] == res]
        if len(exp_subset) > 0:
            exp_id = exp_subset.iloc[0]['experiment_id']
            exp_ids.append(int(exp_id))
            labels.append(f"{res}×{res}")
    
    if exp_ids:
        output_path = VIZ_DIR / "resolution_comparison.png"
        create_comparison_grid(exp_ids, labels, "Comparaison Résolution - t=0.5", output_path, t=0.5)

def create_2d_vs_5d_comparison(df: pd.DataFrame):
    """Crée une visualisation comparant 2D vs 5D."""
    print("\n" + "=" * 80)
    print("CRÉATION DE LA COMPARAISON 2D vs 5D")
    print("=" * 80)
    
    # Filtrer les données à t=0.5, res=64, blur=0.03, reach=0.1
    exp_data = df[(df['t'] == 0.5) & (df['resolution'] == 64) & 
                  (df['blur'] == 0.03) & (df['reach'] == 0.1) &
                  (df['lambda'].isin([0.0, 1.0]))].copy()
    
    if len(exp_data) == 0:
        print("  Aucune donnée disponible pour la comparaison 2D vs 5D")
        return
    
    exp_ids = []
    labels = []
    
    for lam in [0.0, 1.0]:
        exp_subset = exp_data[exp_data['lambda'] == lam]
        if len(exp_subset) > 0:
            exp_id = exp_subset.iloc[0]['experiment_id']
            exp_ids.append(int(exp_id))
            labels.append("2D (λ=0.0)" if lam == 0.0 else "5D (λ=1.0)")
    
    if len(exp_ids) == 2:
        output_path = VIZ_DIR / "2d_vs_5d_comparison.png"
        create_comparison_grid(exp_ids, labels, "Comparaison 2D vs 5D - t=0.5", output_path, t=0.5)
        
        # Créer des GIFs pour les deux
        print("\n  Création de GIFs pour 2D et 5D:")
        for exp_id, label in zip(exp_ids, labels):
            images = load_experiment_images(exp_id)
            if images:
                gif_path = create_gif(exp_id, images, duration=0.15)
                if gif_path:
                    # Renommer avec un nom plus descriptif
                    new_name = f"{label.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')}_interpolation.gif"
                    new_path = GIFS_DIR / new_name
                    gif_path.rename(new_path)
                    print(f"    Renommé: {new_path.name}")

def create_timeline_comparison(exp_id: int, n_frames: int = 10):
    """Crée une visualisation montrant l'évolution temporelle d'une expérience."""
    images = load_experiment_images(exp_id)
    if not images or len(images) < 2:
        return
    
    # Sélectionner n_frames uniformément réparties
    indices = np.linspace(0, len(images) - 1, n_frames, dtype=int)
    selected_images = [images[i] for i in indices]
    
    n_cols = min(5, n_frames)
    n_rows = int(np.ceil(n_frames / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    if n_rows == 1:
        axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (t, img_path) in enumerate(selected_images):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        try:
            img = plt.imread(str(img_path))
            ax.imshow(img)
            ax.set_title(f"t={t:.2f}", fontsize=12)
            ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, f"Erreur", ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    # Cacher les axes inutilisés
    for idx in range(n_frames, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.axis('off')
    
    plt.suptitle(f"Évolution Temporelle - Expérience {exp_id}", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = VIZ_DIR / f"timeline_exp{exp_id}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Timeline sauvegardée: {output_path.name}")
    plt.close()

def create_all_gifs(df: pd.DataFrame, max_experiments: Optional[int] = None):
    """Crée des GIFs pour toutes les expériences disponibles."""
    print("\n" + "=" * 80)
    print("CRÉATION DE TOUS LES GIFS")
    print("=" * 80)
    
    # Obtenir toutes les expériences uniques
    if 'experiment_id' in df.columns:
        exp_ids = sorted(df['experiment_id'].unique())
    else:
        # Extraire depuis les noms de fichiers
        image_files = list(IMAGES_DIR.glob("exp*_t*.png"))
        exp_ids = sorted(set(int(re.search(r'exp(\d+)_', f.name).group(1)) for f in image_files if re.search(r'exp(\d+)_', f.name)))
    
    if max_experiments:
        exp_ids = exp_ids[:max_experiments]
    
    print(f"  Nombre d'expériences à traiter: {len(exp_ids)}")
    
    successful = 0
    failed = 0
    
    for exp_id in exp_ids:
        images = load_experiment_images(exp_id)
        if images:
            gif_path = create_gif(exp_id, images, duration=0.15)
            if gif_path:
                successful += 1
            else:
                failed += 1
        else:
            failed += 1
    
    print(f"\n  Résumé: {successful} GIFs créés, {failed} échecs")

def main():
    """Fonction principale."""
    # Charger les données
    print("Chargement des données...")
    if not CSV_PATH.exists():
        print(f"ERREUR: Le fichier CSV {CSV_PATH} n'existe pas!")
        print("Tentative de création de GIFs sans données CSV...")
        df = pd.DataFrame()
    else:
        df = pd.read_csv(CSV_PATH)
        print(f"✓ {len(df)} lignes chargées")
        if 'experiment_id' in df.columns:
            print(f"✓ {df['experiment_id'].nunique()} expériences uniques")
    
    # Vérifier que le répertoire images existe
    if not IMAGES_DIR.exists():
        print(f"ERREUR: Le répertoire {IMAGES_DIR} n'existe pas!")
        return
    
    # Compter les images disponibles
    image_files = list(IMAGES_DIR.glob("exp*_t*.png"))
    print(f"✓ {len(image_files)} images trouvées dans {IMAGES_DIR}")
    
    # Créer les visualisations
    if len(df) > 0:
        create_lambda_comparison(df)
        create_resolution_comparison(df)
        create_2d_vs_5d_comparison(df)
        
        # Créer des timelines pour quelques expériences intéressantes
        print("\n" + "=" * 80)
        print("CRÉATION DE TIMELINES")
        print("=" * 80)
        
        # Meilleur PSNR
        if 'psnr' in df.columns:
            best_exp = int(df.loc[df['psnr'].idxmax(), 'experiment_id'])
            print(f"\n  Timeline pour meilleur PSNR (exp {best_exp}):")
            create_timeline_comparison(best_exp, n_frames=10)
        
        # Lambda optimal
        exp_data = df[(df['t'] == 0.5) & (df['resolution'] == 64) & 
                      (df['blur'] == 0.03) & (df['reach'] == 0.1)]
        if len(exp_data) > 0 and 'psnr' in exp_data.columns:
            best_lambda_exp = int(exp_data.loc[exp_data['psnr'].idxmax(), 'experiment_id'])
            print(f"\n  Timeline pour lambda optimal (exp {best_lambda_exp}):")
            create_timeline_comparison(best_lambda_exp, n_frames=10)
    
    # Créer tous les GIFs (limiter à 50 pour éviter de surcharger)
    create_all_gifs(df, max_experiments=50)
    
    print("\n" + "=" * 80)
    print("TERMINÉ")
    print("=" * 80)
    print(f"GIFs sauvegardés dans: {GIFS_DIR}")
    print(f"Visualisations sauvegardées dans: {VIZ_DIR}")

if __name__ == "__main__":
    main()

