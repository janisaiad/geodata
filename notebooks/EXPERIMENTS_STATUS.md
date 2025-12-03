# État d'implémentation des expériences 5D

## ✅ Expériences implémentées

### 1. Expérience 1: Ablation Lambda (λ)
- **Status**: ✅ Implémenté
- **Paramètres**: 
  - Résolution: 32×32
  - Blur: 0.03
  - Reach: 0.1
  - Lambda: [0.0, 0.5, 1.0, 1.5, 2.0]
  - Temps: t=0.5
- **Visualisation**: `ablation_lambda.png`

### 2. Expérience 2: Comparaison 2D vs 5D
- **Status**: ✅ Implémenté
- **Paramètres**:
  - Résolution: 32×32
  - Blur: 0.03
  - Reach: 0.1
  - Lambda: 0.0 (2D) vs 1.0 (5D)
  - Temps: [0.0, 0.05, 0.1, ..., 1.0] (17 points)
- **Visualisation**: `comparison_2d_vs_5d.png`

### 3. Expérience 3: Impact du Splatting Adaptatif (Résolution)
- **Status**: ✅ Implémenté
- **Paramètres**:
  - Lambda: 1.0
  - Blur: 0.03
  - Reach: 0.1
  - Résolutions: [32, 48, 64]
  - Temps: t=0.5
- **Visualisation**: `splatting_impact.png`
- **Note**: Le splatting adaptatif est toujours activé (code du notebook)

### 4. Expérience 4: Sensibilité aux paramètres (ε, ρ)
- **Status**: ✅ Implémenté
- **Paramètres**:
  - Résolution: 32×32
  - Lambda: 1.0
  - Blur: [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
  - Reach: [None, 0.01, 0.05, 0.1, 0.3, 0.5]
  - Temps: t=0.5
- **Visualisation**: `parameter_heatmaps.png` (PSNR et ΔE)

### 5. Expérience 5: Scalabilité Résolution
- **Status**: ✅ Implémenté
- **Paramètres**:
  - Lambda: 1.0
  - Blur: 0.03
  - Reach: 0.1
  - Résolutions: Test progressif [32, 48, 64, 96, 128]
  - Temps: t=0.5
- **Note**: Test progressif qui s'arrête si une résolution échoue

## ✅ Fonctionnalités implémentées

### Métriques calculées
- ✅ PSNR (Peak Signal-to-Noise Ratio)
- ✅ ΔE (Delta E color distance, CIE76)
- ✅ Coverage (taux de pixels non-nuls)
- ✅ Mass Error (erreur relative de conservation)
- ✅ Sharpness (variance du Laplacien)
- ⚠️ Tearing % (désactivé par défaut, nécessite get_transport_map)

### Sauvegarde
- ✅ CSV avec toutes les métriques (`all_experiments.csv`)
- ✅ Résumé des temps (`timing_summary.csv` et `.tex`)
- ✅ **TOUTES les images PNG** sauvegardées (format: `exp{id}_t{t:.3f}.png`)
- ✅ Plans de transport sauvegardés (échelle 0.01) dans `transport_plans/`
- ✅ Logs complets dans `logs/`

### Visualisations générées
- ✅ Comparaison 2D vs 5D (séquence temporelle)
- ✅ Ablation Lambda
- ✅ Impact Résolution (Splatting)
- ✅ Courbes métriques (PSNR, ΔE, Coverage, Sharpness)
- ✅ Heatmaps paramètres (ε × ρ)

### Logging
- ✅ Logs dans fichiers (`logs/experiments_YYYYMMDD_HHMMSS.log`)
- ✅ Logs console
- ✅ Logging mémoire GPU
- ✅ Logging temps de calcul (total, Sinkhorn, interpolation)

## 📊 Paramètres configurés

### Résolutions
- `[32, 48, 64, 96, 128]` (augmentées)

### Lambdas
- `[0.0, 0.5, 1.0, 1.5, 2.0]`

### Blurs (ε)
- `[0.01, 0.03, 0.05, 0.1, 0.2, 0.3]` (étendus)

### Reaches (ρ)
- `[None, 0.01, 0.05, 0.1, 0.3, 0.5]` (étendus)

### Temps d'interpolation
- `[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]` (17 points)

## 🔧 Code utilisé

- ✅ Code exact du notebook `5d_transport.ipynb` (OT5DInterpolator)
- ✅ Splatting adaptatif toujours activé
- ✅ Images classiques depuis torchvision (CIFAR-10, MNIST)

## 📁 Structure de sortie

```
refs/reports/results/5d_experiments/
├── metrics/
│   ├── all_experiments.csv
│   ├── timing_summary.csv
│   └── timing_summary.tex
├── images/
│   ├── exp{id}_t{t:.3f}.png (TOUTES les images)
│   ├── comparison_2d_vs_5d.png
│   ├── ablation_lambda.png
│   ├── splatting_impact.png
│   ├── metric_curves.png
│   └── parameter_heatmaps.png
├── transport_plans/
│   └── plan_exp{id}_*.pt
└── logs/
    └── experiments_YYYYMMDD_HHMMSS.log
```

## ✅ Tout est implémenté et prêt à l'emploi !

