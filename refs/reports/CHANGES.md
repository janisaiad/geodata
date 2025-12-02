# Modifications du Rapport LaTeX

## Changements Majeurs

### 1. Titre et Abstract Refondus
- **Nouveau titre** : "Gaussian Splatting Adaptatif pour Transport Optimal d'Images"
- **Sous-titre** : "Reconstruction Géométrique Sans Tearing et Transport 5D Joint Spatial-Couleur"
- **Abstract** : Mise en avant des 2 contributions majeures

### 2. Structure Réorganisée

**Section 1 : Introduction et Motivation**
- Problème 1 : Le Tearing (avec exemple MNIST)
- Problème 2 : Transport RGB marginal
- Contributions du travail

**Section 2 : Cadre Théorique** (inchangé)
- Divergences de Csiszár et UOT
- Dualité et Algorithme de Sinkhorn
- Divergence de Sinkhorn Débiaisée

**Section 3 : Implémentation** (enrichi)
- Architecture GeomLoss + KeOps (nouvelle sous-section)
- Reconstruction du plan π
- Gestion des régimes

**Section 4 : Interpolation Géodésique** (nouvelle)
- Théorème de McCann (displacement interpolation)
- Formulation duale avec plan optimal
- Défi projection Lagrangien-Eulérien

**Section 5 : Contribution 1 - Gaussian Splatting Adaptatif**
- Origine mathématique du tearing (Jacobien, SVD)
- Templates géométriques (expansion, rotation, anisotropie)
- Justification rigoureuse de l'heuristique σ(t)
- Condition de Nyquist-Shannon discrète
- Boost temporel parabolique 4t(1-t)
- Conservation de masse exacte
- Analyse quantitative (métriques)

**Section 6 : Contribution 2 - Transport 5D**
- Motivation (3 défauts du transport marginal)
- Formulation 5D RGB (coût hybride, choix de λ)
- Transport 3D pour MNIST (x,y,i)
- Expérience "1" → "0"
- Transport 5D sur images 16×16
- Tableau comparatif quantitatif

**Section 7 : Expériences** (adapté)
- Focus sur Unbalanced vs Balanced
- Rayon de transport √(2ρ)

**Section 8 : Discussion et Limites**
- Complexité computationnelle 5D
- Stratégies multi-échelles
- Limites du splatting
- Extensions futures

**Section 9 : Conclusion**
- Récapitulatif des 2 contributions
- Impact et applications

### 3. Ajouts Théoriques

#### Equations Importantes Ajoutées
- Théorème de McCann (géodésique W₂)
- Jacobien du transport : ∇X_t = (1-t)I + t∇T
- Critère de tearing : √|det(∇X_t)| > Δ_grid/d̄₀
- Condition Nyquist-Shannon : σ(t) ≥ Δ_grid/2 · √|det(∇X_t)|
- Coût 5D : C = ||x-x'||² + λ||c-c'||²
- Rayon de transport effectif : √(2ρ)

#### Justifications Géométriques
1. **Boost parabolique 4t(1-t)** :
   - Incertitude maximale à t=0.5
   - Non-unicité des trajectoires
   - Annulation aux bords (conditions marginales)

2. **Choix de λ (5D)** :
   - Ratio de variance
   - Échelle perceptuelle (ΔE = 1)
   - Compromis pratique λ ∈ [0.5, 2]

3. **Conservation de masse** :
   - Renormalisation post-splatting
   - Garantie ∑I_final = ∫dπ

### 4. Figures et Tableaux Requis

#### Figures à Générer
1. `template_isotropic.png` - Expansion isotrope
2. `template_rotation.png` - Rotation pure
3. `template_anisotropic.png` - Déformation anisotrope
4. `sigma_evolution.png` - Courbe σ(t) selon Eq.
5. `tearing_comparison.png` - Avec/sans splatting
6. `mnist_1_to_0_comparison.png` - Transport 2D vs 3D vs 3D+splat
7. `5d_marginal.png`, `5d_lambda1.png`, `5d_lambda1_splat.png` - Comparaison 5D
8. `image_5838f6.png` - Balanced vs Unbalanced (déjà référencé)
9. `image_56e3bb.png` - Tearing résolution (déjà référencé)

#### Tableaux à Compléter
1. Tableau métriques tearing (Coverage, Mass Error, Sharpness)
2. Tableau résultats 5D (PSNR, ΔE, Tearing%)
3. Tableau complexité 5D selon résolution

### 5. Bibliographie Complétée
- GeomLoss (Feydy et al. 2019)
- KeOps (Charlier et al. 2021)

## Prochaines Étapes

### Implémentation Code
1. Créer classe `Transport3D_MNIST` dans notebook
2. Créer classe `Transport5D_RGB` pour images 16×16
3. Implémenter estimation automatique de f_exp(t)
4. Ajouter métriques quantitatives (Coverage, PSNR, ΔE)

### Génération Figures
1. Script pour templates géométriques
2. Visualisation courbe σ(t)
3. Expériences MNIST "1"→"0"
4. Downsampling Salamèche-Fraise à 16×16
5. Comparaisons visuelles côte-à-côte

### Validation Expérimentale
1. Mesures quantitatives sur 10 paires MNIST
2. Mesures sur 5 paires images couleur
3. Ablation study λ ∈ [0.1, 0.5, 1.0, 2.0, 5.0]
4. Ablation study γ ∈ [0, 0.1, 0.2, 0.3, 0.5]

