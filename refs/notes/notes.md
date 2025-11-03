## explication du transport optimal non équilibré

le **transport optimal non équilibré** (unbalanced optimal transport) généralise le transport optimal classique en permettant la **création et destruction de masse**.

### différences avec le sinkhorn classique

1. **sinkhorn équilibré** : conserve la masse totale
   - $\pi \mathbf{1} = \alpha$ et $\pi^T \mathbf{1} = \beta$
   - adapté quand les distributions ont la même masse totale

2. **sinkhorn non équilibré** : permet des changements de masse
   - paramètre $\rho$ (reach) contrôle la pénalité pour création/destruction de masse
   - formulation : $\min_{\pi} \langle C, \pi \rangle + \varepsilon H(\pi) + \rho \text{KL}(\pi \mathbf{1} \| \alpha) + \rho \text{KL}(\pi^T \mathbf{1} \| \beta)$

### paramètres utilisés

- **blur** ($\varepsilon = 0.05$) : régularisation entropique, contrôle la "fluidité" du transport
- **reach** ($\rho = 0.5$) : pénalité pour création/destruction de masse
  - $\rho \to 0$ : très permissif (beaucoup de création/destruction)
  - $\rho \to \infty$ : converge vers le transport équilibré classique
- **p = 2** : distance euclidienne au carré dans l'espace $(x, y, r, g, b)$

### interprétation

la distance calculée mesure le coût minimal pour transformer le tenseur aléatoire en l'image cible, en tenant compte :
- du déplacement spatial des pixels (coordonnées $x, y$)
- du changement de couleur (canaux $r, g, b$)
- de la possibilité de créer/détruire de la masse (pixels)
