# Configuration des Expériences Pokemon - Salameche → Strawberry

## Modifications Effectuées

### Images Utilisées
- **Source**: `salameche.webp` (Charmander - pixel art)
- **Target**: `strawberry.jpg` (Fraise - photo réaliste)
- **Paire**: `salameche_strawberry`

### Changements dans le Code

1. **`experiments_pokemon_large_scale.py`**:
   - Mise à jour de `source_image_name`: `"salameche.webp"`
   - Mise à jour de `target_image_name`: `"strawberry.jpg"`
   - Mise à jour de `image_pair_name`: `"salameche_strawberry"`
   - Mise à jour de `output_dir`: `pokemon_experiments_salameche_strawberry`

2. **`analyze_pokemon_experiments.py`**:
   - Mise à jour du chemin `RESULTS_DIR` pour pointer vers le nouveau répertoire

## Structure des Expériences

Le script exécute 6 types d'expériences:

1. **Expérience 1: Ablation Lambda (λ)**
   - Teste différentes valeurs de λ pour trouver l'optimal
   - Paramètres: t=0.5, res=64, blur=0.03, reach=0.1
   - Lambdas: [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

2. **Expérience 2: Comparaison 2D vs 5D**
   - Compare transport spatial (2D) vs spatial+couleur (5D)
   - Paramètres: res=64, blur=0.03, reach=0.1
   - Lambdas: [0.0, 1.0]
   - Temps: séquence complète [0.0, 0.05, ..., 1.0]

3. **Expérience 3: Impact du Splatting Adaptatif**
   - Évalue l'effet de la résolution sur la qualité
   - Paramètres: t=0.5, lambda=1.0, blur=0.03, reach=0.1
   - Résolutions: [32, 48, 64, 96, 128, 160, 192, 256]

4. **Expérience 4: Sensibilité aux Paramètres (ε, ρ)**
   - Cartographie l'impact des paramètres de régularisation
   - Paramètres: t=0.5, res=64, lambda=1.0
   - Blurs: [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]
   - Reaches: [None, 0.01, 0.02, 0.05, 0.1, 0.15, 0.3, 0.5, 0.7, 1.0]

5. **Expérience 5: Scalabilité**
   - Analyse détaillée de la scalabilité en résolution
   - Paramètres: t=0.5, lambda=1.0, blur=0.03, reach=0.1
   - Résolutions: [32, 48, 64, 96, 128, 160, 192, 256]

6. **Expérience 6: Robustesse du Champ de Déplacement**
   - Analyse la régularité du champ selon différents régimes
   - Teste différents paramètres pour évaluer la smoothness

## Métriques Calculées

Pour chaque expérience, les métriques suivantes sont calculées:

- **PSNR** (Peak Signal-to-Noise Ratio): Qualité d'image
- **ΔE (CIE76)**: Distance de couleur perceptuelle
- **Sharpness**: Netteté de l'image
- **Mass Error**: Erreur de conservation de masse
- **Coverage**: Couverture de l'image
- **Tearing %**: Pourcentage de pixels vides
- **Temps de calcul**: Temps total, Sinkhorn, interpolation
- **Mémoire GPU**: Allocation maximale
- **Métriques de smoothness**: Displacement, divergence, curl, Laplacian

## Lancement des Expériences

### Option 1: Script Bash
```bash
cd /Data/janis.aiad/geodata/notebooks
./run_pokemon_experiments.sh
```

### Option 2: Python Direct
```bash
cd /Data/janis.aiad/geodata/notebooks
python experiments_pokemon_large_scale.py
```

## Structure des Résultats

Les résultats sont sauvegardés dans:
```
/Data/janis.aiad/geodata/refs/reports/results/pokemon_experiments_salameche_strawberry/
├── metrics/
│   └── all_experiments.csv          # Toutes les métriques
├── images/
│   └── exp{id}_t{time}.png          # Images interpolées
├── transport_plans/
│   └── plan_exp{id}_*.pt            # Plans de transport
├── logs/
│   └── pokemon_experiments_*.log    # Logs détaillés
└── analysis_summary.json            # Résumé (après analyse)
```

## Analyse des Résultats

Après l'exécution des expériences, lancer l'analyse:

```bash
cd /Data/janis.aiad/geodata/notebooks
python analyze_pokemon_experiments.py
```

Cela génère:
- Graphiques de toutes les expériences
- Matrice de corrélation
- Visualisations des meilleures interpolations
- Résumé JSON avec recommandations

## Notes Importantes

1. **Mémoire GPU**: Les expériences à haute résolution (256×256) peuvent nécessiter beaucoup de mémoire GPU. Le script vérifie automatiquement et peut sauter certaines expériences si la mémoire est insuffisante.

2. **Temps d'exécution**: L'ensemble des expériences peut prendre plusieurs heures selon le matériel. Le script sauvegarde périodiquement les résultats (toutes les 5 minutes).

3. **Interruption**: En cas d'interruption (Ctrl+C), les résultats sont automatiquement sauvegardés.

4. **Reprise**: Le script peut être relancé - il écrasera les résultats existants. Pour reprendre depuis un point spécifique, modifier les grilles de paramètres dans `PokemonExperimentConfig`.

## Configuration Avancée

Pour modifier les paramètres des expériences, éditer la classe `PokemonExperimentConfig` dans `experiments_pokemon_large_scale.py`:

```python
@dataclass
class PokemonExperimentConfig:
    resolutions: List[int] = field(default_factory=lambda: [32, 64, 128])
    lambdas: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 2.0])
    # ... etc
```

## Dépannage

### Erreur: Images non trouvées
- Vérifier que `salameche.webp` et `strawberry.jpg` existent dans `/Data/janis.aiad/geodata/data/pixelart/images/`
- Le script essaie automatiquement différentes extensions (.jpg, .webp, .png)

### Erreur: Mémoire GPU insuffisante
- Réduire la liste des résolutions dans `resolutions`
- Réduire `max_resolution` dans la config
- Le script sautera automatiquement les expériences trop grandes

### Erreur: Segmentation fault
- Vérifier les logs dans `logs/pokemon_experiments_*.log`
- Le script a des mécanismes de récupération, mais certains crashes peuvent nécessiter une reprise manuelle

