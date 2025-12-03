# Guide d'exécution des expériences 5D

## Commande principale

Pour lancer toutes les expériences :

```bash
cd /home/janis/4A/geodata
python notebooks/experiments_5d_massive.py
```

Ou avec `uv` si vous utilisez uv :

```bash
cd /home/janis/4A/geodata
uv run python notebooks/experiments_5d_massive.py
```

## Ce qui sera exécuté

Le script va :

1. **Tests de validation** : Vérifier que toutes les fonctions fonctionnent
2. **Expérience 1** : Ablation Lambda (λ) - 5 configurations
3. **Expérience 2** : Comparaison 2D vs 5D - séquence temporelle complète
4. **Expérience 3** : Impact du Splatting Adaptatif - 3 résolutions
5. **Expérience 4** : Sensibilité aux paramètres (ε, ρ) - grille 3×3
6. **Expérience 5** : Scalabilité - test progressif 16→32→64→128→256

## Résultats générés

Tous les résultats sont sauvegardés dans `refs/reports/results/5d_experiments/` :

- **`logs/experiments_YYYYMMDD_HHMMSS.log`** : Log complet de l'exécution
- **`metrics/all_experiments.csv`** : Toutes les métriques (PSNR, ΔE, temps, mémoire, etc.)
- **`metrics/timing_summary.csv`** : Résumé des temps par résolution
- **`metrics/timing_summary.tex`** : Table LaTeX pour le rapport
- **`images/`** : Toutes les images générées
- **`transport_plans/`** : Plans de transport sauvegardés (échelle 0.01)

## Test progressif des résolutions

Le script teste automatiquement les résolutions de manière progressive :
- Commence par 16×16 et 32×32
- Si ça fonctionne, teste 64×64
- Si ça fonctionne, teste 128×128
- Si ça fonctionne, teste 256×256

Si une résolution échoue (mémoire insuffisante), le script s'arrête et log l'erreur.

## Monitoring

Le script log automatiquement :
- Temps de calcul (total, Sinkhorn, interpolation)
- Utilisation mémoire GPU (allouée, réservée, max)
- Progression des expériences
- Erreurs éventuelles

Tout est visible dans le fichier de log et sur la console.

## Estimation du temps

Avec GPU 16GB :
- 16×16 : ~1-2 sec par expérience
- 32×32 : ~2-5 sec par expérience
- 64×64 : ~5-15 sec par expérience
- 128×128 : ~15-30 sec par expérience
- 256×256 : ~30-60 sec par expérience

**Total estimé** : 30-60 minutes pour toutes les expériences.

