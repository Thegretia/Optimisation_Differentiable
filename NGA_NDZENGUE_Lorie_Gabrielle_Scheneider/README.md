# Devoir d'optimisation différentiable — NGA NDZENGUE Lorie Gabrielle Scheneider

Implémentation *from scratch* en Python pur (sans numpy ni scipy) de :

1. Descente de gradient à pas fixe
2. Descente de gradient à pas optimal (variable)
3. Gradient conjugué (Fletcher-Reeves et Polak-Ribière)
4. Quasi-Newton (BFGS et DFP)

Tout est dans le notebook `Algorithmes_optimisation.ipynb` : explications, code et tests.

## Fonctions test

- Quadratique $f(x) = \frac{1}{2} x^T A x - b^T x$ en dimension 3
- Rosenbrock $f(x, y) = 100 (y - x^2)^2 + (1 - x)^2$, départ $(-1.2, 1)$

## Résultats (nombre d'itérations, $\|\nabla f\| < 10^{-6}$)

| Méthode | Quadratique | Rosenbrock |
|---|---|---|
| Pas fixe | 30 ($\alpha = 0.3$) | ~ 32 000 ($\alpha = 10^{-3}$) |
| Pas optimal | 27 | ~ 15 000 |
| Gradient conjugué | 3 | 34 |
| Quasi-Newton (BFGS) | 9 | 34 |

## Exécution

```bash
pip install jupyter matplotlib
jupyter notebook Algorithmes_optimisation.ipynb
```

`matplotlib` sert uniquement au graphique de comparaison final.
