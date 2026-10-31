# Optimisation différentiable

Implémentation **from scratch** (NumPy uniquement) des principaux algorithmes
de minimisation d'une fonction différentiable f : R^n -> R.

## Contenu

| Fichier | Algorithme |
|---|---|
| `AlgoPasFixe.py` | Descente de gradient à pas fixe |
| `AlgoPasOptimal.py` | Descente de gradient à pas optimal (formule exacte et section dorée) |
| `AlgoPasConjugue.py` | Gradient conjugué (cas quadratique, Fletcher-Reeves, Polak-Ribière+) |
| `AlgoNewton.py` | Méthode de Newton (pure et globalisée avec Armijo) |
| `AlgoQuasiNewton.py` | Quasi-Newton : BFGS et DFP |

Chaque fichier est autonome : il contient l'algorithme et un exemple d'utilisation
(`python AlgoNewton.py`).

## Utilisation

```bash
pip install numpy
python AlgoPasFixe.py
```

Chaque fonction retourne `(x, k, historique)` : le point obtenu, le nombre
d'itérations et la liste des points parcourus.

```python
import numpy as np
from AlgoQuasiNewton import quasi_newton

f = lambda x: (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
grad = lambda x: np.array([-2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
                           200*(x[1] - x[0]**2)])
x, k, _ = quasi_newton(f, grad, np.array([-1.2, 1.0]), methode="BFGS")
```

## Schéma général

Tous les algorithmes suivent x_{k+1} = x_k + alpha_k d_k, avec d_k la direction
de descente et alpha_k le pas.

| Méthode | Direction d_k | Pas alpha_k |
|---|---|---|
| Pas fixe | -g_k | constant |
| Pas optimal | -g_k | argmin f(x_k + alpha d_k) |
| Gradient conjugué | -g_k + beta_{k-1} d_{k-1} | optimal (quadratique) ou Armijo |
| Newton | solution de H_k d = -g_k | 1 ou Armijo |
| Quasi-Newton | -W_k g_k (W_k ~ H_k^-1) | Armijo |

(g_k = grad f(x_k), H_k = hessienne de f en x_k.)

## Fonctions de test

- **Quadratique** : f(x) = 1/2 x^T A x - b^T x avec A = [[4, 1], [1, 3]], b = (1, 2).
- **Rosenbrock** : f(x, y) = (1 - x)^2 + 100 (y - x^2)^2, minimum en (1, 1), départ (-1.2, 1).

## Résultats observés

| Méthode | Quadratique | Rosenbrock |
|---|---|---|
| Pas fixe (1/lambda_max) | 30 it. | non testé (très lent) |
| Pas optimal | 14 it. | non testé (très lent) |
| Gradient conjugué | 2 it. (= n) | 191 it. (FR), 1126 it. (PR+) |
| Newton | 1 it. | 6 it. (pas = 1), 21 it. (Armijo) |
| BFGS | 7 it. | 34 it. |
| DFP | 6 it. | 48 it. |

Observations :
- Pour le pas fixe, la convergence exige 0 < alpha < 2/lambda_max ; au-delà, l'algorithme diverge.
- Le gradient conjugué converge en au plus n itérations sur une quadratique.
- Newton converge quadratiquement près de la solution, mais coûte cher (hessienne + système linéaire).
- BFGS et DFP approchent Newton sans calculer la hessienne.

## Licence

MIT, voir `LICENSE`.
