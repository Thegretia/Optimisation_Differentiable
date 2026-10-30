# Algorithmes d'optimisation sans contrainte (Python)

Implémentation de plusieurs méthodes de minimisation, en n'utilisant que **numpy** et **sympy**.

## Organisation du dépôt

```
.
├── README.md
├── Devoir_AN/                  ← premiers devoirs (exercice du cours)
│   ├── ExoAlgoNewton.py
│   ├── ExoAlgoPasFixe.py
│   ├── ExoAlgoPasOptimale.py
│   └── ExoQuasiNewton.py
├── AlgoNewton.py               ← versions génériques (f fournie par l'utilisateur)
├── AlgoPasFixe.py
├── AlgoPasConjugue.py
└── AlgoQuasiNewton.py
```

- **`Devoir_AN/`** : les premiers devoirs. Chaque script résout l'exercice du cours avec une méthode donnée, et affiche le tableau des itérations et le résultat final.
- **Fichiers à la racine** : versions génériques des algorithmes. On leur fournit n'importe quelle fonction `f` et un point initial.

## Fonction étudiée (exercice du cours)

```
f(x, y, z) = 6x² + 6xy + 3y² + y³ − z³/3 + z⁴/4
X(0) = (−0,6637 ; 0,1758 ; 1,5778)      tolérance : ‖∇f‖ < 10⁻⁵
```

Le minimum local atteint par toutes les méthodes est **X\* = (0, 0, 1)**, avec **f(X\*) = −1/12 ≈ −0,083333**.

> `f` n'est pas bornée inférieurement (terme `y³`) : seul un minimum *local* existe.

## Prérequis et installation

- Python 3.9 ou plus récent
- numpy et sympy

```bash
pip install numpy sympy
```

## Lancer les codes

Chaque fichier se lance directement et exécute une démonstration :

```bash
python Devoir_AN/ExoAlgoNewton.py
python Devoir_AN/ExoAlgoPasFixe.py
python Devoir_AN/ExoAlgoPasOptimale.py
python Devoir_AN/ExoQuasiNewton.py

python AlgoNewton.py
python AlgoPasFixe.py
python AlgoPasConjugue.py
python AlgoQuasiNewton.py
```

## Méthodes et résultats sur l'exercice

| Méthode | Fichier du devoir (`Devoir_AN/`) | Fichier générique | Itérations |
|---|---|---|---|
| Newton | `ExoAlgoNewton.py` | `AlgoNewton.py` | 5 |
| Gradient à pas fixe (α = 0,05) | `ExoAlgoPasFixe.py` | `AlgoPasFixe.py` | 197 |
| Gradient à pas optimal | `ExoAlgoPasOptimale.py` | — | 31 |
| Gradient conjugué (Polak-Ribière+) | — | `AlgoPasConjugue.py` | 14 |
| Quasi-Newton (BFGS) | `ExoQuasiNewton.py` | `AlgoQuasiNewton.py` | 13 |

## Utiliser les versions génériques

Chaque fonction renvoie `(x_opt, f_opt, n_iter, converged, message)`.

```python
import numpy as np
from AlgoQuasiNewton import quasi_newton

def f(X):
    x, y, z = X
    return 6*x**2 + 6*x*y + 3*y**2 + y**3 - (1/3)*z**3 + (1/4)*z**4

x0 = np.array([-0.6637, 0.1758, 1.5778])
x_opt, f_opt, n_iter, converged, message = quasi_newton(f, x0, tol=1e-5)

print(x_opt, f_opt, n_iter, converged, message)
```

`f` peut aussi être une **expression sympy** : les dérivées sont alors exactes.

```python
import sympy as sp
from AlgoNewton import newton

x, y, z = sp.symbols("x y z", real=True)
expr = 6*x**2 + 6*x*y + 3*y**2 + y**3 - sp.Rational(1, 3)*z**3 + sp.Rational(1, 4)*z**4
resultat = newton(expr, x0, variables=(x, y, z))
```

Options utiles : `verbose=True` (affiche les itérations) et `historique=[]` (récupère l'historique).

## Remarques

- Le **message** renvoyé explique l'arrêt : convergence (avec la nature du point : minimum, selle ou dégénéré), divergence, `f` non bornée, stagnation ou nombre maximal d'itérations atteint.
- Pour le **pas fixe**, `α` doit rester inférieur à `2 / λmax` (≈ 0,127 ici), sinon la méthode diverge.
- Avec le **pas optimal**, deux pas alternent, car les gradients successifs sont orthogonaux (propriété de la recherche linéaire exacte).