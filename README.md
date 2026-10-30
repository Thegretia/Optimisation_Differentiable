# Algorithmes d'optimisation différentiable

Projet implémente quatre méthodes de minimisation en Python notamment : Newton, quasi-Newton BFGS, descente de gradient à pas fixe et gradient conjugué non linéaire de Fletcher-Reeves.

Les algorithmes acceptent une fonction Python `f(x)` définie par l'utilisateur. En dimension 1, `x` est un nombre. En dimension supérieure, `x` est une liste de nombres, par exemple `[x1, x2]`. Les dérivées sont approchées par différences finies centrées : aucune bibliothèque de calcul numérique ni de calcul symbolique n'est nécessaire. Cette approche s'applique à de nombreuses fonctions trigonométriques, polynomiales et mixtes, tant que la fonction est définie près des points visités.

## Fichiers

- `Devoir/algorithme_newton.py` : méthode de Newton avec Hessienne numérique et recherche linéaire.
- `Devoir/algorithme_quasi_newton.py` : BFGS avec approximation de la Hessienne inverse.
- `Devoir/algorithme_pas_fixe.py` : descente de gradient avec pas constant.
- `Devoir/algorithme_pas_conjugue.py` : gradient conjugué non linéaire Fletcher-Reeves.
- `Devoir/optimisation_utils.py` : dérivées numériques et opérations vectorielles communes.
- `Devoir/interface_console.py` : menu interactif pour lancer les exemples ou saisir une expression.

## Installation et lancement

Python 3.10 ou plus récent est recommandé

```bash
python3 -m venv .venv
source .venv/bin/activate       # Linux / macOS
# Sous Windows PowerShell : .venv\Scripts\Activate.ps1
cd Devoir
python interface_console.py
```

Depuis la racine du projet, vous pouvez aussi lancer `python -m Devoir.interface_console`.

L'environnement `.venv/` est ignoré par Git. Il ne contient pas de dépendances à installer. Après l'avoir créé et activé, choisissez un algorithme, puis l'exemple ou la saisie personnalisée. L'exemple du menu est `x**2 + 4*cos(x)` avec `x0 = -1`.

## Syntaxe des fonctions personnalisées

En une dimension, utilisez `x` :

```text
x**4 - 3*x**2 + sin(x)
```

En plusieurs dimensions, utilisez `x1`, `x2`, etc. Les noms doivent être consécutifs :

```text
(x1 - 1)**2 + 2*(x2 + 2)**2 + cos(x1*x2)
```

À l'invite du point initial, entrez les valeurs dans le même ordre, séparées par des virgules, par exemple `3, -1`. L'opérateur puissance s'écrit `**` (et non `^`). Les opérateurs disponibles sont `+`, `-`, `*`, `/`, `**`, `%` et les parenthèses. Fonctions disponibles : `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `sqrt`, `exp`, `log`, `log10`, `abs`. Constantes : `pi` et `e`. Les angles sont en radians. Pour un logarithme de base quelconque, écrivez par exemple `log(x)/log(10)`.

L'interface interprète uniquement ces opérateurs, fonctions, constantes et variables. Elle ne permet pas les appels Python arbitraires, les comparaisons, les affectations, les fonctions définies par l'utilisateur ni les variables autres que `x` ou `x1`, `x2`, etc. Les fonctions définies dans un programme Python peuvent, elles, utiliser toute expression Python valide.

## Utiliser les fonctions directement

Chaque méthode est une fonction importable. Exemple :

```python
from Devoir.algorithme_newton import newton

def f(x):
    return x**2 + 4 * math.cos(x)

resultat = newton(f, x0=-1.0)
print(resultat["x"], resultat["f_x"], resultat["converged"])
```

Ajoutez `import math` si votre fonction Python utilise les fonctions trigonométriques. Pour une fonction de deux variables, elle reçoit une liste :

```python
from Devoir.algorithme_quasi_newton import quasi_newton

def f(x):
    x1, x2 = x
    return (x1 - 1)**2 + 2*(x2 + 2)**2

resultat = quasi_newton(f, x0=[4.0, 3.0])
```

Les signatures sont `newton(f, x0, tol=1e-6, max_iter=100, line_search=True, regularisation=1e-8)`, `quasi_newton(f, x0, tol=1e-6, max_iter=500)`, `pas_fixe(f, x0, alpha=0.01, tol=1e-6, max_iter=1000)` et `pas_conjugue(f, x0, tol=1e-6, max_iter=1000)`. Chaque fonction renvoie un dictionnaire avec `x`, `f_x`, `iterations`, `converged` et `history` (points, valeurs et normes du gradient observés).

## Notes sur les méthodes

- Newton calcule gradient et Hessienne numériquement, puis résout le système linéaire par élimination de Gauss avec pivot partiel. Une recherche linéaire d'Armijo et une régularisation limitée aident lorsque le pas de Newton brut ne convient pas.
- BFGS met à jour une approximation de la Hessienne inverse à partir des gradients successifs et emploie une recherche linéaire d'Armijo.
- Le pas fixe utilise exactement le `alpha` donné. Un pas trop grand peut faire diverger la suite; réduisez `alpha` si nécessaire.
- Le gradient conjugué est la variante non linéaire Fletcher-Reeves, avec recherche linéaire et redémarrages périodiques.

Ces méthodes cherchent un minimum local. Les différences finies ont une précision limitée, surtout pour les fonctions mal conditionnées, les points proches d'une singularité ou les fonctions non différentiables. `sqrt(x)` et `log(x)` ne sont évaluables que dans leurs domaines. Une fonction non bornée inférieurement ou un mauvais point initial peut empêcher la convergence.

En cas d'échec dans le menu, le programme indique le type d'erreur, l'expression et le point évalué lorsque c'est possible. Les causes fréquentes (racine carrée ou logarithme hors domaine, division par zéro, dépassement numérique, syntaxe invalide et non-convergence) sont accompagnées de pistes de correction. Une erreur de domaine peut survenir pendant les petites évaluations utilisées pour estimer les dérivées, même si la fonction est définie au point initial; elle doit donc rester calculable dans le voisinage exploré par l'algorithme.
