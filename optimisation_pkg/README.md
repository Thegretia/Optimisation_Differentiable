# optimisation

Petit package Python pédagogique pour **minimiser une fonction** de plusieurs variables avec cinq méthodes :

| Méthode | Fonction | Idée |
|---|---|---|
| Descente à **pas fixe** | `descente_pas_fixe` | on avance toujours de la même longueur dans la direction `-gradient` |
| Descente à **pas variable** | `descente_pas_variable` | le pas est recalculé à chaque itération (Armijo ou pas optimal) |
| Gradient **conjugué** | `gradient_conjugue` | directions successives « conjuguées » (Fletcher-Reeves ou Polak-Ribière) |
| **Newton** | `newton` | utilise le hessien : on résout `H d = -g` |
| **Quasi-Newton** (BFGS) | `quasi_newton` | estime l'inverse du hessien à partir des gradients |

**Particularité :** le calcul du gradient et du hessien est fait par **SymPy** (dérivation symbolique), et toute l'algèbre linéaire (produit matrice-vecteur, produit matriciel, produit extérieur, résolution de système par Gauss...) est **écrite à la main** dans `algebre.py`, sans `@`, `np.dot`, `np.outer` ni `np.linalg`. NumPy ne sert qu'à stocker les tableaux.

## Installation

Dépendances : `numpy` et `sympy`.

```bash
pip install numpy sympy
pip install .          # depuis le dossier optimisation_pkg (optionnel)
```

Sans installer, il suffit de lancer Python depuis le dossier `optimisation_pkg`.

## Utilisation rapide

```python
import sympy as sp
from optimisation import minimiser

x1, x2 = sp.symbols("x1 x2")
f = 100 * (x2 - x1**2) ** 2 + (1 - x1) ** 2        # fonction de Rosenbrock

res = minimiser(f, [x1, x2], x0=[-1.2, 1.0], methode="quasi_newton")
print(res.x, res.iterations, res.converge)          # ~[1, 1]  34  True
```

`methode` peut valoir : `"pas_fixe"`, `"pas_variable"`, `"gradient_conjugue"`, `"newton"`, `"quasi_newton"`.
Les options sont transmises à la méthode choisie :

```python
minimiser(f, [x1, x2], [-1.2, 1.0], "pas_fixe", pas=0.001, max_iter=50000)
minimiser(f, [x1, x2], [-1.2, 1.0], "pas_variable", pas="optimal")
minimiser(f, [x1, x2], [-1.2, 1.0], "gradient_conjugue", formule="FR")
minimiser(f, [x1, x2], [-1.2, 1.0], "newton", amorti=True)
```

### Utilisation directe (sans passer par `minimiser`)

```python
from optimisation import preparer, newton, quasi_newton

f_num, grad, hess = preparer(f, [x1, x2])           # SymPy -> fonctions Python
res = newton(f_num, grad, hess, [-1.2, 1.0])
```

On peut aussi fournir ses propres fonctions `f(x)`, `grad(x)` (et `hess(x)` pour Newton), où `x` est une liste ou un tableau de nombres.

## Le résultat

Chaque méthode renvoie un objet `Resultat` :

| Attribut | Contenu |
|---|---|
| `x` | point trouvé |
| `fx` | valeur de `f` en ce point |
| `iterations` | nombre d'itérations effectuées |
| `converge` | `True` si `‖gradient‖ < tol` avant `max_iter` |
| `historique` | liste de tous les points visités (utile pour tracer le chemin) |
| `methode` | nom de la méthode |

## Les méthodes en détail

Notations : `g` = gradient, `d` = direction, `a` = pas, `x ← x + a·d`.

- **Pas fixe** : `d = -g`, `a` constante. Simple, mais si `a` est trop grand ça diverge, trop petit ça traîne.
  Pour une quadratique de hessien `A`, il faut `a < 2 / λmax(A)`.
- **Pas variable** : `d = -g`, `a` recalculé :
  - `pas="armijo"` : on part de `a = 1` et on divise par 2 tant que `f(x + a d) > f(x) + c·a·(g·d)` (backtracking).
  - `pas="optimal"` : on minimise `f(x + a d)` en 1D (encadrement puis section dorée).
- **Gradient conjugué** : `d₊ = -g₊ + β d`, avec
  - Fletcher-Reeves : `β = (g₊·g₊) / (g·g)`
  - Polak-Ribière : `β = max(0, g₊·(g₊ - g) / (g·g))`

  Redémarrage tous les `n` pas. Sur une quadratique en dimension `n`, il converge en au plus `n` itérations.
- **Newton** : `d` solution de `H(x) d = -g` (Gauss), `a = 1`.
  `amorti=True` choisit `a` par Armijo. Si `H` est singulière ou si `d` ne descend pas, on utilise `d = -g`.
- **Quasi-Newton BFGS** : `d = -H g`, puis
  `H₊ = (I - ρ s yᵀ) H (I - ρ y sᵀ) + ρ s sᵀ`, `ρ = 1/(y·s)`, avec `s = x₊ - x` et `y = g₊ - g`.
  La mise à jour n'est faite que si `y·s > 0` (condition de courbure).

## Organisation du package

```
optimisation_pkg/
├── README.md
├── pyproject.toml
└── optimisation/
    ├── __init__.py
    ├── algebre.py              # algèbre linéaire faite maison
    ├── outils.py               # SymPy (preparer), Armijo, pas optimal, Resultat
    ├── descente.py             # pas fixe et pas variable
    ├── gradient_conjugue.py    # gradient conjugué (FR / PR)
    ├── newton.py               # Newton et BFGS
    └── api.py                  # minimiser(...)
```

## Résultats sur Rosenbrock, départ (-1.2, 1), `tol = 1e-6`

| Méthode | Itérations |
|---|---|
| Newton | 6 |
| Newton amorti | 21 |
| Gradient conjugué (PR ou FR, pas optimal) | 14 |
| Quasi-Newton BFGS | 34 |
| Descente à pas variable (Armijo) | ~14 000 |

## Limites à connaître

- Les méthodes trouvent un **minimum local** (point où le gradient s'annule), pas forcément global.
- Le gradient conjugué avec `pas="armijo"` est moins fiable qu'avec le pas optimal (il peut ne pas converger sur Rosenbrock).
- Le pas fixe demande de choisir `pas` à la main.
- La fonction doit être dérivable ; `preparer` suppose une expression SymPy.
