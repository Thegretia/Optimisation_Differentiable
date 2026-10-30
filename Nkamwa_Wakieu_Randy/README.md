# optimisation différentiable


| Méthode | Fonction | Idée |
|---|---|---|
| Descente à **pas fixe** | `descente_pas_fixe` | on avance toujours de la même longueur dans la direction du gradient |
| Descente à **pas variable** | `descente_pas_variable` | le pas est recalculé à chaque itération (Armijo ou pas optimal) |
| Gradient **conjugué** | `gradient_conjugue` | directions successives « conjuguées » (Fletcher-Reeves ou Polak-Ribière) |
| **Newton** | `newton` | utilise la hessienne |
| **Quasi-Newton** (BFGS) | `quasi_newton` | estime l'inverse de la hessiene à partir des gradients |

Le calcul du gradient et du hessien est fait par **SymPy** (dérivation symbolique), et toute l'algèbre linéaire (produit matrice-vecteur, produit matriciel, produit extérieur, résolution de système par Gauss...) est **écrite à la main** dans `algebre.py`, sans `@`, `np.dot`, `np.outer` ni `np.linalg`. NumPy ne sert qu'à stocker les tableaux.

## Installation

Dépendances : `numpy` et `sympy`.
## Organisation du package

```
optimisation_pkg/
├── README
└── Nkamwa_Wakieu_Randy/
    ├── quasi_newton.py         # SymPy (preparer), Armijo, pas optimal, Resultat
    ├── pas_fixe.py             # pas fixe et pas variable
    ├── pas_conjugue.py         # gradient conjugué (FR / PR)
    ├── newton.py               # Newton et BFGS
```