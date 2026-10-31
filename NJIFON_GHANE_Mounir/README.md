# Optimisation différentiable sans contrainte — algorithmes écrits à la main

**NJIFON GHANE MOUNIR** — SDIA 4, ENSPD, Université de Douala (2026-2027)

Implémentation des méthodes de descente vues en cours (gradient à pas fixe, gradient à pas optimal,
gradient conjugué, Newton, quasi-Newton BFGS/DFP), une série de 8 exercices résolus avec ces
méthodes, et un petit outil qui permet de résoudre n'importe quel exercice (ou n'importe quelle fonction)
en une ligne.

## Contenu du dossier

```
.
├── gradient_pas_fixe.py       algorithme 1 : gradient à pas fixe
├── gradient_pas_optimal.py    algorithme 2 : gradient à pas optimal
├── gradient_conjugue.py       algorithme 3 : gradient conjugué (Fletcher-Reeves / Polak-Ribière+)
├── newton.py                  algorithme 4 : Newton (version du cours et version globalisée)
├── quasi_newton.py            algorithme 5 : quasi-Newton BFGS / DFP
├── minimiser.py               minimiser(f, x0, methode) et comparer(f, x0) : appel commun aux 5 algorithmes
├── resoudre.py                résolution en ligne de commande (ou menu interactif)
├── verifier.py                toutes les méthodes sur tous les exercices
├── outils/                    code commun aux algorithmes
│   ├── algebre.py             Gauss (pivot partiel), Cholesky, valeurs propres (Jacobi)
│   ├── probleme.py            f, gradient, hessienne (sympy exact ou différences finies)
│   ├── recherche_lineaire.py  Armijo, Wolfe forte, pas optimal (encadrement + section dorée)
│   ├── resultat.py            objet Resultat, nature du point, messages d'arrêt
│   ├── exercice.py            classe Exercice (énoncé, solution attendue, réglages)
│   └── graphiques.py          lignes de niveau, trajectoires, convergence (notebook)
├── exercices/                 la série d'exercices, un fichier par exercice
│   ├── exercice_0_cours.py
│   ├── exercice_1_quadratique_zigzag.py
│   ├── exercice_2_booth.py
│   ├── exercice_3_beale.py
│   ├── exercice_4_entrepot_weber.py
│   ├── exercice_5_regression_logistique.py
│   ├── exercice_6_points_critiques.py
│   ├── exercice_7_wood.py
│   └── TD_optimisation.ipynb  énoncés, résolution à la main, résultats et graphiques
├── requirements.txt
└── README.md
```

Chaque fichier d'algorithme se lance seul : `python newton.py` applique Newton à l'exercice du cours,
`python newton.py "x**2 + 3*y**2 - x*y" 1 1` à n'importe quelle fonction.
Chaque fichier d'exercice se lance seul aussi : `python exercices/exercice_2_xxx.py` affiche l'énoncé et
le résout avec les 5 méthodes.

## Ce qui est codé à la main

Aucune fonction d'optimisation toute faite : pas de `scipy.optimize`, pas de `numpy.linalg`.
`numpy` sert à stocker les vecteurs et matrices, `sympy` à dériver f exactement quand f est donnée
sous forme de formule (s'il n'est pas installé, on passe aux différences finies), `matplotlib` uniquement
aux graphiques du notebook.

- résolution de $H d = -g$ par élimination de Gauss avec pivot partiel (Newton pur) ou Cholesky
  (Newton globalisé : si $H$ n'est pas définie positive on factorise $H + \tau I$) ;
- valeurs propres de la hessienne par la méthode de Jacobi, pour donner la nature du point trouvé
  (minimum, maximum, point selle, dégénéré) ;
- recherche linéaire d'Armijo (rebroussement), de Wolfe forte (Nocedal & Wright, algorithmes 3.5-3.6),
  et pas optimal par encadrement puis section dorée ;
- mises à jour BFGS et DFP de l'inverse de la hessienne ;
- gradient par différences finies centrées, hessienne par différences finies d'ordre 2.

## Installation

Python 3.8 ou plus.

```bash
pip install numpy sympy matplotlib jupyter
```

## Utilisation

**Un algorithme seul**

```bash
python gradient_pas_fixe.py                        # exercice du cours
python gradient_pas_fixe.py "x**2 + 3*y**2" 1 1    # autre fonction
python newton.py
python quasi_newton.py "(1-x)**2 + 100*(y-x**2)**2" -1.2 1
```

**Un exercice de la série**

```bash
python exercices/exercice_3_beale.py              # énoncé + les 5 méthodes
python exercices/exercice_3_beale.py newton       # une seule méthode, avec le tableau des itérations
python resoudre.py --liste                       # liste des exercices
python resoudre.py 3 --x0 -1 1                   # exercice 3 depuis un autre point
python resoudre.py 3 -m pas_fixe --alpha 0.02
python resoudre.py                               # menu interactif
```

**Résoudre n'importe quelle fonction**

```bash
python resoudre.py -f "x**2 + 3*y**2 - x*y + exp(x)" --x0 1 1
python resoudre.py -f "(1-x)^2 + 100*(y-x^2)^2" --x0 -1.2 1 -m bfgs -v
```

Les variables sont détectées automatiquement et rangées dans l'ordre naturel
(`x, y, z` ou `x1, x2, ..., x10`) ; on peut imposer l'ordre avec `--variables "y x"`.
Méthodes : `pas_fixe`, `pas_optimal`, `gradient_conjugue`, `newton`, `newton_pur`, `bfgs`, `dfp`.

**Depuis Python**

```python
from newton import newton
from minimiser import minimiser, comparer

r = newton("x**2 + 3*y**2 - x*y", [1, 1])          # appel direct d'un algorithme
r = minimiser("x**2 + 3*y**2 - x*y", x0=[1, 1], methode="newton")   # equivalent
print(r)                     # point, valeur, itérations, message, nature du point
r.afficher_iterations()

comparer("x**2 + 3*y**2 - x*y", [1, 1])        # tableau comparatif des méthodes

# f peut aussi être une fonction python, avec ou sans gradient
import numpy as np
f = lambda X: (X[0] - 1)**2 + 10*(X[1] + 2)**2
g = lambda X: np.array([2*(X[0] - 1), 20*(X[1] + 2)])
minimiser(f, [0, 0], "bfgs", grad=g)
```

Chaque méthode renvoie un objet `Resultat` : `x`, `f`, `iterations`, `converge`, `message`,
`nature`, `historique` (liste des $(k, x_k, f(x_k), \|\nabla f(x_k)\|, \alpha_k)$) et le nombre
d'évaluations de f, du gradient et de la hessienne.

Options utiles : `tol` (défaut $10^{-6}$), `max_iter`, `verbose=True`, `alpha` pour le pas fixe
(sinon $1/\lambda_{\max}(H(x_0))$), `formule="FR"` ou `"PR"` pour le gradient conjugué,
`recherche="wolfe"` pour le gradient conjugué, `formule="DFP"` pour le quasi-Newton.

**Ajouter un exercice** : copier un fichier de `exercices/`, changer la fonction, le point de départ,
l'énoncé et la solution attendue, puis l'ajouter dans la liste de `exercices/__init__.py`.

## Les exercices

| n° | Exercice | Fichier | n | Solution |
|---|---|---|---|---|
| 0 | Exercice du cours | `exercice_0_cours.py` | 3 | (0, 0, 1) |
| 1 | Quadratique mal conditionnee (zigzag) | `exercice_1_quadratique_zigzag.py` | 2 | (2, -1) |
| 2 | Fonction de Booth | `exercice_2_booth.py` | 2 | (1, 3) |
| 3 | Fonction de Beale | `exercice_3_beale.py` | 2 | (3, 0.5) |
| 4 | Localisation d'un entrepot (probleme de Weber) | `exercice_4_entrepot_weber.py` | 2 | (7.84572, 5.45363) |
| 5 | Regression logistique | `exercice_5_regression_logistique.py` | 2 | (0.676706, -3.72188) |
| 6 | Points critiques de x^4 + y^4 + z^4 - 4xyz | `exercice_6_points_critiques.py` | 3 | 4 minima, dont (1, 1, 1) |
| 7 | Fonction de Wood (dimension 4) | `exercice_7_wood.py` | 4 | (1, 1, 1, 1) |

Les énoncés complets, les calculs à la main et les graphiques sont dans `exercices/TD_optimisation.ipynb`.

## Résultats (`python verifier.py`)

Nombre d'itérations pour atteindre $\|\nabla f\| \le 10^{-6}$ ($10^{-5}$ pour l'exercice du cours) :

```
Nombre d'iterations   * = resultat faux ou non converge
                      ~ = echec attendu (limite de la methode, voir l'enonce)

exercice                                        pas fixe    pas opt. grad. conj.      Newton        BFGS
 0 Exercice du cours                                197          31           9           5          13 
 1 Quadratique mal conditionnee (zigzag)            194          17           2           1           8 
 2 Fonction de Booth                                141           9           2           1           6 
 3 Fonction de Beale                               3845         223           8           7          16 
 4 Localisation d'un entrepot (probleme de W        318          25           6           4           7 
 5 Regression logistique                           7201         510           6           5          15 
 6 Points critiques de x^4 + y^4 + z^4 - 4xy        160          14           8           7          13 
 7 Fonction de Wood (dimension 4)                 52401        8050          67          38          38 

```

## Remarques

- Newton est utilisé dans sa version globalisée (hessienne rendue définie positive + pas d'Armijo).
  Près d'un minimum elle coïncide avec Newton pur (pas = 1). `newton_pur` est la version du cours :
  elle peut converger vers un point selle ou un maximum, c'est pour ça que chaque résultat affiche la
  nature du point trouvé.
- Une norme de gradient petite ne suffit pas à conclure : voir le faux point de convergence ou les points
  selles dans le notebook.
- Ce dossier part de la comparaison des deux versions déjà présentes dans le dépôt
  (AAT NDONGO David et TCHOUANANG Dimitri). La version de David, plus complète (messages d'arrêt, nature
  du point, recherche de Wolfe), a servi de base. Le code commun (environ 200 lignes recopiées dans chacun
  de ses 4 fichiers) a été regroupé dans `outils/`, chaque algorithme garde son propre fichier, et on a corrigé :
  ordre des variables `x1 ... x10` (x10 était rangée avant x2), plus de `numpy.linalg`, pas optimal
  générique, détection de la divergence du pas fixe.

## Références

- J. Nocedal, S. J. Wright, *Numerical Optimization*, 2e éd., Springer, 2006.
- J. J. Moré, B. S. Garbow, K. E. Hillstrom, « Testing Unconstrained Optimization Software »,
  *ACM Transactions on Mathematical Software* 7(1), 1981 (fonctions de Rosenbrock, Beale, Wood, Powell).
- Cours d'optimisation différentiable, SDIA 4, ENSPD.

