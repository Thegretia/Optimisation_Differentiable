# Optimisation sans contrainte — 4 méthodes écrites from scratch

Quatre algorithmes de minimisation, un fichier par algorithme, le même exemple
dans chacun. Aucune bibliothèque d'optimisation (`scipy.optimize` n'est pas
utilisé) : uniquement **numpy** pour le calcul numérique et **sympy** pour les
dérivées exactes.

## Contenu

| Fichier | Méthode |
|---|---|
| `gradient_pas_fixe.py` | descente de gradient à pas constant |
| `gradient_pas_optimal.py` | descente de gradient à pas optimal |
| `newton.py` | méthode de Newton |
| `quasi_newton.py` | BFGS (quasi-Newton) |

Chaque fichier est autonome : la petite fonction `_fonctions` (qui transforme
une expression sympy en fonctions numpy) est volontairement recopiée dans les
quatre fichiers, pour qu'on puisse en lire un seul sans ouvrir les autres.

## L'exemple commun

On minimise la même quadratique dans les quatre fichiers :

```
f(x, y) = x² + 10y² + xy - 2x - 4y
```

que l'on écrit aussi sous forme matricielle `f(x) = ½ xᵀAx - bᵀx` avec

```
A = [[2, 1],        b = [2, 4]
     [1, 20]]
```

Son gradient est `grad f(x) = Ax - b`, et le minimum est donné par `Ax = b` :

```
x* = (12/13, 2/13) ≈ (0.9231, 0.1538)
f(x*) = -16/13 ≈ -1.2308
```

La plus grande valeur propre de `A` vaut `L = 11 + √82 ≈ 20.06`. C'est elle qui
fixe la limite de stabilité du pas fixe : `2/L ≈ 0.0997`.

## Comment lire le code

Chaque fichier suit la même structure :

1. **`_fonctions(expr, variables)`** — sympy dérive l'expression une seule fois,
   puis `lambdify` fabrique trois fonctions numpy : `F(x)`, `G(x)` (gradient)
   et `H(x)` (Hessienne). On passe un vecteur `x` et on reçoit un flottant, un
   vecteur ou une matrice.
2. **La fonction d'optimisation** — une boucle courte : calculer le gradient,
   tester l'arrêt, puis appliquer la formule de la méthode.
3. **Le bloc `if __name__ == "__main__"`** — l'exemple, qui affiche `x*`,
   `f(x*)` et le nombre d'itérations.

Le critère d'arrêt est toujours le même : on s'arrête dès que
`||grad f(x)|| <= tol` (le point est stationnaire) ou au bout de `max_iter`
itérations.

## Les quatre méthodes

### 1. Descente de gradient à pas fixe

```
x_{k+1} = x_k - alpha · grad f(x_k)
```

On avance toujours dans la direction opposée au gradient, avec un pas `alpha`
constant. C'est la méthode la plus simple, mais elle n'est stable que si

```
0 < alpha < 2/L
```

où `L` est la plus grande valeur propre de la Hessienne. Dans l'exemple,
`alpha = 0.05` fonctionne, `alpha = 0.1` fait diverger la suite.

**À retenir :** le pas ne s'adapte pas. Trop grand → divergence, trop petit →
convergence très lente.

### 2. Descente de gradient à pas optimal

```
d     = -grad f(x)
alpha = (g · g) / (gᵀ H(x) g)        avec g = grad f(x)
x+    = x + alpha d
```

Au lieu de choisir `alpha` à la main, on prend le pas qui minimise
`phi(a) = f(x - a·g)` dans la direction de descente.

Pour une fonction **quadratique**, la Hessienne `H` est constante et cette
formule donne le minimum exact de `phi`. Pour une fonction générale, `H(x)`
change à chaque itération : la formule n'est qu'une approximation (le « pas de
Cauchy »), et une vraie recherche linéaire en 1D serait nécessaire.

**À retenir :** le pas est optimal à chaque itération, mais la méthode reste
lente sur les fonctions mal conditionnées (elle zigzague).

### 3. Méthode de Newton

```
H(x) d = -grad f(x)
x+     = x + t d
```

On utilise la courbure (la Hessienne) pour viser directement le minimum local
de l'approximation quadratique de `f`. Près de la solution, `t = 1` et la
convergence est **quadratique** (le nombre de chiffres exacts double à chaque
itération).

Loin de la solution, le pas `t = 1` peut faire augmenter `f`. On ajoute donc
une recherche linéaire très simple : tant que `f(x + t·d) > f(x)`, on divise
`t` par 2 (backtracking).

**À retenir :** la méthode la plus rapide quand elle s'applique, mais elle
coûte une Hessienne et une résolution de système linéaire par itération. Si la
Hessienne est singulière, on retombe sur `d = -grad f`.

### 4. BFGS (quasi-Newton)

```
d  = -H grad f(x)
x+ = x + t d

s   = x+ - x
y   = grad f(x+) - grad f(x)
rho = 1 / (y · s)
H+  = (I - rho s yᵀ) H (I - rho y sᵀ) + rho s sᵀ
```

On ne calcule jamais la Hessienne : on entretient une matrice `H` qui
l'approxime (plus précisément son inverse), et on la corrige à chaque pas avec
les variations de `x` et du gradient. La direction `d = -H·grad f` est alors
meilleure qu'une simple descente de gradient.

La mise à jour n'est faite que si `y · s > 0` : c'est la condition de courbure
qui garantit que `H` reste définie positive. Sinon on garde l'ancienne `H`.

**À retenir :** le meilleur compromis en pratique — presque aussi rapide que
Newton, mais avec seulement le gradient.

## Comparaison sur l'exemple

Même point de départ `x0 = (5, 3)`, même tolérance `1e-8` :

| Méthode | Itérations | Ce qui explique le chiffre |
|---|---|---|
| Pas fixe (`alpha = 0.05`) | 200 | pas constant, petite valeur propre `≈ 1.94` |
| Pas optimal | 21 | pas exact, mais trajectoire en zigzag |
| Newton | 1 | quadratique : un seul pas suffit |
| BFGS | 7 | `H` s'affine au fil des itérations |

Les quatre méthodes trouvent bien `x* ≈ (0.9231, 0.1538)` et
`f(x*) ≈ -1.2308`.

## Lancer les fichiers

```bash
python gradient_pas_fixe.py
python gradient_pas_optimal.py
python newton.py
python quasi_newton.py
```

## Limites

Ces versions sont **pédagogiques** : elles montrent le cœur de chaque méthode,
pas une implémentation de production. En particulier :

- le pas fixe exige de connaître `L` (ou de le deviner) ;
- le pas optimal n'est exact que pour une quadratique ;
- Newton échoue si la Hessienne est singulière ou si le backtracking ne trouve
  aucun pas qui diminue `f` ;
- BFGS suppose une fonction différentiable et un gradient exact ;
- aucune gestion des contraintes, des fonctions non convexes ou des points de
  départ très éloignés.

Pour un usage réel, on préférera `scipy.optimize.minimize`.
