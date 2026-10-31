# Optimisation convexe sans contrainte en Python pur

Ce dossier contient des implémentations des algorithmes de l'Appendice A (section A.3,<< Notions d'optimisation convexe>>), écrites sans bibliothèque externe : seul le module standard `math` est utilisé par les solveurs. Les vecteurs sont des listes de flottants et les matrices des listes de listes. Le script `courbes.py` utilise matplotlib uniquement pour tracer les figures, et reste utilisable sans lui (les historiques sont alors exportés au format CSV).

## Organisation du code

`algebre.py` regroupe les operations d'algebre lineaire elémentaires (produit scalaire, norme, produit matrice-vecteur, produit exterieur). `outils.py` definit la structure `Resultat`, commune a tous les solveurs, le compteur d'appels `Compteur` et l'evaluation protégé contre les debordements. `problemes.py` contient trois fonctions convexes de test avec leur gradient, leur hessienne et leur solution exacte.

Les quatre méthodes sont dans `gradient_pas_fixe.py` (Définition 1.8), `gradient_pas_variable.py` (recherche lineaire par rebroussement, Définition 1.9), `gradient_conjugue.py` (gradient conjugué, Definition 1.11, et Newton par gradient conjugue, section A.3.6) et `quasi_newton_bfgs.py` (section A.3.7, equation A.25). `exemples.py` compare les methodes, `courbes.py` trace leur convergence et `tests.py` regroupe les tests unitaires.

Utilisation, depuis ce dossier :

```
python3 exemples.py
python3 courbes.py
python3 -m unittest tests -v
```

Chaque module de solveur possede aussi un exemple autonome, lancé par `python3 gradient_pas_fixe.py` et ses équivalents.

## Problèmes de test

Le premier problème est la quadratique f(u) = 1/2 u^(t)Qu - bᵀu avec Q = [[4, 1], [1, 3]] et b = (1, 2). Sa solution exacte est u* = Q^-(1b) = (1/11, 7/11), et les valeurs propres de Q sont μ = 2,382 et L = 4,618.

Le deuxième est la somme d'exponentielles f(x, y) = e^(x+3y-0,1) + e^(x-3y-0,1) + e^(-x-0,1), convexe et non quadratique, de solution u* = (1/2 ln 1/2, 0) ≈ (-0,34657, 0).

Le troisième est une quadratique diagonale de dimension 30 dont la hessienne a ses valeurs propres réparties de 1 à 100 (conditionnement 100), de solution ui* = 1/di. Il sert à observer l'effet du conditionnement et de la dimension, ce que les deux problèmes en dimension 2 ne permettent pas.

## Méthodes

Le gradient à pas fixe applique u_{k+1} = u_k - αdeltaf(u_k). Pour une quadratique de valeurs propres extremes μ et L, la suite converge si et seulement si α < 2/L, et le pas 2/(μ + L) minimise le taux linéaire (L - μ)/(L + μ). Pour la quadratique 2D, le seuil 2/L vaut 0,433 et le pas optimal 0,286. Le solveur détecte la divergence (valeur non finie ou supérieure à 10^(100)) et s'interrompt avec le statut `divergence`.

Le gradient à pas variable part de α₀ à chaque itération et pose α := betaα tant que f(u - α deltaf) > f(u) - (α/2)‖deltaf‖². La dernière valeur de f calculée dans la recherche linéaire est conservée pour l'itération suivante, ce qui évite une évaluation redondante.

Le gradient conjugué résout Ax = b pour A symétrique définie positive en au plus n itérations en arithmétique exacte (2n sont autorisées pour absorber les arrondis). Le critère d'arrêt est relatif à ‖b‖, et une `ValueError` est levée si la courbure ⟨v, Av⟩ n'est pas strictement positive. Dans `newton_gradient_conjugue`, le système delta carreef(u)sigma = f(u) est résolu par gradient conjugué, sans jamais inverser la hessienne, puis u_{k+1} = u_k - αsigma. Avec `amorti=True` (valeur par défaut), le pas α est déterminé par rebroussement avec la condition d'Armijo ; avec `amorti=False`, on retrouve la méthode de Newton pure, dont la convergence globale n'est pas garantie.

BFGS approxime l'inverse de la hessienne par W, avec W(0) = I, mise à jour par (A.25) à partir de dₜ = uₜ - uₜ₋₁ et δₜ = deltaf(uₜ) - ∇f(uₜ₋₁). La direction est p = -Wdeltaf(u) et le pas est obtenu par rebroussement. La matrice W reste définie positive tant que ⟨δ, d⟩ > 0 ; une recherche de type Armijo ne l'assure pas, la mise à jour est donc ignorée lorsque ⟨δ, d⟩ ≤ 10⁻¹⁰ ‖sigma‖‖d‖. Ce seuil est relatif, de sorte qu'il ne dépend pas de l'échelle du problème. Le nombre de mises à jour ignorées est consigné dans `Resultat.infos`.

## Résultats

Tous les solveurs partagent le même critère d'arrêt ‖delataf(u)‖² ≤ 10⁻¹², ce qui rend les nombres d'itérations comparables. Le coût est mesuré par les appels à f, à deltaf et à la hessienne, et par le nombre de produits matrice-vecteur du gradient conjugué (colonne « prod. MV »). Le pas fixe est le pas optimal 2/(μ + L) pour les quadratiques, et 0,01, choisi empiriquement, pour la somme d'exponentielles.

```
Quadratique 2D, point de départ de dimension 2, critère d'arrêt ||grad f||^2 <= 1e-12
Méthode                           iter  eval f  eval grad  eval hess  prod. MV   erreur ||u-u*||  statut
Gradient à pas fixe                 15      16         16          0         0         2.485e-07  convergence
Gradient à pas variable (BLS)       19      58         20          0         0         2.219e-07  convergence
Newton et gradient conjugué          1       2          2          1         2         3.433e-16  convergence
Quasi-Newton BFGS                    7      10          8          0         0         5.395e-09  convergence

Somme d'exponentielles, point de départ de dimension 2, critère d'arrêt ||grad f||^2 <= 1e-12
Méthode                           iter  eval f  eval grad  eval hess  prod. MV   erreur ||u-u*||  statut
Gradient à pas fixe                563     564        564          0         0         3.828e-07  convergence
Gradient à pas variable (BLS)       30     124         31          0         0         2.737e-07  convergence
Newton et gradient conjugué          6       7          7          6        12         2.680e-12  convergence
Quasi-Newton BFGS                   11      18         12          0         0         3.772e-08  convergence

Quadratique diagonale n=30, cond=100, point de départ de dimension 30, critère d'arrêt ||grad f||^2 <= 1e-12
Méthode                           iter  eval f  eval grad  eval hess  prod. MV   erreur ||u-u*||  statut
Gradient à pas fixe                709     710        710          0         0         6.943e-07  convergence
Gradient à pas variable (BLS)      510    3370        511          0         0         8.941e-07  convergence
Newton et gradient conjugué          1       2          2          1        30         1.349e-16  convergence
Quasi-Newton BFGS                   37     173         38          0         0         8.155e-08  convergence

```

Sur la quadratique 2D, Newton converge en une itération, puisque le modèle de Taylor d'ordre 2 est exact. Le nombre d'itérations ne reflète pas à lui seul le coût : une itération de Newton demande une hessienne et un gradient conjugué interne, alors qu'une itération de gradient ne demande qu'un gradient. Sur le problème de dimension 30, Newton ne fait qu'une itération mais effectue 30 produits matrice-vecteur, tandis que BFGS demande 37 itérations sans jamais évaluer la hessienne. Le pas variable y coûte 3370 évaluations de f pour 510 itérations, à cause des rebroussements successifs imposés par le conditionnement.

Le choix du pas fixe est critique. Sur la quadratique 2D (seuil 2/L = 0,433), les essais de `exemples.py` donnent :

```
   alpha  iterations  statut
  0.0500         131  convergence
  0.2000          26  convergence
  0.2857          15  convergence
  0.4300        1109  convergence
  0.4400        3631  divergence
  0.6000         199  divergence
```

Le pas optimal converge en 15 itérations, un pas de 0,05 en demande 131, et les pas supérieurs à 2/L divergent. À 0,43, juste sous le seuil, la convergence est très lente (1109 itérations).

## Vitesses de convergence

Les figures du dossier `resultats/` (une par problème, avec les données au format CSV) montrent f(uₖ) - f* en échelle logarithmique. Sur la somme d'exponentielles, le rapport d'erreurs successives du gradient à pas fixe tend vers une constante proche de 1 (de 0,54 à 0,92 sur les quatorze premières itérations), ce qui correspond à une convergence linéaire lente. Celui de BFGS décroît vers 0 (de l'ordre de 5·10⁻⁴ à la fin), signe d'une convergence superlinéaire. Pour Newton, l'erreur passe de 4,8·10⁻³ à 5,6·10⁻⁶, puis à 7,9·10⁻¹², comportement caractéristique de la convergence quadratique au voisinage de la solution. Le pas variable se situe entre ces deux régimes : sa convergence est linéaire, avec un taux nettement meilleur que celui du pas fixe.

## Vérifications

`tests.py` compare gradients et hessiennes aux différences finies centrées, contrôle le gradient conjugué (système 2×2, borne de n produits, rejet d'une matrice non définie positive), vérifie que tous les solveurs convergent vers la solution exacte sur les trois problèmes, que Newton converge en une itération sur une quadratique, que la décroissance du pas variable est monotone, que le pas fixe diverge au-delà de 2/L et que BFGS atteint le minimum de la fonction de Rosenbrock (non convexe).

## Limites

Les fonctions sont supposées différentiables (de classe C¹, et C² pour Newton). La convergence de BFGS est classiquement établie dans le cadre convexe avec une recherche linéaire vérifiant les conditions de Wolfe ; le rebroussement d'Armijo utilisé ici ne garantit pas la condition de courbure, d'où le saut de mise à jour, et le bon comportement observé sur la fonction de Rosenbrock reste un constat expérimental. Le code est volontairement pédagogique : la représentation par listes de listes et l'absence de calcul vectorisé le rendent inadapté aux grandes dimensions.
