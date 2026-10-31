# Implémentation d'Algorithmes d'Optimisation Différentiable

Ce document présente l'implémentation et la documentation technique des principaux algorithmes d'optimisation sans contraintes. L'ensemble des méthodes est développé en Python à l'aide des bibliothèques fondamentales pour le calcul matriciel et symbolique.

## Table des matières

1. [Aperçu des algorithmes](#1-aperçu-des-algorithmes)
2. [Prérequis et dépendances](#2-prérequis-et-dépendances)
3. [Détails mathématiques et algorithmiques](#3-détails-mathématiques-et-algorithmiques)
4. [Exécution et résultats](#4-exécution-et-résultats)

## 1. Aperçu des algorithmes

Le projet regroupe quatre algorithmes de descente pour la minimisation de fonctions différentiables :

- **Descente de gradient à pas fixe** : méthode de premier ordre utilisant un taux d'apprentissage constant.
- **Descente de gradient à pas optimal (Armijo)** : recherche linéaire adaptative garantissant la condition de décroissance suffisante.
- **Gradient conjugué (Fletcher-Reeves)** : méthode exploitant des directions de recherche Q-conjuguées pour accélérer la convergence.
- **Quasi-Newton (BFGS)** : algorithme de second ordre approximant la matrice hessienne inverse via la formule de Broyden-Fletcher-Goldfarb-Shanno.

## 2. Prérequis et dépendances

Le projet utilise les bibliothèques scientifiques standards suivantes :

- **NumPy** : gestion des structures vectorielles et opérations d'algèbre linéaire.
- **SymPy** : calcul symbolique exact des dérivées partielles (gradients).

## 3. Détails mathématiques et algorithmiques

Fonction test : $f(x, y) = x^2 + 3y^2$, dont le gradient $\nabla f = (2x,\ 6y)$ est calculé de manière exacte avec SymPy.

- **Pas fixe** : $x_{k+1} = x_k - \alpha \nabla f(x_k)$, arrêt lorsque $\|\nabla f(x_k)\| < \varepsilon$.
- **Pas optimal (Armijo)** : le pas $\alpha$ part de 1 et est divisé par 2 jusqu'à satisfaire
  $f(x_k + \alpha d_k) \le f(x_k) + c\,\alpha\,\nabla f(x_k)^T d_k$, avec $c = 10^{-4}$.
- **Gradient conjugué (Fletcher-Reeves)** : $d_{k+1} = -\nabla f(x_{k+1}) + \beta_k d_k$ avec
  $\beta_k^{FR} = \dfrac{\|\nabla f(x_{k+1})\|^2}{\|\nabla f(x_k)\|^2}$.
- **Quasi-Newton (BFGS)** : $d_k = -H_k \nabla f(x_k)$, avec $s_k = x_{k+1}-x_k$, $y_k = \nabla f_{k+1}-\nabla f_k$, $\rho_k = 1/(y_k^T s_k)$ :

$$H_{k+1} = (I - \rho_k s_k y_k^T)\,H_k\,(I - \rho_k y_k s_k^T) + \rho_k s_k s_k^T$$

## 4. Exécution et résultats

Tests depuis le point $[4, 5]$, avec 100 itérations maximum et une tolérance de $10^{-6}$ (minimum théorique en $[0, 0]$) :

| Méthode | Itérations | f final |
|---|---|---|
| Pas fixe (α = 0,1) | 72 | 1,8e-13 |
| Pas optimal (Armijo) | 25 | 8,1e-14 |
| Gradient conjugué (FR) | 30 | 8,0e-14 |
| Quasi-Newton (BFGS) | 8 | 1,5e-16 |

Le notebook `optimisation_differentiable.ipynb` contient le code complet, l'exécution et les graphiques comparatifs.
