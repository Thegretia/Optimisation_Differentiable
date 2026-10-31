# Optimisation from scratch : descente de gradient, gradient conjugué, BFGS, SGD et sac à dos 0/1

Auteur : Fouodji Tako Prudencia
Langage : Python 3 (testé avec Python 3.10)

## 1. Présentation

Ce dépôt contient deux notebooks Jupyter qui implémentent "from scratch" (avec NumPy uniquement) des algorithmes classiques d'optimisation, puis les valident sur des problèmes de test avec une solution connue.

| Fichier | Contenu |
|---|---|
| `FOUODJI_Prudencia_fichier1.ipynb` | Optimisation sans contrainte : descente à pas fixe, descente à pas optimal, gradient conjugué, quasi-Newton BFGS |
| `FOUODJI_Prudencia_fichier2.ipynb` | Gradient stochastique (SGD) pour la régression linéaire, et sac à dos 0/1 par programmation dynamique |


## 2. Prérequis et installation

Bibliothèques nécessaires :

- numpy
- matplotlib
- jupyter (ou JupyterLab / VS Code avec l'extension Jupyter)

Installation :

```bash
python -m venv venv
source venv/bin/activate          # Windows : venv\Scripts\activate
pip install numpy matplotlib jupyter
```

Lancement :

```bash
jupyter notebook
```

Ouvrir ensuite chaque notebook et exécuter toutes les cellules dans l'ordre (Kernel > Restart & Run All).

## 3. Notebook 1 : méthodes de descente

### 3.1 Problème traité

Minimisation d'une fonction quadratique strictement convexe :

```
f(x) = 0.5 * x^T A x - b^T x          gradient : g(x) = A x - b
```

Jeu de test :

```
A  = [[4, 1],
      [1, 3]]            (symétrique définie positive)
b  = [1, 2]
x0 = [4, -3]
```

Solution exacte (calculée par `np.linalg.solve`) : x* = [0.090909, 0.636364] (soit [1/11, 7/11]), avec f(x*) = -0.681818.

### 3.2 Contenu du notebook

1. **Fonctions utilitaires** : `_validate_vector`, `_validate_max_iter`, `_validate_tol`, `_validate_callable` et `_result`, qui uniformise le format de sortie.
2. **Descente de gradient à pas fixe** (`gradient_descent_fixed_step`) : mise à jour `x_{k+1} = x_k - alpha * g(x_k)` avec alpha constant. Arrêt quand `||g|| < tol`.
3. **Descente à pas optimal** (`gradient_descent_exact_quadratic`) : pour une quadratique, le pas exact est `alpha_k = (g^T g) / (g^T A g)`. Vérifie que A est symétrique définie positive.
4. **Gradient conjugué** (`conjugate_gradient`) : résout `A x = b`. En arithmétique exacte, il converge en au plus n itérations pour une matrice n x n.
5. **Quasi-Newton BFGS** (`bfgs`) : approximation itérative de l'inverse de la Hessienne, avec recherche linéaire par backtracking sous condition d'Armijo (`backtracking_armijo`). La mise à jour n'est faite que si la condition de courbure `y^T s > 0` est satisfaite.
6. **Tests, comparaison et visualisations** : erreur `||x_k - x*||` et écart `f(x_k) - f(x*)` en échelle logarithmique.

### 3.3 Format de sortie des algorithmes

Chaque algorithme renvoie un dictionnaire :

| Clé | Signification |
|---|---|
| `x` | solution trouvée |
| `fun` | valeur de la fonction objectif en `x` |
| `grad_norm` | norme du gradient (ou du résidu) finale |
| `iterations` | nombre d'itérations effectuées |
| `converged` | booléen de convergence |
| `history` | historique (points, valeurs, normes, pas selon la méthode) |
| `message` | message explicatif |

### 3.4 Résultats obtenus

| Méthode | Itérations | Erreur sur x |
|---|---|---|
| Pas fixe (alpha = 0.15, tol = 1e-10) | 59 | 3.77e-11 |
| Pas optimal | 21 | 2.15e-11 |
| Gradient conjugué | 2 | 3.61e-16 |
| BFGS | 11 | 3.51e-16 |

Interprétation : le gradient conjugué converge en 2 itérations (n = 2), conformément à la théorie. BFGS est très rapide sans avoir besoin de la Hessienne explicite. Les méthodes de descente simples convergent linéairement, le pas optimal étant nettement plus efficace qu'un pas fixe bien choisi.

## 4. Notebook 2 : SGD et sac à dos 0/1

### 4.1 Partie A : gradient stochastique (régression linéaire)

Modèle et perte :

```
y_hat = X theta
J(theta) = (1 / (2n)) * ||X theta - y||^2
```

Mise à jour stochastique, une observation i à la fois :

```
theta_{k+1} = theta_k - alpha_k * x_i * (x_i^T theta_k - y_i)
alpha_k = learning_rate / (1 + decay * (k - 1))
```

Fonctions principales :

- `validate_training_data(X, y)` : validation des entrées.
- `mse_half(X, y, theta)` : calcul de J(theta).
- `stochastic_gradient_descent_linear_regression(...)` : SGD avec mélange des données à chaque époque (`shuffle`), décroissance du pas (`decay`), graine aléatoire (`random_state`) et arrêt anticipé par `tol` et `patience`.

Données synthétiques :

```
y = 5 + 3*x1 - 2*x2 + bruit      (bruit gaussien d'écart-type 0.5)
n = 1000 observations, graine 42
theta_true = [5, 3, -2]
```

Hyperparamètres utilisés : `learning_rate = 0.01`, `epochs = 100`, `decay = 1e-4`, `tol = 1e-10`, `patience = 10`.

Résultats :

| Quantité | Valeur |
|---|---|
| theta appris | [5.012639, 3.002069, -2.009166] |
| Perte finale | 0.129035 |
| Distance à theta_true | 0.0157 |
| Solution analytique (équations normales) | [5.016553, 3.001446, -2.010307] |
| Distance SGD / analytique | 0.0041 |

Remarque : `converged` vaut False car la perte continue de fluctuer légèrement à cause du caractère stochastique, sans stabilisation stricte à 1e-10 sur 10 époques consécutives. Les paramètres obtenus sont néanmoins très proches de la solution exacte.

Visualisations : courbe de perte par époque, et évolution de chaque paramètre theta[j] comparée à sa valeur cible.

### 4.2 Partie B : sac à dos 0/1 (programmation dynamique)

Problème : choisir un sous-ensemble d'objets (poids entiers positifs, valeurs réelles) maximisant la valeur totale sans dépasser la capacité.

Récurrence (table `dp` de taille (n+1) x (capacité+1)) :

```
dp[i][c] = dp[i-1][c]                                   si w_i > c
dp[i][c] = max(dp[i-1][c], dp[i-1][c - w_i] + v_i)      sinon
```

La fonction `knapsack_01_dynamic_programming(weights, values, capacity)` renvoie la valeur optimale, le poids total, les indices sélectionnés (reconstruction en remontant la table) et la table `dp`. Complexité : O(n * capacité) en temps et en mémoire.

Exemple de validation :

| Objet | Poids | Valeur |
|---|---|---|
| 0 | 2 | 12 |
| 1 | 1 | 10 |
| 2 | 3 | 20 |
| 3 | 2 | 15 |

Capacité 5. Résultat : objets [0, 1, 3], poids total 5, valeur totale 37.

Des assertions vérifient l'absence de doublons, le respect de la capacité et la cohérence poids/valeur calculés. Une heatmap de la table `dp` est également tracée.

Test plus grand : 30 objets aléatoires (graine 123), poids dans [1, 19], valeurs dans [5, 99], capacité 100. Résultat : valeur optimale 907, poids total 99, 15 objets sélectionnés.

## 5. Structure du dépôt

```
.
|-- FOUODJI_Prudencia_fichier1.ipynb
|-- FOUODJI_Prudencia_fichier2.ipynb
|-- README.md
```

## 6. Reproductibilité

- Notebook 2 : les graines sont fixées (`RANDOM_SEED = 42` pour le SGD, `123` pour le sac à dos), donc les résultats sont reproductibles à version de NumPy égale.
- Notebook 1 : l'exécution est déterministe.

## 7. Limites et remarques

- La descente à pas optimal et le gradient conjugué sont implémentés pour les fonctions quadratiques uniquement (matrice A symétrique définie positive exigée, sinon une exception est levée).
- La descente à pas fixe exige un pas suffisamment petit : pour une quadratique, alpha doit être inférieur à 2 / lambda_max(A), sinon l'algorithme diverge (une erreur explicite est alors levée).
- Dans le notebook 1, la numérotation des sections saute de 4 à 7 (les sections 5 et 6 ne sont pas présentes).
- Dans le notebook 1, la cellule de la section 9 contient une formule LaTeX mal rendue pour alpha_k (caractère de contrôle parasite). Le calcul lui-même n'est pas affecté.
- Le sac à dos par programmation dynamique ne fonctionne qu'avec des poids entiers et devient coûteux pour de très grandes capacités.

## 8. Pistes d'amélioration

- Étendre le SGD au mini-batch et ajouter momentum, Adam ou RMSprop.
- Comparer BFGS avec L-BFGS et avec `scipy.optimize.minimize`.
- Tester les méthodes sur des fonctions non quadratiques (par exemple la fonction de Rosenbrock).
- Ajouter un programme de branch and bound ou une heuristique gloutonne pour le sac à dos, afin de comparer les temps de calcul.

## 9. Références

- Nocedal, J. et Wright, S., Numerical Optimization, Springer.
- Boyd, S. et Vandenberghe, L., Convex Optimization, Cambridge University Press.
- Cormen, T. et al., Introduction to Algorithms (chapitre sur la programmation dynamique).
- Bottou, L., Optimization Methods for Large-Scale Machine Learning (SGD).