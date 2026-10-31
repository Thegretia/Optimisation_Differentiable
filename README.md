# Algorithmes d'optimisation *from scratch*

Implémentation en **Python pur** (sans `numpy`, sans bibliothèque d'optimisation) de quatre algorithmes classiques de minimisation sans contrainte d'une fonction $f:\mathbb{R}^n \to \mathbb{R}$ :

1. **Algorithme de Newton**
2. **Algorithme quasi-Newton (BFGS)**
3. **Algorithme du gradient à pas fixe**
4. **Algorithme du gradient à pas optimal**

Le projet est présenté sous forme d'un notebook Jupyter pédagogique : chaque algorithme est précédé d'une cellule Markdown qui explique le principe et déroule un **exemple numérique à la main**, puis le code est entièrement **commenté ligne par ligne**.

## Contenu du dépôt

```
.
├── algorithmes_optimisation.ipynb   # notebook principal (explications + code commenté)
└── README.md
```

## Ce qui est codé à la main

Aucune fonction pré-existante de calcul n'est utilisée dans les algorithmes. Seul le module standard `math` sert (pour `sqrt`). Sont écrits dans le notebook :

- produit scalaire, norme, addition, soustraction, multiplication par un scalaire ;
- produit matrice-vecteur, produit matriciel, produit extérieur, matrice identité ;
- résolution d'un système linéaire par **élimination de Gauss avec pivot partiel** ;
- recherche linéaire exacte par **section dorée** (avec encadrement) ;
- recherche linéaire d'**Armijo** (rebroussement) ;
- mise à jour **BFGS** de l'approximation de l'inverse de la hessienne.

`matplotlib` est utilisé uniquement pour tracer les graphiques de comparaison à la fin du notebook, jamais dans les algorithmes.

## Les algorithmes

Tous suivent le schéma itératif $x_{k+1} = x_k + \alpha_k d_k$ avec le critère d'arrêt $\|\nabla f(x_k)\| < \varepsilon$.

| Algorithme | Direction $d_k$ | Pas $\alpha_k$ | Convergence |
|---|---|---|---|
| Gradient à pas fixe | $-\nabla f(x_k)$ | constant | linéaire |
| Gradient à pas optimal | $-\nabla f(x_k)$ | $\arg\min_\alpha f(x_k + \alpha d_k)$ | linéaire |
| Quasi-Newton (BFGS) | $-H_k \nabla f(x_k)$, $H_k \approx \nabla^2 f(x_k)^{-1}$ | Armijo | superlinéaire |
| Newton | $-\nabla^2 f(x_k)^{-1} \nabla f(x_k)$ | 1 | quadratique (locale) |

## Problèmes tests

**1. Fonction quadratique** $f(x) = \tfrac12 x^\top A x - b^\top x$ avec

$$A = \begin{pmatrix} 10 & 2 \\ 2 & 1 \end{pmatrix}, \quad b = \begin{pmatrix} 1 \\ 2 \end{pmatrix}, \quad x_0 = \begin{pmatrix} 2 \\ 2 \end{pmatrix}$$

Minimiseur exact : $x^* = (-0.5,\ 3)$, $f(x^*) = -2.75$. Conditionnement $\kappa \approx 18$.

**2. Fonction de Rosenbrock** $f(x_1, x_2) = (1 - x_1)^2 + 100\,(x_2 - x_1^2)^2$, départ $x_0 = (-1.2,\ 1)$, minimum en $(1, 1)$.

## Résultats obtenus

Tolérance $10^{-6}$ sur la norme du gradient pour la quadratique ($10^{-8}$ pour Rosenbrock).

**Fonction quadratique**

| Algorithme | Itérations |
|---|---|
| Newton | 1 |
| Quasi-Newton (BFGS) | 7 |
| Gradient à pas optimal | 9 |
| Gradient à pas fixe ($\alpha = 0.1$) | 231 |

**Rosenbrock**

| Algorithme | Itérations | Remarque |
|---|---|---|
| Newton | 6 | converge |
| Quasi-Newton (BFGS) | 35 | converge |
| Gradient à pas optimal | 500 | non convergé (limite d'itérations atteinte) |

Le pas fixe diverge si $\alpha > 2/\lambda_{\max}(A) \approx 0.192$ ; le notebook le montre avec $\alpha = 0.25$.

## Installation et utilisation

**Prérequis** : Python 3.8 ou plus, Jupyter, matplotlib.

```bash
# 1. Cloner le dépôt
git clone https://github.com/<votre-utilisateur>/<nom-du-depot>.git
cd <nom-du-depot>

# 2. Installer les dépendances
pip install jupyter matplotlib

# 3. Lancer le notebook
jupyter notebook algorithmes_optimisation.ipynb
```

Exécutez ensuite les cellules **dans l'ordre** (la première cellule de code définit les outils d'algèbre linéaire utilisés par toutes les suivantes).

## Structure du notebook

0. Outils d'algèbre linéaire faits maison et problème test
1. Algorithme de Newton
2. Gradient à pas fixe
3. Gradient à pas optimal
4. Quasi-Newton (BFGS)
5. Comparaison sur la fonction quadratique (tableau et graphiques)
6. Test sur Rosenbrock (fonction non quadratique)
7. Synthèse

## Adapter à votre propre fonction

Chaque algorithme prend en entrée des fonctions Python. Pour minimiser votre fonction, définissez `f(x)`, `grad_f(x)` et (pour Newton) `hess_f(x)`, où `x` est une liste de nombres, puis appelez par exemple :

```python
x, xs, normes = newton(grad_f, hess_f, [0.0, 0.0], tol=1e-8)
x, xs, normes = quasi_newton_bfgs(f, grad_f, [0.0, 0.0], tol=1e-8)
x, xs, normes = gradient_pas_fixe(grad_f, [0.0, 0.0], alpha=0.05)
x, xs, normes, pas = gradient_pas_optimal(f, grad_f, [0.0, 0.0])
```

Chaque fonction retourne le dernier itéré, l'historique des itérés et l'historique des normes du gradient.

## Auteur

Projet réalisé par **<Votre nom>**.

## Licence

Ce projet est distribué sous licence MIT (à adapter selon vos besoins).