# Optimisation convexe sans contrainte — Python pur (sans NumPy)

Implémentations *from scratch* des algorithmes de l'Appendice A (« Notions d'optimisation convexe »,
section A.3). Aucune bibliothèque externe : seul le module standard `math` est utilisé.
Vecteurs = listes de floats, matrices = listes de listes.

## Contenu

| Fichier | Rôle | Référence du cours |
|---|---|---|
| `algebre.py` | produit scalaire, norme, matrice×vecteur, produit extérieur… | — |
| `problemes.py` | 2 fonctions convexes de test (f, gradient, hessienne) | — |
| `gradient_pas_fixe.py` | gradient à pas fixe | Déf. 1.8 |
| `gradient_pas_variable.py` | gradient + recherche linéaire par rebroussement (BLS) | Déf. 1.9 |
| `gradient_conjugue.py` | gradient conjugué (résout Ax=b) et Newton par gradient conjugué | Déf. 1.11, A.3.6 |
| `quasi_newton_bfgs.py` | quasi-Newton BFGS | A.3.7, éq. (A.25) |
| `exemples.py` | compare toutes les méthodes sur les 2 problèmes | — |

## Lancer

```bash
python3 exemples.py              # tableau comparatif
python3 gradient_pas_fixe.py     # chaque script a son propre exemple
python3 gradient_pas_variable.py
python3 gradient_conjugue.py
python3 quasi_newton_bfgs.py
```

## Problèmes de test

1. **Quadratique 2D** : f(u) = ½ uᵀQu − bᵀu, Q = [[4,1],[1,3]], b = [1,2].
   Solution exacte u* = Q⁻¹b = [1/11, 7/11].
2. **Somme d'exponentielles** (convexe, non quadratique) :
   f(x,y) = e^(x+3y−0.1) + e^(x−3y−0.1) + e^(−x−0.1).
   Solution exacte u* = [½·ln(½), 0] ≈ [−0.34657, 0].

## Les algorithmes en bref

- **Pas fixe** : `u ← u − α∇f(u)`. Simple, mais α doit être choisi à la main :
  trop petit → lent, trop grand → oscillations/divergence (c'est pourquoi le point de départ
  du problème 2 est modéré : avec α fixe, on diverge si la hessienne est trop grande).
- **Pas variable (BLS)** : à chaque itération on part de α₀ et on pose `α ← βα`
  tant que `f(u − α∇f) > f(u) − (α/2)‖∇f‖²`.
- **Gradient conjugué** : résout `Ax = b` (A symétrique définie positive) en au plus n
  itérations, avec des directions A‑conjuguées (éq. A.16 à A.18). Utilisé dans
  `newton_gradient_conjugue` pour résoudre `∇²f(u) δ = ∇f(u)` puis `u ← u − δ`,
  sans jamais inverser la hessienne.
- **BFGS** : approxime l'**inverse** de la hessienne par W (W(0)=I) mise à jour via (A.25)
  avec dₜ = uₜ − uₜ₋₁ et δₜ = ∇f(uₜ) − ∇f(uₜ₋₁). Direction `p = −W∇f(u)`, pas par
  rebroussement (condition d'Armijo). La mise à jour est sautée si ⟨δ,d⟩ ≤ 10⁻¹² pour
  garder W définie positive.

## Résultats obtenus (critère d'arrêt ‖∇f‖² ≤ ε)

| Méthode | Quadratique (iter.) | Exponentielles (iter.) |
|---|---|---|
| Gradient pas fixe | 26 | 563 |
| Gradient pas variable (BLS) | 19 | 30 |
| Newton + gradient conjugué | 1 | 6 |
| Quasi-Newton BFGS | 7 | 12 |

Observations : Newton converge en **1 itération** sur une quadratique (le modèle de Taylor
d'ordre 2 est exact) ; BFGS s'en approche sans calculer la hessienne ; le pas variable
évite le réglage manuel de α.

## Limites

- Fonctions supposées convexes et différentiables (C¹, C² pour Newton).
- Newton « pur » (sans recherche linéaire) peut diverger loin du minimum sur des fonctions non quadratiques.
- Python pur : volontairement pédagogique, pas optimisé pour de grandes dimensions.
