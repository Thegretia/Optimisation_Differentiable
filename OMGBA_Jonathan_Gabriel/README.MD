# Algorithmes d'optimisation sans contrainte (numpy)

Implémentation de quatre algorithmes de minimisation d'une fonction f : ℝⁿ → ℝ, chacun dans son notebook, avec le principe, un exemple calculé à la main, le code et une conclusion.

## Contenu

- `gradient_pas_fixe.ipynb` : gradient à pas constant
- `gradient_pas_optimal.ipynb` : gradient avec pas optimal (recherche linéaire exacte)
- `newton.ipynb` : algorithme de Newton
- `quasi_newton_bfgs.ipynb` : quasi-Newton (BFGS) avec recherche d'Armijo

## Problème test

f(x) = ½ xᵀAx − bᵀx avec A = [[6, 2], [2, 3]], b = [2, 1], point de départ x₀ = [3, −2].

- Gradient : ∇f(x) = Ax − b ; hessienne : A (constante).
- Valeurs propres de A : 2 et 7 (conditionnement κ = 3,5).
- Minimiseur exact : x* = [2/7, 1/7] ≈ [0,2857 ; 0,1429].

## Résultats (tolérance 10⁻⁶ sur la norme du gradient)

| Algorithme | Itérations | Erreur ‖x − x*‖ |
|---|---|---|
| Gradient à pas fixe (α = 0,1) | 71 | 4,1·10⁻⁷ |
| Gradient à pas optimal | 25 | 2,0·10⁻⁷ |
| Newton | 1 | 3,4·10⁻¹⁶ |
| Quasi-Newton (BFGS) | 6 | 3,9·10⁻¹⁰ |

## Analyse

- Le pas fixe converge si 0 < α < 2/λmax = 2/7 ≈ 0,286 ; avec α = 0,3 il diverge.
- Le pas optimal est bien plus rapide, mais les gradients successifs sont orthogonaux : la trajectoire zigzague.
- Newton converge en une itération car f est quadratique, mais il exige la hessienne et la résolution d'un système linéaire à chaque pas.
- BFGS approche l'inverse de la hessienne à partir des seuls gradients : sa convergence est superlinéaire, sans calcul de dérivées secondes.

## Utilisation

Prérequis : Python 3, Jupyter et numpy (`pip install jupyter numpy`). Chaque notebook est autonome : exécuter les cellules dans l'ordre.

## Auteur

Jonathan Gabriel Omgba