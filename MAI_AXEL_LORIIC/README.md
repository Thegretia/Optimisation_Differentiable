# Optimisation différentiable : minimisation de f(x, y) = x⁴ + y⁴ − 4xy

Étude comparative de cinq méthodes de descente, réparties dans quatre notebooks Jupyter.

## 1. Objet du travail

Ce travail applique cinq algorithmes d'optimisation différentiable sans contrainte à un même problème test, avec les mêmes données initiales et le même critère d'arrêt : la méthode de Newton, la méthode quasi-Newton BFGS, la méthode du gradient à pas fixe, la méthode du gradient conjugué et la méthode du gradient à pas optimal.

L'objectif est de confronter, sur un exemple où tout peut être calculé analytiquement, les garanties théoriques de chaque méthode (ordre de convergence, conditions sur le pas, rôle du conditionnement) à leur comportement numérique (nombre d'itérations, coût par itération, robustesse, limites).

## 2. Problème étudié

On minimise

    f(x, y) = x⁴ + y⁴ − 4xy,

dont le gradient et le Hessien sont

    ∇f(x, y) = (4x³ − 4y, 4y³ − 4x),        H_f(x, y) = [[12x², −4], [−4, 12y²]].

La condition ∇f = 0 conduit à x = x⁹, soit trois points critiques :

| Point | Valeurs propres de H_f | Nature | f |
|---|---|---|---|
| (0, 0) | −4 et 4 | point selle | 0 |
| (1, 1) | 8 et 16 | minimum local strict, global | −2 |
| (−1, −1) | 8 et 16 | minimum local strict, global | −2 |

Cette fonction a été retenue pour trois raisons : elle n'est pas quadratique, de sorte que les méthodes ne terminent pas en un nombre fini d'itérations ; elle possède deux minima globaux et un point selle, ce qui permet d'étudier la dépendance au point initial ; sa courbure varie fortement (les valeurs propres du Hessien croissent avec la distance à l'origine), alors qu'elle est bien conditionnée au minimum (κ = 16/8 = 2).

Paramètres communs à tous les notebooks :

| Élément | Valeur |
|---|---|
| Point initial | x₀ = (2 ; 1,5), avec f(x₀) = 9,0625 et ‖∇f(x₀)‖₂ ≈ 26,58 |
| Critère d'arrêt | ‖∇f(xₖ)‖₂ < 10⁻¹⁰ |
| Calcul symbolique | SymPy (gradient, Hessien, points critiques, coefficients des polynômes de pas) |
| Calcul numérique et graphiques | NumPy, Pandas, Matplotlib |

## 3. Contenu du dépôt

| Fichier | Méthode(s) | Information utilisée |
|---|---|---|
| `Newton_Optimisation.ipynb` | Newton | gradient et Hessien exact |
| `QuasiNewton_BFGS_Optimisation.ipynb` | Quasi-Newton BFGS, recherche linéaire d'Armijo | gradient, approximation de l'inverse du Hessien |
| `Gradient_PasFixe_GradientConjugue_Optimisation.ipynb` | Gradient à pas fixe ; gradient conjugué (Fletcher–Reeves, Polak–Ribière+) | gradient |
| `Gradient_PasOptimal_Optimisation.ipynb` | Gradient à pas optimal (recherche linéaire exacte) | gradient |
| `README.md` | Le présent document | |

Les quatre notebooks suivent la même structure en six parties : introduction, énoncé de l'exercice, résolution mathématique, implémentation Python, résultats et visualisations, interprétation et conclusion. Chaque notebook est autonome et peut être exécuté séparément. Chacun vérifie symboliquement les formules établies à la main, présente le tableau des itérations, la courbe de convergence et le chemin suivi sur les courbes de niveau, puis discute les limites de la méthode.

## 4. Présentation des notebooks

### 4.1 Méthode de Newton

La méthode minimise à chaque itération le modèle de Taylor d'ordre 2 de f en résolvant le système linéaire H_f(xₖ) dₖ = −∇f(xₖ), avec un pas unitaire. Elle converge en 6 itérations vers (1, 1), et le rapport eₖ₊₁/eₖ² reste borné, signature d'une convergence quadratique.

Les limites étudiées sont la dépendance au point initial (depuis (0,3 ; 0,2) et (−0,3 ; 0,1), la méthode converge vers le point selle, car elle recherche un point critique et non un minimum), le Hessien singulier sur la courbe |xy| = 1/3, l'absence de globalisation et le coût par itération en O(n²) pour le Hessien et O(n³) pour le système linéaire. Les remèdes classiques sont rappelés : recherche linéaire, région de confiance, régularisation de Levenberg–Marquardt, méthodes quasi-Newton.

### 4.2 Quasi-Newton BFGS

La méthode remplace le Hessien par une approximation de son inverse, mise à jour à chaque itération à partir des seuls gradients, sous l'équation sécante Hₖ₊₁ yₖ = sₖ. Le notebook établit l'équation sécante et la condition de courbure sₖᵀyₖ > 0 qui préserve la définie positivité, et vérifie numériquement ces propriétés. Le pas est choisi par recherche linéaire d'Armijo, avec un pas initial min(1, 1/‖g₀‖) à la première itération, la matrice initiale H₀ = I ne contenant aucune information d'échelle.

BFGS converge en 13 itérations, avec une convergence superlinéaire. La matrice construite se rapproche de l'inverse exact du Hessien sans jamais calculer celui-ci. Contrairement à Newton, BFGS est une méthode de descente : depuis les départs proches du point selle, il atteint un minimum. Le cas du départ (1 ; −1), qui converge vers (0, 0) par symétrie, est expliqué. Les limites traitées sont la condition de courbure avec Armijo seul, le choix de H₀ et le coût mémoire en O(n²), qui motive L-BFGS.

### 4.3 Gradient à pas fixe et gradient conjugué

Ce notebook traite deux méthodes du premier ordre.

**Gradient à pas fixe.** L'itération est xₖ₊₁ = xₖ − α∇f(xₖ). Le notebook établit la condition de décroissance α < 2/L (avec L ≈ 49,5 sur l'ensemble de sous-niveau de f(x₀), soit α < 0,0404) et l'analyse locale : taux ρ(α) = max |1 − αλᵢ| avec λᵢ ∈ {8, 16}, pas optimal local α = 1/12 et taux 1/3. Il met en évidence le dilemme du pas fixe, qui doit rester petit pour la courbure maximale rencontrée alors qu'un pas plus grand serait efficace près du minimum. Avec α = 0,04, la méthode converge en 61 itérations ; avec α = 1/12, en 29. Une étude de l'influence de α montre une convergence lente pour les petits pas, un cycle persistant à α = 0,13 et la divergence à α = 0,15.

**Gradient conjugué.** Le notebook montre que la méthode termine en n = 2 itérations sur le modèle quadratique local (directions A-conjuguées), établit le pas optimal pour f en résolvant une cubique, et compare les variantes Fletcher–Reeves (13 itérations) et Polak–Ribière+ (6 itérations, convergence superlinéaire). Il illustre la perte de conjugaison hors du cas quadratique (stagnation de FR depuis (−0,3 ; 0,1), corrigée par un redémarrage ou par PR+) et l'effet du conditionnement sur une quadratique de nombre de conditionnement 100 : 1 382 et 2 292 itérations pour le pas fixe, 2 pour le gradient conjugué.

### 4.4 Gradient à pas optimal

La direction est −∇f(xₖ) et le pas αₖ est le minimiseur exact de φₖ(α) = f(xₖ − α∇f(xₖ)). Comme f est polynomiale, φₖ′ est une cubique dont les coefficients sont vérifiés symboliquement ; on retient la racine positive qui minimise φₖ, sans évaluation supplémentaire du gradient.

Le notebook démontre quatre propriétés : l'orthogonalité des gradients successifs ; en dimension 2, la colinéarité de gₖ et gₖ₊₂, donc un chemin en escalier le long de deux axes fixes (g₀ et g₀⊥) ; une décroissance garantie avec αₖ ≥ 1/L sans connaître L ; et, sur un modèle quadratique, la formule αₖ = gᵀg / gᵀAg et le taux de Kantorovich ((κ−1)/(κ+1))².

Résultats : convergence en 22 itérations, soit environ 2,8 fois moins qu'à pas fixe avec α = 0,04 (61) et 24 % de moins qu'avec le meilleur pas fixe local α = 1/12 (29). Le rapport (f − f*)ₖ₊₁/(f − f*)ₖ se stabilise à 0,0946, sous la borne théorique 1/9, et les pas alternent entre 0,0734 et 0,0963, dans l'intervalle [1/16 ; 1/8] prévu par la théorie. Les propriétés d'orthogonalité et de colinéarité sont vérifiées numériquement.

Limites étudiées : dégradation proportionnelle au conditionnement (depuis un départ défavorable sur une quadratique, 22, 128, 1 399 et 15 141 itérations pour κ = 2, 10, 100 et 1 000), coïncidence dans ce pire cas avec le meilleur pas fixe 2/(μ+L), convergence en une itération vers le point selle depuis tout point de la droite y = −x, arrêt prématuré possible au point selle lorsque le départ est à 10⁻⁷ de cette droite (le seul critère du gradient ne distingue pas un point selle d'un minimum), présence de trois racines positives dans 0,66 % des recherches linéaires, et coût de la recherche linéaire exacte, négligeable ici uniquement parce que f est polynomiale. Sur 2 000 départs aléatoires dans [−3, 3]², l'algorithme converge dans tous les cas vers l'un des deux minima globaux, en 21 itérations en médiane.

## 5. Résultats comparatifs

Résultats depuis x₀ = (2 ; 1,5), avec la tolérance 10⁻¹⁰. Tous les algorithmes convergent vers le minimum global (1, 1), avec f = −2.

| Méthode | Itérations | Pas | Convergence | Coût par itération (hors f et ∇f) |
|---|---|---|---|---|
| Newton | 6 | unitaire | quadratique | Hessien O(n²) et système linéaire O(n³) |
| Gradient conjugué PR+ | 6 | exact (cubique) | superlinéaire | O(n) et cubique |
| Gradient conjugué FR | 13 | exact (cubique) | linéaire rapide | O(n) et cubique |
| BFGS | 13 | Armijo | superlinéaire | O(n²) |
| Gradient à pas optimal | 22 | exact (cubique) | linéaire, taux d'erreur moyen ≈ 0,31 | O(n) et cubique |
| Gradient à pas fixe, α = 1/12 | 29 | constant | linéaire, taux 1/3 | O(n) |
| Gradient à pas fixe, α = 0,04 | 61 | constant | linéaire, taux 0,68 | O(n) |

## 6. Enseignements

1. **Information de courbure et vitesse.** Plus la méthode exploite la courbure, moins elle demande d'itérations : le Hessien exact (Newton) et la mémoire des directions (gradient conjugué PR+) donnent 6 itérations, contre 61 pour le gradient à pas fixe. Le gradient conjugué obtient ce résultat avec un coût par itération linéaire, sans Hessien.
2. **Le choix du pas est déterminant.** Le pas fixe exige de connaître une borne sur la courbure et reste inefficace dans une partie du parcours. La recherche linéaire exacte supprime ce réglage et garantit αₖ ≥ 1/L, la recherche d'Armijo en tient lieu pour BFGS, et le pas unitaire de Newton n'est valable que près de la solution.
3. **Conditionnement.** Il est ici favorable (κ = 2), ce qui masque les écarts. Sur une quadratique mal conditionnée, les méthodes de plus forte descente voient leur nombre d'itérations croître comme κ, alors que le gradient conjugué conserve sa terminaison en n itérations.
4. **Point selle et critère d'arrêt.** Newton est attiré par le point selle ; les méthodes de descente l'évitent sauf configuration symétrique exceptionnelle. Un critère d'arrêt portant sur le seul gradient ne distingue pas un point selle d'un minimum : une vérification du second ordre est nécessaire.
5. **Passage à grande dimension.** Le Hessien étant hors de portée, ce sont les descendants des méthodes étudiées qui sont utilisés : gradient conjugué préconditionné, L-BFGS, méthodes à moment et pas adaptatif.

## 7. Reproduction

Environnement utilisé pour la vérification des résultats : Python 3.12, NumPy 2.4, SymPy 1.14, Pandas 3.0, Matplotlib 3.10, Jupyter (nbconvert et ipykernel).

    pip install numpy sympy pandas matplotlib jupyter
    jupyter notebook

Chaque notebook s'exécute de bout en bout avec « Restart & Run All », sans fichier externe. Les résultats sont déterministes ; le seul tirage aléatoire (2 000 points de départ du dernier notebook) utilise une graine fixée. L'exécution du notebook du gradient à pas optimal prend une quinzaine de secondes.
