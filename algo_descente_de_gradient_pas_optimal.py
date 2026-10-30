"""
Algorithme du gradient à pas optimal (recherche linéaire exacte).

    d_k      = -grad f(x_k)
    alpha_k  = argmin_{a >= 0} phi(a) ,   phi(a) = f(x_k + a d_k)
    x_{k+1}  = x_k + alpha_k d_k

Calcul de alpha_k :
  * f quadratique (Hessienne constante, détectée avec sympy) : formule exacte
        alpha_k = (g.g) / (g^T A g)
  * cas général : on résout phi'(a) = -grad f(x_k - a g).g = 0 ; on encadre la
    racine en doublant a, puis on la calcule par dichotomie (phi' est négative
    en 0 puisque phi'(0) = -||g||^2).

Sécurités :
  * valeurs non finies traitées comme "dérivée positive" lors de l'encadrement ;
  * fonction non bornée dans la direction de descente -> arrêt avec message ;
  * si le pas calculé ne fait pas décroître f (erreur numérique), on le réduit
    par backtracking.
"""
from dataclasses import dataclass, field

import numpy as np
import sympy as sp


# ======================================================================
# Structure de données renvoyée par l'algorithme
# ======================================================================
@dataclass
class Resultat:
    """Résultat d'une optimisation."""
    x: np.ndarray            # point final
    fx: float                # f(point final)
    norme_grad: float        # ||grad f(point final)||
    iterations: int          # nombre d'itérations effectuées
    converge: bool           # True si ||grad|| <= tol
    message: str             # motif d'arrêt, lisible
    # Trajectoire x_0, x_1, ... (défaut mutable => default_factory obligatoire).
    historique: list = field(default_factory=list)


def construire_fonctions(expr, variables):
    """Retourne f, grad, hess (numpy) et un booléen : f est-elle quadratique ?"""
    variables = list(variables)
    n = len(variables)

    # Dérivées exactes (symboliques).
    grad_expr = sp.Matrix([sp.diff(expr, v) for v in variables])
    hess_expr = sp.hessian(expr, variables)

    # ----- Détection du cas quadratique -----
    # Une fonction est quadratique (au sens : Hessienne constante) si sa
    # Hessienne ne contient AUCUN des symboles du problème. Les symboles
    # restants éventuels (constantes, paramètres) ne comptent pas.
    # free_symbols est un ensemble => intersection avec set(variables).
    est_quadratique = len(hess_expr.free_symbols & set(variables)) == 0

    # Conversion en fonctions numériques (appel : f_num([x1, ..., xn])).
    f_num = sp.lambdify([variables], expr, "numpy")
    g_num = sp.lambdify([variables], grad_expr, "numpy")
    h_num = sp.lambdify([variables], hess_expr, "numpy")

    # --- Closures protégées : sentinelles au lieu d'exceptions ---
    def f(x):
        with np.errstate(all="ignore"):
            try:
                v = complex(f_num(list(x)))
            except Exception:
                return np.inf
        # Valeur complexe ou non finie => point inutilisable => +infini.
        if v.imag != 0 or not np.isfinite(v.real):
            return np.inf
        return float(v.real)

    def grad(x):
        with np.errstate(all="ignore"):
            try:
                return np.asarray(g_num(list(x)), dtype=float).reshape(n)
            except Exception:
                return np.full(n, np.nan)

    def hess(x):
        with np.errstate(all="ignore"):
            try:
                return np.asarray(h_num(list(x)), dtype=float).reshape(n, n)
            except Exception:
                return np.full((n, n), np.nan)

    # Le booléen est renvoyé en plus : il évite de recalculer la Hessienne
    # à chaque itération dans le cas quadratique.
    return f, grad, hess, est_quadratique


def pas_optimal_general(grad, x, g, a_max=1e12, n_bisect=200):
    """Racine de phi'(a) = -grad(x - a g).g par encadrement + dichotomie.
    Retourne None si phi n'a pas de minimiseur (non bornée)."""
    # phi(a) = f(x - a g) ; sa dérivée vaut phi'(a) = grad(x - a g) . (-g)
    #        = -grad(x - a g) . g.
    # On cherche a > 0 tel que phi'(a) = 0 (a = 0 => phi'(0) = -||g||^2 < 0).
    def dphi(a):
        ga = grad(x - a * g)
        v = -(ga @ g)
        # Si l'évaluation n'est pas finie, on considère phi' "positive" :
        # cela force l'encadrement à s'arrêter plutôt qu'à boucler.
        return v if np.isfinite(v) else np.inf  # non fini : on considère "positif"

    # --- Phase d'encadrement : on cherche [lo, hi] avec phi'(lo) < 0 <= phi'(hi) ---
    lo, hi = 0.0, 1.0
    while dphi(hi) < 0:     # phi décroît encore en hi : on va plus loin
        lo = hi
        hi *= 2.0           # doublement du pas (croissance géométrique)
        if hi > a_max:      # au-delà, on considère f non bornée dans la direction
            return None

    # --- Phase de dichotomie : phi' est supposée croissante sur [lo, hi] ---
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        if dphi(mid) < 0:
            lo = mid        # la racine est à droite
        else:
            hi = mid        # la racine est à gauche
        # Critère d'arrêt : intervalle négligeable devant l'échelle de a.
        if hi - lo <= 1e-15 * max(1.0, hi):
            break
    # On renvoie le milieu de l'intervalle final (meilleure estimation).
    return 0.5 * (lo + hi)


def gradient_pas_optimal(expr, variables, x0, tol=1e-8, max_iter=50000, verbose=False):
    """
    Minimise expr (expression sympy) par descente de gradient à pas optimal.

    Paramètres
    ----------
    expr      : expression sympy de la fonction à minimiser
    variables : liste des symboles sympy
    x0        : point de départ
    tol       : arrêt si ||grad f(x)|| <= tol
    max_iter  : nombre maximum d'itérations
    """
    # On récupère aussi l'indicateur "quadratique".
    f, grad, hess, est_quad = construire_fonctions(expr, variables)

    # Point de départ en vecteur numpy plat.
    x = np.asarray(x0, dtype=float).reshape(-1)
    if len(x) != len(list(variables)):
        raise ValueError("x0 et variables n'ont pas la même dimension")

    # f(x0) doit être finie.
    fx = f(x)
    if not np.isfinite(fx):
        raise ValueError("f(x0) n'est pas finie : choisir un autre point de départ")

    # Si f est quadratique, la Hessienne est constante : on la calcule
    # UNE seule fois et on la réutilise pour tous les pas optimaux.
    A = hess(x) if est_quad else None

    g = grad(x)
    historique = [x.copy()]

    for k in range(max_iter):
        # --- Sécurité : gradient exploitable ? ---
        if not np.all(np.isfinite(g)):
            return Resultat(x, fx, np.nan, k, False, "gradient non fini", historique)

        ng = np.linalg.norm(g)
        if verbose and k % 10 == 0:
            print(f"it {k:5d} | f = {fx:.6e} | ||grad|| = {ng:.3e}")

        # --- Test d'arrêt ---
        if ng <= tol:
            return Resultat(x, fx, ng, k, True, "convergence : ||grad|| <= tol", historique)

        # ------------------------------------------------------------------
        # Calcul du pas optimal alpha_k
        # ------------------------------------------------------------------
        alpha = None
        if est_quad:
            # Cas quadratique : phi(a) = f(x - a g) est une parabole en a.
            # Le long de la direction -g, la courbure est g^T A g.
            courbure = g @ A @ g
            if courbure > 0:
                # Minimisation exacte de la parabole :
                #   phi'(a) = -g^T grad(x - a g) = -g^T(g - a A g) = 0
                #   => a = (g.g) / (g^T A g).
                alpha = (g @ g) / courbure
            else:
                # Courbure <= 0 : f décroît indéfiniment dans la direction -g.
                return Resultat(x, fx, ng, k, False,
                                "f non bornée inférieurement dans la direction -grad "
                                "(courbure <= 0)", historique)
        else:
            # Cas général : racine de phi' par encadrement + dichotomie.
            alpha = pas_optimal_general(grad, x, g)
            if alpha is None:
                return Resultat(x, fx, ng, k, False,
                                "f non bornée inférieurement dans la direction -grad",
                                historique)

        # ------------------------------------------------------------------
        # Garde-fou numérique : le pas théorique doit réellement décroître f.
        # (La dichotomie peut être imprécise, et les arrondis peuvent faire
        # remonter f de quelques ulp.)
        # ------------------------------------------------------------------
        x_new = x - alpha * g
        f_new = f(x_new)
        marge = 1e-14 * (1.0 + abs(fx))
        # Tant que f ne décroît pas (ou est invalide), on divise le pas par 2.
        while (not np.isfinite(f_new) or f_new > fx + marge) and alpha > 1e-20:
            alpha *= 0.5
            x_new = x - alpha * g
            f_new = f(x_new)
        if alpha <= 1e-20:
            # Le pas est devenu négligeable : on est bloqué par la précision.
            return Resultat(x, fx, ng, k, False,
                            "pas devenu négligeable (précision machine atteinte)", historique)

        # --- Mise à jour ---
        x, fx = x_new, f_new
        g = grad(x)
        historique.append(x.copy())

    return Resultat(x, fx, np.linalg.norm(g), max_iter, False,
                    "nombre maximum d'itérations atteint", historique)


# ----------------------------------------------------------------------
# Exemples
# ----------------------------------------------------------------------
if __name__ == "__main__":
    x, y = sp.symbols("x y", real=True)

    # --- Exemple 1 : quadratique => formule exacte du pas ---
    # La Hessienne A = [[2, 1], [1, 20]] est constante et définie positive.
    print("=== Exemple 1 : quadratique (formule exacte du pas optimal) ===")
    f1 = x ** 2 + 10 * y ** 2 - 2 * x - 4 * y + x * y   # minimum unique
    r = gradient_pas_optimal(f1, [x, y], [5.0, 3.0])
    print(r.message)
    print("x* =", r.x, "| f(x*) =", r.fx, "| itérations =", r.iterations)
    # vérification : x* = A^{-1} b
    # (f = 1/2 x^T A x - b^T x + c => grad = A x - b = 0 <=> x = A^{-1} b.)
    A = np.array([[2.0, 1.0], [1.0, 20.0]])
    b = np.array([2.0, 4.0])
    print("solution exacte =", np.linalg.solve(A, b))

    # --- Exemple 2 : non quadratique => dichotomie sur phi'(a) ---
    print("\n=== Exemple 2 : non quadratique (dichotomie sur phi'(a)) ===")
    f2 = x ** 4 + 2 * y ** 2 + x * y - x
    r = gradient_pas_optimal(f2, [x, y], [3.0, 3.0])
    print(r.message)
    print("x* =", r.x, "| f(x*) =", r.fx, "| itérations =", r.iterations)

    # --- Exemple 3 : f non bornée inférieurement ---
    # f = x^2 - y^2 : dans la direction +y, f -> -infini. La courbure
    # g^T A g peut devenir négative => l'algorithme doit le détecter.
    print("\n=== Exemple 3 : fonction non bornée (cas limite détecté) ===")
    f3 = x ** 2 - y ** 2
    r = gradient_pas_optimal(f3, [x, y], [1.0, 1.0])
    print(r.message)
