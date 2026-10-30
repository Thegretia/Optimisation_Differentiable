"""
Algorithme de quasi-Newton : BFGS (approximation de l'inverse de la Hessienne).

    d_k      = -H_k grad f(x_k)
    x_{k+1}  = x_k + t_k d_k
    s_k = x_{k+1} - x_k ,  y_k = grad f(x_{k+1}) - grad f(x_k) ,  rho_k = 1/(y_k.s_k)
    H_{k+1}  = (I - rho s y^T) H_k (I - rho y s^T) + rho s s^T

Aucune Hessienne n'est calculée : seul le gradient (obtenu avec sympy) est utilisé.

Sécurités pour que la méthode marche dans tous les cas :
  * H_0 = I, puis mise à l'échelle (s.y / y.y) I à la première mise à jour ;
  * mise à jour sautée si la condition de courbure s.y > 0 n'est pas satisfaite
    (H_k reste alors définie positive) ;
  * si d n'est pas une direction de descente ou si la recherche linéaire échoue,
    on réinitialise H = I (direction du gradient) ;
  * recherche linéaire d'Armijo et détection des valeurs non finies.
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
    historique: list = field(default_factory=list)  # x_0, x_1, ...


def construire_fonctions(expr, variables):
    """Retourne f(x) et grad(x) (numpy) à partir d'une expression sympy."""
    variables = list(variables)
    n = len(variables)

    # BFGS n'a besoin QUE du gradient : pas de Hessienne symbolique ici
    # (c'est justement l'intérêt de la méthode).
    grad_expr = sp.Matrix([sp.diff(expr, v) for v in variables])

    # Conversion sympy -> numpy (appel : f_num([x1, ..., xn])).
    f_num = sp.lambdify([variables], expr, "numpy")
    g_num = sp.lambdify([variables], grad_expr, "numpy")

    # --- Closures protégées : sentinelles au lieu d'exceptions ---
    def f(x):
        with np.errstate(all="ignore"):
            try:
                v = complex(f_num(list(x)))
            except Exception:
                return np.inf
        # Valeur complexe ou non finie => point inutilisable.
        if v.imag != 0 or not np.isfinite(v.real):
            return np.inf
        return float(v.real)

    def grad(x):
        with np.errstate(all="ignore"):
            try:
                return np.asarray(g_num(list(x)), dtype=float).reshape(n)
            except Exception:
                return np.full(n, np.nan)

    return f, grad


def recherche_lineaire(f, x, fx, g, d, c1=1e-4, beta=0.5, t_min=1e-20):
    """Backtracking d'Armijo."""
    t = 1.0                          # on essaie d'abord le pas complet
    pente = g @ d                    # g.d < 0 pour une direction de descente
    marge = 1e-14 * (1.0 + abs(fx))  # tolérance pour les erreurs d'arrondi

    while t > t_min:
        ft = f(x + t * d)
        # Condition d'Armijo : décroissance suffisante.
        if np.isfinite(ft) and ft <= fx + c1 * t * pente + marge:
            return t, ft            # pas accepté (+ valeur de f en ce point)
        t *= beta                   # sinon, on réduit le pas

    return 0.0, fx                  # aucun pas acceptable


def bfgs(expr, variables, x0, tol=1e-8, max_iter=1000, verbose=False):
    """
    Minimise expr (expression sympy) par la méthode BFGS.

    Paramètres
    ----------
    expr      : expression sympy de la fonction à minimiser
    variables : liste des symboles sympy
    x0        : point de départ
    tol       : arrêt si ||grad f(x)|| <= tol
    max_iter  : nombre maximum d'itérations
    """
    f, grad = construire_fonctions(expr, variables)

    # Point de départ en vecteur numpy plat ; n = dimension.
    x = np.asarray(x0, dtype=float).reshape(-1)
    n = len(x)
    if n != len(list(variables)):
        raise ValueError("x0 et variables n'ont pas la même dimension")

    # f(x0) doit être finie.
    fx = f(x)
    if not np.isfinite(fx):
        raise ValueError("f(x0) n'est pas finie : choisir un autre point de départ")
    g = grad(x)

    # --- État interne de BFGS ---
    I = np.eye(n)
    H = I.copy()            # approximation de l'INVERSE de la Hessienne
    H_est_identite = True   # vrai si H vient d'être (ré)initialisée à I
    premiere_maj = True     # vrai tant qu'aucune mise à jour BFGS n'a eu lieu
    historique = [x.copy()]

    for k in range(max_iter):
        # --- Sécurité : gradient exploitable ? ---
        if not np.all(np.isfinite(g)):
            return Resultat(x, fx, np.nan, k, False, "gradient non fini", historique)

        ng = np.linalg.norm(g)
        if verbose and k % 10 == 0:
            print(f"it {k:4d} | f = {fx:.6e} | ||grad|| = {ng:.3e}")

        # --- Test d'arrêt ---
        if ng <= tol:
            return Resultat(x, fx, ng, k, True, "convergence : ||grad|| <= tol", historique)

        # --- Direction de recherche : d = -H g ---
        d = -H @ g
        # Sécurité : d doit être une direction de descente, c.-à-d. g.d < 0.
        # Le second terme est un critère RELATIF : si |g.d| est négligeable
        # devant ||g|| ||d||, la direction est jugée non fiable (annulation
        # numérique) et on réinitialise.
        if not np.all(np.isfinite(d)) or g @ d >= -1e-12 * ng * np.linalg.norm(d):
            # Réinitialisation complète : H = I => d = -g (descente de gradient).
            H, H_est_identite, premiere_maj = I.copy(), True, True
            d = -g

        # --- Recherche linéaire ---
        t, f_new = recherche_lineaire(f, x, fx, g, d)
        if t == 0.0:
            if H_est_identite:
                # Même avec la direction du gradient, impossible d'avancer :
                # on est bloqué par la précision machine.
                return Resultat(x, fx, ng, k, False,
                                "échec de la recherche linéaire (précision machine atteinte)",
                                historique)
            # Sinon, on retente après réinitialisation à la direction du gradient,
            # sans consommer d'itération "utile" (on passe à l'itération suivante
            # avec le même x, mais H = I).
            H, H_est_identite, premiere_maj = I.copy(), True, True
            continue

        # --- Quantités de la mise à jour BFGS ---
        s = t * d               # déplacement effectif x_{k+1} - x_k
        x_new = x + s
        g_new = grad(x_new)
        if not np.all(np.isfinite(g_new)):
            return Resultat(x, fx, ng, k, False, "gradient non fini", historique)
        y = g_new - g           # variation du gradient
        sy = s @ y              # courbure mesurée le long de s

        # --- Condition de courbure ---
        # On ne met à jour que si s.y > 0 (f localement convexe le long de s).
        # Le seuil relatif 1e-10 évite les mises à jour numériquement instables
        # quand s.y est à peine positif.
        if sy > 1e-10 * np.linalg.norm(s) * np.linalg.norm(y):
            if premiere_maj:
                # Premier pas : H = I n'est pas à l'échelle du problème.
                # On l'initialise à (s.y / y.y) I, qui approxime l'inverse
                # de la courbure moyenne (cf. Nocedal & Wright).
                H = (sy / (y @ y)) * I
                premiere_maj = False
            rho = 1.0 / sy
            # Formule BFGS "inverse" :
            #   H+ = (I - rho s y^T) H (I - rho y s^T) + rho s s^T.
            # np.outer(s, y) construit la matrice s y^T.
            V = I - rho * np.outer(s, y)
            H = V @ H @ V.T + rho * np.outer(s, s)
            # Symétrisation : H doit rester symétrique, les arrondis pouvant
            # introduire une petite asymétrie.
            H = 0.5 * (H + H.T)
            H_est_identite = False

        # --- Mise à jour de l'itéré ---
        x, fx, g = x_new, f_new, g_new
        historique.append(x.copy())

    return Resultat(x, fx, np.linalg.norm(g), max_iter, False,
                    "nombre maximum d'itérations atteint", historique)



# Exemple
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Symboles réels (utilisés par sympy pour simplifier les dérivées).
    x, y, z = sp.symbols("x y z", real=True)

    # --- Exemple 1 : Rosenbrock, test classique de robustesse ---
    print("=== Exemple 1 : Rosenbrock, minimum en (1, 1) ===")
    rosen = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    r = bfgs(rosen, [x, y], [-1.2, 1.0], verbose=True)
    print(r.message)
    print("x* =", r.x, "| f(x*) =", r.fx, "| itérations =", r.iterations)

   