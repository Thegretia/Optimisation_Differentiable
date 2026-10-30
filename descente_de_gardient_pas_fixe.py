"""
Algorithme du gradient à pas fixe.

    x_{k+1} = x_k - alpha * grad f(x_k)       (alpha constant)

Théorie : si f est de classe C^1 avec grad f L-lipschitzien, l'algorithme
décroît pour 0 < alpha < 2/L (et converge si f est bornée inférieurement ;
pour une fonction quadratique elliptique de plus grande valeur propre L,
la condition est exactement 0 < alpha < 2/L).

Sécurités :
  * si alpha n'est pas donné, on prend alpha = 1/L avec L = plus grande valeur
    propre en valeur absolue de la Hessienne en x0 (heuristique si f non quadratique) ;
  * détection de divergence (valeurs non finies, ||x|| ou f explosent) :
    l'algorithme s'arrête proprement et indique de diminuer alpha.
"""
from dataclasses import dataclass, field

import numpy as np
import sympy as sp


# ======================================================================
# Structure de données renvoyée par l'algorithme
# ======================================================================
@dataclass
class Resultat:
    """Résultat d'une optimisation (spécifique ici : on garde le pas utilisé)."""
    x: np.ndarray            # point final x_k
    fx: float                # f(x_k)
    norme_grad: float        # ||grad f(x_k)||
    iterations: int          # nombre d'itérations effectuées
    converge: bool           # True si le critère sur le gradient est atteint
    message: str             # motif d'arrêt, lisible
    alpha: float = 0.0       # pas fixe réellement employé (utile si auto)
    historique: list = field(default_factory=list)  # x_0, x_1, ...


def construire_fonctions(expr, variables):
    """Retourne f(x), grad(x), hess(x) (numpy) à partir d'une expression sympy."""
    # Liste concrète des symboles et dimension du problème.
    variables = list(variables)
    n = len(variables)

    # Dérivées exactes calculées symboliquement, une seule fois.
    grad_expr = sp.Matrix([sp.diff(expr, v) for v in variables])
    hess_expr = sp.hessian(expr, variables)

    # Conversion sympy -> fonctions numpy.
    # Convention d'appel : f_num([x1, ..., xn]).
    f_num = sp.lambdify([variables], expr, "numpy")
    g_num = sp.lambdify([variables], grad_expr, "numpy")
    h_num = sp.lambdify([variables], hess_expr, "numpy")

    # --- Closures "sûres" : jamais d'exception, seulement des sentinelles ---
    def f(x):
        # On ignore les warnings numpy (0/0, overflow, ...) : ils sont gérés.
        with np.errstate(all="ignore"):
            try:
                v = complex(f_num(list(x)))
            except Exception:
                return np.inf          # évaluation impossible => +infini
        # Résultat complexe ou non fini => point rejeté (+infini).
        if v.imag != 0 or not np.isfinite(v.real):
            return np.inf
        return float(v.real)

    def grad(x):
        with np.errstate(all="ignore"):
            try:
                # Vecteur plat de taille n, quoi que renvoie lambdify.
                return np.asarray(g_num(list(x)), dtype=float).reshape(n)
            except Exception:
                return np.full(n, np.nan)  # signal d'échec pour l'appelant

    def hess(x):
        with np.errstate(all="ignore"):
            try:
                return np.asarray(h_num(list(x)), dtype=float).reshape(n, n)
            except Exception:
                return np.full((n, n), np.nan)

    return f, grad, hess


def gradient_pas_fixe(expr, variables, x0, alpha=None, tol=1e-6, max_iter=100000,
                      verbose=False):
    """
    Minimise expr (expression sympy) par descente de gradient à pas fixe.

    Paramètres
    ----------
    expr      : expression sympy de la fonction à minimiser
    variables : liste des symboles sympy
    x0        : point de départ
    alpha     : pas fixe > 0 (si None : 1/L estimé à partir de la Hessienne en x0)
    tol       : arrêt si ||grad f(x)|| <= tol
    max_iter  : nombre maximum d'itérations
    """
    # Fonctions numériques f, gradient, Hessienne.
    f, grad, hess = construire_fonctions(expr, variables)

    # Point de départ en vecteur numpy plat.
    x = np.asarray(x0, dtype=float).reshape(-1)
    if len(x) != len(list(variables)):
        raise ValueError("x0 et variables n'ont pas la même dimension")

    # f(x0) doit être finie, sinon le problème est mal posé au départ.
    fx = f(x)
    if not np.isfinite(fx):
        raise ValueError("f(x0) n'est pas finie : choisir un autre point de départ")

    # ------------------------------------------------------------------
    # Choix automatique du pas si l'utilisateur n'en fournit pas.
    # Idée : pour une quadratique, le pas 1/L (L = plus grande valeur
    # propre de la Hessienne) est le pas "sûr" classique. Hors quadratique,
    # on utilise la Hessienne en x0 comme approximation locale de L.
    # ------------------------------------------------------------------
    if alpha is None:
        H0 = hess(x)
        # eigvalsh : valeurs propres d'une matrice symétrique (réelles).
        # On symétrise d'abord pour éliminer les erreurs d'arrondi.
        L = np.max(np.abs(np.linalg.eigvalsh(0.5 * (H0 + H0.T)))) if np.all(np.isfinite(H0)) else 0.0
        # Si L = 0 (Hessienne nulle ou invalide), on prend alpha = 1 par défaut.
        alpha = 1.0 / L if L > 0 else 1.0
    if alpha <= 0:
        raise ValueError("le pas alpha doit être strictement positif")

    f0 = fx                 # valeur de référence pour détecter une explosion
    g = grad(x)             # gradient courant
    historique = [x.copy()]  # copy() pour ne pas stocker une référence mutable

    for k in range(max_iter):
        # --- Sécurité : gradient exploitable ? ---
        if not np.all(np.isfinite(g)):
            return Resultat(x, fx, np.nan, k, False,
                            "divergence : gradient non fini, diminuer alpha", alpha, historique)

        ng = np.linalg.norm(g)
        # Affichage périodique (pas à chaque itération : max_iter peut être grand).
        if verbose and k % 50 == 0:
            print(f"it {k:6d} | f = {fx:.6e} | ||grad|| = {ng:.3e}")

        # --- Test d'arrêt : gradient suffisamment petit ---
        if ng <= tol:
            return Resultat(x, fx, ng, k, True, "convergence : ||grad|| <= tol", alpha, historique)

        # --- Cœur de l'algorithme : un pas de descente ---
        x = x - alpha * g
        fx = f(x)
        g = grad(x)
        historique.append(x.copy())

        # --- Sécurité anti-divergence ---
        # Trois symptômes : f non finie, x qui part à l'infini, ou f qui
        # explose par rapport à sa valeur initiale.
        if (not np.isfinite(fx)) or np.linalg.norm(x) > 1e12 or fx > f0 + 1e12 * (1 + abs(f0)):
            return Resultat(x, fx, np.linalg.norm(g), k + 1, False,
                            f"divergence : le pas alpha = {alpha:g} est trop grand, "
                            "diminuer alpha", alpha, historique)

    # Nombre maximal d'itérations atteint sans satisfaire le critère.
    return Resultat(x, fx, np.linalg.norm(g), max_iter, False,
                    "nombre maximum d'itérations atteint (essayer un pas plus grand "
                    "ou plus d'itérations)", alpha, historique)


# ----------------------------------------------------------------------
# Exemples
# ----------------------------------------------------------------------
if __name__ == "__main__":
    x, y = sp.symbols("x y", real=True)

    # f(x, y) = x^2 + 10 y^2 - 2x ; Hessienne = diag(2, 20) donc L = 20, 2/L = 0.1
    # minimum en (1, 0)
    # (Le minimum s'obtient en annulant le gradient : 2x - 2 = 0 et 20y = 0.)
    f1 = x ** 2 + 10 * y ** 2 - 2 * x

    # --- Exemple 1 : pas "raisonnable" (sous la borne 2/L) ---
    print("=== Exemple 1 : pas fixe alpha = 0.05 (< 2/L = 0.1) ===")
    r = gradient_pas_fixe(f1, [x, y], [5.0, 3.0], alpha=0.05)
    print(r.message)
    print("x* =", r.x, "| f(x*) =", r.fx, "| itérations =", r.iterations)

    # --- Exemple 2 : pas choisi automatiquement (heuristique alpha = 1/L) ---
    print("\n=== Exemple 2 : pas choisi automatiquement (alpha = 1/L) ===")
    r = gradient_pas_fixe(f1, [x, y], [5.0, 3.0])
    print(r.message, "| alpha =", r.alpha)
    print("x* =", r.x, "| itérations =", r.iterations)

    # --- Exemple 3 : pas trop grand => l'algorithme diverge (et le détecte) ---
    # alpha = 0.2 > 2/L = 0.1 : le facteur d'amplification dépasse 1.
    print("\n=== Exemple 3 : pas trop grand (alpha = 0.2 > 2/L) -> divergence détectée ===")
    r = gradient_pas_fixe(f1, [x, y], [5.0, 3.0], alpha=0.2)
    print(r.message)
    print("itérations avant arrêt =", r.iterations)

    # --- Exemple 4 : fonction non quadratique ---
    # La théorie du pas 1/L n'est qu'une heuristique ici ; on choisit
    # manuellement un petit pas pour rester dans le domaine de convergence.
    print("\n=== Exemple 4 : fonction non quadratique ===")
    f4 = (x - 1) ** 4 + (y + 2) ** 2 + x * y / 10
    r = gradient_pas_fixe(f4, [x, y], [3.0, 3.0], alpha=0.05)
    print(r.message)
    print("x* =", r.x, "| f(x*) =", r.fx, "| itérations =", r.iterations)
