"""
Algorithme de Newton (globalisé) pour la minimisation sans contrainte.

    x_{k+1} = x_k + t_k d_k ,   avec   H(x_k) d_k = -grad f(x_k)

Pour que la méthode marche "dans tous les cas", on ajoute trois sécurités :
  1. Hessienne non définie positive ou singulière -> régularisation H + mu*I
     (testée par Cholesky), ce qui garantit une direction de descente.
  2. Recherche linéaire d'Armijo (backtracking) -> pas t_k = 1 près de la
     solution (convergence quadratique conservée), pas réduit sinon.
  3. Détection des valeurs non finies (log de négatif, overflow, ...).

Bibliothèques : numpy (calcul numérique) et sympy (gradient et Hessienne exacts).
"""
from dataclasses import dataclass, field

import numpy as np
import sympy as sp


# ======================================================================
# Structure de données renvoyée par l'algorithme
# ======================================================================
@dataclass
class Resultat:
    """Tout ce qu'on veut savoir à la fin d'une optimisation."""
    x: np.ndarray            # point final x_k (vecteur numpy)
    fx: float                # valeur de f en ce point
    norme_grad: float        # ||grad f(x_k)||, mesure de l'optimalité
    iterations: int          # nombre d'itérations réellement effectuées
    converge: bool           # True si l'arrêt vient du critère sur le gradient
    message: str             # explication lisible du motif d'arrêt
    # Liste des itérés x_0, x_1, ... (utile pour tracer la trajectoire).
    # On utilise default_factory=list : indispensable pour un défaut mutable,
    # sinon la même liste serait partagée par tous les Resultat.
    historique: list = field(default_factory=list)


# ----------------------------------------------------------------------
# Outils : dérivées symboliques -> fonctions numériques
# ----------------------------------------------------------------------
def construire_fonctions(expr, variables):
    """Retourne f(x), grad(x), hess(x) (x : vecteur numpy) à partir de sympy."""
    # On matérialise la liste des symboles : elle sert plusieurs fois et
    # l'utilisateur peut très bien passer un tuple.
    variables = list(variables)
    n = len(variables)  # dimension du problème

    # --- Calcul symbolique EXACT des dérivées, fait une seule fois ---
    # Gradient : matrice colonne des dérivées partielles.
    grad_expr = sp.Matrix([sp.diff(expr, v) for v in variables])
    # Hessienne : matrice des dérivées secondes d^2 f / (dx_i dx_j).
    hess_expr = sp.hessian(expr, variables)

    # --- Transformation en fonctions numériques ---
    # lambdify traduit l'expression sympy en code Python utilisant numpy.
    # On passe [variables] (une seule liste) : la fonction générée s'appelle
    # donc f_num([x1, ..., xn]), d'où le list(x) dans les closures ci-dessous.
    f_num = sp.lambdify([variables], expr, "numpy")
    g_num = sp.lambdify([variables], grad_expr, "numpy")
    h_num = sp.lambdify([variables], hess_expr, "numpy")

    # ------------------------------------------------------------------
    # Les trois closures ci-dessous "blinder" l'évaluation numérique :
    # au lieu de laisser une exception (log de négatif, division par 0,
    # overflow...) remonter et casser l'algorithme, on renvoie une valeur
    # sentinelle (inf ou NaN) que les routines appellent à traiter.
    # ------------------------------------------------------------------
    def f(x):
        # np.errstate(all="ignore") : on gère nous-mêmes les warnings numpy
        # (division par zéro, invalide, overflow) au lieu de les afficher.
        with np.errstate(all="ignore"):
            try:
                # complex(...) : si l'expression produit un complexe (ex.
                # sqrt d'un négatif), on le détecte au lieu de planter.
                v = complex(f_num(list(x)))
            except Exception:
                # Toute erreur d'évaluation => f vaut +infini (point rejeté).
                return np.inf
        # Une valeur complexe non nulle ou non finie n'a pas de sens ici :
        # on la traite comme +infini (point "inutilisable").
        if v.imag != 0 or not np.isfinite(v.real):
            return np.inf
        return float(v.real)  # on ne garde que la partie réelle, en float

    def grad(x):
        with np.errstate(all="ignore"):
            try:
                # reshape(n) : garantit un vecteur plat de taille n, quel que
                # soit le format renvoyé par lambdify (liste, (n,1), ...).
                return np.asarray(g_num(list(x)), dtype=float).reshape(n)
            except Exception:
                # Échec d'évaluation => vecteur de NaN ; l'appelant arrêtera
                # proprement l'algorithme ("gradient non fini").
                return np.full(n, np.nan)

    def hess(x):
        with np.errstate(all="ignore"):
            try:
                return np.asarray(h_num(list(x)), dtype=float).reshape(n, n)
            except Exception:
                return np.full((n, n), np.nan)

    # On renvoie les trois fonctions ; l'expression symbolique n'est plus
    # nécessaire ensuite.
    return f, grad, hess


# ----------------------------------------------------------------------
# Direction de Newton régularisée
# ----------------------------------------------------------------------
def direction_newton(H, g):
    """Résout (H + mu I) d = -g avec mu >= 0 minimal (ordre de grandeur) tel que
    H + mu I soit définie positive. Retourne (d, mu)."""
    n = len(g)

    # Si la Hessienne contient des NaN/inf, on ne peut rien en tirer :
    # on retombe sur la direction du gradient -g (toujours une descente).
    if not np.all(np.isfinite(H)):
        return -g, np.inf

    # On symétrise la Hessienne : (H + H^T)/2. Les dérivées secondes sont
    # théoriquement symétriques ; ceci neutralise les erreurs d'arrondi.
    H = 0.5 * (H + H.T)
    I = np.eye(n)

    # Échelle caractéristique de H : sert à choisir un mu proportionné
    # (max(1, ...) évite un mu démesuré si H est minuscule).
    echelle = max(1.0, np.linalg.norm(H, "fro"))

    mu = 0.0  # régularisation courante
    for _ in range(60):  # au plus 60 essais, puis on abandonne
        try:
            # Cholesky n'aboutit QUE si la matrice est définie positive :
            # c'est notre test "H + mu I > 0".
            # H = L L^T, avec L triangulaire inférieure.
            L = np.linalg.cholesky(H + mu * I)

            # Résolution de (L L^T) d = -g en deux étapes triangulaires :
            #   L y = -g   puis   L^T d = y.
            # C'est plus stable et plus rapide qu'inverser la matrice.
            y = np.linalg.solve(L, -g)
            d = np.linalg.solve(L.T, y)

            # On vérifie que la direction est utilisable (pas de NaN/inf).
            if np.all(np.isfinite(d)):
                return d, mu
        except np.linalg.LinAlgError:
            # Matrice non définie positive (ou numériquement singulière) :
            # on augmente mu et on réessaie.
            pass

        # Premier échec : petit mu proportionnel à l'échelle de H ;
        # échecs suivants : on multiplie mu par 10 (croissance géométrique).
        mu = 1e-8 * echelle if mu == 0.0 else 10.0 * mu

    # Dernier recours (H vraiment pathologique) : direction du gradient.
    return -g, mu


def recherche_lineaire(f, x, fx, g, d, c1=1e-4, beta=0.5, t_min=1e-20):
    """Backtracking d'Armijo : f(x+td) <= f(x) + c1 t g.d"""
    t = 1.0  # on essaie d'abord le pas complet (Newton pur, t = 1)
    # Pente directionnelle : g.d < 0 pour une direction de descente.
    pente = g @ d
    # Tolérance additive : absorbe les erreurs d'arrondi quand f est
    # proche de son minimum et que la condition devient impossible à
    # satisfaire strictement.
    marge = 1e-14 * (1.0 + abs(fx))

    # Tant que le pas n'est pas devenu négligeable...
    while t > t_min:
        ft = f(x + t * d)
        # Condition d'Armijo (suffisante de décroissance) + f finie.
        if np.isfinite(ft) and ft <= fx + c1 * t * pente + marge:
            return t, ft
        # Sinon on coupe le pas en deux (backtracking).
        t *= beta

    # Aucun pas acceptable : on signale l'échec par t = 0.
    return 0.0, fx


# ----------------------------------------------------------------------
# Algorithme de Newton
# ----------------------------------------------------------------------
def newton(expr, variables, x0, tol=1e-8, max_iter=200, verbose=False):
    """
    Minimise expr (expression sympy) par la méthode de Newton globalisée.

    Paramètres
    ----------
    expr      : expression sympy de la fonction à minimiser
    variables : liste des symboles sympy, ex. [x, y]
    x0        : point de départ (liste ou array)
    tol       : arrêt si ||grad f(x)|| <= tol
    max_iter  : nombre maximum d'itérations
    """
    # Construction des fonctions numériques à partir de l'expression.
    f, grad, hess = construire_fonctions(expr, variables)

    # Point de départ sous forme de vecteur numpy plat.
    x = np.asarray(x0, dtype=float).reshape(-1)
    if len(x) != len(list(variables)):
        raise ValueError("x0 et variables n'ont pas la même dimension")

    # Évaluation initiale : un point de départ invalide est une erreur
    # d'utilisation (on ne peut pas "réparer" silencieusement).
    fx = f(x)
    if not np.isfinite(fx):
        raise ValueError("f(x0) n'est pas finie : choisir un autre point de départ")
    g = grad(x)
    historique = [x.copy()]  # copy() : sinon on stockerait une référence

    for k in range(max_iter):
        # --- Sécurité : gradient exploitable ? ---
        if not np.all(np.isfinite(g)):
            return Resultat(x, fx, np.nan, k, False, "gradient non fini", historique)

        ng = np.linalg.norm(g)
        if verbose:
            print(f"it {k:3d} | f = {fx:.6e} | ||grad|| = {ng:.3e}")

        # --- Test d'arrêt : point stationnaire ---
        if ng <= tol:
            return Resultat(x, fx, ng, k, True, "convergence : ||grad|| <= tol", historique)

        # --- Direction de Newton (régularisée si besoin) ---
        # mu n'est pas utilisé ensuite, mais renvoyé pour information/debug.
        d, mu = direction_newton(hess(x), g)

        # Sécurité : si d n'est pas une direction de descente (g.d >= 0),
        # on force la direction du gradient. Cela ne devrait pas arriver
        # après régularisation, mais protège contre les cas pathologiques.
        if g @ d >= 0:  # sécurité : doit être une direction de descente
            d = -g

        # --- Recherche linéaire : choisit t dans (0, 1] ---
        t, f_new = recherche_lineaire(f, x, fx, g, d)
        if t == 0.0:
            return Resultat(x, fx, ng, k, False,
                            "échec de la recherche linéaire (précision machine atteinte)",
                            historique)

        # --- Mise à jour de l'itéré et des quantités associées ---
        x = x + t * d
        fx = f_new          # on réutilise la valeur déjà calculée (gain)
        g = grad(x)         # nouveau gradient au point x_{k+1}
        historique.append(x.copy())

    # Boucle épuisée sans convergence.
    return Resultat(x, fx, np.linalg.norm(g), max_iter, False,
                    "nombre maximum d'itérations atteint", historique)


# ----------------------------------------------------------------------
# Exemples
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Symboles réels : permet à sympy de simplifier (abs, signe, ...).
    x, y = sp.symbols("x y", real=True)

    # --- Exemple 1 : la classique "vallée de Rosenbrock" ---
    # Minimum global en (1, 1) ; vallée très courbe => bon test de robustesse.
    print("=== Exemple 1 : Rosenbrock, minimum en (1, 1) ===")
    rosen = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    r = newton(rosen, [x, y], [-1.2, 1.0], verbose=True)
    print(r.message)
    print("x* =", r.x, "| f(x*) =", r.fx, "| itérations =", r.iterations)

    # --- Exemple 2 : Hessienne singulière au point de départ ---
    # En (0, 3), d2f/dx2 = 12 x^2 = 0 : la Hessienne n'est pas définie
    # positive => la régularisation H + mu I est indispensable.
    print("\n=== Exemple 2 : Hessienne singulière au départ (cas difficile) ===")
    # f = x^4 + y^4 en (0, 3) : d2f/dx2 = 0 -> Hessienne singulière
    f2 = x ** 4 + y ** 4
    r = newton(f2, [x, y], [0.0, 3.0])
    print(r.message)
    print("x* =", r.x, "| f(x*) =", r.fx, "| itérations =", r.iterations)

    # --- Exemple 3 : cas quadratique ---
    # Pour une quadratique, Newton avec t = 1 atteint le minimum en une
    # seule itération (la Hessienne est exacte et constante).
    print("\n=== Exemple 3 : fonction quadratique (Newton converge en 1 itération) ===")
    f3 = 3 * x ** 2 + 2 * x * y + 5 * y ** 2 - 4 * x - 6 * y
    r = newton(f3, [x, y], [10.0, -7.0])
    print(r.message)
    print("x* =", r.x, "| itérations =", r.iterations)
