"""Outils communs : SymPy (dérivées), recherches linéaires, objet résultat."""
import numpy as np
import sympy as sp


class Resultat:
    """Ce que renvoie chaque méthode."""

    def __init__(self, x, fx, iterations, converge, historique, methode):
        self.x = x                      # point trouvé
        self.fx = fx                    # f(x)
        self.iterations = iterations    # nombre d'itérations faites
        self.converge = converge        # True si ||gradient|| < tol
        self.historique = historique    # liste des points successifs
        self.methode = methode

    def __repr__(self):
        return (f"Resultat(methode={self.methode!r}, x={self.x}, "
                f"fx={self.fx:.3e}, iterations={self.iterations}, "
                f"converge={self.converge})")


def preparer(expression, variables):
    """Avec SymPy, construit f, son gradient et son hessien (fonctions Python).

    expression : expression SymPy, ex. 100*(x2 - x1**2)**2 + (1 - x1)**2
    variables  : liste de symboles, ex. [x1, x2]
    """
    grad_sym = [sp.diff(expression, v) for v in variables]
    hess_sym = [[sp.diff(gi, v) for v in variables] for gi in grad_sym]

    f_l = sp.lambdify([variables], expression, "math")
    g_l = sp.lambdify([variables], grad_sym, "math")
    h_l = sp.lambdify([variables], hess_sym, "math")

    f = lambda x: float(f_l(list(x)))
    grad = lambda x: np.array(g_l(list(x)), dtype=float)
    hess = lambda x: np.array(h_l(list(x)), dtype=float)
    return f, grad, hess


def armijo(f, x, g, d, a0=1.0, c=1e-4, tau=0.5, max_reductions=60):
    """Pas variable par backtracking (Armijo).

    On part de a = a0 et on divise par 2 tant que f ne diminue pas assez :
        f(x + a d) <= f(x) + c * a * (g . d)
    """
    from .algebre import produit_scalaire, vec_ajoute
    fx = f(x)
    pente = produit_scalaire(g, d)
    a = a0
    for _ in range(max_reductions):
        if f(vec_ajoute(x, d, a)) <= fx + c * a * pente:
            break
        a = a * tau
    return a


def pas_optimal(f, x, d, a0=1.0, tol=1e-10, max_iter=100):
    """Pas optimal : on minimise phi(a) = f(x + a d) en 1D.

    1) on double a tant que phi diminue -> on encadre le minimum dans [0, b]
    2) on resserre l'intervalle par la méthode de la section dorée.
    """
    from .algebre import vec_ajoute
    phi = lambda a: f(vec_ajoute(x, d, a))

    # 1) encadrement
    a, fa = a0, phi(a0)
    for _ in range(60):
        b = 2 * a
        fb = phi(b)
        if fb < fa:
            a, fa = b, fb
        else:
            break
    gauche, droite = 0.0, 2 * a

    # 2) section dorée
    r = (5 ** 0.5 - 1) / 2          # ~ 0.618
    c = droite - r * (droite - gauche)
    e = gauche + r * (droite - gauche)
    fc, fe = phi(c), phi(e)
    for _ in range(max_iter):
        if droite - gauche < tol:
            break
        if fc < fe:
            droite, e, fe = e, c, fc
            c = droite - r * (droite - gauche)
            fc = phi(c)
        else:
            gauche, c, fc = c, e, fe
            e = gauche + r * (droite - gauche)
            fe = phi(e)
    return (gauche + droite) / 2
