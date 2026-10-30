"""BFGS (quasi-Newton) : approximation de l'inverse de la Hessienne.

    d   = -H grad f(x)
    x+  = x + t d
    s   = x+ - x ,  y = grad f(x+) - grad f(x) ,  rho = 1 / (y.s)
    H+  = (I - rho s y^T) H (I - rho y s^T) + rho s s^T

Pas de Hessienne : seul le gradient (sympy) sert.
Gardes-fous : H mise a l'echelle au 1er pas, mise a jour sautee si s.y <= 0,
retour a H = I si d n'est pas une descente ou si Armijo echoue, valeurs non
finies detectees.
"""
import numpy as np
import sympy as sp


class Resultat:
    """Sortie d'une optimisation."""

    __slots__ = ("x", "fx", "norme_grad", "iterations", "converge",
                 "message", "historique")

    def __init__(self, x, fx, norme_grad, iterations, converge, message,
                 historique=None):
        self.x = x
        self.fx = fx
        self.norme_grad = norme_grad
        self.iterations = iterations
        self.converge = converge
        self.message = message
        self.historique = [] if historique is None else list(historique)

    def __repr__(self):
        return (f"Resultat(fx={self.fx:.6e}, norme_grad={self.norme_grad:.3e}, "
                f"iterations={self.iterations}, converge={self.converge})")


class _Fonctions:
    """f et gradient numeriques, tolerants aux erreurs d'evaluation."""

    def __init__(self, expr, variables):
        syms = list(variables)
        self.n = len(syms)
        derivees = sp.Matrix([sp.diff(expr, v) for v in syms])
        self._f = sp.lambdify([syms], expr, "numpy")
        self._g = sp.lambdify([syms], derivees, "numpy")

    def f(self, x):
        with np.errstate(all="ignore"):
            try:
                z = complex(self._f(list(x)))
            except Exception:
                return np.inf
        if z.imag != 0 or not np.isfinite(z.real):
            return np.inf
        return float(z.real)

    def grad(self, x):
        with np.errstate(all="ignore"):
            try:
                return np.asarray(self._g(list(x)), dtype=float).reshape(self.n)
            except Exception:
                return np.full(self.n, np.nan)


def _armijo(f, x, fx, g, d, c1=1e-4, beta=0.5, t_min=1e-20):
    """Backtracking d'Armijo ; (0, fx) si aucun pas ne convient."""
    pente = float(g @ d)
    marge = 1e-14 * (1.0 + abs(fx))
    t = 1.0
    while t > t_min:
        ft = f(x + t * d)
        if np.isfinite(ft) and ft <= fx + c1 * t * pente + marge:
            return t, ft
        t *= beta
    return 0.0, fx


def _bfgs_inverse(H, s, y):
    """Nouvelle approximation de H^{-1}, ou None si la courbure est trop faible."""
    sy = float(s @ y)
    if sy <= 1e-10 * np.linalg.norm(s) * np.linalg.norm(y):
        return None
    rho = 1.0 / sy
    V = np.eye(H.shape[0]) - rho * np.outer(s, y)
    Hn = V @ H @ V.T + rho * np.outer(s, s)
    return 0.5 * (Hn + Hn.T)


def bfgs(expr, variables, x0, tol=1e-8, max_iter=1000, verbose=False):
    """Minimise expr (sympy) par BFGS.

    expr      : expression sympy
    variables : symboles sympy
    x0        : point de depart
    tol       : arret si ||grad f(x)|| <= tol
    max_iter  : nombre maximum d'iterations
    """
    fonc = _Fonctions(expr, variables)
    x = np.asarray(x0, dtype=float).reshape(-1)
    if x.size != fonc.n:
        raise ValueError("x0 et variables n'ont pas la meme dimension")

    fx = fonc.f(x)
    if not np.isfinite(fx):
        raise ValueError("f(x0) non finie : choisir un autre point de depart")
    g = fonc.grad(x)

    I = np.eye(fonc.n)
    H = I.copy()          # approximation de l'inverse de la Hessienne
    H_neuve = True        # H vient d'etre remise a I
    premier_pas = True    # aucune mise a jour BFGS encore faite
    trace = [x.copy()]
    k = 0

    while k < max_iter:
        if not np.all(np.isfinite(g)):
            return Resultat(x, fx, np.nan, k, False, "gradient non fini", trace)

        ng = float(np.linalg.norm(g))
        if verbose and k % 10 == 0:
            print(f"it {k:4d} | f = {fx:.6e} | ||grad|| = {ng:.3e}")
        if ng <= tol:
            return Resultat(x, fx, ng, k, True,
                            "convergence : ||grad|| <= tol", trace)

        d = -H @ g
        if not np.all(np.isfinite(d)) or g @ d >= -1e-12 * ng * np.linalg.norm(d):
            H, H_neuve, premier_pas = I.copy(), True, True
            d = -g

        t, f_suivant = _armijo(fonc.f, x, fx, g, d)
        if t == 0.0:
            if H_neuve:
                return Resultat(x, fx, ng, k, False,
                                "echec de la recherche lineaire (precision machine)",
                                trace)
            H, H_neuve, premier_pas = I.copy(), True, True
            k += 1
            continue

        s = t * d
        x_suivant = x + s
        g_suivant = fonc.grad(x_suivant)
        if not np.all(np.isfinite(g_suivant)):
            return Resultat(x, fx, ng, k, False, "gradient non fini", trace)

        y = g_suivant - g
        if premier_pas:
            sy = float(s @ y)
            if sy > 1e-10 * np.linalg.norm(s) * np.linalg.norm(y):
                H = (sy / float(y @ y)) * I
                premier_pas = False

        Hn = _bfgs_inverse(H, s, y)
        if Hn is not None:
            H, H_neuve = Hn, False

        x, fx, g = x_suivant, f_suivant, g_suivant
        trace.append(x.copy())
        k += 1

    return Resultat(x, fx, float(np.linalg.norm(g)), max_iter, False,
                    "nombre maximum d'iterations atteint", trace)


if __name__ == "__main__":
    x, y = sp.symbols("x y", real=True)

    # Rosenbrock : minimum en (1, 1)
    rosen = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    print("=== Rosenbrock, minimum en (1, 1) ===")
    res = bfgs(rosen, [x, y], [-1.2, 1.0], verbose=True)
    print(res.message)
    print("x* =", res.x, "| f(x*) =", res.fx, "| iterations =", res.iterations)
