"""Methode de Newton globalisee pour la minimisation sans contrainte.

    x+ = x + t d ,   avec   H(x) d = -grad f(x)

Trois gardes-fous :
  1. Hessienne non definie positive ou singuliere : regularisation H + mu I
     (testee par Cholesky) pour obtenir une direction de descente.
  2. Recherche lineaire d'Armijo : t = 1 pres de la solution (convergence
     quadratique), t reduit sinon.
  3. Detection des valeurs non finies (log negatif, overflow, ...).

numpy pour le calcul, sympy pour le gradient et la Hessienne exacts.
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
    """f, gradient et Hessienne numeriques, tolerants aux erreurs."""

    def __init__(self, expr, variables):
        syms = list(variables)
        self.n = len(syms)
        derivees = sp.Matrix([sp.diff(expr, v) for v in syms])
        self._f = sp.lambdify([syms], expr, "numpy")
        self._g = sp.lambdify([syms], derivees, "numpy")
        self._h = sp.lambdify([syms], sp.hessian(expr, syms), "numpy")

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

    def hess(self, x):
        with np.errstate(all="ignore"):
            try:
                return np.asarray(self._h(list(x)), dtype=float).reshape(self.n, self.n)
            except Exception:
                return np.full((self.n, self.n), np.nan)


def _direction_newton(H, g):
    """Resout (H + mu I) d = -g avec mu >= 0 minimal tel que H + mu I > 0."""
    n = g.size
    if not np.all(np.isfinite(H)):
        return -g, np.inf

    S = 0.5 * (H + H.T)
    I = np.eye(n)
    echelle = max(1.0, float(np.linalg.norm(S, "fro")))
    mu = 0.0

    for _ in range(60):
        try:
            L = np.linalg.cholesky(S + mu * I)
            d = np.linalg.solve(L.T, np.linalg.solve(L, -g))
            if np.all(np.isfinite(d)):
                return d, mu
        except np.linalg.LinAlgError:
            pass
        mu = 1e-8 * echelle if mu == 0.0 else 10.0 * mu

    return -g, mu


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


def newton(expr, variables, x0, tol=1e-8, max_iter=200, verbose=False):
    """Minimise expr (sympy) par la methode de Newton globalisee.

    expr      : expression sympy
    variables : symboles sympy, ex. [x, y]
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
    trace = [x.copy()]
    k = 0

    while k < max_iter:
        if not np.all(np.isfinite(g)):
            return Resultat(x, fx, np.nan, k, False, "gradient non fini", trace)

        ng = float(np.linalg.norm(g))
        if verbose:
            print(f"it {k:3d} | f = {fx:.6e} | ||grad|| = {ng:.3e}")
        if ng <= tol:
            return Resultat(x, fx, ng, k, True,
                            "convergence : ||grad|| <= tol", trace)

        d, _ = _direction_newton(fonc.hess(x), g)
        if g @ d >= 0:
            d = -g

        t, f_suivant = _armijo(fonc.f, x, fx, g, d)
        if t == 0.0:
            return Resultat(x, fx, ng, k, False,
                            "echec de la recherche lineaire (precision machine)",
                            trace)

        x = x + t * d
        fx = f_suivant
        g = fonc.grad(x)
        trace.append(x.copy())
        k += 1

    return Resultat(x, fx, float(np.linalg.norm(g)), max_iter, False,
                    "nombre maximum d'iterations atteint", trace)


if __name__ == "__main__":
    x, y = sp.symbols("x y", real=True)

    # Rosenbrock : minimum en (1, 1), vallee tres courbe
    rosen = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    print("=== Rosenbrock, minimum en (1, 1) ===")
    res = newton(rosen, [x, y], [-1.2, 1.0], verbose=True)
    print(res.message)
    print("x* =", res.x, "| f(x*) =", res.fx, "| iterations =", res.iterations)
