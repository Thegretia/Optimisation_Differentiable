"""Descente de gradient a pas fixe.

    x_{k+1} = x_k - alpha grad f(x_k)      (alpha constant)

Si grad f est L-lipschitzien, la methode decroit pour 0 < alpha < 2/L
(condition exacte pour une quadratique de plus grande valeur propre L).

Gardes-fous : pas automatique alpha = 1/L (L estime par la Hessienne en x0),
detection de divergence (valeurs non finies, ||x|| ou f qui explosent).
"""
import numpy as np
import sympy as sp


class Resultat:
    """Sortie d'une optimisation (on garde le pas utilise)."""

    __slots__ = ("x", "fx", "norme_grad", "iterations", "converge",
                 "message", "alpha", "historique")

    def __init__(self, x, fx, norme_grad, iterations, converge, message,
                 alpha=0.0, historique=None):
        self.x = x
        self.fx = fx
        self.norme_grad = norme_grad
        self.iterations = iterations
        self.converge = converge
        self.message = message
        self.alpha = alpha
        self.historique = [] if historique is None else list(historique)

    def __repr__(self):
        return (f"Resultat(fx={self.fx:.6e}, norme_grad={self.norme_grad:.3e}, "
                f"iterations={self.iterations}, alpha={self.alpha:g}, "
                f"converge={self.converge})")


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


def _pas_auto(H):
    """alpha = 1/L avec L = plus grande valeur propre en module de H."""
    if not np.all(np.isfinite(H)):
        return 1.0
    valeurs = np.linalg.eigvalsh(0.5 * (H + H.T))
    L = float(np.max(np.abs(valeurs)))
    return 1.0 / L if L > 0 else 1.0


def gradient_pas_fixe(expr, variables, x0, alpha=None, tol=1e-6, max_iter=100000,
                      verbose=False):
    """Minimise expr (sympy) par descente de gradient a pas fixe.

    expr      : expression sympy
    variables : symboles sympy
    x0        : point de depart
    alpha     : pas > 0 (None : 1/L estime en x0)
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

    if alpha is None:
        alpha = _pas_auto(fonc.hess(x))
    if alpha <= 0:
        raise ValueError("le pas alpha doit etre strictement positif")

    f0 = fx
    g = fonc.grad(x)
    trace = [x.copy()]
    seuil = f0 + 1e12 * (1.0 + abs(f0))

    for k in range(max_iter):
        if not np.all(np.isfinite(g)):
            return Resultat(x, fx, np.nan, k, False,
                            "divergence : gradient non fini, reduire alpha",
                            alpha, trace)

        ng = float(np.linalg.norm(g))
        if verbose and k % 50 == 0:
            print(f"it {k:6d} | f = {fx:.6e} | ||grad|| = {ng:.3e}")
        if ng <= tol:
            return Resultat(x, fx, ng, k, True,
                            "convergence : ||grad|| <= tol", alpha, trace)

        x = x - alpha * g
        fx = fonc.f(x)
        g = fonc.grad(x)
        trace.append(x.copy())

        if not np.isfinite(fx) or np.linalg.norm(x) > 1e12 or fx > seuil:
            return Resultat(x, fx, float(np.linalg.norm(g)), k + 1, False,
                            f"divergence : pas alpha = {alpha:g} trop grand, le reduire",
                            alpha, trace)

    return Resultat(x, fx, float(np.linalg.norm(g)), max_iter, False,
                    "nombre maximum d'iterations atteint (pas plus grand ou "
                    "plus d'iterations)", alpha, trace)


if __name__ == "__main__":
    x, y = sp.symbols("x y", real=True)

    # Hessienne diag(2, 20) : L = 20, 2/L = 0.1 ; minimum en (1, 0)
    f1 = x ** 2 + 10 * y ** 2 - 2 * x
    print("=== pas fixe alpha = 0.05 (< 2/L = 0.1) ===")
    res = gradient_pas_fixe(f1, [x, y], [5.0, 3.0], alpha=0.05)
    print(res.message)
    print("x* =", res.x, "| f(x*) =", res.fx, "| iterations =", res.iterations)
