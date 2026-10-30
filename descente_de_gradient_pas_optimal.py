"""Descente de gradient a pas optimal (recherche lineaire exacte).

    d      = -grad f(x)
    alpha  = argmin_{a >= 0} phi(a) ,  phi(a) = f(x + a d)
    x+     = x + alpha d

Calcul de alpha :
  * f quadratique (Hessienne constante, detectee avec sympy) :
        alpha = (g.g) / (g^T A g)
  * cas general : racine de phi'(a) = -grad(x - a g).g = 0, encadree en
    doublant a puis calculee par dichotomie (phi'(0) = -||g||^2 < 0).

Gardes-fous : valeurs non finies vues comme derivee positive, fonction non
bornee dans la direction de descente signalee, backtracking si le pas ne fait
pas decroitre f.
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
    """f, gradient, Hessienne numeriques + indicateur quadratique."""

    def __init__(self, expr, variables):
        syms = list(variables)
        self.n = len(syms)
        derivees = sp.Matrix([sp.diff(expr, v) for v in syms])
        hess = sp.hessian(expr, syms)
        self.quadratique = not (hess.free_symbols & set(syms))
        self._f = sp.lambdify([syms], expr, "numpy")
        self._g = sp.lambdify([syms], derivees, "numpy")
        self._h = sp.lambdify([syms], hess, "numpy")

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


def _pas_dichotomie(grad, x, g, a_max=1e12, n_bisect=200):
    """Racine de phi'(a) = -grad(x - a g).g ; None si f n'est pas bornee."""
    def dphi(a):
        v = -float(grad(x - a * g) @ g)
        return v if np.isfinite(v) else np.inf

    lo, hi = 0.0, 1.0
    while dphi(hi) < 0:
        lo, hi = hi, 2.0 * hi
        if hi > a_max:
            return None

    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        if dphi(mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-15 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def gradient_pas_optimal(expr, variables, x0, tol=1e-8, max_iter=50000,
                         verbose=False):
    """Minimise expr (sympy) par descente de gradient a pas optimal.

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

    A = fonc.hess(x) if fonc.quadratique else None
    g = fonc.grad(x)
    trace = [x.copy()]

    for k in range(max_iter):
        if not np.all(np.isfinite(g)):
            return Resultat(x, fx, np.nan, k, False, "gradient non fini", trace)

        ng = float(np.linalg.norm(g))
        if verbose and k % 10 == 0:
            print(f"it {k:5d} | f = {fx:.6e} | ||grad|| = {ng:.3e}")
        if ng <= tol:
            return Resultat(x, fx, ng, k, True,
                            "convergence : ||grad|| <= tol", trace)

        if fonc.quadratique:
            courbure = float(g @ A @ g)
            if courbure > 0:
                alpha = float(g @ g) / courbure
            else:
                return Resultat(x, fx, ng, k, False,
                                "f non bornee dans la direction -grad (courbure <= 0)",
                                trace)
        else:
            alpha = _pas_dichotomie(fonc.grad, x, g)
            if alpha is None:
                return Resultat(x, fx, ng, k, False,
                                "f non bornee dans la direction -grad", trace)

        x_suivant = x - alpha * g
        f_suivant = fonc.f(x_suivant)
        marge = 1e-14 * (1.0 + abs(fx))
        while (not np.isfinite(f_suivant) or f_suivant > fx + marge) and alpha > 1e-20:
            alpha *= 0.5
            x_suivant = x - alpha * g
            f_suivant = fonc.f(x_suivant)
        if alpha <= 1e-20:
            return Resultat(x, fx, ng, k, False,
                            "pas devenu negligeable (precision machine)", trace)

        x, fx = x_suivant, f_suivant
        g = fonc.grad(x)
        trace.append(x.copy())

    return Resultat(x, fx, float(np.linalg.norm(g)), max_iter, False,
                    "nombre maximum d'iterations atteint", trace)


if __name__ == "__main__":
    x, y = sp.symbols("x y", real=True)

    # Quadratique : Hessienne A = [[2, 1], [1, 20]] constante et definie positive
    f1 = x ** 2 + 10 * y ** 2 - 2 * x - 4 * y + x * y
    print("=== quadratique : pas optimal exact ===")
    res = gradient_pas_optimal(f1, [x, y], [5.0, 3.0])
    print(res.message)
    print("x* =", res.x, "| f(x*) =", res.fx, "| iterations =", res.iterations)

    A = np.array([[2.0, 1.0], [1.0, 20.0]])
    b = np.array([2.0, 4.0])
    print("solution exacte =", np.linalg.solve(A, b))
