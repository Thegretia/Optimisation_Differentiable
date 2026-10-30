"""Descente de gradient a pas fixe.

    x_{k+1} = x_k - alpha * grad f(x_k)        (alpha constant)

Condition de convergence : 0 < alpha < 2/L, ou L est la plus grande valeur
propre de la Hessienne. Ici L = 20.06, donc 2/L = 0.0997 : alpha = 0.05
convient, alpha >= 0.1 fait diverger la suite.
"""
import numpy as np
import sympy as sp


def _fonctions(expr, variables):
    """f, gradient et Hessienne de expr, evaluables sur des vecteurs numpy."""
    f = sp.lambdify(variables, expr, "numpy")
    g = sp.lambdify(variables, [sp.diff(expr, v) for v in variables], "numpy")
    h = sp.lambdify(variables, sp.hessian(expr, variables), "numpy")
    return (lambda x: float(f(*x)),
            lambda x: np.asarray(g(*x), dtype=float),
            lambda x: np.asarray(h(*x), dtype=float))


def gradient_pas_fixe(expr, variables, x0, alpha, tol=1e-8, max_iter=100000):
    """Minimise expr par descente de gradient a pas fixe.

    Retourne (x, f(x), nombre d'iterations).
    """
    F, G, _ = _fonctions(expr, variables)
    x = np.array(x0, dtype=float)

    for k in range(max_iter):
        g = G(x)
        if np.linalg.norm(g) <= tol:          # gradient nul : on s'arrete
            return x, F(x), k
        x = x - alpha * g                     # le pas de descente

    return x, F(x), max_iter


if __name__ == "__main__":
    x, y = sp.symbols("x y", real=True)
    # Hessienne A = [[2, 1], [1, 20]] : L = 20.06, donc 2/L = 0.0997
    f = x**2 + 10 * y**2 + x * y - 2 * x - 4 * y

    x_opt, f_opt, n_iter = gradient_pas_fixe(f, [x, y], [5.0, 3.0], alpha=0.05)
    print("x*         =", x_opt)      # solution exacte : (12/13, 2/13)
    print("f(x*)      =", f_opt)      # solution exacte : -16/13
    print("iterations =", n_iter)
