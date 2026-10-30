"""Descente de gradient a pas optimal (recherche lineaire exacte).

    d     = -grad f(x)
    alpha = (g . g) / (g^T H(x) g)        avec g = grad f(x)
    x+    = x + alpha d

Pour une fonction quadratique, H est constante et alpha minimise exactement
phi(a) = f(x - a g). Pour une fonction generale, H(x) change a chaque pas :
la formule n'est plus qu'une approximation (pas de Cauchy).
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


def gradient_pas_optimal(expr, variables, x0, tol=1e-8, max_iter=100000):
    """Minimise expr par descente de gradient a pas optimal.

    Retourne (x, f(x), nombre d'iterations).
    """
    F, G, H = _fonctions(expr, variables)
    x = np.array(x0, dtype=float)

    for k in range(max_iter):
        g = G(x)
        if np.linalg.norm(g) <= tol:          # gradient nul : on s'arrete
            return x, F(x), k
        alpha = (g @ g) / (g @ H(x) @ g)      # pas optimal exact (quadratique)
        x = x - alpha * g

    return x, F(x), max_iter


if __name__ == "__main__":
    x, y = sp.symbols("x y", real=True)
    # Meme exemple que les autres fichiers : A = [[2, 1], [1, 20]]
    f = x**2 + 10 * y**2 + x * y - 2 * x - 4 * y

    x_opt, f_opt, n_iter = gradient_pas_optimal(f, [x, y], [5.0, 3.0])
    print("x*         =", x_opt)      # solution exacte : (12/13, 2/13)
    print("f(x*)      =", f_opt)      # solution exacte : -16/13
    print("iterations =", n_iter)
