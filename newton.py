"""Methode de Newton.

    H(x) d = -grad f(x)
    x+     = x + t d

Le pas t vaut 1 pres de la solution (convergence quadratique). S'il ne fait
pas diminuer f, on le divise par 2 : c'est cette recherche lineaire qui rend
la methode utilisable loin de la solution.
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


def newton(expr, variables, x0, tol=1e-8, max_iter=100):
    """Minimise expr par la methode de Newton.

    Retourne (x, f(x), nombre d'iterations).
    """
    F, G, H = _fonctions(expr, variables)
    x = np.array(x0, dtype=float)

    for k in range(max_iter):
        g = G(x)
        if np.linalg.norm(g) <= tol:          # gradient nul : on s'arrete
            return x, F(x), k
        try:
            d = np.linalg.solve(H(x), -g)     # direction de Newton
        except np.linalg.LinAlgError:         # Hessienne singuliere
            d = -g

        t = 1.0
        while F(x + t * d) > F(x) and t > 1e-12:   # backtracking
            t *= 0.5
        x = x + t * d

    return x, F(x), max_iter


if __name__ == "__main__":
    x, y = sp.symbols("x y", real=True)
    # Meme exemple que les autres fichiers : A = [[2, 1], [1, 20]]
    f = x**2 + 10 * y**2 + x * y - 2 * x - 4 * y

    x_opt, f_opt, n_iter = newton(f, [x, y], [5.0, 3.0])
    print("x*         =", x_opt)      # solution exacte : (12/13, 2/13)
    print("f(x*)      =", f_opt)      # solution exacte : -16/13
    print("iterations =", n_iter)
