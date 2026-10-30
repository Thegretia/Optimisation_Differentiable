"""BFGS (quasi-Newton) : approximation de l'inverse de la Hessienne.

Au lieu de calculer la Hessienne, on entretient une matrice H ~ (Hess f)^-1
mise a jour avec les variations de x et du gradient :

    s = x+ - x ,   y = grad f(x+) - grad f(x) ,   rho = 1 / (y . s)
    H+ = (I - rho s y^T) H (I - rho y s^T) + rho s s^T

La direction est d = -H grad f(x). On ne fait la mise a jour que si y . s > 0,
sinon H pourrait perdre son caractere defini positif.
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


def bfgs(expr, variables, x0, tol=1e-8, max_iter=1000):
    """Minimise expr par BFGS.

    Retourne (x, f(x), nombre d'iterations).
    """
    F, G, _ = _fonctions(expr, variables)     # pas besoin de la Hessienne
    x = np.array(x0, dtype=float)
    n = len(variables)
    H = np.eye(n)                             # approximation de (Hess f)^-1
    g = G(x)

    for k in range(max_iter):
        if np.linalg.norm(g) <= tol:          # gradient nul : on s'arrete
            return x, F(x), k
        d = -H @ g

        t = 1.0
        while F(x + t * d) > F(x) and t > 1e-12:   # backtracking
            t *= 0.5
        s = t * d

        x_new = x + s
        g_new = G(x_new)
        y = g_new - g
        if s @ y > 1e-12:                     # courbure strictement positive
            rho = 1.0 / (s @ y)
            V = np.eye(n) - rho * np.outer(s, y)
            H = V @ H @ V.T + rho * np.outer(s, s)

        x, g = x_new, g_new

    return x, F(x), max_iter


if __name__ == "__main__":
    x, y = sp.symbols("x y", real=True)
    # Meme exemple que les autres fichiers : A = [[2, 1], [1, 20]]
    f = x**2 + 10 * y**2 + x * y - 2 * x - 4 * y

    x_opt, f_opt, n_iter = bfgs(f, [x, y], [5.0, 3.0])
    print("x*         =", x_opt)      # solution exacte : (12/13, 2/13)
    print("f(x*)      =", f_opt)      # solution exacte : -16/13
    print("iterations =", n_iter)
