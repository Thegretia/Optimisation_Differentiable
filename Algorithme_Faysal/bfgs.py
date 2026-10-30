"""Algorithme quasi-Newton (BFGS) implémenté from scratch avec NumPy."""
import numpy as np


def backtracking(f, grad, x, p, alpha=1.0, rho=0.5, c=1e-4, max_iter=50):
    """Recherche linéaire d'Armijo : réduit alpha jusqu'à décroissance suffisante."""
    fx, slope = f(x), grad(x) @ p
    for _ in range(max_iter):
        if f(x + alpha * p) <= fx + c * alpha * slope:
            break
        alpha *= rho
    return alpha


def bfgs(f, grad, x0, tol=1e-6, max_iter=1000):
    """Minimise f avec BFGS. Retourne (x*, historique des valeurs de f)."""
    x = np.asarray(x0, dtype=float)
    n = x.size
    H = np.eye(n)              # approximation de l'inverse de la hessienne
    g = grad(x)
    history = [f(x)]

    for k in range(max_iter):
        if np.linalg.norm(g) < tol:
            break
        p = -H @ g                              # direction de descente
        alpha = backtracking(f, grad, x, p)     # taille du pas
        x_new = x + alpha * p
        g_new = grad(x_new)

        s, y = x_new - x, g_new - g
        sy = s @ y
        if sy > 1e-10:                          # condition de courbure
            rho = 1.0 / sy
            I = np.eye(n)
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) \
                + rho * np.outer(s, s)

        x, g = x_new, g_new
        history.append(f(x))

    return x, history


if __name__ == "__main__":
    # Fonction de Rosenbrock : minimum global en (1, 1)
    def rosenbrock(x):
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    def rosenbrock_grad(x):
        return np.array([
            -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
            200 * (x[1] - x[0] ** 2),
        ])

    x_opt, hist = bfgs(rosenbrock, rosenbrock_grad, x0=[-1.2, 1.0])
    print("Minimum trouvé :", x_opt)
    print("f(x*) =", rosenbrock(x_opt))
    print("Itérations :", len(hist) - 1)
