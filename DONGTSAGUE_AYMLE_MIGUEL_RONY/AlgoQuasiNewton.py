"""
Méthodes de quasi-Newton : BFGS et DFP
======================================

On n'utilise pas la hessienne : on construit une approximation W_k de
l'INVERSE de la hessienne à partir de s_k et y_k :

    s_k = x_{k+1} - x_k          y_k = grad f(x_{k+1}) - grad f(x_k)

Équation de la sécante : W_{k+1} y_k = s_k.

  BFGS : W+ = (I - r s y^T) W (I - r y s^T) + r s s^T,     r = 1 / (y^T s)
  DFP  : W+ = W + s s^T / (y^T s) - W y y^T W / (y^T W y)

Direction : d_k = -W_k grad f(x_k), pas par recherche linéaire d'Armijo.
La mise à jour n'est faite que si y^T s > 0 (W reste définie positive).
"""
import numpy as np


def armijo(f, x, g, d, a=1.0, c=1e-4, rho=0.5, kmax=60):
    fx, pente = f(x), g @ d
    for _ in range(kmax):
        if f(x + a * d) <= fx + c * a * pente:
            break
        a *= rho
    return a


def quasi_newton(f, grad, x0, methode="BFGS", eps=1e-6, kmax=10000):
    x = np.asarray(x0, dtype=float)
    n = len(x)
    W = np.eye(n)                        # W_0 = identité
    g = grad(x)
    historique = [x.copy()]
    for k in range(kmax):
        if np.linalg.norm(g) < eps:
            return x, k, historique
        d = -W @ g
        a = armijo(f, x, g, d)
        s = a * d
        x_new = x + s
        g_new = grad(x_new)
        y = g_new - g
        ys = y @ s
        if ys > 1e-12:                   # condition de courbure
            if methode == "BFGS":
                r = 1.0 / ys
                I = np.eye(n)
                W = (I - r * np.outer(s, y)) @ W @ (I - r * np.outer(y, s)) \
                    + r * np.outer(s, s)
            else:                        # DFP
                Wy = W @ y
                W = W + np.outer(s, s) / ys - np.outer(Wy, Wy) / (y @ Wy)
        x, g = x_new, g_new
        historique.append(x.copy())
    return x, kmax, historique


if __name__ == "__main__":
    # --- Quadratique ---
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    f = lambda x: 0.5 * x @ A @ x - b @ x
    grad = lambda x: A @ x - b
    print("Quadratique : solution exacte", np.linalg.solve(A, b))
    for m in ("BFGS", "DFP"):
        x, k, _ = quasi_newton(f, grad, np.array([5.0, -5.0]), methode=m)
        print(f"  {m:4s}: x = {x}, iterations = {k}")

    # --- Rosenbrock ---
    f = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    grad = lambda x: np.array([
        -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
        200 * (x[1] - x[0] ** 2),
    ])
    x0 = np.array([-1.2, 1.0])
    print("Rosenbrock :")
    for m in ("BFGS", "DFP"):
        x, k, _ = quasi_newton(f, grad, x0, methode=m)
        print(f"  {m:4s}: x = {x}, iterations = {k}")
