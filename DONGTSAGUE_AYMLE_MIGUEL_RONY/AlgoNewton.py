"""
Méthode de Newton
=================

Principe : à chaque itération on résout  H(x_k) d_k = -grad f(x_k)
puis                                     x_{k+1} = x_k + alpha_k d_k.

  - Convergence quadratique près d'un minimiseur où H est définie positive.
  - Coût : calcul de la hessienne + résolution d'un système linéaire O(n^3).
  - Version globalisée : si d_k n'est pas une direction de descente
    (H non définie positive), on prend d_k = -grad f(x_k), et on peut
    ajouter une recherche linéaire d'Armijo (si f est fournie).
"""
import numpy as np


def armijo(f, x, g, d, a=1.0, c=1e-4, rho=0.5, kmax=60):
    fx, pente = f(x), g @ d
    for _ in range(kmax):
        if f(x + a * d) <= fx + c * a * pente:
            break
        a *= rho
    return a


def newton(grad, hess, x0, f=None, eps=1e-8, kmax=100):
    """
    f est optionnelle : si elle est donnée, on utilise une recherche linéaire
    (Newton globalisé), sinon le pas vaut 1 (Newton pur).
    """
    x = np.asarray(x0, dtype=float)
    historique = [x.copy()]
    for k in range(kmax):
        g = grad(x)
        if np.linalg.norm(g) < eps:
            return x, k, historique
        try:
            d = np.linalg.solve(hess(x), -g)     # on n'inverse pas H
        except np.linalg.LinAlgError:
            d = -g
        if g @ d >= 0:                           # H non définie positive
            d = -g
        alpha = armijo(f, x, g, d) if f is not None else 1.0
        x = x + alpha * d
        historique.append(x.copy())
    return x, kmax, historique


if __name__ == "__main__":
    # --- Quadratique : Newton converge en 1 itération ---
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    x, k, _ = newton(lambda x: A @ x - b, lambda x: A, np.array([5.0, -5.0]))
    print(f"Quadratique : x = {x}, iterations = {k}")

    # --- Rosenbrock ---
    f = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    grad = lambda x: np.array([
        -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
        200 * (x[1] - x[0] ** 2),
    ])
    hess = lambda x: np.array([
        [2 - 400 * x[1] + 1200 * x[0] ** 2, -400 * x[0]],
        [-400 * x[0], 200.0],
    ])
    x0 = np.array([-1.2, 1.0])
    x, k, h_pur = newton(grad, hess, x0)
    print(f"Rosenbrock (pas = 1)    : x = {x}, iterations = {k}")
    x, k, _ = newton(grad, hess, x0, f=f)
    print(f"Rosenbrock (Armijo)     : x = {x}, iterations = {k}")

    # Vitesse de convergence : l'erreur est élevée au carré à chaque itération
    print("\nErreurs ||x_k - x*|| (convergence quadratique) :")
    for i, xi in enumerate(h_pur):
        print(f"  k={i}: {np.linalg.norm(xi - np.array([1.0, 1.0])):.3e}")
