"""
Descente de gradient à pas optimal
==================================

Principe : x_{k+1} = x_k - alpha_k * grad f(x_k)
avec      alpha_k = argmin_{alpha > 0} f(x_k - alpha * grad f(x_k)).

Deux versions :
  - cas quadratique : formule exacte  alpha_k = (g^T g) / (g^T A g)
  - cas général     : minimisation 1D numérique (encadrement + section dorée)

Propriété : deux gradients successifs sont orthogonaux (g_{k+1}^T g_k = 0),
d'où le comportement en "zigzag".
"""
import numpy as np


def descente_pas_optimal_quadratique(A, b, x0, eps=1e-8, kmax=10000):
    """Minimise f(x) = 1/2 x^T A x - b^T x (A symétrique définie positive)."""
    x = np.asarray(x0, dtype=float)
    historique = [x.copy()]
    for k in range(kmax):
        g = A @ x - b
        if np.linalg.norm(g) < eps:
            return x, k, historique
        alpha = (g @ g) / (g @ A @ g)    # pas optimal exact
        x = x - alpha * g
        historique.append(x.copy())
    return x, kmax, historique


def encadrer(phi, b=1.0, kmax=60):
    """Trouve un intervalle [0, B] contenant le minimum de phi (phi unimodale)."""
    for _ in range(kmax):
        if phi(2 * b) >= phi(b):
            break
        b *= 2
    return 2 * b


def section_doree(phi, a, b, tol=1e-10):
    """Minimise phi sur [a, b] par la méthode de la section dorée."""
    r = (np.sqrt(5) - 1) / 2
    c = b - r * (b - a)
    d = a + r * (b - a)
    fc, fd = phi(c), phi(d)
    while b - a > tol:
        if fc < fd:                      # le minimum est dans [a, d]
            b, d, fd = d, c, fc
            c = b - r * (b - a)
            fc = phi(c)
        else:                            # le minimum est dans [c, b]
            a, c, fc = c, d, fd
            d = a + r * (b - a)
            fd = phi(d)
    return (a + b) / 2


def descente_pas_optimal(f, grad, x0, eps=1e-6, kmax=10000):
    """Version générale : le pas optimal est calculé numériquement."""
    x = np.asarray(x0, dtype=float)
    historique = [x.copy()]
    for k in range(kmax):
        g = grad(x)
        if np.linalg.norm(g) < eps:
            return x, k, historique
        phi = lambda a: f(x - a * g)     # fonction d'une seule variable
        alpha = section_doree(phi, 0.0, encadrer(phi))
        x = x - alpha * g
        historique.append(x.copy())
    return x, kmax, historique


if __name__ == "__main__":
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    f = lambda x: 0.5 * x @ A @ x - b @ x
    grad = lambda x: A @ x - b
    x0 = np.array([5.0, -5.0])

    print("Solution exacte :", np.linalg.solve(A, b))
    x, k, h = descente_pas_optimal_quadratique(A, b, x0)
    print(f"Formule exacte   : x = {x}, iterations = {k}")
    x, k, h = descente_pas_optimal(f, grad, x0)
    print(f"Section dorée    : x = {x}, iterations = {k}")

    # Vérification de l'orthogonalité des gradients successifs
    g0, g1 = grad(h[0]), grad(h[1])
    print("g1^T g0 (doit etre ~ 0) :", g1 @ g0)
