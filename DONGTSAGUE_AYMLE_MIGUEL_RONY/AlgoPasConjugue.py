"""
Méthode du gradient conjugué
============================

Direction : d_{k+1} = -g_{k+1} + beta_k d_k
  - Fletcher-Reeves : beta_k = ||g_{k+1}||^2 / ||g_k||^2
  - Polak-Ribière+  : beta_k = max(0, g_{k+1}^T (g_{k+1} - g_k) / ||g_k||^2)

Cas quadratique f(x) = 1/2 x^T A x - b^T x :
  - pas optimal exact  alpha_k = -(g_k^T d_k) / (d_k^T A d_k)
  - directions A-conjuguées : d_i^T A d_j = 0 (i != j)
  - convergence en au plus n itérations (n = dimension), en arithmétique exacte.

Cas général : recherche linéaire d'Armijo + redémarrage si d n'est pas
une direction de descente.
"""
import numpy as np


def gradient_conjugue_quadratique(A, b, x0, eps=1e-10, kmax=None):
    x = np.asarray(x0, dtype=float)
    n = len(x)
    kmax = kmax or 10 * n
    g = A @ x - b
    d = -g
    historique = [x.copy()]
    for k in range(kmax):
        if np.linalg.norm(g) < eps:
            return x, k, historique
        Ad = A @ d
        alpha = -(g @ d) / (d @ Ad)      # pas optimal
        x = x + alpha * d
        g_new = g + alpha * Ad           # gradient mis à jour sans recalcul
        beta = (g_new @ g_new) / (g @ g)
        d = -g_new + beta * d
        g = g_new
        historique.append(x.copy())
    return x, kmax, historique


def armijo(f, x, g, d, a=1.0, c=1e-4, rho=0.5, kmax=60):
    """Recherche linéaire par backtracking : réduit a tant que f ne décroît pas assez."""
    fx, pente = f(x), g @ d
    for _ in range(kmax):
        if f(x + a * d) <= fx + c * a * pente:
            break
        a *= rho
    return a


def gradient_conjugue(f, grad, x0, formule="PR", eps=1e-6, kmax=50000):
    """Gradient conjugué non linéaire (formule = 'FR' ou 'PR')."""
    x = np.asarray(x0, dtype=float)
    g = grad(x)
    d = -g
    historique = [x.copy()]
    for k in range(kmax):
        if np.linalg.norm(g) < eps:
            return x, k, historique
        if g @ d >= 0:                   # pas une direction de descente : redémarrage
            d = -g
        a = armijo(f, x, g, d)
        x = x + a * d
        g_new = grad(x)
        if formule == "FR":
            beta = (g_new @ g_new) / (g @ g)
        else:
            beta = max(0.0, g_new @ (g_new - g) / (g @ g))
        d = -g_new + beta * d
        g = g_new
        historique.append(x.copy())
    return x, kmax, historique


if __name__ == "__main__":
    # --- Cas quadratique ---
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    x0 = np.array([5.0, -5.0])
    x, k, h = gradient_conjugue_quadratique(A, b, x0)
    print("Quadratique : solution exacte", np.linalg.solve(A, b))
    print(f"              GC : x = {x}, iterations = {k}")

    # --- Cas non quadratique : Rosenbrock ---
    f = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    grad = lambda x: np.array([
        -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
        200 * (x[1] - x[0] ** 2),
    ])
    x0 = np.array([-1.2, 1.0])
    for formule in ("FR", "PR"):
        x, k, _ = gradient_conjugue(f, grad, x0, formule=formule)
        print(f"Rosenbrock ({formule}) : x = {x}, iterations = {k}")
