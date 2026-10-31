"""
Descente de gradient à pas fixe
===============================

Principe : x_{k+1} = x_k - alpha * grad f(x_k)   avec alpha constant.

Condition de convergence (f quadratique, A symétrique définie positive) :
    0 < alpha < 2 / lambda_max(A)
Pas fixe "optimal" (qui minimise le facteur de contraction) :
    alpha* = 2 / (lambda_min + lambda_max)
"""
import numpy as np


def descente_pas_fixe(grad, x0, alpha, eps=1e-8, kmax=10000):
    """
    Paramètres
    ----------
    grad  : fonction x -> gradient de f en x
    x0    : point de départ
    alpha : pas fixe (> 0)
    eps   : tolérance sur la norme du gradient
    kmax  : nombre maximal d'itérations

    Retour
    ------
    x          : dernier point calculé
    k          : nombre d'itérations effectuées
    historique : liste des points x_0, x_1, ..., x_k
    """
    x = np.asarray(x0, dtype=float)
    historique = [x.copy()]
    for k in range(kmax):
        g = grad(x)
        if np.linalg.norm(g) < eps:      # critère d'arrêt
            return x, k, historique
        x = x - alpha * g                # mise à jour
        historique.append(x.copy())
    return x, kmax, historique


if __name__ == "__main__":
    # Exemple : f(x) = 1/2 x^T A x - b^T x, minimiseur : A x = b
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    grad = lambda x: A @ x - b
    x0 = np.array([5.0, -5.0])

    valeurs_propres = np.linalg.eigvalsh(A)
    lmin, lmax = valeurs_propres[0], valeurs_propres[-1]
    print("Solution exacte :", np.linalg.solve(A, b))
    print(f"lambda_min = {lmin:.4f}, lambda_max = {lmax:.4f}, 2/lambda_max = {2/lmax:.4f}\n")

    pas = {
        "1/lambda_max": 1 / lmax,
        "2/(lmin+lmax) (meilleur pas fixe)": 2 / (lmin + lmax),
        "1.9/lambda_max (limite)": 1.9 / lmax,
        "2.1/lambda_max (> 2/lambda_max)": 2.1 / lmax,
    }
    for nom, alpha in pas.items():
        x, k, _ = descente_pas_fixe(grad, x0, alpha, kmax=500)
        converge = np.linalg.norm(grad(x)) < 1e-6
        print(f"alpha = {nom:35s} -> iterations: {k:4d} | converge: {converge}")
