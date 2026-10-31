"""
Petits outils d'algebre lineaire ecrits a la main.

numpy sert seulement a stocker les vecteurs/matrices et a faire les
operations element par element ; on n'utilise jamais numpy.linalg.
"""
import math

import numpy as np


class MatriceSinguliere(Exception):
    pass


def norme(v):
    s = 0.0
    for vi in v:
        s += float(vi) * float(vi)
    return math.sqrt(s)


def scalaire(u, v):
    s = 0.0
    for a, b in zip(u, v):
        s += float(a) * float(b)
    return s


def identite(n):
    I = np.zeros((n, n))
    for i in range(n):
        I[i, i] = 1.0
    return I


def resoudre_gauss(A, b):
    """Resout A x = b par elimination de Gauss avec pivot partiel."""
    n = len(b)
    M = np.array(A, dtype=float).copy()
    y = np.array(b, dtype=float).copy()
    echelle = max(1.0, float(np.max(np.abs(M))))

    for k in range(n):
        # pivot : la plus grande valeur absolue de la colonne k
        p = k
        for i in range(k + 1, n):
            if abs(M[i, k]) > abs(M[p, k]):
                p = i
        if abs(M[p, k]) < 1e-14 * echelle:
            raise MatriceSinguliere("matrice singuliere (pivot nul)")
        if p != k:
            M[[k, p]] = M[[p, k]]
            y[k], y[p] = y[p], y[k]
        for i in range(k + 1, n):
            c = M[i, k] / M[k, k]
            M[i, k:] -= c * M[k, k:]
            y[i] -= c * y[k]

    # remontee
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= M[i, j] * x[j]
        x[i] = s / M[i, i]
    return x


def cholesky(A):
    """Renvoie L (triangulaire inf.) avec A = L L^T, ou None si A n'est
    pas definie positive."""
    n = A.shape[0]
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            s = A[i, j]
            for k in range(j):
                s -= L[i, k] * L[j, k]
            if i == j:
                if s <= 0.0:
                    return None
                L[i, i] = math.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    return L


def resoudre_cholesky(L, b):
    n = len(b)
    z = np.zeros(n)
    for i in range(n):              # L z = b
        s = b[i]
        for k in range(i):
            s -= L[i, k] * z[k]
        z[i] = s / L[i, i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):  # L^T x = z
        s = z[i]
        for k in range(i + 1, n):
            s -= L[k, i] * x[k]
        x[i] = s / L[i, i]
    return x


def valeurs_propres(A, tol=1e-12, max_balayages=100):
    """Valeurs propres d'une matrice symetrique (methode de Jacobi).
    Renvoyees dans l'ordre croissant."""
    M = np.array(A, dtype=float).copy()
    M = 0.5 * (M + M.T)
    n = M.shape[0]
    for _ in range(max_balayages):
        hors_diag = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                hors_diag += M[i, j] ** 2
        if hors_diag < tol ** 2 * max(1.0, float(np.sum(M * M))):
            break
        for p in range(n):
            for q in range(p + 1, n):
                if abs(M[p, q]) < 1e-300:
                    continue
                # angle qui annule M[p, q]
                theta = (M[q, q] - M[p, p]) / (2.0 * M[p, q])
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    mkp, mkq = M[k, p], M[k, q]
                    M[k, p] = c * mkp - s * mkq
                    M[k, q] = s * mkp + c * mkq
                for k in range(n):
                    mpk, mqk = M[p, k], M[q, k]
                    M[p, k] = c * mpk - s * mqk
                    M[q, k] = s * mpk + c * mqk
    return sorted(float(M[i, i]) for i in range(n))
