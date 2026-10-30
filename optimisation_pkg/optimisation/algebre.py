"""Algèbre linéaire écrite à la main (boucles for uniquement).

NumPy ne sert ici qu'à stocker les vecteurs/matrices (np.zeros, np.array).
Aucune fonction toute faite du type @, np.dot, np.outer, np.linalg.solve.
"""
import numpy as np


def identite(n):
    I = np.zeros((n, n))
    for i in range(n):
        I[i, i] = 1.0
    return I


def produit_scalaire(u, v):
    """u . v"""
    somme = 0.0
    for i in range(len(u)):
        somme += u[i] * v[i]
    return somme


def norme(u):
    """||u|| (norme euclidienne)"""
    return produit_scalaire(u, u) ** 0.5


def vec_ajoute(u, v, coef=1.0):
    """u + coef * v"""
    r = np.zeros(len(u))
    for i in range(len(u)):
        r[i] = u[i] + coef * v[i]
    return r


def vec_fois_scalaire(u, c):
    """c * u"""
    r = np.zeros(len(u))
    for i in range(len(u)):
        r[i] = c * u[i]
    return r


def mat_vec(A, v):
    """A * v"""
    r = np.zeros(len(A))
    for i in range(len(A)):
        for j in range(len(v)):
            r[i] += A[i][j] * v[j]
    return r


def mat_mat(A, B):
    """A * B"""
    n, m, p = len(A), len(B), len(B[0])
    C = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            for k in range(m):
                C[i, j] += A[i][k] * B[k][j]
    return C


def produit_externe(u, v):
    """u * v^T (donne une matrice)"""
    M = np.zeros((len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            M[i, j] = u[i] * v[j]
    return M


def mat_ajoute(A, B, coef=1.0):
    """A + coef * B"""
    C = np.zeros((len(A), len(A[0])))
    for i in range(len(A)):
        for j in range(len(A[0])):
            C[i, j] = A[i][j] + coef * B[i][j]
    return C


def resoudre_systeme(A, b):
    """Résout A x = b par élimination de Gauss avec pivot partiel.

    Lève ValueError si la matrice est (quasi) singulière.
    """
    n = len(b)
    # matrice augmentée [A | b] (copie : on ne modifie pas A)
    M = np.zeros((n, n + 1))
    for i in range(n):
        for j in range(n):
            M[i, j] = A[i][j]
        M[i, n] = b[i]

    # élimination vers l'avant
    for k in range(n):
        # pivot partiel : ligne avec le plus grand |M[i, k]|
        p = k
        for i in range(k + 1, n):
            if abs(M[i, k]) > abs(M[p, k]):
                p = i
        if abs(M[p, k]) < 1e-14:
            raise ValueError("Matrice singulière : système non résoluble")
        if p != k:
            for j in range(n + 1):
                M[k, j], M[p, j] = M[p, j], M[k, j]
        for i in range(k + 1, n):
            facteur = M[i, k] / M[k, k]
            for j in range(k, n + 1):
                M[i, j] -= facteur * M[k, j]

    # remontée
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = M[i, n]
        for j in range(i + 1, n):
            s -= M[i, j] * x[j]
        x[i] = s / M[i, i]
    return x
