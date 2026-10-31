#Algèbre linéaire minimale en Python pur (module standard `math` uniquement).


import math


def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def add(u, v):
    return [a + b for a, b in zip(u, v)]


def sub(u, v):
    return [a - b for a, b in zip(u, v)]


def scale(c, u):
    return [c * a for a in u]


def norme2_carre(u):
    return dot(u, u)


def norme2(u):
    return math.sqrt(dot(u, u))


def matvec(A, x):
    return [dot(ligne, x) for ligne in A]


def outer(u, v):
    return [[a * b for b in v] for a in u]


def mat_add(A, B):
    return [[a + b for a, b in zip(la, lb)] for la, lb in zip(A, B)]


def mat_sub(A, B):
    return [[a - b for a, b in zip(la, lb)] for la, lb in zip(A, B)]


def mat_scale(c, A):
    return [[c * a for a in ligne] for ligne in A]


def identite(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
