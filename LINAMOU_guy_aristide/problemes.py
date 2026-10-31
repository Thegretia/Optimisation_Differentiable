#Fonctions convexes de test : valeur, gradient, hessienne, solution exacte.

import math


class Probleme:
    def __init__(self, nom, f, grad, hess, u0, solution, mu=None, L=None):
        self.nom, self.f, self.grad, self.hess = nom, f, grad, hess
        self.u0, self.solution, self.mu, self.L = u0, solution, mu, L

    @property
    def f_etoile(self):
        return self.f(self.solution)


# 1) Quadratique 2D : f(u) = 1/2 u^T Q u - b^T u, minimum en Q u* = b.
Q = [[4.0, 1.0], [1.0, 3.0]]
b = [1.0, 2.0]


def f_quad(u):
    Qu = [Q[0][0] * u[0] + Q[0][1] * u[1], Q[1][0] * u[0] + Q[1][1] * u[1]]
    return 0.5 * (u[0] * Qu[0] + u[1] * Qu[1]) - (b[0] * u[0] + b[1] * u[1])


def grad_quad(u):
    return [Q[0][0] * u[0] + Q[0][1] * u[1] - b[0],
            Q[1][0] * u[0] + Q[1][1] * u[1] - b[1]]


def hess_quad(u):
    return [ligne[:] for ligne in Q]


# Valeurs propres de Q (formule explicite pour une matrice symétrique 2x2).
_demi_trace = 0.5 * (Q[0][0] + Q[1][1])
_rayon = math.sqrt((0.5 * (Q[0][0] - Q[1][1])) ** 2 + Q[0][1] ** 2)

QUADRATIQUE = Probleme(
    "Quadratique 2D", f_quad, grad_quad, hess_quad,
    u0=[5.0, -4.0], solution=[1 / 11, 7 / 11],
    mu=_demi_trace - _rayon, L=_demi_trace + _rayon)


# 2) Convexe non quadratique (Boyd et Vandenberghe) :
#    f(x, y) = exp(x + 3y - 0.1) + exp(x - 3y - 0.1) + exp(-x - 0.1).
def f_exp(u):
    x, y = u
    return math.exp(x + 3 * y - 0.1) + math.exp(x - 3 * y - 0.1) + math.exp(-x - 0.1)


def grad_exp(u):
    x, y = u
    e1, e2, e3 = math.exp(x + 3 * y - 0.1), math.exp(x - 3 * y - 0.1), math.exp(-x - 0.1)
    return [e1 + e2 - e3, 3 * e1 - 3 * e2]


def hess_exp(u):
    x, y = u
    e1, e2, e3 = math.exp(x + 3 * y - 0.1), math.exp(x - 3 * y - 0.1), math.exp(-x - 0.1)
    return [[e1 + e2 + e3, 3 * e1 - 3 * e2],
            [3 * e1 - 3 * e2, 9 * e1 + 9 * e2]]


EXPONENTIELLE = Probleme(
    "Somme d'exponentielles", f_exp, grad_exp, hess_exp,
    u0=[-1.0, 1.0], solution=[0.5 * math.log(0.5), 0.0])


# 3) Quadratique diagonale de dimension n, mal conditionnée :
#    f(u) = 1/2 sum_i d_i u_i^2 - sum_i u_i, avec d_i répartis de 1 à kappa.
#    Solution exacte : u_i* = 1 / d_i.
def quadratique_diagonale(n=30, kappa=100.0):
    d = [1.0 + (kappa - 1.0) * i / (n - 1) for i in range(n)]

    def f(u):
        return sum(0.5 * di * ui * ui - ui for di, ui in zip(d, u))

    def grad(u):
        return [di * ui - 1.0 for di, ui in zip(d, u)]

    def hess(u):
        return [[d[i] if i == j else 0.0 for j in range(n)] for i in range(n)]

    return Probleme(
        f"Quadratique diagonale n={n}, cond={kappa:g}", f, grad, hess,
        u0=[0.0] * n, solution=[1.0 / di for di in d], mu=min(d), L=max(d))
