"""Fonctions convexes de test : f, gradient, hessienne."""
import math

class Probleme:
    def __init__(self, nom, f, grad, hess, u0):
        self.nom, self.f, self.grad, self.hess, self.u0 = nom, f, grad, hess, u0

# --- 1) Quadratique : f(u) = 1/2 u^T Q u - b^T u  (minimum : Q u* = b) ---
Q = [[4.0, 1.0], [1.0, 3.0]]
b = [1.0, 2.0]

def f_quad(u):
    Qu = [Q[0][0]*u[0] + Q[0][1]*u[1], Q[1][0]*u[0] + Q[1][1]*u[1]]
    return 0.5 * (u[0]*Qu[0] + u[1]*Qu[1]) - (b[0]*u[0] + b[1]*u[1])

def grad_quad(u):
    return [Q[0][0]*u[0] + Q[0][1]*u[1] - b[0],
            Q[1][0]*u[0] + Q[1][1]*u[1] - b[1]]

def hess_quad(u):
    return [ligne[:] for ligne in Q]

QUADRATIQUE = Probleme("Quadratique 2D", f_quad, grad_quad, hess_quad, [5.0, -4.0])
# solution exacte : Q^{-1} b = [1/11, 7/11]

# --- 2) Convexe non quadratique (Boyd & Vandenberghe) ---
# f(x,y) = exp(x+3y-0.1) + exp(x-3y-0.1) + exp(-x-0.1)
def f_exp(u):
    x, y = u
    return math.exp(x + 3*y - 0.1) + math.exp(x - 3*y - 0.1) + math.exp(-x - 0.1)

def grad_exp(u):
    x, y = u
    e1, e2, e3 = math.exp(x+3*y-0.1), math.exp(x-3*y-0.1), math.exp(-x-0.1)
    return [e1 + e2 - e3, 3*e1 - 3*e2]

def hess_exp(u):
    x, y = u
    e1, e2, e3 = math.exp(x+3*y-0.1), math.exp(x-3*y-0.1), math.exp(-x-0.1)
    return [[e1 + e2 + e3, 3*e1 - 3*e2],
            [3*e1 - 3*e2, 9*e1 + 9*e2]]

EXPONENTIELLE = Probleme("Somme d'exponentielles", f_exp, grad_exp, hess_exp, [-1.0, 1.0])
# solution : x* = 0.5*ln(0.5) , y* = 0
