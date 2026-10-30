"""Outils numériques partagés par les algorithmes d'optimisation.

Implémentation avec la seule bibliothèque standard Python.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

Vector = list[float]


def as_vector(x: float | Sequence[float]) -> Vector:
    if isinstance(x, (int, float)):
        return [float(x)]
    return [float(value) for value in x]


def restore_shape(values: Sequence[float], scalar_input: bool):
    return values[0] if scalar_input else list(values)


def evaluate(f: Callable, x: Sequence[float]) -> float:
    """Appelle f avec un nombre en 1D et une liste en dimension supérieure."""
    arg = x[0] if len(x) == 1 else list(x)
    value = float(f(arg))
    if not math.isfinite(value):
        raise ValueError("La fonction renvoie une valeur non finie au point considéré.")
    return value


def norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in v))


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def add_scaled(x: Sequence[float], direction: Sequence[float], scale: float) -> Vector:
    return [a + scale * b for a, b in zip(x, direction)]


def gradient(f: Callable, x: Sequence[float], step: float = 1e-5) -> Vector:
    result = []
    for i in range(len(x)):
        h = step * max(1.0, abs(x[i]))
        plus, minus = list(x), list(x)
        plus[i] += h
        minus[i] -= h
        result.append((evaluate(f, plus) - evaluate(f, minus)) / (2.0 * h))
    return result


def hessian(f: Callable, x: Sequence[float], step: float = 1e-4) -> list[Vector]:
    """Hessienne par différences finies centrées (dérivées secondes mixtes incluses)."""
    n = len(x)
    matrix = [[0.0] * n for _ in range(n)]
    f0 = evaluate(f, x)
    hs = [step * max(1.0, abs(value)) for value in x]
    for i in range(n):
        xp, xm = list(x), list(x)
        xp[i] += hs[i]
        xm[i] -= hs[i]
        matrix[i][i] = (evaluate(f, xp) - 2.0 * f0 + evaluate(f, xm)) / (hs[i] ** 2)
        for j in range(i):
            pp, pm, mp, mm = (list(x) for _ in range(4))
            pp[i] += hs[i]; pp[j] += hs[j]
            pm[i] += hs[i]; pm[j] -= hs[j]
            mp[i] -= hs[i]; mp[j] += hs[j]
            mm[i] -= hs[i]; mm[j] -= hs[j]
            value = (evaluate(f, pp) - evaluate(f, pm) - evaluate(f, mp) + evaluate(f, mm)) / (4.0 * hs[i] * hs[j])
            matrix[i][j] = matrix[j][i] = value
    return matrix


def solve_linear(matrix: Sequence[Sequence[float]], rhs: Sequence[float], pivot_tol: float = 1e-12) -> Vector:
    """Résout Ax=b par élimination de Gauss avec pivot partiel."""
    n = len(rhs)
    augmented = [list(matrix[i]) + [float(rhs[i])] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < pivot_tol:
            raise ValueError("Matrice singulière ou presque singulière.")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        divisor = augmented[col][col]
        for j in range(col, n + 1):
            augmented[col][j] /= divisor
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            for j in range(col, n + 1):
                augmented[row][j] -= factor * augmented[col][j]
    return [augmented[i][n] for i in range(n)]


def backtracking(f: Callable, x: Sequence[float], direction: Sequence[float],
                 grad: Sequence[float], initial: float = 1.0,
                 contraction: float = 0.5, armijo: float = 1e-4,
                 min_step: float = 1e-12) -> float:
    """Recherche linéaire d'Armijo; la direction doit être une direction de descente."""
    alpha = initial
    fx = evaluate(f, x)
    slope = dot(grad, direction)
    if slope >= 0:
        return min(initial, 1e-3)
    while alpha >= min_step:
        candidate = add_scaled(x, direction, alpha)
        if evaluate(f, candidate) <= fx + armijo * alpha * slope:
            return alpha
        alpha *= contraction
    return min_step
