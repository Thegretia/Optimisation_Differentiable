"""Méthode de Newton pour la minimisation d'une fonction différentiable."""
from optimisation_utils import (as_vector, restore_shape, evaluate, gradient, hessian,
                                norm, dot, add_scaled, solve_linear, backtracking)


def newton(f, x0, tol=1e-6, max_iter=100, line_search=True, regularisation=1e-8):
    """Minimise f avec une Hessienne calculée par différences finies.

    Une petite régularisation diagonale est ajoutée si la Hessienne ne donne pas
    directement un système résoluble. Une recherche d'Armijo amortit les pas.
    """
    scalar = isinstance(x0, (int, float))
    x = as_vector(x0)
    history = []
    for iteration in range(max_iter + 1):
        fx, g = evaluate(f, x), gradient(f, x)
        gnorm = norm(g)
        history.append((restore_shape(x, scalar), fx, gnorm))
        if gnorm <= tol:
            return {"x": restore_shape(x, scalar), "f_x": fx, "iterations": iteration,
                    "converged": True, "history": history}
        if iteration == max_iter:
            break
        hess = hessian(f, x)
        direction = None
        for shift in (0.0, regularisation, regularisation * 100, regularisation * 10000):
            adjusted = [row[:] for row in hess]
            for i in range(len(x)):
                adjusted[i][i] += shift
            try:
                candidate = solve_linear(adjusted, [-value for value in g])
                if dot(g, candidate) < 0:
                    direction = candidate
                    break
            except ValueError:
                pass
        if direction is None:
            direction = [-value for value in g]
        alpha = backtracking(f, x, direction, g) if line_search else 1.0
        x = add_scaled(x, direction, alpha)
    return {"x": restore_shape(x, scalar), "f_x": evaluate(f, x), "iterations": max_iter,
            "converged": False, "history": history}
