"""Descente de gradient à directions conjuguées (Fletcher-Reeves)."""
from optimisation_utils import (as_vector, restore_shape, evaluate, gradient, norm, dot,
                                add_scaled, backtracking)


def pas_conjugue(f, x0, tol=1e-6, max_iter=1000):
    """Minimise une fonction par gradient conjugué non linéaire Fletcher-Reeves."""
    scalar = isinstance(x0, (int, float))
    x = as_vector(x0)
    g = gradient(f, x)
    direction = [-value for value in g]
    history = []
    for iteration in range(max_iter + 1):
        fx = evaluate(f, x)
        gnorm = norm(g)
        history.append((restore_shape(x, scalar), fx, gnorm))
        if gnorm <= tol:
            return {"x": restore_shape(x, scalar), "f_x": fx, "iterations": iteration,
                    "converged": True, "history": history}
        if iteration == max_iter:
            break
        if dot(g, direction) >= 0:
            direction = [-value for value in g]
        alpha = backtracking(f, x, direction, g)
        next_x = add_scaled(x, direction, alpha)
        next_g = gradient(f, next_x)
        denominator = dot(g, g)
        beta = max(0.0, dot(next_g, next_g) / denominator) if denominator else 0.0
        direction = [-next_g[i] + beta * direction[i] for i in range(len(x))]
        # Redémarrage périodique pour limiter la perte de conjugaison numérique.
        if (iteration + 1) % max(1, len(x)) == 0:
            direction = [-value for value in next_g]
        x, g = next_x, next_g
    return {"x": restore_shape(x, scalar), "f_x": evaluate(f, x), "iterations": max_iter,
            "converged": False, "history": history}
