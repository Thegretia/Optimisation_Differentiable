"""Méthode quasi-Newton BFGS avec approximation inverse de la Hessienne."""
try:  # lancement depuis la racine avec python -m Devoir.interface_console
    from .optimisation_utils import (as_vector, restore_shape, evaluate, gradient, norm, dot,
                                     add_scaled, backtracking)
except ImportError:  # lancement direct depuis le dossier Devoir
    from optimisation_utils import (as_vector, restore_shape, evaluate, gradient, norm, dot,
                                    add_scaled, backtracking)


def quasi_newton(f, x0, tol=1e-6, max_iter=500):
    """Minimise f avec BFGS, sans calculer explicitement la Hessienne."""
    scalar = isinstance(x0, (int, float))
    x = as_vector(x0)
    n = len(x)
    inverse = [[float(i == j) for j in range(n)] for i in range(n)]
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
        direction = [-sum(inverse[i][j] * g[j] for j in range(n)) for i in range(n)]
        if dot(direction, g) >= 0:
            inverse = [[float(i == j) for j in range(n)] for i in range(n)]
            direction = [-value for value in g]
        alpha = backtracking(f, x, direction, g)
        next_x = add_scaled(x, direction, alpha)
        next_g = gradient(f, next_x)
        s = [next_x[i] - x[i] for i in range(n)]
        y = [next_g[i] - g[i] for i in range(n)]
        ys = dot(y, s)
        if ys > 1e-12 * norm(y) * norm(s):
            rho = 1.0 / ys
            # H_{k+1}=(I-rho*s*y^T) H_k (I-rho*y*s^T)+rho*s*s^T
            left = [[(1.0 if i == j else 0.0) - rho * s[i] * y[j] for j in range(n)] for i in range(n)]
            right = [[(1.0 if i == j else 0.0) - rho * y[i] * s[j] for j in range(n)] for i in range(n)]
            temp = [[sum(left[i][k] * inverse[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
            inverse = [[sum(temp[i][k] * right[k][j] for k in range(n)) + rho * s[i] * s[j]
                        for j in range(n)] for i in range(n)]
        x = next_x
    return {"x": restore_shape(x, scalar), "f_x": evaluate(f, x), "iterations": max_iter,
            "converged": False, "history": history}
