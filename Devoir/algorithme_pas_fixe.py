"""Descente de gradient à pas fixe."""
try:  # lancement depuis la racine avec python -m Devoir.interface_console
    from .optimisation_utils import as_vector, restore_shape, evaluate, gradient, norm, add_scaled
except ImportError:  # lancement direct depuis le dossier Devoir
    from optimisation_utils import as_vector, restore_shape, evaluate, gradient, norm, add_scaled


def pas_fixe(f, x0, alpha=0.01, tol=1e-6, max_iter=1000):
    """Minimise f par x[k+1] = x[k] - alpha * grad(f)(x[k]).

    ``x0`` est un nombre en une dimension ou une séquence en dimension supérieure.
    Le résultat est un dictionnaire contenant le point, la valeur et le statut.
    """
    scalar = isinstance(x0, (int, float))
    x = as_vector(x0)
    history = []
    for iteration in range(max_iter + 1):
        fx = evaluate(f, x)
        g = gradient(f, x)
        history.append((restore_shape(x, scalar), fx, norm(g)))
        if norm(g) <= tol:
            return {"x": restore_shape(x, scalar), "f_x": fx, "iterations": iteration,
                    "converged": True, "history": history}
        if iteration == max_iter:
            break
        x = add_scaled(x, g, -alpha)
    return {"x": restore_shape(x, scalar), "f_x": evaluate(f, x), "iterations": max_iter,
            "converged": False, "history": history}
