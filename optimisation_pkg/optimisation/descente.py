"""Descente de gradient : pas fixe et pas variable."""
from .algebre import norme, vec_ajoute, vec_fois_scalaire
from .outils import Resultat, armijo, pas_optimal


def _boucle(f, grad, x0, choisir_pas, tol, max_iter, nom):
    """Boucle commune : x_{k+1} = x_k - a_k * grad f(x_k)."""
    x = [float(v) for v in x0]
    historique = [list(x)]
    converge = False
    k = 0
    try:
        for k in range(max_iter):
            g = grad(x)
            if norme(g) < tol:                  # gradient ~ 0 -> minimum
                converge = True
                break
            d = vec_fois_scalaire(g, -1.0)      # direction de descente
            a = choisir_pas(x, g, d)            # taille du pas
            x = vec_ajoute(x, d, a)
            historique.append(list(x))
        else:
            k = max_iter
    except (OverflowError, ValueError):         # divergence (nombres trop grands)
        converge = False
    return Resultat(x, f(x) if _fini(x) else float("inf"),
                    k, converge, historique, nom)


def _fini(x):
    return all(abs(v) < 1e150 for v in x)


def descente_pas_fixe(f, grad, x0, pas=0.1, tol=1e-6, max_iter=10000):
    """x_{k+1} = x_k - pas * grad f(x_k)   (pas constant).

    Attention : si `pas` est trop grand, la méthode diverge.
    """
    return _boucle(f, grad, x0, lambda x, g, d: pas, tol, max_iter,
                   "descente à pas fixe")


def descente_pas_variable(f, grad, x0, pas="armijo", tol=1e-6, max_iter=10000):
    """Descente de gradient où le pas est recalculé à chaque itération.

    pas = "armijo"  : backtracking (on divise le pas par 2 jusqu'à ce que f baisse assez)
    pas = "optimal" : on minimise f le long de la direction (section dorée)
    """
    if pas == "armijo":
        choisir = lambda x, g, d: armijo(f, x, g, d)
    elif pas == "optimal":
        choisir = lambda x, g, d: pas_optimal(f, x, d)
    else:
        raise ValueError("pas doit valoir 'armijo' ou 'optimal'")
    return _boucle(f, grad, x0, choisir, tol, max_iter,
                   f"descente à pas variable ({pas})")
