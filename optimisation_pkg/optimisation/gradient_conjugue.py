"""Gradient conjugué non linéaire (Fletcher-Reeves ou Polak-Ribière)."""
from .algebre import norme, produit_scalaire, vec_ajoute, vec_fois_scalaire
from .outils import Resultat, armijo, pas_optimal


def gradient_conjugue(f, grad, x0, formule="PR", pas="optimal",
                      tol=1e-6, max_iter=5000):
    """Méthode du gradient conjugué.

    Direction : d_{k+1} = -g_{k+1} + beta * d_k, avec
        Fletcher-Reeves ("FR") : beta = (g_{k+1}.g_{k+1}) / (g_k.g_k)
        Polak-Ribière  ("PR") : beta = max(0, g_{k+1}.(g_{k+1}-g_k) / (g_k.g_k))

    pas = "optimal" (section dorée) ou "armijo" (backtracking).
    On repart de -gradient tous les n pas, ou si d n'est plus une direction
    de descente.
    """
    if formule not in ("FR", "PR"):
        raise ValueError("formule doit valoir 'FR' ou 'PR'")
    if pas not in ("optimal", "armijo"):
        raise ValueError("pas doit valoir 'optimal' ou 'armijo'")

    n = len(x0)
    x = [float(v) for v in x0]
    g = grad(x)
    d = vec_fois_scalaire(g, -1.0)
    historique = [list(x)]
    converge = False
    k = 0

    for k in range(max_iter):
        if norme(g) < tol:
            converge = True
            break

        if produit_scalaire(g, d) >= 0:          # d ne descend pas -> on repart
            d = vec_fois_scalaire(g, -1.0)

        if pas == "optimal":
            a = pas_optimal(f, x, d)
        else:
            a = armijo(f, x, g, d)

        x = vec_ajoute(x, d, a)
        g_new = grad(x)

        if formule == "FR":
            beta = produit_scalaire(g_new, g_new) / produit_scalaire(g, g)
        else:
            diff = vec_ajoute(g_new, g, -1.0)
            beta = produit_scalaire(g_new, diff) / produit_scalaire(g, g)
            beta = max(0.0, beta)

        if (k + 1) % n == 0:                     # redémarrage périodique
            beta = 0.0

        d = vec_ajoute(vec_fois_scalaire(g_new, -1.0), d, beta)
        g = g_new
        historique.append(list(x))
    else:
        k = max_iter

    nom = f"gradient conjugué ({formule}, pas {pas})"
    return Resultat(x, f(x), k, converge, historique, nom)
