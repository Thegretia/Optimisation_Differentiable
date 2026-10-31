# Algorithme du gradient à pas fixe 

import math

from algebre import sub, scale, norme2_carre
from outils import Resultat, valeur


def gradient_pas_fixe(f, grad, u0, alpha=0.1, eps=1e-12, max_iter=100000,
                      seuil_divergence=1e100):
    u = list(u0)
    res = Resultat(u=list(u0))
    fu = valeur(f, u)
    for k in range(max_iter + 1):
        if not math.isfinite(fu) or fu > seuil_divergence:
            res.statut = "divergence"
            break
        g = grad(u)
        g2 = norme2_carre(g)
        res.u, res.iterations = u, k
        res.historique_f.append(fu)
        res.historique_g2.append(g2)
        if g2 <= eps:
            res.statut = "convergence"
            break
        if k == max_iter:
            break
        u = sub(u, scale(alpha, g))
        fu = valeur(f, u)
    return res


if __name__ == "__main__":
    from problemes import QUADRATIQUE as P
    alpha = 2.0 / (P.mu + P.L)
    r = gradient_pas_fixe(P.f, P.grad, P.u0, alpha=alpha)
    print(f"{P.nom} : alpha = {alpha:.4f}, u = {r.u}, f = {P.f(r.u):.10f}, "
          f"itérations = {r.iterations}, statut = {r.statut}")
