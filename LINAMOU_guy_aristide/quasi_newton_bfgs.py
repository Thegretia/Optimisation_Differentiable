"""Methode de quasi-Newton BFGS (Section A.3.7, equation A.25).

W approxime l'inverse de la hessienne, avec W(0) = I. La direction est
p = -W grad f(u) et le pas est choisi par rebroussement (condition d'Armijo).
"""
from algebre import (dot, add, sub, scale, matvec, outer, identite,
                     mat_add, mat_sub, mat_scale, norme2, norme2_carre)
from outils import Resultat, valeur


def bfgs(f, grad, u0, eps=1e-12, max_iter=1000, beta=0.5, c=1e-4,
         tol_courbure=1e-10, alpha_min=1e-16):
    n = len(u0)
    u = list(u0)
    W = identite(n)
    res = Resultat(u=list(u0), infos={"maj_ignorees": 0})
    fu, g = f(u), grad(u)
    for k in range(max_iter + 1):
        g2 = norme2_carre(g)
        res.u, res.iterations = u, k
        res.historique_f.append(fu)
        res.historique_g2.append(g2)
        if g2 <= eps:
            res.statut = "convergence"
            break
        if k == max_iter:
            break
        p = scale(-1.0, matvec(W, g))
        gp = dot(g, p)
        if gp >= 0.0:                    # direction non descendante : retour à W = I
            W = identite(n)
            p = scale(-1.0, g)
            gp = -g2
        alpha = 1.0
        u_new = add(u, p)
        f_new = valeur(f, u_new)
        while f_new > fu + c * alpha * gp and alpha > alpha_min:
            alpha *= beta
            u_new = add(u, scale(alpha, p))
            f_new = valeur(f, u_new)
        d = sub(u_new, u)                # d_t = u(t) - u(t-1)
        g_new = grad(u_new)
        delta = sub(g_new, g)            # delta_t = grad f(u(t)) - grad f(u(t-1))
        dd = dot(delta, d)               # <delta_t, d_t>
        if dd > tol_courbure * norme2(delta) * norme2(d):
            Wdelta = matvec(W, delta)
            # (A.25) : W := W - (d delta^T W + W delta d^T) / <delta, d>
            #                  + (1 + <delta, W delta> / <delta, d>) d d^T / <delta, d>
            terme1 = mat_scale(1.0 / dd, mat_add(outer(d, Wdelta), outer(Wdelta, d)))
            terme2 = mat_scale((1.0 + dot(delta, Wdelta) / dd) / dd, outer(d, d))
            W = mat_add(mat_sub(W, terme1), terme2)
        else:
            res.infos["maj_ignorees"] += 1
        u, fu, g = u_new, f_new, g_new
    return res


if __name__ == "__main__":
    from problemes import EXPONENTIELLE as P
    r = bfgs(P.f, P.grad, P.u0)
    print(f"{P.nom} (BFGS) : u = {r.u}, f = {P.f(r.u):.10f}, "
          f"itérations = {r.iterations}, statut = {r.statut}, "
          f"mises à jour ignorées = {r.infos['maj_ignorees']}")
