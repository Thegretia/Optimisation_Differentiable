"""Méthode de quasi-Newton BFGS (Section A.3.7, équation A.25).
W approxime l'inverse de la hessienne. Direction p = -W grad f(u), pas par rebroussement."""
from algebre import (dot, add, sub, scale, matvec, outer, identite,
                     mat_add, mat_sub, mat_scale, norme2_carre)

def bfgs(f, grad, u0, eps=1e-14, max_iter=1000, beta=0.5, c=1e-4):
    n = len(u0)
    u = u0[:]
    W = identite(n)                      # W(0) = I
    g = grad(u)
    historique = [f(u)]
    for k in range(max_iter):
        if norme2_carre(g) <= eps:
            return u, historique, k
        p = scale(-1.0, matvec(W, g))    # direction de descente
        # rebroussement (condition d'Armijo)
        alpha, fu, gp = 1.0, f(u), dot(g, p)
        while f(add(u, scale(alpha, p))) > fu + c * alpha * gp and alpha > 1e-16:
            alpha *= beta
        d = scale(alpha, p)              # d_t = u(t) - u(t-1)
        u_new = add(u, d)
        g_new = grad(u_new)
        delta = sub(g_new, g)            # delta_t = grad f(u(t)) - grad f(u(t-1))
        dd = dot(delta, d)               # <delta_t, d_t>
        if dd > 1e-12:                   # condition de courbure => W reste définie positive
            Wdelta = matvec(W, delta)
            # (A.25) : W <- W - (d delta^T W + W delta d^T)/<delta,d>
            #                 + (1 + <delta, W delta>/<delta,d>) d d^T / <delta,d>
            terme1 = mat_scale(1.0 / dd, mat_add(outer(d, Wdelta), outer(Wdelta, d)))
            terme2 = mat_scale((1.0 + dot(delta, Wdelta) / dd) / dd, outer(d, d))
            W = mat_add(mat_sub(W, terme1), terme2)
        u, g = u_new, g_new
        historique.append(f(u))
    return u, historique, max_iter

if __name__ == "__main__":
    from problemes import EXPONENTIELLE as P
    u, h, k = bfgs(P.f, P.grad, P.u0)
    print(f"{P.nom} (BFGS) : u* ≈ {u}, f = {P.f(u):.10f}, itérations = {k}")
