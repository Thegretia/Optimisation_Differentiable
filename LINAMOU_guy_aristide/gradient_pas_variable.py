#Gradient à pas variable : recherche linéaire par rebroussement 
from algebre import sub, scale, norme2_carre
from outils import Resultat, valeur


def gradient_pas_variable(f, grad, u0, alpha0=1.0, beta=0.5, eps=1e-12,
                          max_iter=100000, alpha_min=1e-16):
    u = list(u0)
    res = Resultat(u=list(u0))
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
        alpha = alpha0
        u_new = sub(u, scale(alpha, g))
        f_new = valeur(f, u_new)
        while f_new > fu - 0.5 * alpha * g2 and alpha > alpha_min:
            alpha *= beta
            u_new = sub(u, scale(alpha, g))
            f_new = valeur(f, u_new)
        u, fu, g = u_new, f_new, grad(u_new)
    return res


if __name__ == "__main__":
    from problemes import EXPONENTIELLE as P
    r = gradient_pas_variable(P.f, P.grad, P.u0)
    print(f"{P.nom} : u = {r.u}, f = {P.f(r.u):.10f}, "
          f"itérations = {r.iterations}, statut = {r.statut}")
