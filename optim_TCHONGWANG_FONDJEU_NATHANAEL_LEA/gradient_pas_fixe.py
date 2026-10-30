"""Algorithme du gradient à pas fixe (Définition 1.8) : u <- u - alpha * grad f(u)."""
from algebre import sub, scale, norme2_carre

def gradient_pas_fixe(f, grad, u0, alpha=0.1, eps=1e-12, max_iter=100000):
    u = u0[:]
    historique = [f(u)]
    for k in range(max_iter):
        g = grad(u)
        if norme2_carre(g) <= eps:
            return u, historique, k
        u = sub(u, scale(alpha, g))
        historique.append(f(u))
    return u, historique, max_iter

if __name__ == "__main__":
    from problemes import QUADRATIQUE as P
    u, h, k = gradient_pas_fixe(P.f, P.grad, P.u0, alpha=0.2)
    print(f"{P.nom} : u* ≈ {u}, f = {P.f(u):.10f}, itérations = {k}")
