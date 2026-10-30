"""Gradient à pas variable : recherche linéaire par rebroussement (BLS, Définition 1.9).
À chaque itération on repart de alpha0 puis on réduit alpha <- beta*alpha tant que
f(u - alpha*g) > f(u) - (alpha/2) * g^T g."""
from algebre import sub, scale, norme2_carre

def gradient_pas_variable(f, grad, u0, alpha0=1.0, beta=0.5, eps=1e-12, max_iter=100000):
    u = u0[:]
    historique = [f(u)]
    for k in range(max_iter):
        g = grad(u)
        ng2 = norme2_carre(g)
        if ng2 <= eps:
            return u, historique, k
        alpha, fu = alpha0, f(u)
        # rebroussement
        while f(sub(u, scale(alpha, g))) > fu - 0.5 * alpha * ng2 and alpha > 1e-16:
            alpha *= beta
        u = sub(u, scale(alpha, g))
        historique.append(f(u))
    return u, historique, max_iter

if __name__ == "__main__":
    from problemes import EXPONENTIELLE as P
    u, h, k = gradient_pas_variable(P.f, P.grad, P.u0)
    print(f"{P.nom} : u* ≈ {u}, f = {P.f(u):.10f}, itérations = {k}")
