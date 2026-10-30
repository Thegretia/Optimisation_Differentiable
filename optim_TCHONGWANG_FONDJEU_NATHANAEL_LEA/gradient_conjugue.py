"""Méthode du gradient conjugué (Définition 1.11) pour A x = b, A symétrique définie positive,
et méthode de Newton par gradient conjugué (Section A.3.6) : on résout H(u) delta = grad f(u)
par gradient conjugué puis u <- u - delta."""
from algebre import dot, add, sub, scale, matvec, norme2_carre

def gradient_conjugue(A, b, x0=None, tol=1e-20):
    n = len(b)
    x = x0[:] if x0 is not None else [0.0] * n
    r = sub(b, matvec(A, x))
    v = r[:]
    for _ in range(n):
        rr = dot(r, r)
        if rr <= tol:
            break
        Av = matvec(A, v)
        alpha = rr / dot(v, Av)
        x = add(x, scale(alpha, v))          # (A.16)
        r = sub(b, matvec(A, x))             # (A.17)
        beta = dot(r, r) / rr
        v = add(r, scale(beta, v))           # (A.18)
    return x

def newton_gradient_conjugue(grad, hess, u0, eps=1e-14, max_iter=200):
    u = u0[:]
    historique = []
    for k in range(max_iter):
        g = grad(u)
        if norme2_carre(g) <= eps:
            return u, k
        delta = gradient_conjugue(hess(u), g)   # H delta = g
        u = sub(u, delta)
    return u, max_iter

if __name__ == "__main__":
    A = [[4.0, 1.0], [1.0, 3.0]]; b = [1.0, 2.0]
    x = gradient_conjugue(A, b)
    print("Solution de Ax=b :", x, "(exact : [1/11, 7/11] =", [1/11, 7/11], ")")
    from problemes import QUADRATIQUE as P
    u, k = newton_gradient_conjugue(P.grad, P.hess, P.u0)
    print(f"{P.nom} (Newton-CG) : u* ≈ {u}, itérations de Newton = {k}")
