#Methode du gradient conjugu2 pour A x = b, avec A symétrique

from algebre import dot, add, sub, scale, matvec, norme2_carre
from outils import Resultat, valeur


def gradient_conjugue(A, b, x0=None, tol_rel=1e-12, max_iter=None):
    """Résout A x = b. Renvoie (x, nombre de produits matrice-vecteur).

    """L'arrêt a lieu lorsque ||r||^2 <= tol_rel^2 * ||b||^2. """
   
    n = len(b)
    max_iter = 2 * n if max_iter is None else max_iter
    produits = 0
    if x0 is None:
        x = [0.0] * n
        r = list(b)
    else:
        x = list(x0)
        r = sub(b, matvec(A, x))
        produits += 1
    v = list(r)
    rr = dot(r, r)
    seuil = tol_rel ** 2 * dot(b, b)
    for _ in range(max_iter):
        if rr <= seuil:
            break
        Av = matvec(A, v)
        produits += 1
        courbure = dot(v, Av)
        if courbure <= 0.0:
            raise ValueError("matrice non définie positive (courbure <= 0)")
        alpha = rr / courbure
        x = add(x, scale(alpha, v))             # (A.16)
        r = sub(r, scale(alpha, Av))            # (A.17), forme récurrente
        rr_new = dot(r, r)
        v = add(r, scale(rr_new / rr, v))       # (A.18)
        rr = rr_new
    return x, produits


def newton_gradient_conjugue(f, grad, hess, u0, eps=1e-12, max_iter=200,
                             amorti=True, beta=0.5, c=1e-4, alpha_min=1e-16):
   
    u = list(u0)
    res = Resultat(u=list(u0), infos={"produits_mv": 0})
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
        delta, produits = gradient_conjugue(hess(u), g)
        res.infos["produits_mv"] += produits
        p = scale(-1.0, delta)
        alpha = 1.0
        u_new = add(u, p)
        f_new = valeur(f, u_new)
        if amorti:
            gp = dot(g, p)
            while f_new > fu + c * alpha * gp and alpha > alpha_min:
                alpha *= beta
                u_new = add(u, scale(alpha, p))
                f_new = valeur(f, u_new)
        u, fu, g = u_new, f_new, grad(u_new)
    return res


if __name__ == "__main__":
    A = [[4.0, 1.0], [1.0, 3.0]]
    b = [1.0, 2.0]
    x, produits = gradient_conjugue(A, b)
    print("Solution de Ax = b :", x, "(exacte :", [1 / 11, 7 / 11], ")",
          f"produits matrice-vecteur : {produits}")
    from problemes import QUADRATIQUE as P
    r = newton_gradient_conjugue(P.f, P.grad, P.hess, P.u0)
    print(f"{P.nom} (Newton et gradient conjugué) : u = {r.u}, "
          f"itérations de Newton = {r.iterations}, statut = {r.statut}")
