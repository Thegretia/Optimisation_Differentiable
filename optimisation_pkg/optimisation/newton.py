"""Méthode de Newton et méthode quasi-Newton (BFGS)."""
from .algebre import (identite, mat_ajoute, mat_mat, mat_vec, norme,
                      produit_externe, produit_scalaire, resoudre_systeme,
                      vec_ajoute, vec_fois_scalaire)
from .outils import Resultat, armijo


def newton(f, grad, hess, x0, amorti=False, tol=1e-6, max_iter=100):
    """Méthode de Newton : x_{k+1} = x_k + a * d, avec  H(x_k) d = -grad f(x_k).

    On ne calcule pas l'inverse du hessien : on résout le système H d = -g
    (élimination de Gauss écrite à la main).

    amorti=False : pas a = 1 (Newton pur, très rapide près du minimum)
    amorti=True  : pas choisi par Armijo (plus robuste loin du minimum)
    Si le hessien est singulier ou si d ne descend pas, on prend d = -g.
    """
    x = [float(v) for v in x0]
    historique = [list(x)]
    converge = False
    k = 0

    for k in range(max_iter):
        g = grad(x)
        if norme(g) < tol:
            converge = True
            break

        try:
            d = resoudre_systeme(hess(x), vec_fois_scalaire(g, -1.0))
        except ValueError:
            d = vec_fois_scalaire(g, -1.0)
        if produit_scalaire(g, d) >= 0:          # pas une direction de descente
            d = vec_fois_scalaire(g, -1.0)

        a = armijo(f, x, g, d) if amorti else 1.0
        x = vec_ajoute(x, d, a)
        historique.append(list(x))
    else:
        k = max_iter

    nom = "Newton amorti" if amorti else "Newton"
    return Resultat(x, f(x), k, converge, historique, nom)


def quasi_newton(f, grad, x0, tol=1e-6, max_iter=200):
    """Quasi-Newton BFGS : on estime l'inverse du hessien H (pas besoin du hessien).

    H+ = (I - rho s y^T) H (I - rho y s^T) + rho s s^T,   rho = 1 / (y . s)
        s = x_new - x      (déplacement)
        y = g_new - g      (variation du gradient)
    """
    n = len(x0)
    x = [float(v) for v in x0]
    I = identite(n)
    H = identite(n)
    g = grad(x)
    historique = [list(x)]
    converge = False
    k = 0

    for k in range(max_iter):
        if norme(g) < tol:
            converge = True
            break

        d = vec_fois_scalaire(mat_vec(H, g), -1.0)      # d = -H g
        a = armijo(f, x, g, d)
        x_new = vec_ajoute(x, d, a)
        g_new = grad(x_new)

        s = vec_ajoute(x_new, x, -1.0)
        y = vec_ajoute(g_new, g, -1.0)
        ys = produit_scalaire(y, s)

        if ys > 1e-12:                                  # condition de courbure
            rho = 1.0 / ys
            A = mat_ajoute(I, produit_externe(s, y), -rho)
            B = mat_ajoute(I, produit_externe(y, s), -rho)
            H = mat_ajoute(mat_mat(mat_mat(A, H), B),
                           produit_externe(s, s), rho)

        x, g = x_new, g_new
        historique.append(list(x))
    else:
        k = max_iter

    return Resultat(x, f(x), k, converge, historique, "quasi-Newton (BFGS)")
