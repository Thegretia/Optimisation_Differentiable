"""
Methode quasi-Newton : BFGS (par defaut) ou DFP.

Utilisation :
    from quasi_newton import quasi_newton
    r = quasi_newton("x**2 + 3*y**2 - x*y", [1, 1])
    print(r)
"""
import numpy as np

from outils.algebre import identite, norme, scalaire
from outils.recherche_lineaire import armijo, wolfe
from outils.resultat import (SEUIL_NON_BORNEE, MSG_NON_BORNEE, MSG_PAS_DE_PAS, Resultat,
                             msg_max_iter, noter, preparer)


def quasi_newton(f, x0, formule="BFGS", tol=1e-6, max_iter=1000, verbose=False, **kw):
    """d_k = -W_k grad f(x_k) ou W_k approche l'inverse de la hessienne.
    Avec s = x_{k+1} - x_k, y = g_{k+1} - g_k, rho = 1 / <y, s> :
      BFGS : W <- (I - rho s y^T) W (I - rho y s^T) + rho s s^T
      DFP  : W <- W - (W y y^T W) / <y, W y> + rho s s^T
    Pas de Wolfe (c2 = 0.9) pour garantir <y, s> > 0."""
    formule = formule.upper()
    if formule not in ("BFGS", "DFP"):
        raise ValueError("formule = 'BFGS' ou 'DFP'")
    pb, x, fx, gx = preparer(f, x0, **kw)
    n = x.size
    I = identite(n)
    W = I.copy()
    premier = True
    historique = []
    pas = None
    nom = f"quasi-Newton {formule}"
    for k in range(max_iter + 1):
        ng = noter(historique, verbose, k, x, fx, gx, pas)
        if ng <= tol:
            return Resultat(nom, pb, x, fx, gx, k, True, "convergence", historique)
        if k == max_iter:
            break
        if fx < SEUIL_NON_BORNEE:
            return Resultat(nom, pb, x, fx, gx, k, False, MSG_NON_BORNEE, historique)

        d = -(W @ gx)
        if scalaire(gx, d) >= 0:          # W a perdu la positivite : on repart de I
            W, premier, d = I.copy(), True, -gx
        a0 = min(1.0, 1.0 / ng) if premier else 1.0
        r = wolfe(pb, x, d, fx, gx, a0=a0, c2=0.9) or armijo(pb, x, d, fx, gx, a0=a0)
        if r is None and not premier:
            W, premier, d = I.copy(), True, -gx
            a0 = min(1.0, 1.0 / ng)
            r = wolfe(pb, x, d, fx, gx, a0=a0, c2=0.9) or armijo(pb, x, d, fx, gx, a0=a0)
        if r is None:
            return Resultat(nom, pb, x, fx, gx, k, False, MSG_PAS_DE_PAS, historique)
        pas, xn, fn, gn = r

        s = xn - x
        y = gn - gx
        if norme(s) <= 1e-15 * (1.0 + norme(x)):
            return Resultat(nom, pb, x, fx, gx, k, False,
                            "stagnation numerique avant d'atteindre tol", historique)
        sy = scalaire(s, y)
        if sy > 1e-12 * norme(s) * norme(y):
            if premier:                   # mise a l'echelle de W_0 (N&W eq. 6.20)
                W = (sy / scalaire(y, y)) * I
                premier = False
            rho = 1.0 / sy
            if formule == "BFGS":
                A = I - rho * np.outer(s, y)
                W = A @ W @ A.T + rho * np.outer(s, s)
            else:
                Wy = W @ y
                W = W - np.outer(Wy, Wy) / scalaire(y, Wy) + rho * np.outer(s, s)
        x, fx, gx = xn, fn, gn

    return Resultat(nom, pb, x, fx, gx, max_iter, False,
                    msg_max_iter(historique, gx), historique)


if __name__ == "__main__":
    # python quasi_newton.py                          -> exercice du cours
    # python quasi_newton.py "x**2 + 3*y**2 - x*y" 1 1 -> n'importe quelle fonction
    import sys
    if len(sys.argv) > 2:
        f, x0 = sys.argv[1], [float(v) for v in sys.argv[2:]]
        options = {}
    else:
        f = "6*x**2 + 6*x*y + 3*y**2 + y**3 - z**3/3 + z**4/4"
        x0 = [-0.6637, 0.1758, 1.5778]
        options = {"tol": 1e-5}
    r = quasi_newton(f, x0, **options)
    r.afficher_iterations()
    print()
    print(r)
