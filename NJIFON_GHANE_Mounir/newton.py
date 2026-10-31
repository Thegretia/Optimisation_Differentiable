"""
Methode de Newton (version du cours 'pur=True' et version globalisee).

Utilisation :
    from newton import newton
    r = newton("x**2 + 3*y**2 - x*y", [1, 1])
    print(r)
"""
import math

from outils.algebre import (MatriceSinguliere, cholesky, identite, norme, resoudre_cholesky,
                            resoudre_gauss)
from outils.recherche_lineaire import armijo
from outils.resultat import (SEUIL_DIVERGENCE, SEUIL_NON_BORNEE, MSG_NON_BORNEE, MSG_PAS_DE_PAS,
                             Resultat, msg_max_iter, noter, preparer)


def newton(f, x0, tol=1e-6, max_iter=100, pur=False, verbose=False, **kw):
    """Direction de Newton : H(x_k) d_k = -grad f(x_k).

    pur=True  : methode du cours, pas = 1 et systeme resolu par Gauss.
                Converge tres vite pres d'un minimum mais peut aller vers un
                point selle ou un maximum, ou echouer si H est singuliere.
    pur=False : Newton 'globalise' : si H n'est pas definie positive on
                utilise H + tau I (Cholesky), puis pas d'Armijo. Pres d'un
                minimum on retrouve exactement Newton pur (pas = 1)."""
    pb, x, fx, gx = preparer(f, x0, **kw)
    n = x.size
    I = identite(n)
    historique = []
    pas = None
    nom = "Newton pur" if pur else "Newton"
    for k in range(max_iter + 1):
        ng = noter(historique, verbose, k, x, fx, gx, pas)
        if ng <= tol:
            return Resultat(nom, pb, x, fx, gx, k, True, "convergence", historique)
        if k == max_iter:
            break
        if fx < SEUIL_NON_BORNEE:
            return Resultat(nom, pb, x, fx, gx, k, False, MSG_NON_BORNEE, historique)
        H = pb.hess(x)
        if H is None:
            return Resultat(nom, pb, x, fx, gx, k, False, "hessienne non calculable", historique)

        if pur:
            try:
                d = resoudre_gauss(H, -gx)
            except MatriceSinguliere:
                return Resultat(nom, pb, x, fx, gx, k, False,
                                "hessienne singuliere : la direction de Newton n'existe pas",
                                historique)
            pas = 1.0
            xn = x + d
            fn = pb.f(xn)
            gn = pb.grad(xn)
            if not math.isfinite(fn) or gn is None or norme(xn) > SEUIL_DIVERGENCE:
                return Resultat(nom, pb, x, fx, gx, k, False, "divergence", historique)
        else:
            # H + tau I definie positive (Nocedal & Wright, algo 3.3)
            diag_min = min(H[i, i] for i in range(n))
            tau = 0.0 if diag_min > 0 else -diag_min + 1e-3
            L = cholesky(H + tau * I)
            while L is None and tau < 1e12:
                tau = max(2.0 * tau, 1e-3)
                L = cholesky(H + tau * I)
            d = resoudre_cholesky(L, -gx) if L is not None else -gx
            r = armijo(pb, x, d, fx, gx)
            if r is None:
                r = armijo(pb, x, -gx, fx, gx)
            if r is None:
                return Resultat(nom, pb, x, fx, gx, k, False, MSG_PAS_DE_PAS, historique)
            pas, xn, fn, gn = r

        if norme(xn - x) <= 1e-15 * (1.0 + norme(x)):
            return Resultat(nom, pb, x, fx, gx, k, False,
                            "stagnation numerique avant d'atteindre tol", historique)
        x, fx, gx = xn, fn, gn
    return Resultat(nom, pb, x, fx, gx, max_iter, False,
                    msg_max_iter(historique, gx), historique)


if __name__ == "__main__":
    # python newton.py                          -> exercice du cours
    # python newton.py "x**2 + 3*y**2 - x*y" 1 1 -> n'importe quelle fonction
    import sys
    if len(sys.argv) > 2:
        f, x0 = sys.argv[1], [float(v) for v in sys.argv[2:]]
        options = {}
    else:
        f = "6*x**2 + 6*x*y + 3*y**2 + y**3 - z**3/3 + z**4/4"
        x0 = [-0.6637, 0.1758, 1.5778]
        options = {"tol": 1e-5}
    r = newton(f, x0, **options)
    r.afficher_iterations()
    print()
    print(r)
