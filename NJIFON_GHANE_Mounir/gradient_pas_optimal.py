"""
Methode du gradient a pas optimal (recherche lineaire exacte par section doree).

Utilisation :
    from gradient_pas_optimal import gradient_pas_optimal
    r = gradient_pas_optimal("x**2 + 3*y**2 - x*y", [1, 1])
    print(r)
"""
from outils.algebre import norme
from outils.recherche_lineaire import pas_optimal
from outils.resultat import (SEUIL_NON_BORNEE, MSG_NON_BORNEE, MSG_PAS_DE_PAS, Resultat,
                             msg_max_iter, noter, preparer)


def gradient_pas_optimal(f, x0, tol=1e-6, max_iter=5000, verbose=False, **kw):
    """d_k = -grad f(x_k) et a_k = argmin_a f(x_k + a d_k) (section doree)."""
    pb, x, fx, gx = preparer(f, x0, **kw)
    historique = []
    pas = None
    nom = "gradient a pas optimal"
    for k in range(max_iter + 1):
        ng = noter(historique, verbose, k, x, fx, gx, pas)
        if ng <= tol:
            return Resultat(nom, pb, x, fx, gx, k, True, "convergence", historique)
        if k == max_iter:
            break
        if fx < SEUIL_NON_BORNEE:
            return Resultat(nom, pb, x, fx, gx, k, False, MSG_NON_BORNEE, historique)
        r = pas_optimal(pb, x, -gx, fx, a0=min(1.0, 1.0 / ng))
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
    # python gradient_pas_optimal.py                          -> exercice du cours
    # python gradient_pas_optimal.py "x**2 + 3*y**2 - x*y" 1 1 -> n'importe quelle fonction
    import sys
    if len(sys.argv) > 2:
        f, x0 = sys.argv[1], [float(v) for v in sys.argv[2:]]
        options = {}
    else:
        f = "6*x**2 + 6*x*y + 3*y**2 + y**3 - z**3/3 + z**4/4"
        x0 = [-0.6637, 0.1758, 1.5778]
        options = {"tol": 1e-5}
    r = gradient_pas_optimal(f, x0, **options)
    r.afficher_iterations()
    print()
    print(r)
