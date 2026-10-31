"""
Methode du gradient a pas fixe.

Utilisation :
    from gradient_pas_fixe import gradient_pas_fixe
    r = gradient_pas_fixe("x**2 + 3*y**2 - x*y", [1, 1])
    print(r)
"""
import math

from outils.algebre import norme
from outils.resultat import (SEUIL_DIVERGENCE, SEUIL_NON_BORNEE, MSG_NON_BORNEE, Resultat,
                             lambda_max, msg_max_iter, noter, preparer)


def gradient_pas_fixe(f, x0, alpha=None, tol=1e-6, max_iter=10000, verbose=False, **kw):
    """x_{k+1} = x_k - alpha grad f(x_k).

    Si alpha n'est pas donne, on prend alpha = 1 / lambda_max(H(x0)).
    Pour une quadratique il faut 0 < alpha < 2 / lambda_max sinon ca diverge."""
    pb, x, fx, gx = preparer(f, x0, **kw)
    lmax = lambda_max(pb, x)
    note = ""
    if alpha is None:
        alpha = 1.0 / lmax if lmax is not None and lmax > 1e-12 else 1e-3
        note = f" (alpha choisi automatiquement = 1/lambda_max = {alpha:.4g})"
    if alpha <= 0:
        raise ValueError("alpha doit etre > 0")

    def conseil():
        lm = lambda_max(pb, x)
        if lm is not None and lm > 0 and alpha >= 2.0 / lm:
            return f" alpha = {alpha:g} >= 2/lambda_max = {2.0 / lm:.4g} : prendre un pas plus petit."
        return ""

    historique = []
    nom = f"gradient a pas fixe (alpha={alpha:g})"
    for k in range(max_iter + 1):
        ng = noter(historique, verbose, k, x, fx, gx, alpha if k else None)
        if ng <= tol:
            return Resultat(nom, pb, x, fx, gx, k, True, "convergence" + note, historique)
        if k == max_iter:
            break
        if fx < SEUIL_NON_BORNEE:
            return Resultat(nom, pb, x, fx, gx, k, False, MSG_NON_BORNEE + conseil(), historique)
        xn = x - alpha * gx
        fn = pb.f(xn)
        if not math.isfinite(fn) or abs(fn) > SEUIL_DIVERGENCE or norme(xn) > SEUIL_DIVERGENCE:
            return Resultat(nom, pb, x, fx, gx, k, False,
                            "divergence." + (conseil() or " Essayer un alpha plus petit."), historique)
        gn = pb.grad(xn)
        if gn is None:
            return Resultat(nom, pb, x, fx, gx, k, False, "gradient non calculable", historique)
        x, fx, gx = xn, fn, gn

    c = conseil()
    msg = msg_max_iter(historique, gx) + c
    if not c and "lente" in msg:
        msg += " Ou augmenter alpha (sans depasser 2/lambda_max)."
    return Resultat(nom, pb, x, fx, gx, max_iter, False, msg, historique)


if __name__ == "__main__":
    # python gradient_pas_fixe.py                          -> exercice du cours
    # python gradient_pas_fixe.py "x**2 + 3*y**2 - x*y" 1 1 -> n'importe quelle fonction
    import sys
    if len(sys.argv) > 2:
        f, x0 = sys.argv[1], [float(v) for v in sys.argv[2:]]
        options = {}
    else:
        f = "6*x**2 + 6*x*y + 3*y**2 + y**3 - z**3/3 + z**4/4"
        x0 = [-0.6637, 0.1758, 1.5778]
        options = {"alpha": 0.05, "tol": 1e-5}
    r = gradient_pas_fixe(f, x0, **options)
    r.afficher_iterations()
    print()
    print(r)
