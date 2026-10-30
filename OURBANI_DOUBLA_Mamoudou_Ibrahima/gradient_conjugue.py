"""
Methode du gradient conjugue non lineaire (Fletcher-Reeves / Polak-Ribiere+).

Utilisation :
    from gradient_conjugue import gradient_conjugue
    r = gradient_conjugue("x**2 + 3*y**2 - x*y", [1, 1])
    print(r)
"""
from outils.algebre import norme, scalaire
from outils.recherche_lineaire import armijo, pas_optimal, wolfe
from outils.resultat import (SEUIL_NON_BORNEE, MSG_NON_BORNEE, MSG_PAS_DE_PAS, Resultat,
                             msg_max_iter, noter, preparer)


def gradient_conjugue(f, x0, formule="PR", recherche="exacte", tol=1e-6, max_iter=5000,
                      verbose=False, **kw):
    """d_0 = -g_0,  d_{k+1} = -g_{k+1} + beta_k d_k

    formule : "FR" (Fletcher-Reeves)  beta = ||g_{k+1}||^2 / ||g_k||^2
              "PR" (Polak-Ribiere+)   beta = max(0, <g_{k+1}, g_{k+1} - g_k> / ||g_k||^2)
    recherche : "exacte" (pas optimal) ou "wolfe".
    On redemarre (d = -g) toutes les n iterations.
    Sur une quadratique en dimension n avec pas exact : n iterations au plus."""
    formule = formule.upper()
    if formule not in ("FR", "PR"):
        raise ValueError("formule = 'FR' ou 'PR'")
    pb, x, fx, gx = preparer(f, x0, **kw)
    n = x.size
    d = -gx
    historique = []
    pas = None
    nom = f"gradient conjugue ({formule}, pas {'exact' if recherche != 'wolfe' else 'de Wolfe'})"
    for k in range(max_iter + 1):
        ng = noter(historique, verbose, k, x, fx, gx, pas)
        if ng <= tol:
            return Resultat(nom, pb, x, fx, gx, k, True, "convergence", historique)
        if k == max_iter:
            break
        if fx < SEUIL_NON_BORNEE:
            return Resultat(nom, pb, x, fx, gx, k, False, MSG_NON_BORNEE, historique)
        if scalaire(gx, d) >= 0:          # plus une direction de descente
            d = -gx
        a0 = min(1.0, 1.0 / ng) if k == 0 else 1.0
        if recherche == "wolfe":
            r = wolfe(pb, x, d, fx, gx, a0=a0, c2=0.1) or armijo(pb, x, d, fx, gx, a0=a0)
        else:
            r = pas_optimal(pb, x, d, fx, a0=a0)
        if r is None and scalaire(d, -gx) < norme(d) * ng * (1 - 1e-12):
            d = -gx                       # dernier recours : direction du gradient
            r = pas_optimal(pb, x, d, fx, a0=a0)
        if r is None:
            return Resultat(nom, pb, x, fx, gx, k, False, MSG_PAS_DE_PAS, historique)
        pas, xn, fn, gn = r
        if norme(xn - x) <= 1e-15 * (1.0 + norme(x)):
            return Resultat(nom, pb, x, fx, gx, k, False,
                            "stagnation numerique avant d'atteindre tol", historique)

        g2 = scalaire(gx, gx)
        if formule == "FR":
            beta = scalaire(gn, gn) / g2
        else:
            beta = max(0.0, scalaire(gn, gn - gx) / g2)
        if (k + 1) % n == 0:
            beta = 0.0                    # redemarrage periodique
        d = -gn + beta * d
        x, fx, gx = xn, fn, gn
    return Resultat(nom, pb, x, fx, gx, max_iter, False,
                    msg_max_iter(historique, gx), historique)


if __name__ == "__main__":
    # python gradient_conjugue.py                          -> exercice du cours
    # python gradient_conjugue.py "x**2 + 3*y**2 - x*y" 1 1 -> n'importe quelle fonction
    import sys
    if len(sys.argv) > 2:
        f, x0 = sys.argv[1], [float(v) for v in sys.argv[2:]]
        options = {}
    else:
        f = "6*x**2 + 6*x*y + 3*y**2 + y**3 - z**3/3 + z**4/4"
        x0 = [-0.6637, 0.1758, 1.5778]
        options = {"tol": 1e-5}
    r = gradient_conjugue(f, x0, **options)
    r.afficher_iterations()
    print()
    print(r)
