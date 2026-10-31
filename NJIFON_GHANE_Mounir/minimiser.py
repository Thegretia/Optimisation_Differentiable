"""
Point d'entree commun : minimiser(f, x0, methode) et comparer(f, x0).

    from minimiser import minimiser, comparer
    r = minimiser("x**2 + 3*y**2 - x*y", [1, 1], methode="newton")
    comparer("x**2 + 3*y**2 - x*y", [1, 1])
"""
from gradient_conjugue import gradient_conjugue
from gradient_pas_fixe import gradient_pas_fixe
from gradient_pas_optimal import gradient_pas_optimal
from newton import newton
from outils.resultat import afficher_comparaison
from quasi_newton import quasi_newton


METHODES = {
    "pas_fixe": gradient_pas_fixe,
    "pas_optimal": gradient_pas_optimal,
    "gradient_conjugue": gradient_conjugue,
    "newton": newton,
    "newton_pur": lambda f, x0, **o: newton(f, x0, pur=True, **o),
    "bfgs": quasi_newton,
    "dfp": lambda f, x0, **o: quasi_newton(f, x0, formule="DFP", **o),
}

ALIAS = {
    "fixe": "pas_fixe", "gradient_pas_fixe": "pas_fixe",
    "optimal": "pas_optimal", "gradient_pas_optimal": "pas_optimal",
    "gc": "gradient_conjugue", "conjugue": "gradient_conjugue",
    "quasi_newton": "bfgs", "quasi-newton": "bfgs",
}


def minimiser(f, x0, methode="bfgs", **options):
    """Minimise f a partir de x0.

    f       : chaine ("x**2 + y**2"), expression sympy, fonction python ou Probleme
    methode : pas_fixe, pas_optimal, gradient_conjugue, newton, newton_pur, bfgs, dfp
    options : tol, max_iter, verbose, alpha (pas fixe), formule (GC / quasi-Newton),
              grad, hess (si f est une fonction python), variables ("x y z")"""
    cle = methode.lower().replace(" ", "_")
    cle = ALIAS.get(cle, cle)
    if cle not in METHODES:
        raise ValueError(f"methode inconnue '{methode}'. Choix : {', '.join(METHODES)}")
    return METHODES[cle](f, x0, **options)


def comparer(f, x0, methodes=None, options=None, afficher=True, **communes):
    """Lance plusieurs methodes sur le meme probleme et affiche un tableau.
    options : dict {methode: {options propres}}, ex. {"pas_fixe": {"alpha": 0.05}}"""
    methodes = methodes or ["pas_fixe", "pas_optimal", "gradient_conjugue", "newton", "bfgs"]
    options = options or {}
    resultats = []
    for m in methodes:
        opts = dict(communes)
        opts.update(options.get(m, {}))
        resultats.append(minimiser(f, x0, methode=m, **opts))
    if afficher:
        afficher_comparaison(resultats)
    return resultats
