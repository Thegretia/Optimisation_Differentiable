"""
Exercice 6 - Maximisation d'un profit
Serie de OURBANI DOUBLA MAMOUDOU IBRAHIMA

    python exercices/exercice_6_profit.py                 toutes les methodes
    python exercices/exercice_6_profit.py newton          une seule methode, avec les iterations
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from outils.exercice import Exercice  # noqa: E402


EXERCICE = Exercice(
    6, "Maximisation d'un profit",
    "-(50*x + 60*y - x**2 - x*y - y**2)",
    x0=[0, 0],
    enonce="""
Une entreprise fabrique deux produits en quantites x et y (centaines d'unites).
Son profit est P(x, y) = 50x + 60y - x^2 - xy - y^2 (milliers FCFA).
Maximiser P revient a minimiser f = -P.
a) Trouver le point critique et verifier que c'est un maximum de P.
b) Retrouver le resultat numeriquement. Quel est le profit maximal ?""",
    solution=[40 / 3, 70 / 3], f_min=-3100 / 3,
    options={"pas_fixe": {"alpha": 0.3}},
    remarque="profit maximal = 3100/3 ~ 1033.33",
)


if __name__ == "__main__":
    EXERCICE.afficher_enonce()
    print()
    if len(sys.argv) > 1:
        r = EXERCICE.resoudre(sys.argv[1])
        r.afficher_iterations()
        print()
        print(r)
    else:
        EXERCICE.comparer()
