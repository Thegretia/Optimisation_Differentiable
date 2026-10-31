"""
Exercice 6 - Points critiques de x^4 + y^4 + z^4 - 4xyz
Serie de NJIFON GHANE MOUNIR

    python exercices/exercice_6_points_critiques.py                 toutes les methodes
    python exercices/exercice_6_points_critiques.py newton          une seule methode, avec les iterations
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from outils.exercice import Exercice  # noqa: E402


EXERCICE = Exercice(
    6, "Points critiques de x^4 + y^4 + z^4 - 4xyz",
    "x**4 + y**4 + z**4 - 4*x*y*z",
    x0=[2, 1.5, 0.5],
    enonce="""
f(x, y, z) = x^4 + y^4 + z^4 - 4xyz.
a) Trouver tous les points critiques reels et leur nature.
b) Minimiser f depuis (2, 1.5, 0.5).
c) Lancer Newton pur depuis (0.1, 0.2, 0.3) : que se passe-t-il ?""",
    solution=[[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], f_min=-1.0,
    options={"pas_fixe": {"alpha": 0.02}},
    remarque="4 minima (f = -1) ; l'origine est un point critique degenere qui n'est pas un minimum",
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
