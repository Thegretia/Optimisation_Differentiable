"""
Exercice 5 - Fonction non quadratique de trois variables
Serie de OURBANI DOUBLA MAMOUDOU IBRAHIMA

    python exercices/exercice_5_fonction_3_variables.py                 toutes les methodes
    python exercices/exercice_5_fonction_3_variables.py newton          une seule methode, avec les iterations
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from outils.exercice import Exercice  # noqa: E402


EXERCICE = Exercice(
    5, "Fonction non quadratique de trois variables",
    "(x - 1)**2 + (y - 2)**2 + (z - 3)**2 + (x*y*z - 6)**2",
    x0=[2, 1, 1],
    enonce="""
f(x, y, z) = (x-1)^2 + (y-2)^2 + (z-3)^2 + (xyz - 6)^2.
a) Montrer que f(x, y, z) >= 0 et trouver un point ou f = 0. Conclure.
b) Calculer grad f.
c) Minimiser f a partir de (2, 1, 1).
   (Depuis (0, 0, 0) le pas optimal tombe sur la solution en une seule
   iteration : expliquer pourquoi.)""",
    solution=[1, 2, 3], f_min=0.0,
    options={"pas_fixe": {"alpha": 0.01}},
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
