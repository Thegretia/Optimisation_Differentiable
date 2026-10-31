"""
Exercice 3 - Fonction de Beale
Serie de NJIFON GHANE MOUNIR

    python exercices/exercice_3_beale.py                 toutes les methodes
    python exercices/exercice_3_beale.py newton          une seule methode, avec les iterations
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from outils.exercice import Exercice  # noqa: E402


EXERCICE = Exercice(
    3, "Fonction de Beale",
    "(1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2",
    x0=[1, 1],
    enonce="""
f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2.
a) Verifier que f(3, 0.5) = 0. Pourquoi est-ce un minimum global ?
b) Minimiser f depuis (1, 1) et comparer les methodes.""",
    solution=[3, 0.5], f_min=0.0,
    options={"pas_fixe": {"alpha": 0.01, "max_iter": 50000}},
    remarque="f est tres plate dans certaines directions : le pas fixe est lent",
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
