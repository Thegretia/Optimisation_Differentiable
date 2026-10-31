"""
Exercice 1 - Quadratique mal conditionnee (zigzag)
Serie de NJIFON GHANE MOUNIR

    python exercices/exercice_1_quadratique_zigzag.py                 toutes les methodes
    python exercices/exercice_1_quadratique_zigzag.py newton          une seule methode, avec les iterations
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from outils.exercice import Exercice  # noqa: E402


EXERCICE = Exercice(
    1, "Quadratique mal conditionnee (zigzag)",
    "(x - 2)**2 + 20*(y + 1)**2",
    x0=[-3, 2],
    enonce="""
f(x, y) = (x - 2)^2 + 20 (y + 1)^2.
a) Donner le minimum et le conditionnement de la hessienne.
b) Minimiser f depuis (-3, 2) par le gradient a pas optimal et tracer les
   iteres : observer le zigzag (deux gradients successifs sont orthogonaux).
c) Pour le pas fixe, quelle est la plus grande valeur de alpha permise ?""",
    solution=[2, -1], f_min=0.0,
    options={"pas_fixe": {"alpha": 0.04}},
    remarque="H = diag(2, 40), conditionnement 20 ; pas fixe stable si alpha < 2/40 = 0.05",
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
