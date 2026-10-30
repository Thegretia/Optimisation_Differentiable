"""
Exercice 1 - Quadratique en dimension 2
Serie de OURBANI DOUBLA MAMOUDOU IBRAHIMA

    python exercices/exercice_1_quadratique_2d.py                 toutes les methodes
    python exercices/exercice_1_quadratique_2d.py newton          une seule methode, avec les iterations
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from outils.exercice import Exercice  # noqa: E402


EXERCICE = Exercice(
    1, "Quadratique en dimension 2",
    "2*x**2 + x*y + y**2 - 3*x + 2*y",
    x0=[3, -3],
    enonce="""
Soit f(x, y) = 2x^2 + xy + y^2 - 3x + 2y.
a) Ecrire f sous la forme 1/2 X^T A X - b^T X et montrer que A est definie positive.
b) En deduire le minimum exact (resoudre A X = b).
c) Comparer le nombre d'iterations des differentes methodes depuis X0 = (3, -3).""",
    solution=[8 / 7, -11 / 7], f_min=-23 / 7,
    options={"pas_fixe": {"alpha": 0.2}},
    remarque="A = [[4, 1], [1, 2]], valeurs propres 3 -/+ sqrt(2) ; il faut alpha < 2/(3+sqrt 2) = 0.453 pour le pas fixe",
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
