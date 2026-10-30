"""
Exercice 3 - Fonction de Himmelblau (plusieurs minima)
Serie de OURBANI DOUBLA MAMOUDOU IBRAHIMA

    python exercices/exercice_3_himmelblau.py                 toutes les methodes
    python exercices/exercice_3_himmelblau.py newton          une seule methode, avec les iterations
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from outils.exercice import Exercice  # noqa: E402


EXERCICE = Exercice(
    3, "Fonction de Himmelblau (plusieurs minima)",
    "(x**2 + y - 11)**2 + (x + y**2 - 7)**2",
    x0=[0, 0],
    enonce="""
f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2 admet quatre minima globaux (f = 0).
a) Verifier que (3, 2) est un minimum.
b) Lancer les methodes depuis (0, 0), (-1, 1), (-4, -4) et (4, -2) : vers quel
   minimum converge-t-on ? Conclure sur le role du point initial.""",
    solution=[[3, 2], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]],
    f_min=0.0,
    options={"pas_fixe": {"alpha": 0.01}},
    remarque="un minimum local de f n'est pas forcement unique : le resultat depend de x0",
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
