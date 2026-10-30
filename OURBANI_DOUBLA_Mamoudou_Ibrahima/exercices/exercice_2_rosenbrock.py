"""
Exercice 2 - Fonction de Rosenbrock
Serie de OURBANI DOUBLA MAMOUDOU IBRAHIMA

    python exercices/exercice_2_rosenbrock.py                 toutes les methodes
    python exercices/exercice_2_rosenbrock.py newton          une seule methode, avec les iterations
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from outils.exercice import Exercice  # noqa: E402


EXERCICE = Exercice(
    2, "Fonction de Rosenbrock",
    "(1 - x)**2 + 100*(y - x**2)**2",
    x0=[-1.2, 1],
    enonce="""
f(x, y) = (1 - x)^2 + 100 (y - x^2)^2 (vallee tres etroite en forme de banane).
Verifier que (1, 1) est l'unique minimum, puis minimiser f depuis (-1.2, 1).
Expliquer pourquoi les methodes de gradient sont tres lentes ici.""",
    solution=[1, 1], f_min=0.0,
    options={"pas_fixe": {"alpha": 0.001, "max_iter": 100000},
             "pas_optimal": {"max_iter": 20000}},
    remarque="hessienne en (1,1) : valeurs propres 0.4 et 1001.6, conditionnement ~ 2500",
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
