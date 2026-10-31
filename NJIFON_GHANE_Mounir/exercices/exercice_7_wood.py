"""
Exercice 7 - Fonction de Wood (dimension 4)
Serie de NJIFON GHANE MOUNIR

    python exercices/exercice_7_wood.py                 toutes les methodes
    python exercices/exercice_7_wood.py newton          une seule methode, avec les iterations
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from outils.exercice import Exercice  # noqa: E402

# --- exercice 7 : fonction de Wood ------------------------------------------
WOOD = ("100*(x1**2 - x2)**2 + (x1 - 1)**2 + (x3 - 1)**2 + 90*(x3**2 - x4)**2"
        " + 10.1*((x2 - 1)**2 + (x4 - 1)**2) + 19.8*(x2 - 1)*(x4 - 1)")


EXERCICE = Exercice(
    7, "Fonction de Wood (dimension 4)",
    WOOD,
    x0=[-3, -1, -3, -1],
    enonce="""
f(x1, x2, x3, x4) = 100(x1^2 - x2)^2 + (x1 - 1)^2 + (x3 - 1)^2 + 90(x3^2 - x4)^2
                + 10.1[(x2 - 1)^2 + (x4 - 1)^2] + 19.8(x2 - 1)(x4 - 1)
Probleme test classique (More, Garbow et Hillstrom, 1981). Minimum en (1, 1, 1, 1).
Comparer le nombre d'iterations et d'evaluations de f et du gradient.""",
    solution=[1, 1, 1, 1], f_min=0.0,
    options={"pas_fixe": {"alpha": 0.0005, "max_iter": 100000},
             "pas_optimal": {"max_iter": 50000}},
    remarque="les variables sont x1, x2, x3, x4 ; avec alpha = 0.001 le pas fixe diverge des la 2e iteration",
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
