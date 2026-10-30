"""
Exercice 0 - Exercice du cours
Serie de OURBANI DOUBLA MAMOUDOU IBRAHIMA

    python exercices/exercice_0_cours.py                 toutes les methodes
    python exercices/exercice_0_cours.py newton          une seule methode, avec les iterations
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from outils.exercice import Exercice  # noqa: E402


EXERCICE = Exercice(
    0, "Exercice du cours",
    "6*x**2 + 6*x*y + 3*y**2 + y**3 - z**3/3 + z**4/4",
    x0=[-0.6637, 0.1758, 1.5778],
    enonce="""
Minimiser f(x, y, z) = 6x^2 + 6xy + 3y^2 + y^3 - z^3/3 + z^4/4
a partir de X0 = (-0.6637 ; 0.1758 ; 1.5778) avec ||grad f|| < 1e-5.
f n'est pas bornee inferieurement (terme y^3) : on cherche un minimum local.""",
    solution=[0, 0, 1], f_min=-1 / 12, tol=1e-5,
    options={"pas_fixe": {"alpha": 0.05}},
    remarque="points critiques : (0,0,0) degenere, (0,0,1) minimum, (0.5,-1,0) et (0.5,-1,1) selles",
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
