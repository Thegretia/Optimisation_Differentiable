"""
Exercice 7 - Quadratique en dimension 10
Serie de OURBANI DOUBLA MAMOUDOU IBRAHIMA

    python exercices/exercice_7_quadratique_dim10.py                 toutes les methodes
    python exercices/exercice_7_quadratique_dim10.py newton          une seule methode, avec les iterations
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from outils.exercice import Exercice  # noqa: E402

# --- exercice 7 : quadratique en dimension 10 -------------------------------
N = 10
_termes = [f"x{i}**2" for i in range(1, N + 1)]
_termes += [f"(-x{i}*x{i + 1})" for i in range(1, N)]
_termes += [f"(-x{i})" for i in range(1, N + 1)]
QUADRA_N = " + ".join(_termes)        # = 1/2 x^T A x - b^T x, A = tridiag(-1, 2, -1), b = 1


EXERCICE = Exercice(
    7, "Quadratique en dimension 10",
    QUADRA_N,
    x0=[0] * N,
    enonce="""
f(x) = 1/2 x^T A x - b^T x avec A = tridiag(-1, 2, -1) de taille 10 et b = (1, ..., 1).
(probleme -u'' = 1 discretise.) La solution exacte est x_i = i (11 - i) / 2.
a) Verifier que A est definie positive.
b) Verifier qu'avec le pas exact le gradient conjugue converge en au plus 10 iterations.
c) Comparer avec le gradient a pas optimal.""",
    solution=[i * (N + 1 - i) / 2 for i in range(1, N + 1)], f_min=-55.0,
    options={"pas_fixe": {"alpha": 0.25}},
    remarque="les variables sont x1, ..., x10 (dans cet ordre)",
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
