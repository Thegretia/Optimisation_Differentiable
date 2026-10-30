"""
Exercice 4 - Regression lineaire par moindres carres
Serie de OURBANI DOUBLA MAMOUDOU IBRAHIMA

    python exercices/exercice_4_regression_lineaire.py                 toutes les methodes
    python exercices/exercice_4_regression_lineaire.py newton          une seule methode, avec les iterations
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from outils.exercice import Exercice  # noqa: E402

# --- donnees de l'exercice 4 (regression lineaire) -------------------------
T_PUB = [1, 2, 3, 4, 5, 6]                     # budget publicite (centaines de milliers FCFA)
VENTES = [3.1, 4.9, 7.2, 8.8, 11.1, 13.2]       # ventes (millions FCFA)
MOINDRES_CARRES = " + ".join(f"(a*{t} + b - {v})**2" for t, v in zip(T_PUB, VENTES))


EXERCICE = Exercice(
    4, "Regression lineaire par moindres carres",
    MOINDRES_CARRES,
    x0=[0, 0], variables="a b",
    enonce=f"""
Une boutique note ses ventes y (millions FCFA) selon le budget publicite t :
   t = {T_PUB}
   y = {VENTES}
On cherche la droite y = a t + b qui minimise S(a, b) = somme (a t_i + b - y_i)^2.
a) Ecrire les equations normales et les resoudre a la main.
b) Retrouver (a, b) avec les algorithmes et comparer.""",
    solution=[2.02, 0.98], f_min=0.128,
    options={"pas_fixe": {"alpha": 0.005, "max_iter": 50000}},
    remarque="S est quadratique : Newton trouve la solution en 1 iteration",
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
