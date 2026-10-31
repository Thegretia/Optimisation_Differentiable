"""
Exercice 5 - Regression logistique
Serie de NJIFON GHANE MOUNIR

    python exercices/exercice_5_regression_logistique.py                 toutes les methodes
    python exercices/exercice_5_regression_logistique.py newton          une seule methode, avec les iterations
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from outils.exercice import Exercice  # noqa: E402

# --- exercice 5 : regression logistique -----------------------------------
HEURES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]          # heures de revision
REUSSITE = [-1, -1, -1, 1, -1, 1, -1, 1, 1, 1]    # +1 = admis, -1 = recale
LOGISTIQUE = " + ".join(f"log(1 + exp({-r}*(w*{h} + b)))" for h, r in zip(HEURES, REUSSITE))


EXERCICE = Exercice(
    5, "Regression logistique",
    LOGISTIQUE,
    x0=[0, 0], variables="w b",
    enonce=f"""
On veut predire la reussite a un examen en fonction des heures de revision h :
   h      = {HEURES}
   y      = {REUSSITE}   (+1 admis, -1 recale)
Le modele est P(admis) = 1 / (1 + exp(-(w h + b))). On estime (w, b) en
minimisant la perte logistique  L(w, b) = somme log(1 + exp(-y_i (w h_i + b))).
a) Calculer grad L et montrer que L est convexe.
b) Minimiser L depuis (0, 0). A partir de combien d'heures P(admis) > 0.5 ?""",
    solution=[0.676706, -3.721882], f_min=4.335111,
    options={"pas_fixe": {"alpha": 0.01, "max_iter": 100000}},
    remarque="seuil P > 0.5 pour h > -b/w ~ 5.5 h ; reference calculee avec une precision 1e-12",
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
