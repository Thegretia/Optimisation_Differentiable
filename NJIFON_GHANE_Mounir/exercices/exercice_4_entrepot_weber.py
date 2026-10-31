"""
Exercice 4 - Localisation d'un entrepot (probleme de Weber)
Serie de NJIFON GHANE MOUNIR

    python exercices/exercice_4_entrepot_weber.py                 toutes les methodes
    python exercices/exercice_4_entrepot_weber.py newton          une seule methode, avec les iterations
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from outils.exercice import Exercice  # noqa: E402

# --- exercice 4 : localisation d'un entrepot ------------------------------
# (abscisse, ordonnee) en km et nombre de livraisons par jour
CLIENTS = [(0, 0, 3), (10, 0, 2), (10, 8, 4), (2, 9, 1)]
WEBER = " + ".join(f"{w}*sqrt((x - {a})**2 + (y - {b})**2)" for a, b, w in CLIENTS)


EXERCICE = Exercice(
    4, "Localisation d'un entrepot (probleme de Weber)",
    WEBER,
    x0=[5, 4],
    enonce=f"""
Une entreprise de livraison veut placer un entrepot au point (x, y) pour
servir quatre clients (coordonnees en km, nombre de livraisons par jour) :
   {CLIENTS}
Chaque jour, la distance parcourue est D(x, y) = somme w_i * ||(x, y) - (a_i, b_i)||.
a) D est-elle differentiable partout ? Ou ne l'est-elle pas ?
b) Minimiser D depuis (5, 4). Comparer avec le barycentre des clients
   (qui minimise la somme des distances AU CARRE, pas des distances).""",
    solution=[7.845724, 5.453632], f_min=60.571261,
    options={"pas_fixe": {"alpha": 0.1}},
    remarque="solution de reference calculee par l'algorithme de Weiszfeld",
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
