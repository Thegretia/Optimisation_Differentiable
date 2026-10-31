"""
Exercice 2 - Fonction de Booth
Serie de NJIFON GHANE MOUNIR

    python exercices/exercice_2_booth.py                 toutes les methodes
    python exercices/exercice_2_booth.py newton          une seule methode, avec les iterations
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from outils.exercice import Exercice  # noqa: E402


EXERCICE = Exercice(
    2, "Fonction de Booth",
    "(x + 2*y - 7)**2 + (2*x + y - 5)**2",
    x0=[0, 0],
    enonce="""
f(x, y) = (x + 2y - 7)^2 + (2x + y - 5)^2.
a) Montrer que f >= 0 et trouver le point ou f = 0 en resolvant un systeme 2x2.
b) Calculer la hessienne. Combien d'iterations faut-il a Newton ? au gradient
   conjugue avec pas exact ? Verifier numeriquement.""",
    solution=[1, 3], f_min=0.0,
    options={"pas_fixe": {"alpha": 0.05}},
    remarque="H = [[10, 8], [8, 10]], valeurs propres 2 et 18",
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
