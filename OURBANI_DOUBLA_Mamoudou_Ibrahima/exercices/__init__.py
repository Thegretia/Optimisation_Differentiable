"""
Serie d'exercices de OURBANI DOUBLA MAMOUDOU IBRAHIMA (SDIA 4, ENSPD) : un fichier par exercice.

Pour ajouter un exercice : creer exercice_8_xxx.py sur le meme modele
(une variable EXERCICE) et l'ajouter dans la liste ci-dessous.
"""
from . import (exercice_0_cours, exercice_1_quadratique_2d, exercice_2_rosenbrock, exercice_3_himmelblau, exercice_4_regression_lineaire, exercice_5_fonction_3_variables, exercice_6_profit, exercice_7_quadratique_dim10)

EXERCICES = [m.EXERCICE for m in (exercice_0_cours, exercice_1_quadratique_2d, exercice_2_rosenbrock, exercice_3_himmelblau, exercice_4_regression_lineaire, exercice_5_fonction_3_variables, exercice_6_profit, exercice_7_quadratique_dim10)]


def exercice(numero):
    for e in EXERCICES:
        if e.numero == numero:
            return e
    raise KeyError(f"pas d'exercice numero {numero}")
