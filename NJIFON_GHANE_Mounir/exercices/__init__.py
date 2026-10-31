"""
Serie d'exercices de NJIFON GHANE MOUNIR (SDIA 4, ENSPD) : un fichier par exercice.

Pour ajouter un exercice : creer exercice_8_xxx.py sur le meme modele
(une variable EXERCICE) et l'ajouter dans la liste ci-dessous.
"""
from . import (exercice_0_cours, exercice_1_quadratique_zigzag, exercice_2_booth, exercice_3_beale, exercice_4_entrepot_weber, exercice_5_regression_logistique, exercice_6_points_critiques, exercice_7_wood)

EXERCICES = [m.EXERCICE for m in (exercice_0_cours, exercice_1_quadratique_zigzag, exercice_2_booth, exercice_3_beale, exercice_4_entrepot_weber, exercice_5_regression_logistique, exercice_6_points_critiques, exercice_7_wood)]


def exercice(numero):
    for e in EXERCICES:
        if e.numero == numero:
            return e
    raise KeyError(f"pas d'exercice numero {numero}")
