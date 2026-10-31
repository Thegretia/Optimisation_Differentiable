# Outils communs aux solveurs

import math
from dataclasses import dataclass, field


@dataclass
class Resultat:
    """Sortie d'un solveur.

    u              : dernier itéré enregistré
    iterations     : nombre d'itérations effectuées
    statut         : "convergence", "max_iter" ou "divergence"
    historique_f   : valeurs f(u_k), k = 0, ..., iterations
    historique_g2  : valeurs ||grad f(u_k)||^2, k = 0, ..., iterations
    infos          : compteurs propres à chaque méthode (produits matrice-vecteur,
                     mises à jour BFGS ignorées, etc.)
    """
    u: list
    iterations: int = 0
    statut: str = "max_iter"
    historique_f: list = field(default_factory=list)
    historique_g2: list = field(default_factory=list)
    infos: dict = field(default_factory=dict)

    @property
    def converge(self):
        return self.statut == "convergence"


class Compteur:
    """Enveloppe une fonction et compte le nombre de ses appels."""

    def __init__(self, fonction):
        self.fonction = fonction
        self.appels = 0

    def __call__(self, *args):
        self.appels += 1
        return self.fonction(*args)


def valeur(f, u):
    """Évalue f(u) en renvoyant +inf si l'évaluation provoque un débordement."""
    try:
        return f(u)
    except OverflowError:
        return math.inf
