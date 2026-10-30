"""
Un exercice = une fonction, un point de depart, l'enonce et la solution
attendue (quand on la connait). Sert a resoudre.py et verifier.py.
"""
import numpy as np

from .algebre import norme
from .resultat import afficher_comparaison
from .probleme import Probleme

TOUTES = ["pas_fixe", "pas_optimal", "gradient_conjugue", "newton", "bfgs"]


class Exercice:

    def __init__(self, numero, titre, f, x0, enonce, solution=None, f_min=None,
                 variables=None, options=None, tol=1e-6, precision=1e-3, limites=None, remarque=""):
        self.numero = numero
        self.titre = titre
        self.f = f
        self.x0 = list(x0)
        self.enonce = enonce.strip()
        self.solution = None if solution is None else np.array(solution, dtype=float)
        self.f_min = f_min
        self.variables = variables
        self.options = options or {}       # options propres a chaque methode
        self.tol = tol
        self.precision = precision      # ecart tolere avec la solution
        self.remarque = remarque.strip()
        # methodes dont on sait qu'elles echouent ou sont trop lentes ici (et pourquoi)
        self.limites = limites or {}

    def probleme(self):
        return Probleme(self.f, variables=self.variables, nom=self.titre)

    def resoudre(self, methode="bfgs", x0=None, **options):
        from minimiser import minimiser      # import ici pour eviter un import circulaire
        opts = {"tol": self.tol}
        opts.update(self.options.get(methode, {}))
        opts.update(options)
        return minimiser(self.probleme(), self.x0 if x0 is None else x0, methode=methode, **opts)

    def comparer(self, methodes=None, x0=None, afficher=True, **options):
        res = [self.resoudre(m, x0=x0, **options) for m in (methodes or TOUTES)]
        if afficher:
            afficher_comparaison(res)
        return res

    def est_correct(self, resultat, precision=None):
        precision = precision or self.precision
        if not resultat.converge:
            return False
        if self.solution is not None:
            # plusieurs minima possibles : solution peut etre une liste de points
            sols = self.solution if self.solution.ndim == 2 else [self.solution]
            return any(norme(resultat.x - s) <= precision * (1 + norme(s)) for s in sols)
        if self.f_min is not None:
            return abs(resultat.f - self.f_min) <= precision * (1 + abs(self.f_min))
        return True

    def afficher_enonce(self):
        print(f"Exercice {self.numero} - {self.titre}")
        print("-" * 70)
        print(self.enonce)
        pb = self.probleme()
        print(f"\n{pb}")
        print(f"Point de depart x0 = {tuple(self.x0)}")
        if self.solution is not None:
            print(f"Solution attendue : {np.round(self.solution, 6).tolist()}"
                  + ("" if self.f_min is None else f"   f* = {self.f_min:.6g}"))
        if self.remarque:
            print(f"Remarque : {self.remarque}")
        for m, raison in self.limites.items():
            print(f"Limite connue ({m}) : {raison}")
