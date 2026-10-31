"""
Objet Resultat renvoye par tous les algorithmes, et petites fonctions
communes (preparation, affichage des iterations, messages d'arret).
"""
import math

from .algebre import norme, valeurs_propres
from .probleme import Probleme

SEUIL_NON_BORNEE = -1e12
SEUIL_DIVERGENCE = 1e15


class Resultat:

    def __init__(self, methode, pb, x, fx, gx, iterations, converge, message, historique):
        self.methode = methode
        self.probleme = pb
        self.x = x
        self.f = fx
        self.norme_grad = norme(gx) if gx is not None else math.nan
        self.iterations = iterations
        self.converge = converge
        self.message = message
        self.historique = historique       # liste de (k, x_k, f(x_k), ||g_k||, pas)
        self.evaluations = (pb.nf, pb.ng, pb.nh)
        self._nature = None

    @property
    def nature(self):
        if self._nature is None:
            self._nature = nature_point(self.probleme, self.x) if self.converge else "-"
        return self._nature

    def afficher_iterations(self, max_lignes=25):
        h = self.historique
        print(f"{'k':>5} {'f(x_k)':>18} {'||grad f||':>12} {'pas':>11}   x_k")
        if len(h) <= max_lignes:
            lignes = h
        else:
            debut = h[:max_lignes - 5]
            lignes = debut + [None] + h[-5:]
        for ligne in lignes:
            if ligne is None:
                print(f"{'...':>5}")
                continue
            k, xk, fk, ng, pas = ligne
            p = "-" if pas is None else f"{pas:.5g}"
            print(f"{k:>5} {fk:>18.10g} {ng:>12.3e} {p:>11}   {_vecteur(xk)}")

    def __str__(self):
        nf, ng, nh = self.evaluations
        lignes = [
            f"Methode            : {self.methode}",
            f"Converge           : {'oui' if self.converge else 'NON'}",
            f"Iterations         : {self.iterations}",
            f"x*                 : {_vecteur(self.x, 8)}",
            f"f(x*)              : {self.f:.10g}",
            f"||grad f(x*)||     : {self.norme_grad:.3e}",
            f"Evaluations f/g/H  : {nf} / {ng} / {nh}",
            f"Message            : {self.message}",
        ]
        if self.converge:
            lignes.append(f"Nature du point    : {self.nature}")
        return "\n".join(lignes)

    def __repr__(self):
        return (f"Resultat({self.methode}, x={_vecteur(self.x)}, f={self.f:.6g}, "
                f"iter={self.iterations}, converge={self.converge})")


def _vecteur(x, chiffres=6):
    return "(" + ", ".join(f"{v:.{chiffres}g}" for v in x) + ")"


def nature_point(pb, x):
    """Nature d'un point critique d'apres les valeurs propres de la hessienne."""
    compteurs = (pb.nf, pb.ng, pb.nh)
    H = pb.hess(x)
    pb.nf, pb.ng, pb.nh = compteurs
    if H is None:
        return "inconnue (hessienne non calculable)"
    vp = valeurs_propres(H)
    eps = 1e-6 * max(1.0, max(abs(v) for v in vp))
    pos = sum(v > eps for v in vp)
    neg = sum(v < -eps for v in vp)
    if pos == len(vp):
        return "minimum local strict (hessienne definie positive)"
    if neg == len(vp):
        return "ATTENTION : maximum local"
    if pos and neg:
        return "ATTENTION : point selle"
    return "degenere (valeur propre nulle) : il faut une etude d'ordre superieur"


def preparer(f, x0, grad=None, hess=None, variables=None):
    pb = f if isinstance(f, Probleme) else Probleme(f, grad=grad, hess=hess, variables=variables)
    x = pb.verifier_point(x0)
    pb.reinitialiser_compteurs()
    fx = pb.f(x)
    gx = pb.grad(x)
    if gx is None:
        raise ValueError("le gradient n'est pas calculable en x0")
    return pb, x, fx, gx


def afficher_ligne(verbose, k, x, fx, ng, pas):
    if not verbose:
        return
    if k == 0:
        print(f"{'k':>5} {'f(x_k)':>18} {'||grad f||':>12} {'pas':>11}")
    p = "-" if pas is None else f"{pas:.5g}"
    print(f"{k:>5} {fx:>18.10g} {ng:>12.3e} {p:>11}")


def noter(historique, verbose, k, x, fx, gx, pas):
    ng = norme(gx)
    historique.append((k, x.copy(), fx, ng, pas))
    afficher_ligne(verbose, k, x, fx, ng, pas)
    return ng


def lambda_max(pb, x):
    compteurs = (pb.nf, pb.ng, pb.nh)
    H = pb.hess(x)
    pb.nf, pb.ng, pb.nh = compteurs
    if H is None:
        return None
    return valeurs_propres(H)[-1]


def msg_max_iter(historique, gx):
    msg = f"nombre maximal d'iterations atteint (||grad|| = {norme(gx):.2e})."
    if len(historique) >= 40:
        q = len(historique) // 4
        _, _, f_a, g_a, _ = historique[-q]
        _, _, f_b, g_b, _ = historique[-1]
        if g_b > 0.5 * g_a and f_b < f_a - 1e-6 * (1 + abs(f_a)):
            msg += (" Le gradient ne diminue plus alors que f continue de baisser :"
                    " f n'a peut-etre pas de minimum.")
        else:
            msg += " Convergence lente : augmenter max_iter."
    return msg


MSG_PAS_DE_PAS = ("aucun pas ne fait suffisamment diminuer f : verifier que f admet "
                  "un minimum et changer de point initial")
MSG_NON_BORNEE = "f descend sans limite (f non bornee inferieurement ?) : pas de minimum"


def afficher_comparaison(resultats):
    print(f"{'methode':<38} {'iter':>6} {'f(x*)':>16} {'||grad||':>10} "
          f"{'eval f/g/H':>16} {'ok':>4}   x*")
    for r in resultats:
        nf, ng, nh = r.evaluations
        print(f"{r.methode:<38} {r.iterations:>6} {r.f:>16.10g} {r.norme_grad:>10.2e} "
              f"{f'{nf}/{ng}/{nh}':>16} {'oui' if r.converge else 'NON':>4}   {_vecteur(r.x)}")
