#Comparaison des quatre methodes sur les problemes de test.
from algebre import sub, norme2
from outils import Compteur
from problemes import QUADRATIQUE, EXPONENTIELLE, quadratique_diagonale
from gradient_pas_fixe import gradient_pas_fixe
from gradient_pas_variable import gradient_pas_variable
from gradient_conjugue import newton_gradient_conjugue
from quasi_newton_bfgs import bfgs

EPS = 1e-12
ALPHA_EXPONENTIELLE = 0.01   # pas choisi empiriquement (hessienne non bornée a priori)


def pas_fixe_pour(P):
    """Pas optimal 2/(mu + L) si les constantes de la quadratique sont connues."""
    return 2.0 / (P.mu + P.L) if P.mu is not None else ALPHA_EXPONENTIELLE


def lancer_methodes(P, eps=EPS):
    """Exécute les quatre méthodes sur P. Renvoie une liste de
    (nom, Resultat, compteur_f, compteur_grad, compteur_hess)."""
    sorties = []
    methodes = [
        ("Gradient à pas fixe",
         lambda f, g, h: gradient_pas_fixe(f, g, P.u0, alpha=pas_fixe_pour(P), eps=eps)),
        ("Gradient à pas variable (BLS)",
         lambda f, g, h: gradient_pas_variable(f, g, P.u0, eps=eps)),
        ("Newton et gradient conjugué",
         lambda f, g, h: newton_gradient_conjugue(f, g, h, P.u0, eps=eps)),
        ("Quasi-Newton BFGS",
         lambda f, g, h: bfgs(f, g, P.u0, eps=eps)),
    ]
    for nom, lancer in methodes:
        cf, cg, ch = Compteur(P.f), Compteur(P.grad), Compteur(P.hess)
        sorties.append((nom, lancer(cf, cg, ch), cf, cg, ch))
    return sorties


def afficher_tableau(P, sorties):
    print(f"\n{P.nom}, point de départ de dimension {len(P.u0)}, "
          f"critère d'arrêt ||grad f||^2 <= {EPS:g}")
    print(f"{'Méthode':32}{'iter':>6}{'eval f':>8}{'eval grad':>11}{'eval hess':>11}"
          f"{'prod. MV':>10}{'erreur ||u-u*||':>18}  statut")
    for nom, r, cf, cg, ch in sorties:
        erreur = norme2(sub(r.u, P.solution))
        mv = r.infos.get("produits_mv", 0)
        print(f"{nom:32}{r.iterations:6d}{cf.appels:8d}{cg.appels:11d}{ch.appels:11d}"
              f"{mv:10d}{erreur:18.3e}  {r.statut}")


def sensibilite_pas_fixe():
    """Effet du pas sur la quadratique 2D : convergence si alpha < 2/L, divergence sinon."""
    P = QUADRATIQUE
    seuil = 2.0 / P.L
    print(f"\nSensibilité au pas fixe sur {P.nom} "
          f"(mu = {P.mu:.4f}, L = {P.L:.4f}, seuil 2/L = {seuil:.4f}, "
          f"pas optimal 2/(mu+L) = {2 / (P.mu + P.L):.4f})")
    print(f"{'alpha':>8}{'iterations':>12}  statut")
    for alpha in (0.05, 0.2, 2 / (P.mu + P.L), 0.43, 0.44, 0.6):
        r = gradient_pas_fixe(P.f, P.grad, P.u0, alpha=alpha, eps=EPS)
        print(f"{alpha:8.4f}{r.iterations:12d}  {r.statut}")


if __name__ == "__main__":
    for P in (QUADRATIQUE, EXPONENTIELLE, quadratique_diagonale(30, 100.0)):
        afficher_tableau(P, lancer_methodes(P))
    sensibilite_pas_fixe()
