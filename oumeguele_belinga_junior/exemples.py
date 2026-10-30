"""Compare les méthodes sur les deux problèmes de test."""
import math
from problemes import QUADRATIQUE, EXPONENTIELLE
from gradient_pas_fixe import gradient_pas_fixe
from gradient_pas_variable import gradient_pas_variable
from gradient_conjugue import newton_gradient_conjugue
from quasi_newton_bfgs import bfgs

exacts = {QUADRATIQUE.nom: [1/11, 7/11], EXPONENTIELLE.nom: [0.5*math.log(0.5), 0.0]}

for P, alpha in ((QUADRATIQUE, 0.2), (EXPONENTIELLE, 0.01)):
    print(f"\n=== {P.nom} | départ {P.u0} | solution exacte {exacts[P.nom]} ===")
    resultats = []
    u, h, k = gradient_pas_fixe(P.f, P.grad, P.u0, alpha=alpha);  resultats.append(("Gradient pas fixe", u, k))
    u, h, k = gradient_pas_variable(P.f, P.grad, P.u0);           resultats.append(("Gradient pas variable (BLS)", u, k))
    u, k = newton_gradient_conjugue(P.grad, P.hess, P.u0);        resultats.append(("Newton + grad. conjugué", u, k))
    u, h, k = bfgs(P.f, P.grad, P.u0);                            resultats.append(("Quasi-Newton BFGS", u, k))
    print(f"{'Méthode':32}{'u*':38}{'f(u*)':>16}{'iter':>8}")
    for nom, u, k in resultats:
        print(f"{nom:32}{str([round(x, 6) for x in u]):38}{P.f(u):16.10f}{k:8d}")
