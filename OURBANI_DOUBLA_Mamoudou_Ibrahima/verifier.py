"""
Lance toutes les methodes sur tous les exercices et compare a la solution
attendue.   python verifier.py
"""
import time

from outils.exercice import TOUTES
from exercices import EXERCICES

COURT = {"pas_fixe": "pas fixe", "pas_optimal": "pas opt.", "gradient_conjugue": "grad. conj.",
         "newton": "Newton", "bfgs": "BFGS"}


def main():
    print("Nombre d'iterations   * = resultat faux ou non converge")
    print("                      ~ = echec attendu (limite de la methode, voir l'enonce)\n")
    print(f"{'exercice':<44}" + "".join(f"{COURT[m]:>12}" for m in TOUTES))
    echecs = 0
    debut = time.time()
    for e in EXERCICES:
        ligne = f"{e.numero:>2} {e.titre[:41]:<41}"
        for m in TOUTES:
            r = e.resoudre(m)
            ok = e.est_correct(r)
            if ok:
                marque = " "
            elif m in e.limites:
                marque = "~"
            else:
                marque = "*"
                echecs += 1
            ligne += f"{r.iterations:>11}{marque}"
        print(ligne)
    print(f"\n{echecs} echec(s) - {time.time() - debut:.1f} s")


if __name__ == "__main__":
    main()
