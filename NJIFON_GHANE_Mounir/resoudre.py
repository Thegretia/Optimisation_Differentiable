"""
Resolution des exercices (ou de n'importe quelle fonction) en ligne de commande.

  python resoudre.py                              menu interactif
  python resoudre.py --liste                      liste des exercices
  python resoudre.py 3                            exercice 3, toutes les methodes
  python resoudre.py 3 -m newton -v               exercice 3 par Newton, avec les iterations
  python resoudre.py 3 --x0 -1 1                  exercice 3 depuis un autre point
  python resoudre.py -f "x**2 + 3*y**2 - x*y" --x0 1 1
  python resoudre.py -f "x**2 + 3*y**2" --x0 1 1 -m pas_fixe --alpha 0.1

Methodes : pas_fixe, pas_optimal, gradient_conjugue, newton, newton_pur, bfgs, dfp
"""
import argparse
import sys

from minimiser import minimiser
from outils.probleme import Probleme
from outils.resultat import afficher_comparaison
from outils.exercice import TOUTES
from exercices import EXERCICES, exercice

METHODES = ["pas_fixe", "pas_optimal", "gradient_conjugue", "newton", "newton_pur", "bfgs", "dfp"]


def lister():
    print("Exercices disponibles :")
    for e in EXERCICES:
        print(f"  {e.numero:>2}  {e.titre}")


def lancer(exo, f, x0, methode, options, verbose):
    """exo : un Exercice ou None (fonction libre f)."""
    if methode in (None, "toutes"):
        if exo is not None:
            res = exo.comparer(x0=x0, **options)
        else:
            res = [minimiser(f, x0, m, **options) for m in TOUTES]
            afficher_comparaison(res)
        meilleur = min((r for r in res if r.converge), key=lambda r: r.iterations, default=None)
        if meilleur is not None:
            print(f"\nNature du point trouve : {meilleur.nature}")
            if exo is not None and exo.solution is not None:
                print("Resultat conforme a la solution attendue :",
                      "oui" if exo.est_correct(meilleur) else "NON")
        return res

    if exo is not None:
        r = exo.resoudre(methode, x0=x0, **options)
    else:
        r = minimiser(f, x0, methode, **options)
    if verbose:
        r.afficher_iterations()
        print()
    print(r)
    if exo is not None and exo.solution is not None:
        print("Conforme a la solution attendue :", "oui" if exo.est_correct(r) else "NON")
    return r


def lire_vecteur(texte):
    return [float(v) for v in texte.replace(",", " ").replace(";", " ").split()]


def menu():
    while True:
        print()
        lister()
        print("   f  saisir sa propre fonction")
        print("   q  quitter")
        choix = input("\nVotre choix : ").strip().lower()
        if choix in ("q", ""):
            return
        try:
            if choix == "f":
                texte = input("f = ").strip()
                pb = Probleme(texte)
                print(f"variables (dans cet ordre) : {pb.variables}")
                x0 = lire_vecteur(input(f"x0 ({len(pb.variables)} valeurs) = "))
                exo, f = None, texte
            else:
                exo, f = exercice(int(choix)), None
                print()
                exo.afficher_enonce()
                t = input(f"\nx0 (Entree = {tuple(exo.x0)}) : ").strip()
                x0 = lire_vecteur(t) if t else None
            print("Methodes :", ", ".join(METHODES), "(Entree = toutes)")
            m = input("Methode : ").strip() or None
            options = {}
            if m == "pas_fixe":
                a = input("alpha (Entree = valeur conseillee) : ").strip()
                if a:
                    options["alpha"] = float(a)
            v = m is not None and input("Afficher les iterations ? (o/n) ").strip().lower() == "o"
            print()
            lancer(exo, f, x0, m, options, v)
        except (ValueError, KeyError) as err:
            print("Erreur :", err)
        input("\n(Entree pour continuer)")


def main():
    p = argparse.ArgumentParser(description="Minimisation sans contrainte - exercices")
    p.add_argument("numero", nargs="?", type=int, help="numero de l'exercice")
    p.add_argument("-f", "--fonction", help='fonction a minimiser, ex. "x**2 + 3*y**2"')
    p.add_argument("--x0", nargs="+", type=float, help="point de depart")
    p.add_argument("-m", "--methode", choices=METHODES + ["toutes"], help="methode (defaut : toutes)")
    p.add_argument("--tol", type=float)
    p.add_argument("--max-iter", type=int)
    p.add_argument("--alpha", type=float, help="pas pour le gradient a pas fixe")
    p.add_argument("--variables", help='ordre des variables, ex. "x y z"')
    p.add_argument("-v", "--verbose", action="store_true", help="afficher les iterations")
    p.add_argument("--liste", action="store_true", help="lister les exercices")
    p.add_argument("--enonce", action="store_true", help="afficher seulement l'enonce")
    args = p.parse_args()

    if args.liste:
        lister()
        return
    if args.numero is None and args.fonction is None:
        menu()
        return

    options = {}
    if args.tol is not None:
        options["tol"] = args.tol
    if args.max_iter is not None:
        options["max_iter"] = args.max_iter
    if args.alpha is not None:
        if args.methode not in ("pas_fixe", None, "toutes"):
            p.error("--alpha ne sert qu'au gradient a pas fixe")
        if args.methode == "pas_fixe":
            options["alpha"] = args.alpha

    try:
        if args.fonction is not None:
            if args.x0 is None:
                p.error("donner le point de depart avec --x0")
            f = args.fonction
            if args.variables:
                f = Probleme(f, variables=args.variables)
            print(Probleme(args.fonction, variables=args.variables), "\n")
            if args.alpha is not None and args.methode in (None, "toutes"):
                r = minimiser(f, args.x0, "pas_fixe", alpha=args.alpha, **options)
                print(r)
                return
            lancer(None, f, args.x0, args.methode, options, args.verbose)
        else:
            exo = exercice(args.numero)
            exo.afficher_enonce()
            if args.enonce:
                return
            print()
            lancer(exo, None, args.x0, args.methode, options, args.verbose)
    except (ValueError, KeyError) as err:
        print("Erreur :", err)
        sys.exit(1)


if __name__ == "__main__":
    main()
