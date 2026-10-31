#Courbes de convergence : f(u_k) - f* en fonction de k, échelle logarithmique.
import csv
import os

from problemes import QUADRATIQUE, EXPONENTIELLE, quadratique_diagonale
from exemples import lancer_methodes

DOSSIER = "resultats"
PLANCHER = 1e-16     # en dessous, f(u_k) - f* est dominé par les erreurs d'arrondi


def nom_fichier(P):
    brut = P.nom.lower().replace(" ", "_").replace("'", "").replace("=", "").replace(",", "")
    return brut


def exporter_csv(P, sorties):
    os.makedirs(DOSSIER, exist_ok=True)
    chemin = os.path.join(DOSSIER, f"convergence_{nom_fichier(P)}.csv")
    f_etoile = P.f_etoile
    with open(chemin, "w", newline="", encoding="utf-8") as fichier:
        w = csv.writer(fichier)
        w.writerow(["methode", "iteration", "f_moins_f_etoile", "norme_gradient_carre"])
        for nom, r, *_ in sorties:
            for k, (fk, g2) in enumerate(zip(r.historique_f, r.historique_g2)):
                w.writerow([nom, k, f"{fk - f_etoile:.6e}", f"{g2:.6e}"])
    return chemin


def tracer(P, sorties):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    f_etoile = P.f_etoile
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    styles = ["-", "--", "-.", ":"]
    for (nom, r, *_), style in zip(sorties, styles):
        ecarts = [max(fk - f_etoile, PLANCHER) for fk in r.historique_f]
        ax.semilogy(range(len(ecarts)), ecarts, style, linewidth=1.8, label=nom)
    ax.set_xscale("symlog", linthresh=10)   # échelle linéaire jusqu'à 10, puis logarithmique
    ax.set_xlabel("Itération k (échelle symlog)")
    ax.set_ylabel("f(u_k) - f*")
    ax.set_title(P.nom)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    chemin = os.path.join(DOSSIER, f"convergence_{nom_fichier(P)}.png")
    fig.savefig(chemin, dpi=150)
    plt.close(fig)
    return chemin


if __name__ == "__main__":
    for P in (QUADRATIQUE, EXPONENTIELLE, quadratique_diagonale(30, 100.0)):
        sorties = lancer_methodes(P)
        print(exporter_csv(P, sorties))
        figure = tracer(P, sorties)
        if figure:
            print(figure)
