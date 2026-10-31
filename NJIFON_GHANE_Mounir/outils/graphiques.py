"""
Graphiques pour le notebook (matplotlib n'est utilise qu'ici, jamais dans
les algorithmes).
"""
import numpy as np


def tracer_trajectoires(pb, resultats, marge=0.25, cadre=None, niveaux=30, titre=None, ax=None):
    """Lignes de niveau de f (en dimension 2) et chemin suivi par chaque methode."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5.5))
    if cadre is None:
        pts = np.vstack([h[1] for r in resultats for h in r.historique])
        (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
        dx, dy = max(x1 - x0, 1e-3), max(y1 - y0, 1e-3)
        cadre = (x0 - marge * dx, x1 + marge * dx, y0 - marge * dy, y1 + marge * dy)
    xs = np.linspace(cadre[0], cadre[1], 160)
    ys = np.linspace(cadre[2], cadre[3], 160)
    Z = np.empty((ys.size, xs.size))
    compteurs = (pb.nf, pb.ng, pb.nh)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            Z[i, j] = pb.f(np.array([x, y]))
    pb.nf, pb.ng, pb.nh = compteurs

    finies = Z[np.isfinite(Z)]
    lv = np.unique(np.quantile(finies, np.linspace(0.0, 0.97, niveaux)))
    ax.contour(xs, ys, Z, levels=lv, cmap="viridis", linewidths=0.7)
    for r in resultats:
        chemin = np.array([h[1] for h in r.historique])
        ax.plot(chemin[:, 0], chemin[:, 1], ".-", ms=3, lw=1,
                label=f"{r.methode} ({r.iterations} it.)")
    depart = resultats[0].historique[0][1]
    ax.plot(depart[0], depart[1], "ks", ms=6)
    for r in resultats:
        ax.plot(r.x[0], r.x[1], "r*", ms=11)
    noms = pb.variables or ["x", "y"]
    ax.set_xlabel(noms[0])
    ax.set_ylabel(noms[1])
    ax.set_xlim(cadre[0], cadre[1])
    ax.set_ylim(cadre[2], cadre[3])
    ax.legend(fontsize=8)
    if titre:
        ax.set_title(titre)
    return ax


def tracer_convergence(resultats, titre=None, ax=None, max_iter=None):
    """||grad f(x_k)|| en fonction de k, echelle logarithmique."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for r in resultats:
        k = [h[0] for h in r.historique]
        g = [max(h[3], 1e-16) for h in r.historique]
        if max_iter is not None:
            k, g = k[:max_iter + 1], g[:max_iter + 1]
        ax.semilogy(k, g, label=r.methode)
    ax.set_xlabel("iteration k")
    ax.set_ylabel("||grad f(x_k)||")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    if titre:
        ax.set_title(titre)
    return ax
