import numpy as np
import sympy as sp

SEUIL_NON_BORNEE = -1e12


class Modele:

    def __init__(self, f, dimension, variables=None):
        self.symbolique = isinstance(f, sp.Basic)
        if self.symbolique:
            vs = tuple(variables) if variables is not None else \
                tuple(sorted(f.free_symbols, key=lambda s: s.name))
            if len(vs) != dimension:
                raise ValueError(f"L'expression sympy a {len(vs)} variable(s) "
                                 f"mais x0 en a {dimension}.")
            self._f = sp.lambdify(vs, f, modules="numpy")
            self._g = sp.lambdify(vs, sp.Matrix([sp.diff(f, v) for v in vs]), modules="numpy")
            self._h = sp.lambdify(vs, sp.hessian(f, vs), modules="numpy")
            self.fonction = lambda X: self._f(*X)
        else:
            if not callable(f):
                raise TypeError("f doit être une fonction Python ou une expression sympy.")
            self.fonction = f

    def valeur_initiale(self, x):
        try:
            with np.errstate(all="ignore"):
                return float(self.fonction(x)), None
        except Exception as erreur:
            return np.nan, (f"Impossible d'evaluer f(x0): {erreur}. "
                            "Verifiez la dimension de x0 et la definition de f.")

    def valeur(self, x):
        try:
            with np.errstate(all="ignore"):
                return float(self.fonction(x))
        except (ArithmeticError, ValueError):
            return np.nan

    def gradient(self, x):
        if self.symbolique:
            with np.errstate(all="ignore"):
                g = np.asarray(self._g(*x), dtype=float).ravel()
            return g if np.all(np.isfinite(g)) else None
        g = np.empty_like(x)
        base = np.cbrt(np.finfo(float).eps)
        for i in range(x.size):
            h = base * max(1.0, abs(x[i]))
            xp, xm = x.copy(), x.copy()
            xp[i] += h
            xm[i] -= h
            fp, fm = self.valeur(xp), self.valeur(xm)
            if not (np.isfinite(fp) and np.isfinite(fm)):
                return None
            g[i] = (fp - fm) / (2*h)
        return g if np.all(np.isfinite(g)) else None

    def hessienne(self, x):
        if self.symbolique:
            with np.errstate(all="ignore"):
                H = np.asarray(self._h(*x), dtype=float)
        else:
            n = x.size
            H = np.empty((n, n))
            base = np.cbrt(np.finfo(float).eps)
            for i in range(n):
                h = base * max(1.0, abs(x[i]))
                xp, xm = x.copy(), x.copy()
                xp[i] += h
                xm[i] -= h
                gp, gm = self.gradient(xp), self.gradient(xm)
                if gp is None or gm is None:
                    return None
                H[:, i] = (gp - gm) / (2*h)
            H = 0.5 * (H + H.T)
        return H if np.all(np.isfinite(H)) else None


def valider(x0, tol, max_iter):
    x = np.array(x0, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("x0 doit etre un vecteur non vide de valeurs finies.")
    if tol <= 0 or max_iter < 0:
        raise ValueError("tol doit etre positif et max_iter positif ou nul.")
    return x


def nature_point(modele, x):
    H = modele.hessienne(x)
    if H is None:
        return ""
    vp = np.linalg.eigvalsh(H)
    eps = 1e-6 * max(1.0, float(np.max(np.abs(vp))))
    if np.all(vp > eps):
        nat = "minimum local (Hessienne definie positive)"
    elif np.any(vp < -eps) and np.any(vp > eps):
        nat = "ATTENTION : point selle (valeurs propres de signes opposes)"
    elif np.all(vp < -eps):
        nat = "ATTENTION : maximum local"
    else:
        nat = "point critique degenere (valeur propre nulle) : etude d'ordre superieur requise"
    return f" Nature du point : {nat}."


def trace(verbose, k, valeur, ng, pas=None):
    if not verbose:
        return
    if k == 0:
        print(f"{'k':>5} {'f(x)':>16} {'||grad||':>12} {'pas':>10}")
    p = "-" if pas is None else f"{pas:.4g}"
    print(f"{k:>5} {valeur:>16.8e} {ng:>12.3e} {p:>10}")


def recherche_armijo(m, x, valeur, gradient, direction, alpha0, c1, beta, max_essais=60):
    pente = float(gradient @ direction)
    if not pente < 0:
        return None
    a = alpha0
    for _ in range(max_essais):
        with np.errstate(all="ignore"):
            xs = x + a * direction
        if np.all(np.isfinite(xs)):
            v = m.valeur(xs)
            if np.isfinite(v) and v <= valeur + c1 * a * pente:
                g = m.gradient(xs)
                if g is not None:
                    return a, xs, v, g
        a *= beta
    return None


def recherche_wolfe(m, x, valeur, gradient, direction, alpha_init, c1, c2,
                    alpha_max=1e10, max_eval=60):
    dphi0 = float(gradient @ direction)
    if not dphi0 < 0:
        return None

    def evaluer(a):
        with np.errstate(all="ignore"):
            xa = x + a * direction
        if not np.all(np.isfinite(xa)):
            return None
        va = m.valeur(xa)
        if not np.isfinite(va):
            return None
        ga = m.gradient(xa)
        if ga is None:
            return None
        return xa, va, ga, float(ga @ direction)

    def zoom(lo, phi_lo, hi):
        meilleur = None
        for _ in range(max_eval):
            a = 0.5 * (lo + hi)
            r = evaluer(a)
            if r is None or r[1] > valeur + c1 * a * dphi0 or r[1] >= phi_lo:
                hi = a
            else:
                xa, va, ga, da = r
                meilleur = (a, xa, va, ga)
                if abs(da) <= -c2 * dphi0:
                    return meilleur
                if da * (hi - lo) >= 0:
                    hi = lo
                lo, phi_lo = a, va
            if abs(hi - lo) < 1e-16 * max(1.0, abs(lo)):
                break
        return meilleur

    a_prec, phi_prec = 0.0, valeur
    a = alpha_init
    for i in range(max_eval):
        r = evaluer(a)
        if r is None:
            return zoom(a_prec, phi_prec, a)
        xa, va, ga, da = r
        if va < SEUIL_NON_BORNEE:
            return a, xa, va, ga
        if va > valeur + c1 * a * dphi0 or (i > 0 and va >= phi_prec):
            return zoom(a_prec, phi_prec, a)
        if abs(da) <= -c2 * dphi0:
            return a, xa, va, ga
        if da >= 0:
            return zoom(a, va, a_prec)
        a_prec, phi_prec = a, va
        if a >= alpha_max:
            break
        a = min(2.0 * a, alpha_max)
    return None


def afficher_resultat(titre, resultat):
    x_opt, f_opt, n_iter, converge, message = resultat
    print(f"\n--- {titre} ---")
    print("Point obtenu :", x_opt)
    print("Valeur de la fonction :", f_opt)
    print("Nombre d'iterations :", n_iter)
    print("Convergence :", converge)
    print("Message :", message)


def pas_fixe(f, x0, alpha=0.05, tol=1e-5, max_iter=500, seuil_divergence=1e12,
             variables=None, verbose=False, historique=None):
    x = valider(x0, tol, max_iter)
    if alpha <= 0:
        raise ValueError("alpha doit etre strictement positif.")
    m = Modele(f, x.size, variables)

    valeur, erreur = m.valeur_initiale(x)
    if erreur:
        return x, np.nan, 0, False, erreur
    if not np.isfinite(valeur):
        return x, valeur, 0, False, "La fonction n'est pas finie au point initial."
    gradient = m.gradient(x)
    if gradient is None:
        return x, valeur, 0, False, "Le gradient n'a pas pu etre evalue."

    def conseil():
        H = m.hessienne(x)
        if H is None:
            return ""
        lam_max = float(np.linalg.eigvalsh(H)[-1])
        if lam_max > 0 and alpha >= 2.0 / lam_max:
            return (f" alpha = {alpha:g} >= 2/lambda_max = {2.0/lam_max:.4g} : "
                    "le pas est trop grand, reduisez alpha.")
        return ""

    hausses = 0
    fenetre = []
    for iteration in range(max_iter):
        ng = float(np.linalg.norm(gradient))
        fenetre = (fenetre + [(ng, valeur)])[-50:]
        trace(verbose, iteration, valeur, ng, alpha)
        if historique is not None:
            historique.append((iteration, valeur, ng, alpha))
        if ng <= tol:
            return x, valeur, iteration, True, "Convergence atteinte." + nature_point(m, x)

        with np.errstate(all="ignore"):
            deplacement = alpha * gradient
            x_suivant = x - deplacement
        if not np.all(np.isfinite(x_suivant)):
            return x, valeur, iteration, False, (
                "Les iterates sont devenus non finis; arret de l'algorithme." + conseil())
        valeur_suivante = m.valeur(x_suivant)
        if not np.isfinite(valeur_suivante):
            return x, valeur, iteration, False, (
                "La fonction est devenue non finie; arret de l'algorithme." + conseil())
        if valeur_suivante < SEUIL_NON_BORNEE:
            return x_suivant, valeur_suivante, iteration + 1, False, (
                "f chute sans fin (alpha trop grand, ou f non bornee inferieurement); "
                "arret." + conseil())
        if valeur_suivante > seuil_divergence:
            return x_suivant, valeur_suivante, iteration + 1, False, (
                "Divergence : f depasse le seuil de securite." + conseil())
        if np.linalg.norm(deplacement) <= 1e-15 * (1.0 + np.linalg.norm(x)):
            return x, valeur, iteration, False, "Stagnation numerique avant d'atteindre tol."
        gradient_suivant = m.gradient(x_suivant)
        if gradient_suivant is None:
            return x_suivant, valeur_suivante, iteration + 1, False, (
                "Gradient non fini; arret de l'algorithme.")
        if valeur_suivante > valeur:
            hausses += 1
        x, valeur, gradient = x_suivant, valeur_suivante, gradient_suivant

    ng = float(np.linalg.norm(gradient))
    trace(verbose, max_iter, valeur, ng, alpha)
    if ng <= tol:
        return x, valeur, max_iter, True, "Convergence atteinte." + nature_point(m, x)
    indice = ""
    if len(fenetre) >= 20 and fenetre[-1][0] > 0.9 * fenetre[0][0] \
            and fenetre[-1][1] < fenetre[0][1] - 1e-9:
        indice = (" Le gradient ne diminue plus alors que f decroit encore : "
                  "f est peut-etre non bornee inferieurement (pas de minimum).")
    if not indice:
        indice = conseil()
    if not indice and hausses > 0.1 * max_iter:
        indice = " f oscille souvent : alpha est probablement trop grand."
    if not indice:
        indice = " Convergence lente : augmentez max_iter ou alpha (sans depasser 2/lambda_max)."
    return x, valeur, max_iter, False, (
        f"Nombre maximal d'iterations atteint sans convergence (||grad|| = {ng:.2e})." + indice)


def main():
    def f(X):
        x, y, z = X
        return 6*x**2 + 6*x*y + 3*y**2 + y**3 - (1/3)*z**3 + (1/4)*z**4

    x0 = np.array([-0.6637, 0.1758, 1.5778])

    afficher_resultat("Pas fixe - alpha = 0.05 (500 iterations max)",
                      pas_fixe(f, x0, alpha=0.05, tol=1e-5, max_iter=500))

    xs, ys, zs = sp.symbols("x y z", real=True)
    expr = 6*xs**2 + 6*xs*ys + 3*ys**2 + ys**3 - sp.Rational(1, 3)*zs**3 + sp.Rational(1, 4)*zs**4
    afficher_resultat("Pas fixe - derivees exactes (sympy)",
                      pas_fixe(expr, x0, alpha=0.05, tol=1e-5, max_iter=500,
                               variables=(xs, ys, zs)))

    print("\nBalayage de alpha :")
    print(f"{'alpha':>8} {'iterations':>11} {'converge':>9}  message")
    for a in (0.01, 0.05, 0.1, 0.12, 0.15, 0.2):
        _, _, n, ok, msg = pas_fixe(f, x0, alpha=a, tol=1e-5, max_iter=5000)
        print(f"{a:>8.2f} {n:>11} {str(ok):>9}  {msg[:70]}")

    # Cas du notebook d'origine : f(x,y,z,w) = ... + w est NON bornee inferieurement
    # (derivee en w constante egale a 1) : aucun minimum n'existe, l'algorithme doit
    # le dire clairement au lieu de s'epuiser en silence.
    def f_non_bornee(X):
        x, y, z, w = X
        return 6*x**2 + 6*x*y + 3*y**2 + y**3 - (1/3)*z**3 + (1/4)*z**4 + w

    afficher_resultat("Pas fixe - f(x,y,z,w) avec terme +w (pas de minimum)",
                      pas_fixe(f_non_bornee, np.append(x0, 0.5), alpha=0.05,
                               tol=1e-5, max_iter=500))


if __name__ == "__main__":
    main()
