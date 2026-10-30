import numpy as np
import sympy as sp

x, y, z = sp.symbols("x y z", real=True)
VARIABLES = (x, y, z)
F_SYM = (6*x**2 + 6*x*y + 3*y**2 + y**3
         - sp.Rational(1, 3)*z**3 + sp.Rational(1, 4)*z**4)
GRAD_SYM = sp.Matrix([sp.diff(F_SYM, v) for v in VARIABLES])
HESS_SYM = sp.hessian(F_SYM, VARIABLES)

_f = sp.lambdify(VARIABLES, F_SYM, modules="numpy")
_grad = sp.lambdify(VARIABLES, GRAD_SYM, modules="numpy")
_hess = sp.lambdify(VARIABLES, HESS_SYM, modules="numpy")


def f(X):
    return float(_f(*X))


def grad_f(X):
    return np.asarray(_grad(*X), dtype=float).ravel()


def hessian_f(X):
    return np.asarray(_hess(*X), dtype=float)


def verifier_gradient(X, h=1e-6):
    X = np.asarray(X, dtype=float)
    approx = np.empty_like(X)
    for i in range(X.size):
        e = np.zeros_like(X)
        e[i] = h
        approx[i] = (f(X + e) - f(X - e)) / (2*h)
    return float(np.max(np.abs(approx - grad_f(X))))


def nature_point(valeurs_propres, eps=1e-9):
    pos = np.any(valeurs_propres > eps)
    neg = np.any(valeurs_propres < -eps)
    nul = np.any(np.abs(valeurs_propres) <= eps)
    if pos and neg:
        return "point selle" + (" (dégénéré)" if nul else "")
    if pos and not nul:
        return "minimum local strict"
    if neg and not nul:
        return "maximum local strict"
    return "dégénéré (Hessienne semi-définie : étude d'ordre supérieur requise)"


def points_critiques():
    solutions = sp.solve(list(GRAD_SYM), VARIABLES, dict=True)
    resultat = []
    for s in solutions:
        P = np.array([float(s[v]) for v in VARIABLES])
        vp = np.linalg.eigvalsh(hessian_f(P))
        resultat.append((P, vp, nature_point(vp)))
    return resultat


def bfgs(X0, tol=1e-5, max_iter=100, c1=1e-4):

    X = np.array(X0, dtype=float)
    n = X.size
    I = np.eye(n)
    W = I.copy()
    g = grad_f(X)
    historique = [(0, f(X), float(np.linalg.norm(g)), np.nan, "-", 1.0)]
    converge = False

    for k in range(1, max_iter + 1):
        if np.linalg.norm(g) < tol:
            converge = True
            break

        d = -W @ g
        if g @ d >= 0:
            W = I.copy()
            d = -g

        # recherche linéaire d'Armijo (premier pas réduit pour la norme du gradient)
        t = min(1.0, 1.0 / np.linalg.norm(g)) if k == 1 else 1.0
        fX, pente = f(X), float(g @ d)
        while f(X + t*d) > fX + c1*t*pente and t > 1e-14:
            t *= 0.5

        X_nouveau = X + t*d
        g_nouveau = grad_f(X_nouveau)
        s = X_nouveau - X
        yk = g_nouveau - g
        sy = float(s @ yk)

        if sy > 1e-10 * np.linalg.norm(s) * np.linalg.norm(yk):
            if k == 1:
                W = (sy / float(yk @ yk)) * I
            rho = 1.0 / sy
            W = (I - rho*np.outer(s, yk)) @ W @ (I - rho*np.outer(yk, s)) \
                + rho*np.outer(s, s)
            maj = "oui"
        else:
            maj = "non"

        X, g = X_nouveau, g_nouveau
        historique.append((k, f(X), float(np.linalg.norm(g)), t, maj,
                           float(np.linalg.cond(W))))
    else:
        converge = np.linalg.norm(g) < tol

    return {"X": X, "f": f(X), "iterations": len(historique) - 1,
            "converge": bool(converge), "W": W, "historique": historique}


def afficher_historique(res):
    print(f"{'k':>3} {'f(X)':>14} {'||grad f||':>13} {'pas':>9} {'maj W':>6} {'cond(W)':>10}")
    for k, fk, ng, t, maj, c in res["historique"]:
        t_s = "-" if np.isnan(t) else f"{t:.5f}"
        print(f"{k:>3} {fk:>14.8f} {ng:>13.3e} {t_s:>9} {maj:>6} {c:>10.3f}")


def main():
    X0 = np.array([-0.6637, 0.1758, 1.5778])
    tol = 1e-5

    print("f(x,y,z) =", F_SYM)
    print("Gradient symbolique :", list(GRAD_SYM))
    print(f"Écart gradient symbolique / différences finies en X0 : "
          f"{verifier_gradient(X0):.2e}")

    print("\nPoints critiques (résolution symbolique de grad f = 0) :")
    for P, vp, nat in points_critiques():
        print(f"  X = {np.round(P, 6)}   f = {f(P):+.6f}   -> {nat}")

    print(f"QUASI-NEWTON (BFGS)  -  X0 = {X0}  -  tol = {tol:g}")
    print(f"Itération 0 : f(X0) = {f(X0):.8f}\n")
    res = bfgs(X0, tol=tol)
    afficher_historique(res)

    X_star, W = res["X"], res["W"]
    H = hessian_f(X_star)
    vp = np.linalg.eigvalsh(H)
    print("\n Résultat final ")
    print("Convergence atteinte :", "oui" if res["converge"] else "NON")
    print(f"Itérations : {res['iterations']}")
    print(f"X* : {X_star}")
    print(f"f(X*) : {f(X_star):.10f}")
    print(f"||grad f(X*)|| : {np.linalg.norm(grad_f(X_star)):.3e}")
    print(f"Valeurs propres H(X*): {np.round(vp, 6)}  -> {nature_point(vp)}")
    print("\nW (approximation de l'inverse de la Hessienne) :")
    print(np.round(W, 4))
    print("Inverse exacte de H(X*) (contrôle) :")
    print(np.round(np.linalg.inv(H), 4))
    print(f"Écart relatif ||W - H^-1|| / ||H^-1|| = "
          f"{np.linalg.norm(W - np.linalg.inv(H)) / np.linalg.norm(np.linalg.inv(H)):.3f}")


if __name__ == "__main__":
    main()
