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



def newton(X0, tol=1e-5, max_iter=50, securise=True, c1=1e-4):
    X = np.array(X0, dtype=float)
    g = grad_f(X)
    n = X.size
    historique = [(0, f(X), float(np.linalg.norm(g)), np.nan, 0.0, np.nan)]
    converge = False

    for k in range(1, max_iter + 1):
        norme_g = np.linalg.norm(g)
        if norme_g < tol:
            converge = True
            break

        H = hessian_f(X)
        mu = 0.0
        if securise:
            lam_min = np.linalg.eigvalsh(H)[0]
            if lam_min <= 1e-8:
                mu = 1e-2 - lam_min

        try:
            d = np.linalg.solve(H + mu*np.eye(n), -g)
        except np.linalg.LinAlgError:
            print("Hessienne singulière : arrêt.")
            break

        t = 1.0
        if securise:
            fX, pente = f(X), float(g @ d)
            while f(X + t*d) > fX + c1*t*pente and t > 1e-12:
                t *= 0.5

        X = X + t*d
        g = grad_f(X)
        ng = float(np.linalg.norm(g))
        ratio = ng / norme_g**2 if norme_g > 0 else np.nan
        historique.append((k, f(X), ng, t, mu, ratio))
    else:
        converge = np.linalg.norm(g) < tol

    return {"X": X, "f": f(X), "iterations": len(historique) - 1,
            "converge": bool(converge), "historique": historique}


def afficher_historique(res):
    print(f"{'k':>3} {'f(X)':>14} {'||grad f||':>13} {'pas t':>8} "
          f"{'mu':>8} {'||g_k||/||g_k-1||^2':>20}")
    for k, fk, ng, t, mu, r in res["historique"]:
        t_s = "-" if np.isnan(t) else f"{t:.4f}"
        r_s = "-" if np.isnan(r) else f"{r:.4f}"
        print(f"{k:>3} {fk:>14.8f} {ng:>13.3e} {t_s:>8} {mu:>8.3f} {r_s:>20}")


def main():
    X0 = np.array([-0.6637, 0.1758, 1.5778])
    tol = 1e-5

    print("=" * 74)
    print("f(x,y,z) =", F_SYM)
    print("Gradient symbolique :", list(GRAD_SYM))
    print("Hessienne symbolique :", HESS_SYM.tolist())
    print(f"Écart gradient symbolique / différences finies en X0 : "
          f"{verifier_gradient(X0):.2e}")

    print("\nPoints critiques (résolution symbolique de grad f = 0) :")
    for P, vp, nat in points_critiques():
        print(f"  X = {np.round(P, 6)}   f = {f(P):+.6f}   "
              f"valeurs propres H = {np.round(vp, 4)}   -> {nat}")

    print("\n" + "=" * 74)
    print(f"NEWTON SÉCURISÉ  |  X0 = {X0}  |  tol = {tol:g}")
    print("=" * 74)
    print(f"Itération 0 : f(X0) = {f(X0):.8f}\n")
    res = newton(X0, tol=tol, securise=True)
    afficher_historique(res)

    X_star = res["X"]
    vp = np.linalg.eigvalsh(hessian_f(X_star))
    print("\n--- Résultat final ---")
    print("Convergence atteinte :", "oui" if res["converge"] else "NON")
    print(f"Itérations           : {res['iterations']}")
    print(f"X*                   : {X_star}")
    print(f"f(X*)                : {f(X_star):.10f}")
    print(f"||grad f(X*)||       : {np.linalg.norm(grad_f(X_star)):.3e}")
    print(f"Valeurs propres H(X*): {np.round(vp, 6)}  -> {nature_point(vp)}")

    print("\n--- Comparaison Newton pur / Newton sécurisé ---")
    for nom, flag in (("Newton pur", False), ("Newton sécurisé", True)):
        r = newton(X0, tol=tol, securise=flag)
        vp_r = np.linalg.eigvalsh(hessian_f(r["X"]))
        print(f"{nom:<16}: {r['iterations']:>2} itérations | X* = {np.round(r['X'], 6)} "
              f"| f = {r['f']:.6f} | {nature_point(vp_r)} | converge = {r['converge']}")


if __name__ == "__main__":
    main()
