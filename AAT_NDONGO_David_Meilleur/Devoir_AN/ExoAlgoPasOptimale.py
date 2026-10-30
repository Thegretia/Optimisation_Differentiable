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



_a = sp.Symbol("a", real=True)
_x0, _y0, _z0, _gx, _gy, _gz = sp.symbols("x0 y0 z0 gx gy gz", real=True)
_phi = F_SYM.subs({x: _x0 - _a*_gx, y: _y0 - _a*_gy, z: _z0 - _a*_gz},
                  simultaneous=True)
_dphi = sp.expand(sp.diff(_phi, _a))
_coeffs_dphi = sp.lambdify((_x0, _y0, _z0, _gx, _gy, _gz),
                           sp.Poly(_dphi, _a).all_coeffs(), modules="numpy")


def pas_optimal(X, g):
    c = np.array(_coeffs_dphi(*X, *g), dtype=float)
    if not np.any(c):
        return None
    dphi = np.poly1d(c)
    d2phi = dphi.deriv()
    racines = np.roots(c)
    reelles = sorted(r.real for r in racines
                     if abs(r.imag) <= 1e-9 * (1.0 + abs(r.real)) and r.real > 1e-14)
    for r in reelles:
        # une itération de Newton pour polir la racine
        if abs(d2phi(r)) > 1e-14:
            r = r - dphi(r) / d2phi(r)
        if r > 1e-14 and d2phi(r) > 0:
            return float(r)
    return None


def pas_armijo(X, g, c1=1e-4):
    a, fX, pente = 1.0, f(X), -float(g @ g)
    while f(X - a*g) > fX + c1*a*pente and a > 1e-14:
        a *= 0.5
    return a


def descente_pas_optimal(X0, tol=1e-5, max_iter=500):
    X = np.array(X0, dtype=float)
    g = grad_f(X)
    historique = [(0, f(X), float(np.linalg.norm(g)), np.nan, np.nan, "-")]
    converge = False

    for k in range(1, max_iter + 1):
        if np.linalg.norm(g) < tol:
            converge = True
            break
        alpha = pas_optimal(X, g)
        methode = "exact"
        if alpha is None:
            alpha, methode = pas_armijo(X, g), "armijo"
        X = X - alpha * g
        g_nouveau = grad_f(X)
        historique.append((k, f(X), float(np.linalg.norm(g_nouveau)), alpha,
                           float(g_nouveau @ g), methode))
        g = g_nouveau
    else:
        converge = np.linalg.norm(g) < tol

    return {"X": X, "f": f(X), "iterations": len(historique) - 1,
            "converge": bool(converge), "historique": historique}


def afficher_historique(res):
    print(f"{'k':>4} {'f(X)':>14} {'||grad f||':>13} {'alpha':>10} "
          f"{'<g_k+1,g_k>':>13} {'pas':>8}")
    for k, fk, ng, a, ps, m in res["historique"]:
        a_s = "-" if np.isnan(a) else f"{a:.6f}"
        p_s = "-" if np.isnan(ps) else f"{ps:.2e}"
        print(f"{k:>4} {fk:>14.8f} {ng:>13.3e} {a_s:>10} {p_s:>13} {m:>8}")


def main():
    X0 = np.array([-0.6637, 0.1758, 1.5778])
    tol = 1e-5

    print("f(x,y,z) =", F_SYM)
    print("Gradient symbolique :", list(GRAD_SYM))
    print("phi'(a) = d/da f(X - a g) =", sp.collect(_dphi, _a))
    print(f"Écart gradient symbolique / différences finies en X0 : "
          f"{verifier_gradient(X0):.2e}")

    print("\nPoints critiques (résolution symbolique de grad f = 0) :")
    for P, vp, nat in points_critiques():
        print(f"  X = {np.round(P, 6)}   f = {f(P):+.6f}   -> {nat}")

    print(f"DESCENTE À PAS OPTIMAL (exact)  -  X0 = {X0}  -  tol = {tol:g}")
    print(f"Itération 0 : f(X0) = {f(X0):.8f}\n")
    res = descente_pas_optimal(X0, tol=tol)
    afficher_historique(res)

    X_star = res["X"]
    vp = np.linalg.eigvalsh(hessian_f(X_star))
    print("\n Résultat final ")
    print("Convergence atteinte :", "oui" if res["converge"] else "NON")
    print(f"Itérations : {res['iterations']}")
    print(f"X* : {X_star}")
    print(f"f(X*) : {f(X_star):.10f}")
    print(f"||grad f(X*)|| : {np.linalg.norm(grad_f(X_star)):.3e}")
    print(f"Valeurs propres H(X*): {np.round(vp, 6)}  -> {nature_point(vp)}")
    f_vals = [h[1] for h in res["historique"]]
    print("f strictement décroissante :", all(b <= a + 1e-15 for a, b in zip(f_vals, f_vals[1:])))


if __name__ == "__main__":
    main()
