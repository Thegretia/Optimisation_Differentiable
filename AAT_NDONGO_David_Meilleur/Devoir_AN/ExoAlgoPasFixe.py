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


def descente_pas_fixe(X0, alpha, tol=1e-5, max_iter=5000, seuil_divergence=1e8):
    X = np.array(X0, dtype=float)
    g = grad_f(X)
    historique = [(0, f(X), float(np.linalg.norm(g)))]
    statut = "max_iter"

    for k in range(1, max_iter + 1):
        if np.linalg.norm(g) < tol:
            statut = "convergence"
            break
        X = X - alpha * g
        if not np.all(np.isfinite(X)) or np.linalg.norm(X) > seuil_divergence:
            statut = "divergence"
            historique.append((k, np.inf, np.inf))
            break
        g = grad_f(X)
        historique.append((k, f(X), float(np.linalg.norm(g))))
    else:
        if np.linalg.norm(g) < tol:
            statut = "convergence"

    return {"X": X, "f": f(X) if statut != "divergence" else np.inf,
            "iterations": len(historique) - 1, "statut": statut,
            "historique": historique}


def taux_observe(historique, fenetre=20):
    h = [ng for _, _, ng in historique if np.isfinite(ng) and ng > 0]
    if len(h) <= fenetre:
        return np.nan
    return (h[-1] / h[-1 - fenetre]) ** (1.0 / fenetre)



def afficher_historique(res, pas_affichage=25):
    print(f"{'k':>5} {'f(X)':>14} {'||grad f||':>13}")
    h = res["historique"]
    for i, (k, fk, ng) in enumerate(h):
        if k == 0 or k % pas_affichage == 0 or i == len(h) - 1:
            print(f"{k:>5} {fk:>14.8f} {ng:>13.3e}")


def main():
    X0 = np.array([-0.6637, 0.1758, 1.5778])
    tol = 1e-5
    alpha = 0.05


    print("f(x,y,z) =", F_SYM)
    print("Gradient symbolique :", list(GRAD_SYM))
    print(f"Écart gradient symbolique / différences finies en X0 : "
          f"{verifier_gradient(X0):.2e}")

    print("\nPoints critiques (résolution symbolique de grad f = 0) :")
    for P, vp, nat in points_critiques():
        print(f"  X = {np.round(P, 6)}   f = {f(P):+.6f}   -> {nat}")

    print(f"DESCENTE À PAS FIXE  -  alpha = {alpha}  -  X0 = {X0}  -  tol = {tol:g}")
    res = descente_pas_fixe(X0, alpha, tol=tol)
    afficher_historique(res)

    X_star = res["X"]
    vp = np.linalg.eigvalsh(hessian_f(X_star))
    print("\n Résultat final")
    print("Statut :", res["statut"])
    print(f"Itérations : {res['iterations']}")
    print(f"X* : {X_star}")
    print(f"f(X*) : {f(X_star):.10f}")
    print(f"||grad f(X*)|| : {np.linalg.norm(grad_f(X_star)):.3e}")
    print(f"Valeurs propres H(X*): {np.round(vp, 6)}  -> {nature_point(vp)}")

    print("\n Analyse de stabilité au point trouvé ")
    alpha_crit = 2.0 / vp[-1]
    rho = np.max(np.abs(1.0 - alpha * vp))
    print(f"alpha critique 2/lambda_max = {alpha_crit:.4f}  "
          f"(alpha = {alpha} {'<' if alpha < alpha_crit else '>='} alpha critique)")
    print(f"Taux linéaire théorique max|1 - alpha*lambda_i| = {rho:.4f}")
    print(f"Taux observé sur les 20 dernières itérations     = "
          f"{taux_observe(res['historique']):.4f}")
    alpha_opt = 2.0 / (vp[0] + vp[-1])
    print(f"Pas optimal théorique (cas quadratique local) 2/(lambda_min+lambda_max) "
          f"= {alpha_opt:.4f}")

    print("\n Balayage de alpha ")
    print(f"{'alpha':>8} {'statut':>12} {'iter':>7} {'f(X*)':>14}")
    for a in (0.005, 0.01, 0.02, 0.05, 0.08, 0.12, 0.13, 0.15, 0.2):
        r = descente_pas_fixe(X0, a, tol=tol)
        print(f"{a:>8.3f} {r['statut']:>12} {r['iterations']:>7} {r['f']:>14.8f}")


if __name__ == "__main__":
    main()
