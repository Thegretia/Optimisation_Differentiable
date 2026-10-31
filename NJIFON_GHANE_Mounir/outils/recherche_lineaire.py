"""
Recherches lineaires : on cherche un pas a > 0 le long d'une direction de
descente d, c'est-a-dire on regarde phi(a) = f(x + a d).

Chaque fonction renvoie (a, x_nouveau, f_nouveau, g_nouveau) ou None si
aucun pas acceptable n'a ete trouve.
"""
import math

from .algebre import scalaire

NOMBRE_OR = (math.sqrt(5.0) - 1.0) / 2.0     # 0.618...


def armijo(pb, x, d, fx, gx, a0=1.0, c1=1e-4, rho=0.5, max_essais=60):
    """Rebroussement : on divise le pas par 1/rho jusqu'a avoir
    f(x + a d) <= f(x) + c1 a <g, d>."""
    pente = scalaire(gx, d)
    if pente >= 0:
        return None
    a = a0
    for _ in range(max_essais):
        xn = x + a * d
        fn = pb.f(xn)
        if math.isfinite(fn) and fn <= fx + c1 * a * pente:
            gn = pb.grad(xn)
            if gn is not None:
                return a, xn, fn, gn
        a *= rho
    return None


def wolfe(pb, x, d, fx, gx, a0=1.0, c1=1e-4, c2=0.9, a_max=1e8, max_eval=50):
    """Pas verifiant les conditions de Wolfe fortes (Nocedal & Wright,
    algorithmes 3.5 et 3.6). La phase 'zoom' utilise une interpolation
    quadratique protegee, et la dichotomie en secours."""
    dphi0 = scalaire(gx, d)
    if dphi0 >= 0:
        return None

    def evaluer(a):
        xa = x + a * d
        fa = pb.f(xa)
        if not math.isfinite(fa):
            return None
        ga = pb.grad(xa)
        if ga is None:
            return None
        return a, xa, fa, ga, scalaire(ga, d)

    def zoom(a_lo, f_lo, d_lo, a_hi, f_hi):
        meilleur = None
        for _ in range(max_eval):
            ecart = a_hi - a_lo
            a = 0.5 * (a_lo + a_hi)
            if math.isfinite(f_hi):
                denom = 2.0 * (f_hi - f_lo - d_lo * ecart)
                if denom > 0:
                    a = a_lo - d_lo * ecart * ecart / denom
            bas, haut = min(a_lo, a_hi), max(a_lo, a_hi)
            marge = 0.1 * (haut - bas)
            if not (bas + marge <= a <= haut - marge):
                a = 0.5 * (a_lo + a_hi)

            r = evaluer(a)
            if r is None or r[2] > fx + c1 * a * dphi0 or r[2] >= f_lo:
                a_hi, f_hi = a, (r[2] if r else math.inf)
            else:
                meilleur = r
                if abs(r[4]) <= -c2 * dphi0:
                    return r
                if r[4] * (a_hi - a_lo) >= 0:
                    a_hi, f_hi = a_lo, f_lo
                a_lo, f_lo, d_lo = a, r[2], r[4]
            if abs(a_hi - a_lo) < 1e-14 * max(1.0, abs(a_lo)):
                break
        return meilleur

    a_prec, f_prec, d_prec = 0.0, fx, dphi0
    a = a0
    for i in range(max_eval):
        r = evaluer(a)
        if r is None:
            res = zoom(a_prec, f_prec, d_prec, a, math.inf)
            break
        _, _, fa, _, da = r
        if fa > fx + c1 * a * dphi0 or (i > 0 and fa >= f_prec):
            res = zoom(a_prec, f_prec, d_prec, a, fa)
            break
        if abs(da) <= -c2 * dphi0:
            res = r
            break
        if da >= 0:
            res = zoom(a, fa, da, a_prec, f_prec)
            break
        if fa < -1e12 or a >= a_max:      # f semble descendre sans fin
            res = r
            break
        a_prec, f_prec, d_prec = a, fa, da
        a = min(2.0 * a, a_max)
    else:
        res = None

    if res is None:
        return None
    return res[:4]


def pas_optimal(pb, x, d, fx, a0=1.0, tol=1e-10, a_max=1e10):
    """Recherche lineaire 'exacte' : minimise phi(a) = f(x + a d).
    1) encadrement du minimum, 2) section doree."""

    def phi(a):
        v = pb.f(x + a * d)
        return v if math.isfinite(v) else math.inf

    # 1) encadrement : on veut a < b < c avec phi(b) < phi(a) et phi(b) <= phi(c)
    b = a0
    fb = phi(b)
    while fb >= fx and b > 1e-16:
        b *= 0.5
        fb = phi(b)
    if fb >= fx:
        return None                     # d n'est pas une direction de descente utile
    a, c = 0.0, 2.0 * b
    fc = phi(c)
    while fc < fb:
        a, b, fb = b, c, fc
        c = 2.0 * c
        if c > a_max or fb < -1e12:     # f ne remonte jamais : pas de minimum dans cette direction
            xn = x + b * d
            gn = pb.grad(xn)
            return (b, xn, fb, gn) if gn is not None else None
        fc = phi(c)

    # 2) section doree sur [a, c]
    u = c - NOMBRE_OR * (c - a)
    v = a + NOMBRE_OR * (c - a)
    fu, fv = phi(u), phi(v)
    while (c - a) > tol * (1.0 + abs(u)):
        if fu < fv:
            c, v, fv = v, u, fu
            u = c - NOMBRE_OR * (c - a)
            fu = phi(u)
        else:
            a, u, fu = u, v, fv
            v = a + NOMBRE_OR * (c - a)
            fv = phi(v)

    pas, fpas = (u, fu) if fu < fv else (v, fv)
    if fpas > fb:                       # securite : on garde le meilleur point vu
        pas, fpas = b, fb
    xn = x + pas * d
    gn = pb.grad(xn)
    if gn is None:
        return None
    return pas, xn, fpas, gn
