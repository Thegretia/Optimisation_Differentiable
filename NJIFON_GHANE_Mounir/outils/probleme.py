"""
Le probleme a minimiser : f, son gradient et sa hessienne.

On accepte trois facons de donner f :
  1) une chaine de caracteres : "x**2 + 3*y**2 - x*y"   (le plus simple)
  2) une expression sympy
  3) une fonction python f(X) ou X est un vecteur ; on peut fournir en plus
     grad(X) et hess(X), sinon ils sont calcules par differences finies.

Avec 1) et 2), si sympy est installe, les derivees sont exactes.
"""
import math
import re

import numpy as np

try:
    import sympy as sp
except ImportError:          # sympy n'est pas obligatoire
    sp = None


FONCTIONS_CONNUES = {
    "exp": np.exp, "log": np.log, "ln": np.log, "sqrt": np.sqrt,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "atan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh, "abs": np.abs,
    "pi": math.pi, "e": math.e, "E": math.e,
}


def _cle_naturelle(nom):
    # x2 avant x10 (le tri alphabetique simple mettait x10 avant x2)
    m = re.match(r"^(.*?)(\d+)$", nom)
    if m:
        return (m.group(1), int(m.group(2)))
    return (nom, -1)


def ordonner_variables(noms):
    return sorted(noms, key=_cle_naturelle)


class Probleme:

    def __init__(self, f, grad=None, hess=None, variables=None, nom=None):
        self.nom = nom
        self.n = None
        self.variables = None
        self.expression = None
        self.exact = False
        self._f = self._g = self._h = None
        self.nf = self.ng = self.nh = 0

        if isinstance(variables, str):
            variables = variables.replace(",", " ").split()

        if isinstance(f, str) or (sp is not None and isinstance(f, sp.Basic)):
            if sp is not None:
                self._depuis_sympy(f, variables)
            else:
                self._depuis_chaine_sans_sympy(f, variables)
        elif callable(f):
            self._f = f
            self._g = grad
            self._h = hess
            self.exact = grad is not None
            if variables is not None:
                self.variables = list(variables)
                self.n = len(self.variables)
        else:
            raise TypeError("f doit etre une chaine, une expression sympy ou une fonction python")

    # ------------------------------------------------------------------
    def _depuis_sympy(self, f, variables):
        expr = sp.sympify(f, locals={"ln": sp.log, "e": sp.E}) if isinstance(f, str) else f
        if variables is None:
            symboles = ordonner_variables([s.name for s in expr.free_symbols])
            # on reprend les symboles de l'expression (et pas Symbol(nom)) pour
            # garder leurs hypotheses (real=True, ...)
            par_nom = {s.name: s for s in expr.free_symbols}
            vs = [par_nom[nm] for nm in symboles]
        else:
            par_nom = {s.name: s for s in expr.free_symbols}
            vs = [par_nom.get(str(v), sp.Symbol(str(v))) if not isinstance(v, sp.Symbol) else v
                  for v in variables]
            inconnues = set(par_nom) - {v.name for v in vs}
            if inconnues:
                raise ValueError(f"variables non declarees dans l'expression : {sorted(inconnues)}")
        if not vs:
            raise ValueError("l'expression ne contient aucune variable")

        self.expression = expr
        self.variables = [v.name for v in vs]
        self.n = len(vs)
        self.exact = True
        grad = [sp.diff(expr, v) for v in vs]
        hess = [[sp.diff(gi, v) for v in vs] for gi in grad]
        self.grad_symbolique = grad
        f_num = sp.lambdify(vs, expr, modules="numpy")
        g_num = sp.lambdify(vs, grad, modules="numpy")
        h_num = sp.lambdify(vs, hess, modules="numpy")
        self._f = lambda X: f_num(*X)
        self._g = lambda X: g_num(*X)
        self._h = lambda X: h_num(*X)

    def _depuis_chaine_sans_sympy(self, texte, variables):
        texte = texte.replace("^", "**")
        noms = set(re.findall(r"[A-Za-z_]\w*", texte)) - set(FONCTIONS_CONNUES)
        if variables is None:
            variables = ordonner_variables(noms)
        self.variables = list(variables)
        self.n = len(self.variables)
        self.expression = texte
        code = compile(texte, "<f>", "eval")
        noms_vars = self.variables

        def f(X):
            env = dict(FONCTIONS_CONNUES)
            env.update(zip(noms_vars, X))
            return eval(code, {"__builtins__": {}}, env)

        self._f = f

    # ------------------------------------------------------------------
    def verifier_point(self, x0):
        x = np.array(x0, dtype=float).ravel()
        if x.size == 0 or not np.all(np.isfinite(x)):
            raise ValueError("x0 doit etre un vecteur non vide de nombres finis")
        if self.n is not None and x.size != self.n:
            raise ValueError(f"x0 a {x.size} composante(s) mais f depend de {self.n} "
                             f"variable(s) : {self.variables}")
        try:
            v = float(self._f(x))
        except (TypeError, IndexError, ValueError) as err:
            raise ValueError(f"impossible d'evaluer f(x0) : {err}. Verifier la dimension de x0.")
        if not math.isfinite(v):
            raise ValueError("f(x0) n'est pas un nombre fini")
        if self.n is None:
            self.n = x.size
            self.variables = [f"x{i + 1}" for i in range(x.size)]
        return x

    def reinitialiser_compteurs(self):
        self.nf = self.ng = self.nh = 0

    # ------------------------------------------------------------------
    def f(self, x):
        self.nf += 1
        try:
            with np.errstate(all="ignore"):
                v = float(self._f(x))
        except (ArithmeticError, ValueError):
            return math.nan
        return v

    def grad(self, x):
        self.ng += 1
        if self._g is not None:
            try:
                with np.errstate(all="ignore"):
                    g = np.array(self._g(x), dtype=float).ravel()
            except (ArithmeticError, ValueError):
                return None
            return g if np.all(np.isfinite(g)) else None
        # differences finies centrees
        n = x.size
        g = np.zeros(n)
        for i in range(n):
            h = 6.055e-6 * max(1.0, abs(x[i]))      # ~ eps^(1/3)
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            fp, fm = self.f(xp), self.f(xm)
            if not (math.isfinite(fp) and math.isfinite(fm)):
                return None
            g[i] = (fp - fm) / (2.0 * h)
        return g

    def hess(self, x):
        self.nh += 1
        n = x.size
        if self._h is not None:
            try:
                with np.errstate(all="ignore"):
                    H = np.array(self._h(x), dtype=float).reshape(n, n)
            except (ArithmeticError, ValueError):
                return None
            return H if np.all(np.isfinite(H)) else None

        H = np.zeros((n, n))
        if self._g is not None:
            # derivee du gradient exact
            for j in range(n):
                h = 6.055e-6 * max(1.0, abs(x[j]))
                xp = x.copy(); xp[j] += h
                xm = x.copy(); xm[j] -= h
                gp, gm = self.grad(xp), self.grad(xm)
                if gp is None or gm is None:
                    return None
                H[:, j] = (gp - gm) / (2.0 * h)
        else:
            # differences finies d'ordre 2 directement sur f
            pas = [1.22e-4 * max(1.0, abs(xi)) for xi in x]     # ~ eps^(1/4)
            f0 = self.f(x)
            for i in range(n):
                hi = pas[i]
                xp = x.copy(); xp[i] += hi
                xm = x.copy(); xm[i] -= hi
                H[i, i] = (self.f(xp) - 2.0 * f0 + self.f(xm)) / (hi * hi)
                for j in range(i + 1, n):
                    hj = pas[j]
                    v = []
                    for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                        xx = x.copy(); xx[i] += si * hi; xx[j] += sj * hj
                        v.append(self.f(xx))
                    H[i, j] = H[j, i] = (v[0] - v[1] - v[2] + v[3]) / (4.0 * hi * hj)
        H = 0.5 * (H + H.T)
        return H if np.all(np.isfinite(H)) else None

    def __repr__(self):
        if self.expression is not None:
            return f"f({', '.join(self.variables)}) = {self.expression}"
        return self.nom or "f (fonction python)"
