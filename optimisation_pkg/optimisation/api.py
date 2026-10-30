"""Point d'entrée simple : on donne une expression SymPy, le package fait le reste."""
from .outils import preparer
from .descente import descente_pas_fixe, descente_pas_variable
from .gradient_conjugue import gradient_conjugue
from .newton import newton, quasi_newton

METHODES = ("pas_fixe", "pas_variable", "gradient_conjugue", "newton", "quasi_newton")


def minimiser(expression, variables, x0, methode="quasi_newton", **options):
    """Minimise une expression SymPy avec la méthode choisie.

    methode : "pas_fixe", "pas_variable", "gradient_conjugue",
              "newton" ou "quasi_newton"
    options : transmises à la méthode (tol, max_iter, pas, amorti, formule...)
    """
    f, grad, hess = preparer(expression, variables)

    if methode == "pas_fixe":
        return descente_pas_fixe(f, grad, x0, **options)
    if methode == "pas_variable":
        return descente_pas_variable(f, grad, x0, **options)
    if methode == "gradient_conjugue":
        return gradient_conjugue(f, grad, x0, **options)
    if methode == "newton":
        return newton(f, grad, hess, x0, **options)
    if methode == "quasi_newton":
        return quasi_newton(f, grad, x0, **options)
    raise ValueError(f"méthode inconnue : {methode!r}. Choix : {METHODES}")
