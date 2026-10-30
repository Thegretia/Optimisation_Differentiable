"""Package d'optimisation sans contrainte : Newton, quasi-Newton, descentes de gradient."""
from .outils import Resultat, preparer, armijo, pas_optimal
from .descente import descente_pas_fixe, descente_pas_variable
from .gradient_conjugue import gradient_conjugue
from .newton import newton, quasi_newton
from .api import minimiser, METHODES

__all__ = [
    "Resultat", "preparer", "armijo", "pas_optimal",
    "descente_pas_fixe", "descente_pas_variable", "gradient_conjugue",
    "newton", "quasi_newton", "minimiser", "METHODES",
]
__version__ = "1.0.0"
