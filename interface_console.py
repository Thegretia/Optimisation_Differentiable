"""Interface console et saisie sécurisée d'expressions mathématiques."""
import ast
import math
import operator

from algorithme_newton import newton
from algorithme_quasi_newton import quasi_newton
from algorithme_pas_conjugue import pas_conjugue
from algorithme_pas_fixe import pas_fixe


FUNCTIONS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log,
    "log10": math.log10, "abs": abs,
}
CONSTANTS = {"pi": math.pi, "e": math.e}
BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}
UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class EvaluationError(ValueError):
    """Erreur d'expression enrichie avec l'opération et le point évalué."""


def _eval_node(node, variables):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, variables)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        raise ValueError(f"Symbole inconnu : {node.id}")
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARYOPS:
        return UNARYOPS[type(node.op)](_eval_node(node.operand, variables))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS and not node.keywords:
        args = [_eval_node(arg, variables) for arg in node.args]
        try:
            return FUNCTIONS[node.func.id](*args)
        except (ValueError, OverflowError, ZeroDivisionError) as error:
            call = f"{node.func.id}({', '.join(map(str, args))})"
            if node.func.id == "sqrt":
                detail = "sqrt exige un argument positif ou nul dans les réels."
            elif node.func.id in ("log", "log10"):
                detail = f"{node.func.id} exige un argument strictement positif."
            elif node.func.id in ("asin", "acos"):
                detail = f"{node.func.id} exige un argument compris entre -1 et 1."
            elif isinstance(error, OverflowError):
                detail = "le résultat dépasse la plage numérique représentable."
            else:
                detail = str(error) or "les arguments ne sont pas valides pour cette fonction."
            raise EvaluationError(f"{call} ne peut pas être calculé : {detail}") from error
    if isinstance(node, ast.BinOp) and type(node.op) in BINOPS:
        left, right = _eval_node(node.left, variables), _eval_node(node.right, variables)
        try:
            return BINOPS[type(node.op)](left, right)
        except ZeroDivisionError as error:
            raise EvaluationError(f"Division par zéro dans {ast.unparse(node)}.") from error
        except OverflowError as error:
            raise EvaluationError(f"Calcul trop grand dans {ast.unparse(node)} : dépassement numérique.") from error
        except ValueError as error:
            raise EvaluationError(
                f"Opération invalide dans {ast.unparse(node)} : {error}. "
                "Vérifiez notamment les puissances fractionnaires appliquées à un nombre négatif."
            ) from error
    raise ValueError("Expression non autorisée. Consultez la syntaxe présentée dans le menu.")


def compile_expression(expression):
    tree = ast.parse(expression, mode="eval")
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id.startswith("x"):
            if node.id == "x":
                names.add(1)
            elif node.id[1:].isdigit() and int(node.id[1:]) >= 1:
                names.add(int(node.id[1:]))
    if not names:
        raise ValueError("La fonction doit utiliser au moins une variable : x ou x1, x2, ...")
    uses_x = any(isinstance(n, ast.Name) and n.id == "x" for n in ast.walk(tree))
    if uses_x and any(isinstance(n, ast.Name) and n.id != "x" and n.id.startswith("x")
                      for n in ast.walk(tree)):
        raise ValueError("Ne mélangez pas x avec les variables vectorielles x1, x2, ...")
    if uses_x:
        dimension, scalar_variable = 1, True
    else:
        dimension, scalar_variable = max(names), False
        if names != set(range(1, dimension + 1)):
            raise ValueError("Les variables vectorielles doivent être consécutives : x1, x2, ...")

    def function(point):
        if scalar_variable:
            values = {"x": point}
        else:
            if not isinstance(point, (list, tuple)) or len(point) != dimension:
                raise ValueError(f"Cette expression attend {dimension} valeur(s) initiale(s).")
            values = {f"x{i + 1}": point[i] for i in range(dimension)}
        try:
            result = float(_eval_node(tree, values))
        except EvaluationError as error:
            point = ", ".join(f"{key}={value:.8g}" for key, value in values.items())
            raise EvaluationError(
                f"{error}\n  Expression : {expression}\n  Point testé : ({point})\n"
                "  Cela peut arriver pendant le calcul des dérivées ou l'essai d'un pas."
            ) from error
        except (ValueError, TypeError, OverflowError, ZeroDivisionError) as error:
            point = ", ".join(f"{key}={value:.8g}" for key, value in values.items())
            raise EvaluationError(
                f"Échec de l'évaluation de l'expression au point ({point}) : {error}."
            ) from error
        if not math.isfinite(result):
            point = ", ".join(f"{key}={value:.8g}" for key, value in values.items())
            raise EvaluationError(
                f"L'expression produit {result} au point ({point}). Vérifiez le domaine, "
                "les divisions par zéro et les puissances trop grandes."
            )
        return result

    return function, dimension, scalar_variable


def lire_point(dimension, scalar_variable):
    while True:
        raw = input("Point initial x0 (ex. 2 ou 1, -1) : ").strip()
        try:
            values = [float(part.strip()) for part in raw.split(",")]
            if len(values) != dimension:
                raise ValueError(f"Il faut fournir exactement {dimension} valeur(s).")
            return values[0] if scalar_variable else values
        except ValueError as error:
            print(f"Saisie invalide : {error}")


EXAMPLES = {
    "1": ("x**2 + 4*cos(x)", -1.0),
    "2": ("(x1 - 1)**2 + 2*(x2 + 2)**2", [4.0, 3.0]),
}
ALGORITHMS = {
    "1": ("Newton", newton), "2": ("Quasi-Newton (BFGS)", quasi_newton),
    "3": ("Descente à pas fixe", pas_fixe), "4": ("Gradient à pas conjugué", pas_conjugue),
}


def main():
    print("\n=== Optimisation différentiable ===")
    while True:
        print("\nChoisissez un algorithme :")
        for key, (name, _) in ALGORITHMS.items():
            print(f"  {key}. {name}")
        print("  0. Quitter")
        choice = input("Votre choix : ").strip()
        if choice == "0":
            print("Au revoir.")
            return
        if choice not in ALGORITHMS:
            print("Choix invalide.")
            continue
        name, algorithm = ALGORITHMS[choice]
        print(f"\n{name} — choisissez une fonction :\n  1. Choisir un exemple fourni\n  2. Saisir ma fonction")
        mode = input("Votre choix : ").strip()
        current_stage = "la préparation de la fonction"
        expression = "(non définie)"
        x0 = "(non défini)"
        try:
            if mode == "1":
                print("\nExemples :")
                for example_key, (example_expression, example_x0) in EXAMPLES.items():
                    print(f"  {example_key}. f = {example_expression}, x0 = {example_x0}")
                example_choice = input("Votre choix : ").strip()
                if example_choice not in EXAMPLES:
                    print("Choix invalide.")
                    continue
                expression, x0 = EXAMPLES[example_choice]
                function, _, _ = compile_expression(expression)
            elif mode == "2":
                print("Syntaxe : variables x (1D) ou x1, x2, ... (plusieurs dimensions).")
                print("Opérateurs : + - * / ** % et parenthèses. Fonctions : sin, cos, tan,")
                print("asin, acos, atan, sinh, cosh, tanh, sqrt, exp, log, log10, abs.")
                print("Constantes : pi, e. Exemple : x1**2 + sin(x2) + sqrt(x1**2 + 1)")
                expression = input("f = ").strip()
                function, dimension, scalar_variable = compile_expression(expression)
                x0 = lire_point(dimension, scalar_variable)
            else:
                print("Choix invalide.")
                continue
            if choice == "3":
                alpha = float(input("Pas fixe alpha (ex. 0.01) : "))
                if not math.isfinite(alpha) or alpha <= 0:
                    raise ValueError("Le pas alpha doit être un nombre strictement positif et fini.")
                current_stage = "l'optimisation"
                result = algorithm(function, x0, alpha=alpha)
            else:
                current_stage = "l'optimisation"
                result = algorithm(function, x0)
            print(f"\nFonction : {expression}")
            print(f"Point initial : {x0}")
            print(f"Résultat : x* = {result['x']}")
            print(f"f(x*) = {result['f_x']:.10g}")
            print(f"Itérations : {result['iterations']} — convergence : {'oui' if result['converged'] else 'non'}")
            if not result["converged"]:
                print("\nLe calcul s'est terminé sans atteindre la tolérance demandée.")
                print("Causes possibles : nombre maximal d'itérations atteint, pas mal adapté, "
                      "fonction mal conditionnée ou absence de minimum local accessible.")
                if choice == "3":
                    print("Pour le pas fixe, essayez une valeur de alpha plus petite.")
                print("Essayez un autre point initial ou une tolérance moins stricte.")
        except SyntaxError as error:
            print("\nERREUR DE SYNTAXE DANS LA FONCTION")
            print(f"Python n'a pas pu analyser l'expression : {error.msg} (position {error.offset}).")
            print("Vérifiez les parenthèses, les virgules et la puissance écrite avec ** (pas ^).")
            print("Exemple valide : x1**2 + sin(x2)")
        except EvaluationError as error:
            print("\nERREUR DE CALCUL DE LA FONCTION")
            print(error)
            print("Pistes : choisissez un point initial dans le domaine, puis vérifiez que "
                  "la fonction reste définie autour des points explorés.")
            print("Les dérivées numériques testent de petits déplacements autour du point courant; "
                  "la recherche linéaire peut aussi essayer plusieurs pas.")
        except ZeroDivisionError as error:
            print("\nERREUR : division par zéro pendant " + current_stage + ".")
            print(f"Détail technique : {error or 'un dénominateur est nul'}.")
            print("Vérifiez les dénominateurs de votre fonction et évitez leurs zéros dans la zone explorée.")
        except OverflowError as error:
            print("\nERREUR : dépassement numérique pendant " + current_stage + ".")
            print(f"Détail technique : {error or 'une valeur est trop grande'}.")
            print("Réduisez l'échelle des variables ou choisissez un point initial plus proche de la zone utile.")
        except (ValueError, TypeError) as error:
            print("\nERREUR DE SAISIE OU DE PARAMÉTRAGE")
            print(f"{error}")
            print("Vérifiez la syntaxe de la fonction, le nombre de coordonnées du point initial "
                  "et les valeurs numériques demandées.")
        except Exception as error:
            print("\nERREUR INATTENDUE PENDANT " + current_stage.upper())
            print(f"Type d'erreur : {type(error).__name__}")
            print(f"Détail technique : {error or 'aucun détail fourni'}")
            print("Vérifiez que l'expression respecte la syntaxe documentée. Si le problème revient, "
                  "notez l'algorithme, la fonction et le point initial pour pouvoir le reproduire.")


if __name__ == "__main__":
    main()
