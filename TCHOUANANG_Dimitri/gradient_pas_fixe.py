import numpy as np
import sympy as sp


def gradient_pas_fixe(expression, x0, alpha, tol=1e-6, max_iter=100000):
    f = sp.sympify(expression)
    variables = sorted(f.free_symbols, key=lambda v: v.name)
    f_num = sp.lambdify(variables, f, "numpy")
    grad = sp.lambdify(variables, [sp.diff(f, v) for v in variables], "numpy")

    x = np.array(x0, dtype=float)
    for k in range(max_iter):
        g = np.array(grad(*x), dtype=float)
        if np.linalg.norm(g) < tol:
            break
        x = x - alpha * g
    return x, f_num(*x), k


if __name__ == "__main__":
    expression = input("f = ")
    x0 = [float(v) for v in input("x0 = ").split()]
    alpha = float(input("alpha = "))
    x, fx, k = gradient_pas_fixe(expression, x0, alpha)
    print("x* =", x)
    print("f(x*) =", fx)
    print("iterations =", k)
