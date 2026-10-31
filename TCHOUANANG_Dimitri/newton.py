import numpy as np
import sympy as sp


def newton(expression, x0, tol=1e-6, max_iter=100):
    f = sp.sympify(expression)
    variables = sorted(f.free_symbols, key=lambda v: v.name)
    f_num = sp.lambdify(variables, f, "numpy")
    grad = sp.lambdify(variables, [sp.diff(f, v) for v in variables], "numpy")
    hess = sp.lambdify(variables, sp.hessian(f, variables), "numpy")

    x = np.array(x0, dtype=float)
    for k in range(max_iter):
        g = np.array(grad(*x), dtype=float)
        if np.linalg.norm(g) < tol:
            break
        H = np.array(hess(*x), dtype=float)
        x = x + np.linalg.solve(H, -g)
    return x, f_num(*x), k


if __name__ == "__main__":
    expression = input("f = ")
    x0 = [float(v) for v in input("x0 = ").split()]
    x, fx, k = newton(expression, x0)
    print("x* =", x)
    print("f(x*) =", fx)
    print("iterations =", k)
