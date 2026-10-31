import numpy as np
import sympy as sp


def quasi_newton(expression, x0, tol=1e-6, max_iter=1000):
    f = sp.sympify(expression)
    variables = sorted(f.free_symbols, key=lambda v: v.name)
    f_num = sp.lambdify(variables, f, "numpy")
    grad = sp.lambdify(variables, [sp.diff(f, v) for v in variables], "numpy")

    x = np.array(x0, dtype=float)
    n = len(x)
    I = np.eye(n)
    H = np.eye(n)
    g = np.array(grad(*x), dtype=float)
    for k in range(max_iter):
        if np.linalg.norm(g) < tol:
            break
        d = -H @ g

        phi = lambda a: f_num(*(x + a * d))
        b = 1
        while phi(b) >= phi(0) and b > 1e-12:
            b = b / 2
        while phi(2 * b) < phi(b) and b < 1e6:
            b = 2 * b
        a, b = 0, 2 * b
        r = (np.sqrt(5) - 1) / 2
        for _ in range(60):
            c, e = b - r * (b - a), a + r * (b - a)
            if phi(c) < phi(e):
                b = e
            else:
                a = c
        alpha = (a + b) / 2

        x_new = x + alpha * d
        g_new = np.array(grad(*x_new), dtype=float)
        s = x_new - x
        y = g_new - g
        if y @ s > 1e-12:
            rho = 1 / (y @ s)
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        x, g = x_new, g_new
    return x, f_num(*x), k


if __name__ == "__main__":
    expression = input("f = ")
    x0 = [float(v) for v in input("x0 = ").split()]
    x, fx, k = quasi_newton(expression, x0)
    print("x* =", x)
    print("f(x*) =", fx)
    print("iterations =", k)
