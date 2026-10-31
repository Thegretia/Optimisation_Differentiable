import numpy as np
import sympy as sp


def gradient_conjugue(expression, x0, tol=1e-6, max_iter=10000):
    f = sp.sympify(expression)
    variables = sorted(f.free_symbols, key=lambda v: v.name)
    f_num = sp.lambdify(variables, f, "numpy")
    grad = sp.lambdify(variables, [sp.diff(f, v) for v in variables], "numpy")

    x = np.array(x0, dtype=float)
    n = len(x)
    g = np.array(grad(*x), dtype=float)
    d = -g
    for k in range(max_iter):
        if np.linalg.norm(g) < tol:
            break

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

        x = x + alpha * d
        g_new = np.array(grad(*x), dtype=float)
        beta = (g_new @ g_new) / (g @ g)
        d = -g_new + beta * d
        if (k + 1) % n == 0 or g_new @ d >= 0:
            d = -g_new
        g = g_new
    return x, f_num(*x), k


if __name__ == "__main__":
    expression = input("f = ")
    x0 = [float(v) for v in input("x0 = ").split()]
    x, fx, k = gradient_conjugue(expression, x0)
    print("x* =", x)
    print("f(x*) =", fx)
    print("iterations =", k)
