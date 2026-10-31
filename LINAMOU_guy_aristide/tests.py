#Tests unitaires 

import math
import unittest

from algebre import sub, norme2, matvec
from problemes import QUADRATIQUE, EXPONENTIELLE, quadratique_diagonale
from gradient_pas_fixe import gradient_pas_fixe
from gradient_pas_variable import gradient_pas_variable
from gradient_conjugue import gradient_conjugue, newton_gradient_conjugue
from quasi_newton_bfgs import bfgs
from exemples import pas_fixe_pour, lancer_methodes

PROBLEMES = [QUADRATIQUE, EXPONENTIELLE, quadratique_diagonale(10, 50.0)]


def point_test(P):
    """Point différent de u0 et de la solution, pour les vérifications différentielles."""
    return [0.3 + 0.1 * i for i in range(len(P.u0))] if len(P.u0) > 2 else [0.3, -0.2]


class TestDerivees(unittest.TestCase):
    """Le gradient et la hessienne sont comparés à des différences finies centrées."""

    def test_gradient(self):
        h = 1e-6
        for P in PROBLEMES:
            u = point_test(P)
            g = P.grad(u)
            for i in range(len(u)):
                up, um = list(u), list(u)
                up[i] += h
                um[i] -= h
                approx = (P.f(up) - P.f(um)) / (2 * h)
                self.assertAlmostEqual(g[i], approx, delta=1e-5 * max(1.0, abs(g[i])),
                                       msg=f"{P.nom}, composante {i}")

    def test_hessienne(self):
        h = 1e-6
        for P in PROBLEMES:
            u = point_test(P)
            H = P.hess(u)
            for j in range(len(u)):
                up, um = list(u), list(u)
                up[j] += h
                um[j] -= h
                gp, gm = P.grad(up), P.grad(um)
                for i in range(len(u)):
                    approx = (gp[i] - gm[i]) / (2 * h)
                    self.assertAlmostEqual(H[i][j], approx, delta=1e-5 * max(1.0, abs(H[i][j])),
                                           msg=f"{P.nom}, entrée ({i}, {j})")

    def test_gradient_nul_a_la_solution(self):
        for P in PROBLEMES:
            self.assertLess(norme2(P.grad(P.solution)), 1e-12, msg=P.nom)


class TestGradientConjugue(unittest.TestCase):
    def test_systeme_2x2(self):
        x, _ = gradient_conjugue([[4.0, 1.0], [1.0, 3.0]], [1.0, 2.0])
        self.assertAlmostEqual(x[0], 1 / 11, places=12)
        self.assertAlmostEqual(x[1], 7 / 11, places=12)

    def test_convergence_en_au_plus_n_produits(self):
        P = quadratique_diagonale(8, 20.0)
        A = P.hess(None)
        b = [1.0] * 8
        x, produits = gradient_conjugue(A, b)
        self.assertLessEqual(produits, 8)
        residu = sub(matvec(A, x), b)
        self.assertLess(norme2(residu), 1e-10)

    def test_matrice_non_definie_positive(self):
        with self.assertRaises(ValueError):
            gradient_conjugue([[1.0, 0.0], [0.0, -1.0]], [1.0, 1.0])


class TestSolveurs(unittest.TestCase):
    def test_tous_les_solveurs_convergent(self):
        for P in PROBLEMES:
            for nom, r, *_ in lancer_methodes(P):
                self.assertTrue(r.converge, msg=f"{nom} sur {P.nom}")
                self.assertLess(norme2(sub(r.u, P.solution)), 1e-5, msg=f"{nom} sur {P.nom}")

    def test_newton_une_iteration_sur_quadratique(self):
        P = QUADRATIQUE
        r = newton_gradient_conjugue(P.f, P.grad, P.hess, P.u0)
        self.assertEqual(r.iterations, 1)

    def test_decroissance_monotone_du_pas_variable(self):
        P = EXPONENTIELLE
        r = gradient_pas_variable(P.f, P.grad, P.u0)
        h = r.historique_f
        self.assertTrue(all(h[i + 1] <= h[i] + 1e-15 for i in range(len(h) - 1)))

    def test_pas_fixe_diverge_au_dela_de_2_sur_L(self):
        P = QUADRATIQUE
        r = gradient_pas_fixe(P.f, P.grad, P.u0, alpha=1.5 * 2.0 / P.L)
        self.assertEqual(r.statut, "divergence")

    def test_pas_fixe_optimal_plus_rapide_que_pas_trop_petit(self):
        P = QUADRATIQUE
        opt = gradient_pas_fixe(P.f, P.grad, P.u0, alpha=pas_fixe_pour(P))
        petit = gradient_pas_fixe(P.f, P.grad, P.u0, alpha=0.05)
        self.assertLess(opt.iterations, petit.iterations)

    def test_bfgs_sur_rosenbrock(self):
        """Fonction non convexe : BFGS avec rebroussement atteint tout de même le minimum."""
        def f(u):
            return (1 - u[0]) ** 2 + 100 * (u[1] - u[0] ** 2) ** 2

        def g(u):
            return [-2 * (1 - u[0]) - 400 * u[0] * (u[1] - u[0] ** 2),
                    200 * (u[1] - u[0] ** 2)]

        r = bfgs(f, g, [-1.2, 1.0])
        self.assertTrue(r.converge)
        self.assertLess(norme2(sub(r.u, [1.0, 1.0])), 1e-4)


if __name__ == "__main__":
    unittest.main()
