# Fonctions communes aux quatre méthodes d'optimisation.
# Les vecteurs sont des listes et les matrices des listes de listes.

def produit_scalaire(u, v):
    """Multiplie deux vecteurs composante par composante puis additionne."""
    return sum(a * b for a, b in zip(u, v))


def somme(u, v):
    """Additionne deux vecteurs."""
    return [a + b for a, b in zip(u, v)]


def difference(u, v):
    """Soustrait le deuxième vecteur du premier."""
    return [a - b for a, b in zip(u, v)]


def produit_par_scalaire(c, u):
    """Multiplie chaque coordonnée du vecteur par le nombre c."""
    return [c * a for a in u]


def norme(u):
    """Calcule la longueur euclidienne du vecteur."""
    return produit_scalaire(u, u) ** 0.5


def produit_matrice_vecteur(M, v):
    """Multiplie une matrice par un vecteur."""
    return [produit_scalaire(ligne, v) for ligne in M]


def matrice_identite(n):
    """Construit la matrice identité de taille n."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def resoudre_systeme(A, b):
    """Résout le système linéaire A x = b avec l'élimination de Gauss."""
    n = len(b)
    matrice = [A[i][:] + [b[i]] for i in range(n)]

    # Met la matrice sous forme triangulaire.
    for k in range(n):
        ligne_pivot = max(range(k, n), key=lambda i: abs(matrice[i][k]))
        matrice[k], matrice[ligne_pivot] = matrice[ligne_pivot], matrice[k]

        for i in range(k + 1, n):
            facteur = matrice[i][k] / matrice[k][k]
            for j in range(k, n + 1):
                matrice[i][j] -= facteur * matrice[k][j]

    # Remonte depuis la dernière ligne pour calculer les coordonnées de x.
    solution = [0.0] * n
    for i in range(n - 1, -1, -1):
        valeurs_connues = sum(
            matrice[i][j] * solution[j] for j in range(i + 1, n)
        )
        solution[i] = (matrice[i][n] - valeurs_connues) / matrice[i][i]
    return solution


def valeurs_propres_symetrique(A, tol=1e-12, max_iter=10000):
    """Renvoie la plus petite et la plus grande valeur propre de A."""
    n = len(A)

    # La méthode de Jacobi s'applique aux matrices carrées et symétriques.
    if n == 0 or any(len(ligne) != n for ligne in A):
        raise ValueError("A doit être une matrice carrée non vide")
    if any(abs(A[i][j] - A[j][i]) > tol for i in range(n) for j in range(i)):
        raise ValueError("A doit être symétrique")

    matrice = [ligne[:] for ligne in A]

    for _ in range(max_iter):
        # Cherche le plus grand nombre hors de la diagonale.
        p, q = 0, 0
        plus_grand = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(matrice[i][j]) > plus_grand:
                    p, q = i, j
                    plus_grand = abs(matrice[i][j])

        # Quand ces nombres sont proches de zéro, la diagonale contient
        # une bonne approximation des valeurs propres.
        if plus_grand <= tol:
            valeurs = [matrice[i][i] for i in range(n)]
            return min(valeurs), max(valeurs)

        # Calcule la rotation qui annule le nombre situé en (p, q).
        valeur_pp = matrice[p][p]
        valeur_qq = matrice[q][q]
        valeur_pq = matrice[p][q]
        tau = (valeur_qq - valeur_pp) / (2 * valeur_pq)
        signe = 1.0 if tau >= 0 else -1.0
        tangente = signe / (abs(tau) + (1 + tau * tau) ** 0.5)
        cosinus = 1 / (1 + tangente * tangente) ** 0.5
        sinus = tangente * cosinus

        # Applique la rotation aux lignes et aux colonnes p et q.
        for k in range(n):
            if k != p and k != q:
                valeur_kp = matrice[k][p]
                valeur_kq = matrice[k][q]
                matrice[k][p] = matrice[p][k] = (
                    cosinus * valeur_kp - sinus * valeur_kq
                )
                matrice[k][q] = matrice[q][k] = (
                    sinus * valeur_kp + cosinus * valeur_kq
                )

        matrice[p][p] = valeur_pp - tangente * valeur_pq
        matrice[q][q] = valeur_qq + tangente * valeur_pq
        matrice[p][q] = matrice[q][p] = 0.0

    raise RuntimeError("Les rotations de Jacobi n'ont pas convergé")


# Pour une fonction f(x), un gradient manuel grad_f(x) renvoie une liste
# de dérivées, une par coordonnée de x. Si grad_f n'est pas fourni,
# cette fonction estime les dérivées en évaluant f autour du point.

def gradient_numerique(f, x, h=1e-6):
    gradient = []
    for i in range(len(x)):
        point_plus = x[:]
        point_moins = x[:]
        point_plus[i] += h
        point_moins[i] -= h

        derivee = (f(point_plus) - f(point_moins)) / (2 * h)
        gradient.append(derivee)
    return gradient


def backtracking(f, x, g, d, alpha0=1.0, beta=0.5, c=1e-4):
    """Réduit le pas jusqu'à ce que la fonction diminue suffisamment."""
    valeur_depart = f(x)
    pente = produit_scalaire(g, d)
    alpha = alpha0

    while True:
        nouveau_point = somme(x, produit_par_scalaire(alpha, d))
        valeur_suivante = f(nouveau_point)
        limite = valeur_depart + c * alpha * pente

        if valeur_suivante <= limite:
            break

        alpha *= beta
        if alpha < 1e-16:
            break
    return alpha


def section_doree(phi, a, b, tol=1e-10):
    """Cherche le minimum de phi sur l'intervalle [a, b]."""
    proportion = (5 ** 0.5 - 1) / 2
    point_gauche = b - proportion * (b - a)
    point_droit = a + proportion * (b - a)
    valeur_gauche = phi(point_gauche)
    valeur_droite = phi(point_droit)

    while b - a > tol:
        if valeur_gauche < valeur_droite:
            b = point_droit
            point_droit = point_gauche
            valeur_droite = valeur_gauche
            point_gauche = b - proportion * (b - a)
            valeur_gauche = phi(point_gauche)
        else:
            a = point_gauche
            point_gauche = point_droit
            valeur_gauche = valeur_droite
            point_droit = a + proportion * (b - a)
            valeur_droite = phi(point_droit)

    return (a + b) / 2


def pas_optimal(f, x, d):
    """Cherche le meilleur pas en minimisant f dans la direction d."""
    def valeur_sur_direction(alpha):
        point = somme(x, produit_par_scalaire(alpha, d))
        return f(point)

    borne = 1e-3
    while borne < 1e6 and valeur_sur_direction(2 * borne) < valeur_sur_direction(borne):
        borne *= 2
    return section_doree(valeur_sur_direction, 0.0, 2 * borne)


def pas_optimal_quadratique(A, g):
    """Calcule le pas exact d'une quadratique dans la direction -g."""
    return produit_scalaire(g, g) / produit_scalaire(g, produit_matrice_vecteur(A, g))

# Fonctions de test : une quadratique et la fonction de Rosenbrock.
A_QUAD = [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]
B_QUAD = [1.0, 2.0, 3.0]
X0_QUAD = [5.0, -5.0, 2.0]


def f_quadratique(x):
    """Valeur de f(x) = 1/2 x^T A x - b^T x."""
    A_x = produit_matrice_vecteur(A_QUAD, x)
    terme_quadratique = 0.5 * produit_scalaire(x, A_x)
    terme_lineaire = produit_scalaire(B_QUAD, x)
    return terme_quadratique - terme_lineaire


def grad_quadratique(x):
    """Gradient de la fonction quadratique : A x - b."""
    return difference(produit_matrice_vecteur(A_QUAD, x), B_QUAD)


X_EXACT_QUAD = resoudre_systeme(A_QUAD, B_QUAD)     # solution exacte, calculée par Gauss


# La fonction de Rosenbrock atteint son minimum en [1, 1, 1].
X0_ROSEN = [-1.2, 1.0, -1.2]
X_EXACT_ROSEN = [1.0, 1.0, 1.0]


def f_rosenbrock(x):
    """Valeur de la fonction de Rosenbrock pour un vecteur x."""
    valeur = 0.0
    for i in range(len(x) - 1):
        ecart = x[i + 1] - x[i] ** 2
        valeur += 100 * ecart ** 2 + (1 - x[i]) ** 2
    return valeur


def grad_rosenbrock(x):
    """Gradient de Rosenbrock : une dérivée pour chaque coordonnée."""
    n = len(x)
    if n < 2:
        raise ValueError("La fonction de Rosenbrock nécessite au moins 2 coordonnées")

    # Chaque paire de coordonnées contribue aux deux dérivées voisines.
    gradient = [0.0] * n
    for i in range(n - 1):
        ecart = x[i + 1] - x[i] ** 2
        derivee_gauche = -400 * x[i] * ecart - 2 * (1 - x[i])
        derivee_droite = 200 * ecart
        gradient[i] += derivee_gauche
        gradient[i + 1] += derivee_droite
    return gradient


def hessienne_rosenbrock(x):
    """Matrice des dérivées secondes de la fonction de Rosenbrock."""
    n = len(x)
    if n < 2:
        raise ValueError("La fonction de Rosenbrock nécessite au moins 2 coordonnées")
    hessienne = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n - 1):
        hessienne[i][i] += 1200 * x[i] ** 2 - 400 * x[i + 1] + 2
        hessienne[i][i + 1] = -400 * x[i]
        hessienne[i + 1][i] = -400 * x[i]
        hessienne[i + 1][i + 1] += 200
    return hessienne


def afficher_historique(historique, pas_affichage=None):
    """Affiche les valeurs enregistrées pendant les itérations."""
    n = len(historique)
    if pas_affichage is None:
        pas_affichage = max(1, n // 10)

    print(f"{'k':>8} {'f(x_k)':>18} {'||grad f(x_k)||':>18}")
    for i, (k, fx, ng) in enumerate(historique):
        if i % pas_affichage == 0 or i == n - 1:
            print(f"{k:>8d} {fx:>18.10e} {ng:>18.3e}")


def resume(nom, x, historique, x_exact, eps):
    """Affiche le résultat final et indique si la méthode a convergé."""
    k, fx, ng = historique[-1]
    statut = "convergé" if ng <= eps else "NON convergé (max_iter atteint)"
    print(f"--- {nom} ---")
    print(f"Statut        : {statut}")
    print(f"Itérations    : {k}")
    print(f"x trouvé      : {[round(v, 8) for v in x]}")
    print(f"x exact       : {x_exact}")
    print(f"||x - x*||    : {norme(difference(x, x_exact)):.3e}")
    print(f"f(x)          : {fx:.10e}")
    print(f"||grad f(x)|| : {ng:.3e}")
