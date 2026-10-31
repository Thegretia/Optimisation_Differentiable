import numpy as np

# 1. Définition de la fonction objectif
def f(X):
    #Dans le devoir du cours x = x1, y = x2 et z = x3
    x, y, z = X
    return 6*x**2 + 6*x*y + 3*y**2 + y**3 - (1/3)*z**3 + (1/4)*z**4

# 2. Définition du gradient (vecteur des dérivées partielles)
def grad_f(X):
    x, y, z = X
    gx = 12*x + 6*y #par rapport à x
    gy = 6*x + 6*y + 3*y**2
    gz = z**3 - z**2
    return np.array([gx, gy, gz], dtype=float)

# 3. Définition de la matrice Hessienne
def hessian_f(X):
    x, y, z = X
    return np.array([ #calcul deja fait dans les questons 1 et 2 de l'exercise, on implemente juste
        [12.0, 6.0, 0.0],
        [6.0, 6.0*y + 6.0, 0.0],
        [0.0, 0.0, 3.0*z**2 - 2.0*z]
    ], dtype=float)

# --- Configuration de l'algorithme ---
X = np.array([-0.6637, 0.1758, 1.5778], dtype=float)  # Point initial X(0)
tol = 1e-5                                            # Tolérance demandée (10 puissance -5)
max_iter = 50                                         # Sécurité anti-boucle infinie

print("--- Début de l'optimisation par la méthode de Newton ---")
print(f"Itération 0 : X = {X}, f(X) = {f(X):.6f}\n")

# --- Boucle "Tant que" (implémentée avec un for et un break) ---
for k in range(1, max_iter + 1):
    g = grad_f(X)
    norm_g = np.linalg.norm(g) # Calcul de la norme du gradient (Condition d'arrêt)
    
    # Vérification de la condition d'arrêt
    if norm_g < tol:
        print(f"-> Convergence réussie en {k-1} itérations ! <-")
        break
        
    H = hessian_f(X)
    
    try:
        # Résolution de H * delta_X = -g (équivaut à delta_X = -H^-1 * g)
        delta_X = np.linalg.solve(H, -g)
    except np.linalg.LinAlgError:
        print("Erreur : La matrice Hessienne n'est pas inversible à cette étape.")
        break
        
    # Mise à jour du point : X(k+1) = X(k) + delta_X
    X = X + delta_X     #X(k+1) = X(k) - HkGk (delta_X)
    
    # Affichage du suivi des itérations
    print(f"Itération {k} :")
    print(f"  X = {X}")
    print(f"  f(X) = {f(X):.6f}")
    print(f"  ||Grad f|| = {norm_g:.6f}\n")

print("--- Résultat Final ---")
print(f"Point optimal trouvé : X* = {X}")
print(f"Valeur de la fonction au minimum : f(X*) = {f(X):.6f}")
