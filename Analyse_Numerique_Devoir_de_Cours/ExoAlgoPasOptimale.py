import numpy as np

# 1. Définition de la fonction objectif
def f(X):
    x, y, z = X
    return 6*x**2 + 6*x*y + 3*y**2 + y**3 - (1/3)*z**3 + (1/4)*z**4

# 2. Définition du gradient
def grad_f(X):
    x, y, z = X
    gx = 12*x + 6*y
    gy = 6*x + 6*y + 3*y**2
    gz = z**3 - z**2
    return np.array([gx, gy, gz], dtype=float)

# 3. MISE À JOUR DU PAS : Algorithme de recherche du pas optimal (Dichotomie)
def trouver_pas_optimal(X, g, alpha_max=1.0, tol_pas=1e-5):
    """
    Recherche le meilleur pas 'alpha' qui minimise f(X - alpha * g).
    On utilise une recherche par dichotomie sur la dérivée directionnelle.
    """
    bas = 0.0
    haut = alpha_max
    
    # On itère pour resserrer l'intervalle du pas idéal
    for _ in range(50):
        milieu = (bas + haut) / 2.0
        
        # On regarde un tout petit peu à gauche et à droite du milieu pour estimer la pente
        eps = 1e-6
        f_gauche = f(X - (milieu - eps) * g)
        f_droite = f(X - (milieu + eps) * g)
        
        # Si la fonction augmente vers la droite, le minimum est à gauche
        if f_droite > f_gauche:
            haut = milieu
        else:
            bas = milieu
            
        if (haut - bas) < tol_pas:
            break
            
    return (bas + haut) / 2.0

# --- Configuration de l'algorithme ---
X = np.array([-0.6637, 0.1758, 1.5778], dtype=float)  # Même point initial X(0)
tol = 1e-5                                            # Même tolérance
max_iter = 100                                        # Moins d'itérations que le pas fixe

print("--- Début de l'optimisation par la Descente de Gradient à Pas Optimal ---")
print(f"Itération 0 : X = {X}, f(X) = {f(X):.6f}\n")

# --- Boucle principale ---
for k in range(1, max_iter + 1):
    g = grad_f(X)
    norm_g = np.linalg.norm(g)
    
    # Condition d'arrêt
    if norm_g < tol:
        print(f"-> Convergence réussie en {k-1} itérations ! <-")
        break
        
    # CALCUL DYNAMIQUE DU PAS :
    # Au lieu d'avoir un alpha constant, on appelle notre fonction de recherche linéaire
    alpha_optimal = trouver_pas_optimal(X, g)
    
    # Mise à jour du point avec le pas calculé sur mesure
    X = X - alpha_optimal * g
    
    # Affichage du suivi à chaque itération pour voir le pas fluctuer
    print(f"Itération {k} :")
    print(f"  Pas calculé (alpha) = {alpha_optimal:.5f}")
    print(f"  X = {X}")
    print(f"  f(X) = {f(X):.6f}")
    print(f"  ||Grad f|| = {norm_g:.6f}\n")

print("--- Résultat Final ---")
print(f"Point optimal trouvé : X* = {X}")
print(f"Valeur de la fonction au minimum : f(X*) = {f(X):.6f}")
