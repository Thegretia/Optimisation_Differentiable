import numpy as np

# 1. Définition de la fonction objectif
def f(X):
    x, y, z = X
    return 6*x**2 + 6*x*y + 3*y**2 + y**3 - (1/3)*z**3 + (1/4)*z**4

# 2. Définition du gradient (On n'a plus besoin de la Hessienne !)
def grad_f(X):
    x, y, z = X
    gx = 12*x + 6*y
    gy = 6*x + 6*y + 3*y**2
    gz = z**3 - z**2
    return np.array([gx, gy, gz], dtype=float)

# --- Configuration de l'algorithme ---
X = np.array([-0.6637, 0.1758, 1.5778], dtype=float)  # Même point initial X(0)
tol = 1e-5                                            # Même tolérance (10 puissance -5)

# CHANGEMENTS CLÉS POUR LE PAS FIXE :
alpha = 0.05                                          # Taille du pas fixe (si trop grand, l'algo explose !)
max_iter = 500                                        # On augmente le max car le gradient est plus lent que Newton

print("--- Début de l'optimisation par la Descente de Gradient à Pas Fixe ---")
print(f"Itération 0 : X = {X}, f(X) = {f(X):.6f}\n")

# --- Boucle itérative ---
for k in range(1, max_iter + 1):
    g = grad_f(X)
    norm_g = np.linalg.norm(g) # Calcul de la norme du gradient
    
    # Vérification de la condition d'arrêt
    if norm_g < tol:
        print(f"-> Convergence réussie en {k-1} itérations ! <-")
        break
        
    # MISE À JOUR DE PAS FIXE :
    # Au lieu de résoudre un système avec la Hessienne, on avance simplement 
    # dans la direction opposée au gradient avec l'intensité dictée par alpha.
    X = X - alpha * g
    
    # Affichage du suivi toutes les 10 itérations pour ne pas surcharger l'écran
    if k % 10 == 0 or k == 1:
        print(f"Itération {k} :")
        print(f"  X = {X}")
        print(f"  f(X) = {f(X):.6f}")
        print(f"  ||Grad f|| = {norm_g:.6f}\n")

print("--- Résultat Final ---")
print(f"Point optimal trouvé : X* = {X}")
print(f"Valeur de la fonction au minimum : f(X*) = {f(X):.6f}")
