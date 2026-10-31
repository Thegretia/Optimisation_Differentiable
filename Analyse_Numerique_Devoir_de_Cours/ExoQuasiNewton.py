import numpy as np

# 1. Définition de la fonction objectif
def f(X):
    x, y, z = X
    return 6*x**2 + 6*x*y + 3*y**2 + y**3 - (1/3)*z**3 + (1/4)*z**4

# 2. Définition du gradient (Pas de Hessienne !)
def grad_f(X):
    x, y, z = X
    gx = 12*x + 6*y
    gy = 6*x + 6*y + 3*y**2
    gz = z**3 - z**2
    return np.array([gx, gy, gz], dtype=float)

# --- Configuration de l'algorithme ---
X = np.array([-0.6637, 0.1758, 1.5778], dtype=float)  # Même point initial X(0)
tol = 1e-5                                            # Même tolérance
max_iter = 50                                         # Moins d'itérations car très rapide

# INITIALISATION DE L'APPROXIMATION DE L'INVERSE DE LA HESSIENNE (W)
# Au début, on ne sait rien, on l'initialise avec la matrice Identité (3x3)
W = np.eye(3, dtype=float)

print("--- Début de l'optimisation par la méthode Quasi-Newton (BFGS) ---")
print(f"Itération 0 : X = {X}, f(X) = {f(X):.6f}\n")

g = grad_f(X) # Gradient initial

for k in range(1, max_iter + 1):
    norm_g = np.linalg.norm(g)
    
    # Condition d'arrêt
    if norm_g < tol:
        print(f"-> Convergence réussie en {k-1} itérations ! <-")
        break
        
    # DIRECTION DE DESCENTE QUASI-NEWTON
    # Au lieu de résoudre un système avec la vraie Hessienne H, on fait un simple 
    # produit matrice-vecteur avec notre approximation de l'inverse W.
    # d = -W * g
    d = -W.dot(g)
    
    # RECHERCHE DE LIGNE (Un pas fixe ou une ligne simple suffit pour BFGS)
    # Pour que BFGS fonctionne de manière stable, le pas doit respecter certaines conditions.
    # Ici, un petit pas adaptatif ou fixe proche de 1.0 (ici 0.5 pour la sécurité) fonctionne.
    alpha = 0.5 
    
    # Sauvegarde des anciennes valeurs avant la mise à jour
    X_ancien = X.copy()
    g_ancien = g.copy()
    
    # MISE À JOUR DU POINT X
    X = X_ancien + alpha * d
    g = grad_f(X) # Nouveau gradient
    
    # CALCUL DE L'APPROXIMATION DE LA HESSIENNE (Formule BFGS)
    s = X - X_ancien     # Variation de la position
    y = g - g_ancien     # Variation du gradient
    
    dot_ys = np.dot(y, s)
    if abs(dot_ys) > 1e-10: # Sécurité mathématique pour éviter la division par zéro
        rho = 1.0 / dot_ys
        I = np.eye(3)
        
        # Application stricte de la formule BFGS pour mettre à jour W (l'inverse de la Hessienne)
        W = (I - rho * np.outer(s, y)).dot(W).dot(I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        
    # Affichage du suivi
    print(f"Itération {k} :")
    print(f"  X = {X}")
    print(f"  f(X) = {f(X):.6f}")
    print(f"  ||Grad f|| = {norm_g:.6f}")
    print(f"  Approximation W (élément) = {W[0,0]:.4f}\n")

print("--- Résultat Final ---")
print(f"Point optimal trouvé : X* = {X}")
print(f"Valeur de la fonction au minimum : f(X*) = {f(X):.6f}")
