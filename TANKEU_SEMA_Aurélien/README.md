# Exercices d'optimisation

Ce workspace contient quatre notebooks Python présentant des méthodes itératives pour minimiser une fonction de trois variables. Les exemples utilisent NumPy pour les calculs numériques et SymPy pour définir symboliquement la fonction et obtenir ses dérivées.

## Notebooks

- `descente_de_gradient_pas_fixe.ipynb` : déplace le point courant dans la direction opposée au gradient avec une longueur de pas constante. Le processus s'arrête lorsque le gradient est suffisamment petit ou que le nombre maximal d'itérations est atteint.
- `descente_de_gradient_pas_optimal.ipynb` : choisit à chaque étape la longueur du déplacement qui minimise la fonction le long de la direction de descente. La recherche présentée est adaptée à une fonction quadratique dont la courbure reste constante.
- `descente_de_gradient_conjugué.ipynb` : construit des directions de recherche successives qui tiennent compte des directions déjà explorées. Cette méthode converge rapidement sur les problèmes quadratiques bien conditionnés.
- `quasi_newton.ipynb` : utilise la méthode BFGS. Elle améliore progressivement une approximation de l'inverse de la courbure de la fonction, puis s'en sert pour choisir la direction de recherche.

## Utilisation

Dans chaque notebook, la fonction d'exemple est définie symboliquement à partir de trois variables, depuis la deuxième cellule de code. Remplacez cette expression par la fonction à étudier avant d'exécuter les cellules dans l'ordre, du haut vers le bas. Le point initial et la tolérance d'arrêt sont regroupés dans la cellule de paramètres.

Les résultats affichés indiquent le nombre d'itérations, le point final, la valeur de la fonction en ce point et la norme du gradient final.
