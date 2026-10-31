# Méthodes de descente et optimisation

Ce projet présente plusieurs méthodes pour minimiser une fonction de plusieurs
variables. Les algorithmes sont écrits en Python et utilisent des listes pour
représenter les vecteurs et les matrices.

## Contenu du projet

- `descente_pas_fixe.ipynb` : descente de gradient avec un pas constant.
- `descente_pas_optimal.ipynb` : descente de gradient avec recherche du meilleur
  pas par section dorée.
- `descente_backtracking.ipynb` : descente de gradient avec un pas réduit
  automatiquement jusqu'à obtenir une diminution suffisante de la fonction.
- `quasi_newton_bfgs.ipynb` : méthode BFGS, qui apprend une approximation de la
  courbure de la fonction.
- `fonctions.py` : fonctions communes aux notebooks : opérations sur les
  vecteurs et matrices, gradients, fonctions de test et affichage des résultats.

## Prérequis

- Python 3.
- Jupyter Notebook ou Visual Studio Code avec l'extension Jupyter pour ouvrir
  et exécuter les notebooks.

Les calculs sont réalisés avec Python standard ; aucune bibliothèque externe
n'est nécessaire.

## Démarrer

1. Ouvrez le dossier du projet dans Visual Studio Code ou Jupyter.
2. Ouvrez l'un des fichiers `.ipynb`.
3. Vérifiez que `fonctions.py` se trouve dans le même dossier que le notebook.
4. Sélectionnez un environnement Python 3.
5. Exécutez les cellules dans l'ordre, en commençant par la première.

Chaque notebook contient une implémentation de la méthode, puis des exemples
sur une fonction quadratique et sur la fonction de Rosenbrock. Un exemple
supplémentaire montre comment utiliser le gradient numérique.

## Utiliser une fonction personnelle

Une fonction objectif reçoit un vecteur `x` sous forme de liste et renvoie un
nombre. Son gradient reçoit le même vecteur et renvoie une liste de dérivées,
dans le même ordre que les coordonnées.

Par exemple, pour
`f(x) = (x[0] - 2)² + (x[1] + 1)²` :

```python
def f_perso(x):
    return (x[0] - 2) ** 2 + (x[1] + 1) ** 2


def grad_perso(x):
    return [2 * (x[0] - 2), 2 * (x[1] + 1)]
```

Après avoir défini ces fonctions dans une cellule du notebook, vous pouvez les
transmettre à une méthode. Voici un exemple avec la descente à pas fixe :

```python
import fonctions as fct

x_trouve, historique = descente_pas_fixe(
    f_perso,
    x0=[0.0, 0.0],
    alpha=0.1,
    grad=grad_perso,
)
```

Le point de départ `x0` doit avoir autant de coordonnées que le problème.
L'argument `grad` est facultatif : si vous l'omettez ou indiquez `grad=None`,
l'algorithme estime le gradient à partir de la fonction. Cette estimation est
pratique, mais peut être moins précise et demande davantage d'évaluations de
la fonction.

Les fonctions d'objectif et les gradients personnels doivent être définis dans
le notebook avant l'appel à l'algorithme.

## Résultats

Les méthodes renvoient deux valeurs :

- `x_trouve` : le point estimé comme minimum ;
- `historique` : les valeurs enregistrées à chaque itération sous la forme
  `(numéro, valeur de la fonction, norme du gradient)`.

Les fonctions `fct.afficher_historique(...)` et `fct.resume(...)` permettent
d'afficher le déroulement et le bilan de la minimisation.
