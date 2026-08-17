"""
Méthode d'Euler implicite (rétrograde) pour résoudre y' = f(t, y).
Formule : y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})

Deux résolutions de l'équation non linéaire à chaque pas sont proposées :
    - point fixe   : simple, mais NE CONVERGE QUE SI h * L < 1,
                     où L est la constante de Lipschitz de f en y.
                     Pour le problème raide (k = 500), cela impose h < 1/500 = 0.002.
    - Newton       : convergence quadratique, pas de restriction pratique sur h,
                     recommandé pour les problèmes raides.
"""

import numpy as np


# ============================================================
# EXCEPTION PERSONNALISÉE
# ============================================================

class DivergenceError(RuntimeError):
    """
    Exception levée quand l'itération de point fixe ou Newton diverge.
    
    Hérite de RuntimeError pour être cohérente avec les erreurs d'exécution.
    Utilisée dans main.py pour capturer les échecs de convergence.
    """
    pass


# ============================================================
# FONCTION PRINCIPALE : EULER IMPLICITE
# ============================================================

def euler_implicite(f, t0, y0, T, N, tol=1e-8, max_iter=100, methode='newton'):
    """
    Résout y' = f(t, y) avec la méthode d'Euler implicite (ou rétrograde).
    
    Formule : y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})
    
    Particularités :
    - Implicite car y_{n+1} apparaît des deux côtés de l'équation
    - Nécessite la résolution d'une équation non linéaire à chaque pas
    - A-stable : stable pour tout h (pour la solution exacte de l'équation)
    - Ordre 1 (même ordre qu'Euler explicite)
    
    Remarque importante sur l'A-stabilité :
    ---------------------------------------
    La méthode d'Euler implicite est A-stable pour la solution EXACTE
    de l'équation non linéaire. Si on la résout par point fixe,
    cette A-stabilité n'est garantie que si l'itération de point fixe
    elle-même converge (condition h*L < 1). Pour un problème raide,
    préférer methode='newton'.
    
    Paramètres
    ----------
    f        : fonction f(t, y) définissant l'EDO
    t0, y0   : conditions initiales
    T, N     : temps final et nombre de pas
    tol      : tolérance de convergence pour les itérations internes (défaut 1e-8)
    max_iter : nombre maximal d'itérations internes (défaut 100)
    methode  : 'point_fixe' ou 'newton' (défaut 'newton')
    
    Retourne
    --------
    t : vecteur des temps (N+1)
    y : solution aux temps t (scalaire ou vectoriel)
    
    Lève
    ----
    ValueError      : si methode n'est pas reconnue
    DivergenceError : si l'itération interne ne converge pas
    """
    # Vérification que la méthode demandée est valide
    if methode not in ('point_fixe', 'newton'):
        raise ValueError("methode doit être 'point_fixe' ou 'newton'")

    # Pas de temps constant
    h = (T - t0) / N
    
    # Grille temporelle
    t = np.linspace(t0, T, N + 1)

    # Initialisation du tableau de solution (scalaire ou vectoriel)
    if hasattr(y0, '__len__'):
        y = np.zeros((N + 1, len(y0)))
    else:
        y = np.zeros(N + 1)
    y[0] = y0

    # Boucle sur les pas de temps
    for n in range(N):
        if methode == 'point_fixe':
            # Appel à l'itération de point fixe
            # On passe n (numéro du pas) pour les messages d'erreur
            y[n + 1] = _point_fixe(f, t[n + 1], y[n], h, tol, max_iter, n)
        else:
            # Appel à la méthode de Newton
            y[n + 1] = _newton(f, t[n + 1], y[n], h, tol, max_iter)

    return t, y


# ============================================================
# 1. MÉTHODE DU POINT FIXE
# ============================================================

def _point_fixe(f, tn1, yn, h, tol, max_iter, n_pas):
    """
    Résout y = yn + h*f(tn1, y) par itération de point fixe.
    
    Principe : on itère la fonction contractante
        y_{k+1} = yn + h*f(tn1, y_k)
    
    Condition de convergence (théorème du point fixe de Banach) :
        h * L < 1, où L est la constante de Lipschitz de f en y.
    
    Pour f_raide avec k=500, L=500, donc il faut h < 0.002.
    
    Particularités :
    - Simple à implémenter
    - Pas de calcul de dérivée
    - Convergence linéaire (lente)
    - Diverge si h*L >= 1 (d'où les tests dans main.py)
    
    Paramètres
    ----------
    f     : fonction f(t, y)
    tn1   : temps t_{n+1} (où on cherche y_{n+1})
    yn    : y_n (connu)
    h     : pas de temps
    tol   : tolérance de convergence
    max_iter : nombre max d'itérations
    n_pas : numéro du pas (pour les messages d'erreur)
    
    Retourne
    --------
    y_{n+1} (solution convergée)
    
    Lève
    ----
    DivergenceError : si l'itération diverge ou ne converge pas en max_iter
    """
    # Initialisation : on part de y_n comme approximation initiale
    y_old = yn
    y_new = yn

    # Boucle d'itération de point fixe
    for i in range(max_iter):
        # Itération : y_{k+1} = yn + h*f(t_{n+1}, y_k)
        y_new = yn + h * f(tn1, y_old)

        # Calcul de l'écart entre deux itérations successives
        # np.atleast_1d permet de gérer les scalaires comme des vecteurs de taille 1
        ecart = np.linalg.norm(np.atleast_1d(y_new - y_old))
        
        # Vérification des valeurs non finies (NaN, Inf)
        # Cela arrive quand h*L >= 1 : l'itération explose
        if not np.isfinite(ecart):
            raise DivergenceError(
                f"Point fixe a diverge au pas n={n_pas} (t={tn1:.4f}) : "
                f"valeurs non finies. h*L est probablement >= 1 "
                f"(condition de convergence du point fixe non respectee). "
                f"Utiliser methode='newton' ou reduire h."
            )
        
        # Test de convergence
        if ecart < tol:
            return y_new

        # Préparation de la prochaine itération
        y_old = y_new

    # Si on sort de la boucle sans avoir convergé
    # Note : ecart est défini car la boucle a tourné au moins une fois
    raise DivergenceError(
        f"Point fixe n'a pas converge en {max_iter} iterations au pas "
        f"n={n_pas} (t={tn1:.4f}), ecart final = {ecart:.3e}. "
        f"Utiliser methode='newton' ou reduire h."
    )


# ============================================================
# 2. MÉTHODE DE NEWTON
# ============================================================

def _newton(f, tn1, yn, h, tol, max_iter):
    """
    Résout F(y) = y - yn - h*f(tn1, y) = 0 par la méthode de Newton.
    
    Principe : itération
        y_{k+1} = y_k - [J_F(y_k)]^{-1} * F(y_k)
    
    où J_F est la Jacobienne de F (approchée par différences finies).
    
    Avantages :
    - Convergence quadratique (très rapide)
    - Pas de condition de convergence restrictive sur h
    - Idéal pour les problèmes raides
    
    Inconvénients :
    - Plus coûteux (calcul de la Jacobienne à chaque pas)
    - Nécessite de résoudre un système linéaire à chaque itération
    
    Paramètres
    ----------
    f     : fonction f(t, y)
    tn1   : temps t_{n+1}
    yn    : y_n (connu)
    h     : pas de temps
    tol   : tolérance de convergence
    max_iter : nombre max d'itérations
    
    Retourne
    --------
    y_{n+1} (solution convergée)
    """
    # Détection du cas scalaire vs vectoriel
    scalaire = not hasattr(yn, '__len__')
    
    # Conversion en array numpy pour faciliter les calculs
    y = np.atleast_1d(np.array(yn, dtype=float))
    yn_arr = np.atleast_1d(np.array(yn, dtype=float))

    # Boucle de Newton
    for i in range(max_iter):
        # Calcul de F(y) = y - yn - h*f(t_{n+1}, y)
        # Si scalaire, on passe y[0] à f, sinon y directement
        F = y - yn_arr - h * np.atleast_1d(f(tn1, y if not scalaire else y[0]))
        
        # Calcul de la Jacobienne de F (approchée par différences finies)
        J = _jacobienne_approx(f, tn1, y, h, scalaire)

        # Résolution du système linéaire J * dy = -F
        if J.shape == (1, 1):
            # Cas scalaire : division simple
            dy = -F / J[0, 0]
        else:
            # Cas vectoriel : résolution par décomposition LU
            dy = np.linalg.solve(J, -F)

        # Mise à jour de la solution
        y = y + dy

        # Test de convergence sur la norme de dy
        if np.linalg.norm(dy) < tol:
            break

    # Retourne un scalaire si c'était un scalaire, sinon un array
    return y[0] if scalaire else y


def _jacobienne_approx(f, t, y, h, scalaire, eps=1e-7):
    """
    Approxime la Jacobienne de F(y) = y - yn - h*f(t,y) par différences finies.
    
    Principe :
        ∂F_i/∂y_j ≈ (F_i(y + eps*e_j) - F_i(y - eps*e_j)) / (2*eps)
    
    où e_j est le j-ème vecteur de la base canonique.
    
    Formule simplifiée utilisée ici :
        J[:, i] = -h * ∂f/∂y_i  +  (1 si i=j)
    
    Les dérivées de f sont calculées par différences finies centrées.
    
    Paramètres
    ----------
    f        : fonction f(t, y)
    t        : temps actuel
    y        : point où on évalue la Jacobienne (array 1D)
    h        : pas de temps (pour le terme -h * ∂f/∂y)
    scalaire : booléen indiquant si y est scalaire
    eps      : pas de différences finies (défaut 1e-7)
    
    Retourne
    --------
    J : matrice Jacobienne (dim x dim)
    
    Remarque technique
    ------------------
    La Jacobienne de F est :
        J_F = I - h * J_f
    où J_f est la Jacobienne de f.
    
    On approxime J_f par différences finies :
        (J_f)[:, i] ≈ (f(y + eps*e_i) - f(y - eps*e_i)) / (2*eps)
    """
    # Dimension du problème : 1 si scalaire, len(y) sinon
    dim = 1 if scalaire else len(y)
    
    # Matrice Jacobienne initialisée à zéro
    J = np.zeros((dim, dim))

    # Pour chaque composante de y, on approxime la dérivée partielle
    for i in range(dim):
        # Vecteur de perturbation e_i
        e_i = np.zeros(dim)
        e_i[i] = eps
        
        # Points perturbés : y+eps*e_i et y-eps*e_i
        y_plus = y + e_i
        y_moins = y - e_i

        # Évaluation de f aux points perturbés
        # Si scalaire, on extrait l'unique composante
        f_plus = f(t, y_plus[0] if scalaire else y_plus)
        f_moins = f(t, y_moins[0] if scalaire else y_moins)

        # Approche des dérivées partielles par différences finies centrées
        # df = ∂f/∂y_i
        df = (np.atleast_1d(f_plus) - np.atleast_1d(f_moins)) / (2 * eps)
        
        # J_F[:, i] = -h * ∂f/∂y_i
        J[:, i] = -h * df

    # Ajout de l'identité : J_F = I - h * J_f
    J += np.eye(dim)
    
    return J