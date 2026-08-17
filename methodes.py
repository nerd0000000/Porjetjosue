"""
Méthodes numériques pour la résolution des équations différentielles
ordinaires (EDO) : Euler explicite, Runge-Kutta ordre 2 (RK2),
Runge-Kutta ordre 4 (RK4).

Chaque fonction résout y' = f(t, y), y(t0) = y0 sur [t0, T] avec N pas,
et renvoie les vecteurs (t, y).
"""

import numpy as np


# ============================================================
# FONCTION UTILITAIRE PRIVÉE
# ============================================================

def _initialiser(y0, N):
    """
    Prépare le tableau de solution en gérant le cas scalaire et vectoriel.

    Paramètres
    ----------
    y0 : condition initiale (scalaire ou array-like)
    N  : nombre de pas

    Retourne
    --------
    y : tableau numpy de zéros de taille (N+1,) si y0 est scalaire
        ou de taille (N+1, dim) si y0 est vectoriel

    Remarque
    --------
    Le préfixe '_' indique une fonction interne (non destinée à l'usage externe).
    """
    # hasattr(y0, '__len__') détecte si y0 est un conteneur (liste, tuple, array)
    # Pour un scalaire (int, float), cela renvoie False
    if hasattr(y0, '__len__'):
        # Cas vectoriel : on crée un tableau 2D (N+1 lignes, dim colonnes)
        y = np.zeros((N + 1, len(y0)))
    else:
        # Cas scalaire : tableau 1D
        y = np.zeros(N + 1)
    
    # On place la condition initiale en première position (t0)
    y[0] = y0
    return y


# ============================================================
# EULER EXPLICITE
# ============================================================

def euler_explicite(f, t0, y0, T, N):
    """
    Résout y' = f(t, y) avec la méthode d'Euler explicite (ou progressive).
    
    Formule : y_{n+1} = y_n + h * f(t_n, y_n)
    
    C'est la méthode la plus simple : elle est explicite car y_{n+1}
    est calculé directement à partir de quantités connues au temps t_n.
    
    Ordre de convergence : 1 (erreur globale en O(h))
    Stabilité : conditionnelle (nécessite h petit pour les problèmes raides)

    Paramètres
    ----------
    f  : fonction f(t, y) définissant l'EDO y' = f(t, y)
    t0 : temps initial (float)
    y0 : condition initiale (scalaire ou array numpy)
    T  : temps final (float > t0)
    N  : nombre de pas de discrétisation (int > 0)

    Retourne
    --------
    t : vecteur des temps de taille N+1 (numpy array 1D)
    y : solution aux temps t (numpy array : 1D si scalaire, 2D si vectoriel)
    """
    # Pas de temps constant
    h = (T - t0) / N
    
    # Grille temporelle régulière de t0 à T (N+1 points)
    t = np.linspace(t0, T, N + 1)
    
    # Initialisation du tableau de solution (scalaire ou vectoriel)
    y = _initialiser(y0, N)

    # Boucle principale sur les N intervalles
    for n in range(N):
        # Euler explicite : y_{n+1} = y_n + h * f(t_n, y_n)
        # f(t[n], y[n]) donne la pente au point courant
        y[n + 1] = y[n] + h * f(t[n], y[n])

    return t, y


# ============================================================
# RUNGE-KUTTA ORDRE 2 (RK2, méthode de Heun)
# ============================================================

def rk2(f, t0, y0, T, N):
    """
    Résout y' = f(t, y) avec la méthode de Runge-Kutta d'ordre 2,
    aussi appelée méthode de Heun (ou d'Euler améliorée).
    
    Formule (schéma à 2 étages) :
        k1 = f(t_n, y_n)
        k2 = f(t_n + h, y_n + h*k1)
        y_{n+1} = y_n + (h/2)*(k1 + k2)
    
    Interprétation : on calcule une première pente k1 (Euler),
    puis une pente corrigée k2 à la fin du pas, et on prend la moyenne.
    
    Ordre de convergence : 2 (erreur globale en O(h²))
    Stabilité : meilleure qu'Euler explicite mais toujours conditionnelle

    Paramètres
    ----------
    Identiques à euler_explicite

    Retourne
    --------
    t, y : mêmes formats que euler_explicite
    """
    h = (T - t0) / N
    t = np.linspace(t0, T, N + 1)
    y = _initialiser(y0, N)

    for n in range(N):
        # Premier coefficient : pente au début du pas
        k1 = f(t[n], y[n])
        
        # Deuxième coefficient : pente estimée à la fin du pas
        # (en utilisant k1 pour prédire y à t_n + h)
        k2 = f(t[n] + h, y[n] + h * k1)
        
        # Moyenne pondérée des deux pentes (poids 1/2 chacun)
        y[n + 1] = y[n] + (h / 2) * (k1 + k2)

    return t, y


# ============================================================
# RUNGE-KUTTA ORDRE 4 (RK4, classique)
# ============================================================

def rk4(f, t0, y0, T, N):
    """
    Résout y' = f(t, y) avec la méthode de Runge-Kutta d'ordre 4 (classique).
    
    C'est la méthode la plus utilisée en pratique pour les problèmes non raides.
    
    Formule (schéma à 4 étages) :
        k1 = f(t_n, y_n)
        k2 = f(t_n + h/2, y_n + (h/2)*k1)
        k3 = f(t_n + h/2, y_n + (h/2)*k2)
        k4 = f(t_n + h, y_n + h*k3)
        y_{n+1} = y_n + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    Interprétation : combinaison de 4 évaluations de f avec des poids
    [1/6, 2/6, 2/6, 1/6] (symétrique).
    
    Ordre de convergence : 4 (erreur globale en O(h⁴))
    Stabilité : bonne mais toujours conditionnelle pour les problèmes raides
    
    Avantage : excellent compromis précision/coût pour la plupart des EDO.
    Inconvénient : 4 évaluations de f par pas (plus coûteux que Euler/RK2).

    Paramètres
    ----------
    Identiques à euler_explicite

    Retourne
    --------
    t, y : mêmes formats que euler_explicite
    """
    h = (T - t0) / N
    t = np.linspace(t0, T, N + 1)
    y = _initialiser(y0, N)

    for n in range(N):
        # k1 : pente au début du pas
        k1 = f(t[n], y[n])
        
        # k2 : pente au milieu du pas (prédiction avec k1)
        k2 = f(t[n] + h / 2, y[n] + (h / 2) * k1)
        
        # k3 : pente au milieu du pas (prédiction avec k2, plus précise)
        k3 = f(t[n] + h / 2, y[n] + (h / 2) * k2)
        
        # k4 : pente à la fin du pas (prédiction avec k3)
        k4 = f(t[n] + h, y[n] + h * k3)
        
        # Combinaison pondérée des 4 pentes
        # Les poids 1/6, 2/6, 2/6, 1/6 sont issus du développement de Taylor
        y[n + 1] = y[n] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return t, y