"""
Définition des problèmes tests utilisés pour valider et comparer
les méthodes numériques.

Ce module contient :
- Un problème linéaire simple avec solution exacte (décroissance radioactive)
- Un problème raide (stiff) pour tester la stabilité des méthodes
- Un problème conservatif (oscillateur harmonique) pour tester la conservation d'énergie
"""

import numpy as np


# ============================================================
# 4.1 - DÉCROISSANCE RADIOACTIVE
# ============================================================

# Constante de désintégration (demi-vie ~ 1.386 unités de temps)
LAMBDA = 0.5

def f_radioactive(t, y):
    """
    Second membre pour la décroissance radioactive.
    
    Équation différentielle : y' = -λ * y, avec λ = 0.5
    
    Solution analytique : y(t) = y₀ * exp(-λ*t)
    
    Ce problème est :
    - Linéaire
    - Stable (toutes les solutions tendent vers 0)
    - Non raide (car λ est petit)
    - Idéal pour vérifier l'ordre de convergence des méthodes
    
    Paramètres
    ----------
    t : float (temps) - non utilisé ici car l'équation est autonome
    y : float (concentration)
    
    Retourne
    --------
    float : dérivée dy/dt
    """
    return -LAMBDA * y

def sol_radioactive(t, y0):
    """
    Solution exacte de l'équation de décroissance radioactive.
    
    Formule : y(t) = y₀ * exp(-λ*t)
    
    Cette fonction sert de référence pour calculer les erreurs
    dans les études de convergence.
    
    Paramètres
    ----------
    t : float ou numpy array (temps)
    y0 : float (condition initiale)
    
    Retourne
    --------
    float ou numpy array : solution exacte au(x) temps t
    """
    return y0 * np.exp(-LAMBDA * t)


# ============================================================
# 4.2 - PROBLÈME RAIDE (STIFF)
# ============================================================

# Constante de rigidité élevée → problème raide
K_RAIDE = 500.0

def f_raide(t, y):
    """
    Second membre pour le problème raide.
    
    Équation : y' = -k*(y - cos(t)), avec k = 500
    
    Particularités :
    - Solution exacte : y(t) = cos(t) + (y₀ - 1)*exp(-k*t)
    - Le terme exp(-k*t) décroît extrêmement vite (constante de temps = 1/k = 0.002)
    - Le régime transitoire disparaît en ~0.01 unités de temps
    - Après le transitoire, la solution suit cos(t)
    
    Pourquoi est-ce raide ?
    - Le terme exp(-k*t) nécessite un pas de temps très petit
    - Les méthodes explicites (Euler, RK4) sont instables si h*k > ~2
    - Les méthodes implicites (Euler implicite) restent stables
    
    Ce problème illustre parfaitement la différence de stabilité
    entre méthodes explicites et implicites.
    
    Paramètres
    ----------
    t : float (temps) - utilisé dans cos(t)
    y : float (solution)
    
    Retourne
    --------
    float : dérivée dy/dt
    """
    return -K_RAIDE * (y - np.cos(t))


# ============================================================
# 4.3 - OSCILLATEUR HARMONIQUE
# ============================================================

# Fréquence propre de l'oscillateur (pulsation)
OMEGA = 1.0

def f_oscillateur(t, y):
    """
    Second membre pour l'oscillateur harmonique (sans amortissement).
    
    Système du premier ordre équivalent :
        x' = v           (position)
        v' = -ω² * x     (vitesse)
    
    où y = [x, v] est le vecteur d'état.
    
    Équation du second ordre originale : x'' + ω²*x = 0
    
    Propriétés physiques :
    - Système conservatif (énergie mécanique constante)
    - Solution exacte : x(t) = x₀*cos(ωt) + (v₀/ω)*sin(ωt)
    - Période T = 2π/ω = 2π ≈ 6.28
    
    Pourquoi ce problème est intéressant :
    - Teste la conservation d'énergie (erreur de phase)
    - Les méthodes explicites ont une dérive d'énergie (amplification)
    - RK4 conserve mieux l'énergie qu'Euler ou RK2
    - Permet de visualiser l'erreur de phase sur des temps longs
    
    Paramètres
    ----------
    t : float (temps) - non utilisé car le système est autonome
    y : numpy array de taille 2 [x, v]
    
    Retourne
    --------
    numpy array de taille 2 : [dx/dt, dv/dt] = [v, -ω²*x]
    """
    x, v = y[0], y[1]  # décompose le vecteur d'état
    
    # Retourne le vecteur des dérivées
    return np.array([v, -OMEGA**2 * x])

def energie_oscillateur(t, y):
    """
    Calcule l'énergie mécanique totale de l'oscillateur.
    
    Énergie cinétique : Ec = ½*v²
    Énergie potentielle : Ep = ½*ω²*x²
    Énergie totale : E = Ec + Ep = ½*(v² + ω²*x²)
    
    Pour ω = 1, on a simplement E = ½*(v² + x²)
    
    Attention à la forme de y :
    --------
    y est attendu sous forme (dimension, N) où :
    - y[0] : positions (x)
    - y[1] : vitesses (v)
    
    Cela correspond à la transposition de ce que retournent
    les solveurs (qui donnent (N+1, dimension)).
    
    Exemple d'utilisation :
    >>> t, y = rk4(f_oscillateur, 0, [1, 0], 50, 1000)
    >>> E = energie_oscillateur(t, y.T)  # On transpose !
    
    Paramètres
    ----------
    t : numpy array (temps) - non utilisé, mais gardé pour signature uniforme
    y : numpy array de forme (2, N) - [positions ; vitesses]
    
    Retourne
    --------
    numpy array : énergie mécanique à chaque instant
    """
    x, v = y[0], y[1]  # x : positions, v : vitesses
    
    # Formule : E = ½*v² + ½*ω²*x²
    return 0.5 * v**2 + 0.5 * OMEGA**2 * x**2