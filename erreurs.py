"""
Calcul d'erreurs, génération de solution de référence, tracé des courbes
de convergence et affichage des tableaux d'erreurs.

Ce module fournit les outils d'analyse de convergence pour comparer
les différentes méthodes numériques du projet.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from euler_implicite import DivergenceError


# ============================================================
# 1. ERREUR L-INFINI (NORME INFINIE)
# ============================================================

def erreur_Linf(y_exact, y_approx):
    """
    Calcule l'erreur L_infini entre deux solutions.
    
    Définition : ||e||_∞ = max_i |y_exact(t_i) - y_approx(t_i)|
    
    Pour les systèmes vectoriels, on prend la norme euclidienne
    de l'erreur à chaque instant, puis le max sur tous les instants.
    
    Formule vectorielle :
        ||e||_∞ = max_i ||y_exact(t_i) - y_approx(t_i)||_2
    
    Cette norme est plus sévère que l'erreur moyenne (L2) car elle
    capture les pires écarts locaux.
    
    Paramètres
    ----------
    y_exact  : solution exacte (array 1D ou 2D)
    y_approx : solution approchée (array 1D ou 2D)
    
    Retourne
    --------
    float : erreur L_infini (norme infinie)
    """
    # Conversion en array numpy si ce n'est pas déjà fait
    y_exact = np.asarray(y_exact)
    y_approx = np.asarray(y_approx)
    
    # Cas scalaire (y est un vecteur 1D)
    if y_exact.ndim == 1:
        # Erreur absolue en chaque point
        return np.max(np.abs(y_exact - y_approx))
    else:
        # Cas vectoriel : chaque ligne correspond à un instant t_i
        # et chaque colonne à une composante de l'état
        diff = y_exact - y_approx
        
        # Norme euclidienne de l'erreur à chaque instant
        # axis=1 : on somme sur les colonnes (les composantes)
        norme_pts = np.linalg.norm(diff, axis=1)
        
        # Erreur L_infini = max sur tous les instants
        return np.max(norme_pts)


# ============================================================
# 2. SOLUTION DE RÉFÉRENCE (SOLVEUR HAUTE PRÉCISION)
# ============================================================

def solution_reference(f, t0, y0, T, N_ref=20000):
    """
    Génère une solution de référence très précise avec solve_ivp (RK45).
    
    Pourquoi utiliser une référence numérique plutôt que la solution exacte ?
    - Certains problèmes n'ont pas de solution analytique (ex: problème raide)
    - On veut comparer les méthodes sur un "oracle" fiable
    - RK45 avec tolérance serrée (1e-12) donne une solution quasi-exacte
    
    Paramètres
    ----------
    f      : fonction f(t, y) définissant l'EDO
    t0, y0 : conditions initiales
    T      : temps final
    N_ref  : nombre de points pour la sortie (défaut 20000, très fin)
    
    Retourne
    --------
    t_ref : vecteur des temps (taille N_ref+1)
    y_ref : solution associée (taille N_ref+1)
    
    Remarque
    --------
    solve_ivp exige que y0 soit un array 1D. On utilise np.atleast_1d
    pour gérer proprement les conditions initiales scalaires.
    """
    # Conversion de y0 en array 1D (solve_ivp ne gère pas les scalaires)
    y0_arr = np.atleast_1d(np.asarray(y0, dtype=float))
    
    # Grille temporelle régulière (comme les solveurs du projet)
    t_ref = np.linspace(t0, T, N_ref + 1)

    # Appel à l'intégrateur de SciPy avec des tolérances extrêmement serrées
    sol = solve_ivp(
        f,                    # second membre
        [t0, T],              # intervalle de temps
        y0_arr,               # condition initiale (array 1D)
        t_eval=t_ref,         # on demande les valeurs sur la grille
        method='RK45',        # méthode de Runge-Kutta d'ordre 4/5
        rtol=1e-12,           # tolérance relative (très petite)
        atol=1e-12            # tolérance absolue (très petite)
    )

    # solve_ivp retourne sol.y de forme (dim, N+1)
    # On transpose pour avoir (N+1, dim) comme nos solveurs
    y_ref = sol.y.T
    
    # Si le problème était scalaire à l'origine, on extrait la colonne 0
    # pour revenir à un array 1D (compatibilité avec les fonctions d'erreur)
    if not hasattr(y0, '__len__'):
        y_ref = y_ref[:, 0]

    return sol.t, y_ref


# ============================================================
# 3. CALCUL DES ERREURS POUR UNE LISTE DE PAS
# ============================================================

def _calculer_erreurs(methode, h_list, f, t0, y0, T, sol_exacte):
    """
    Calcule l'erreur L_infini pour une méthode donnée sur une liste de pas h.
    
    Cette fonction est interne (préfixe '_') et est utilisée par
    trace_convergence et tableau_erreurs.
    
    Particularité importante :
    --------------------------
    Si la méthode lève une DivergenceError (cas du point fixe qui diverge),
    l'erreur est notée comme NaN et un avertissement est affiché.
    Cela permet de continuer l'exécution même si une méthode échoue
    pour certains pas, au lieu de faire planter tout le script.
    
    Paramètres
    ----------
    methode   : fonction solveur (euler_explicite, rk4, etc.)
    h_list    : liste des pas de temps à tester
    f, t0, y0, T : paramètres du problème
    sol_exacte : fonction solution exacte (ou None si référence numérique)
    
    Retourne
    --------
    liste des erreurs (des NaN pour les pas divergents)
    """
    erreurs = []
    
    for h in h_list:
        # Calcul du nombre de pas (arrondi pour éviter les erreurs d'arrondi)
        N = int(round((T - t0) / h))
        
        try:
            # Résolution avec la méthode demandée
            t, y = methode(f, t0, y0, T, N)
        except DivergenceError as e:
            # Le point fixe a divergé : on note NaN et on continue
            print(f"  [ATTENTION] h={h} : {e}")
            erreurs.append(np.nan)
            continue

        # Calcul de l'erreur
        if sol_exacte is not None:
            # Cas 1 : solution exacte disponible (décroissance radioactive)
            y_exact = sol_exacte(t)
            err = erreur_Linf(y_exact, y)
        else:
            # Cas 2 : on utilise la référence numérique (problème raide, oscillateur)
            # Attention : référence calculée à chaque appel → coûteux !
            # Pour optimiser, on pourrait la mettre en cache avec @lru_cache
            t_ref, y_ref = solution_reference(f, t0, y0, T)
            
            # Interpolation de la référence sur la grille t de la méthode
            if np.asarray(y_ref).ndim == 1:
                # Cas scalaire : interpolation 1D standard
                y_ref_interp = np.interp(t, t_ref, y_ref)
            else:
                # Cas vectoriel : interpolation composante par composante
                y_ref_interp = np.zeros_like(y)
                for i in range(y.shape[1]):
                    y_ref_interp[:, i] = np.interp(t, t_ref, y_ref[:, i])
            
            # Calcul de l'erreur L_infini
            err = erreur_Linf(y_ref_interp, y)

        erreurs.append(err)

    return erreurs


def _ordre_observe(erreurs, h_list):
    """
    Estime l'ordre de convergence par régression sur les deux plus petits pas valides.
    
    Principe :
    ----------
    Si erreur ≈ C * h^p, alors p = log(err(h1)/err(h2)) / log(h1/h2)
    
    On utilise les deux plus petits pas (où l'erreur est la plus petite
    et donc la plus fiable) pour estimer p.
    
    Paramètres
    ----------
    erreurs : liste des erreurs (contient potentiellement des NaN)
    h_list  : liste des pas correspondants
    
    Retourne
    --------
    float : ordre estimé (ou NaN si pas assez de données valides)
    """
    # Conversion en array pour faciliter les opérations vectorielles
    erreurs = np.array(erreurs, dtype=float)
    h_arr = np.array(h_list, dtype=float)
    
    # Filtrage des valeurs valides (erreurs finies et > 0)
    valides = np.isfinite(erreurs) & (erreurs > 0)
    
    # Il faut au moins 2 points pour estimer une pente
    if valides.sum() < 2:
        return np.nan
    
    # On prend les deux derniers pas valides (généralement les plus petits)
    e = erreurs[valides]
    hh = h_arr[valides]
    
    # Formule de l'ordre : p = log(e1/e2) / log(h1/h2)
    return (np.log(e[-1]) - np.log(e[-2])) / (np.log(hh[-1]) - np.log(hh[-2]))


# ============================================================
# 4. TRACÉ DES COURBES DE CONVERGENCE
# ============================================================

def trace_convergence(methodes, h_list, f, t0, y0, T,
                       sol_exacte=None, titre="Convergence"):
    """
    Trace la courbe de convergence log-log pour chaque méthode.
    
    Graphique en échelle logarithmique :
        - Axe x : pas h (log)
        - Axe y : erreur L_infini (log)
    
    Interprétation :
    ---------------
    - La pente donne l'ordre de convergence de la méthode
    - Méthode d'ordre 1 → pente -1
    - Méthode d'ordre 2 → pente -2
    - Méthode d'ordre 4 → pente -4 (idéal : RK4)
    
    Les points plus hauts sur le graphe = plus d'erreur (moins précis).
    Les pentes plus raides = convergence plus rapide.
    
    Paramètres
    ----------
    methodes   : dictionnaire {nom: fonction_solveur}
    h_list     : liste des pas à tester
    f, t0, y0, T : paramètres du problème
    sol_exacte : fonction de solution exacte (None → référence numérique)
    titre      : titre du graphique
    """
    plt.figure(figsize=(8, 6))

    for nom, methode in methodes.items():
        # Calcul des erreurs pour tous les pas
        erreurs = _calculer_erreurs(methode, h_list, f, t0, y0, T, sol_exacte)
        
        # Conversion en arrays
        h_arr = np.array(h_list)
        err_arr = np.array(erreurs)
        
        # Filtrage des valeurs valides (on enlève les NaN)
        valides = np.isfinite(err_arr) & (err_arr > 0)

        # Tracé en échelle log-log des points valides
        if valides.sum() > 0:
            plt.loglog(
                h_arr[valides],           # pas h (échelle log)
                err_arr[valides],         # erreurs (échelle log)
                'o-',                     # points + ligne
                label=nom,
                linewidth=2,
                markersize=8
            )

        # Affichage de l'ordre estimé dans la console
        ordre = _ordre_observe(erreurs, h_list)
        if np.isfinite(ordre):
            print(f"Ordre de convergence pour {nom} : {ordre:.3f}")
        else:
            print(f"Ordre de convergence pour {nom} : indéterminé (donnees insuffisantes)")

    # Mise en forme du graphique
    plt.xlabel('Pas h (log)', fontsize=12)
    plt.ylabel('Erreur L_infini (log)', fontsize=12)
    plt.title(titre, fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)  # grille majeure et mineure
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()


# ============================================================
# 5. TABLEAU DES ERREURS (AFFICHAGE CONSOLE)
# ============================================================

def tableau_erreurs(methodes, h_list, f, t0, y0, T, sol_exacte=None):
    """
    Affiche un tableau des erreurs et de l'ordre de convergence observé.
    
    Format du tableau :
        Méthode          | h=1.0e-1   | h=5.0e-2   | ... | Ordre
        -------------------------------------------------------
        Euler explicite  | 1.23e-2    | 6.15e-3    | ... | 1.001
        RK4              | 3.45e-5    | 2.16e-6    | ... | 4.002
    
    Interprétation :
    ---------------
    - Les erreurs doivent diminuer quand h diminue
    - L'ordre estimé doit se rapprocher de l'ordre théorique
    - Les 'diverge' indiquent que la méthode est instable pour ce pas
    
    Paramètres
    ----------
    Identiques à trace_convergence
    """
    print("\n" + "=" * 90)
    print("TABLEAU D'ERREURS")
    print("=" * 90)

    # En-tête du tableau
    print(f"{'Methode':<15}", end='')
    for h in h_list:
        print(f"| h={h:<8.1e} ", end='')
    print("| Ordre")
    print("-" * 90)

    # Ligne pour chaque méthode
    for nom, methode in methodes.items():
        erreurs = _calculer_erreurs(methode, h_list, f, t0, y0, T, sol_exacte)
        ordre = _ordre_observe(erreurs, h_list)

        # Affichage du nom de la méthode
        print(f"{nom:<15}", end='')
        
        # Affichage des erreurs pour chaque pas
        for err in erreurs:
            if np.isfinite(err):
                print(f"| {err:<8.2e} ", end='')   # notation scientifique
            else:
                print(f"| {'diverge':<8} ", end='')  # cas du point fixe divergent
        
        # Affichage de l'ordre estimé
        if np.isfinite(ordre):
            print(f"| {ordre:.3f}")
        else:
            print("| N/A")

    print("=" * 90)