"""
Fichier principal : lance les trois problèmes tests du projet
(décroissance radioactive, problème raide, oscillateur harmonique)
et génère les figures/tableaux du rapport.
"""

# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
# NumPy : calcul numérique vectoriel
# Matplotlib : tracé de courbes

# Import des schémas numériques depuis les modules dédiés
from methodes import euler_explicite, rk2, rk4
# rk2 = Runge-Kutta d'ordre 2 (méthode de Heun)
# rk4 = Runge-Kutta classique d'ordre 4

from euler_implicite import euler_implicite, DivergenceError
# euler_implicite : solveur avec choix entre point fixe et Newton
# DivergenceError : exception levée quand la résolution échoue

from problemes import (
    f_radioactive,      # second membre pour la décroissance
    sol_radioactive,    # solution analytique associée
    f_raide,            # second membre du problème raide (k=500)
    f_oscillateur,      # second membre pour l'oscillateur harmonique
    energie_oscillateur,# fonction qui calcule l'énergie mécanique
)

from erreurs import trace_convergence, tableau_erreurs, solution_reference
# trace_convergence : graphe log-log des erreurs en fonction du pas
# tableau_erreurs : affiche un tableau récapitulatif des erreurs
# solution_reference : intégrateur RK45 de SciPy pour la référence

# ============================================================
# PARAMÈTRES GÉNÉRAUX
# ============================================================

t0 = 0.0          # temps initial (commun à tous les problèmes)
T = 5.0           # temps final (commun)

y0_radio = 1.0    # condition initiale scalaire pour la radio
y0_raide = 2.0    # condition initiale scalaire pour le problème raide
y0_osc = np.array([1.0, 0.0])  # CI vectorielle [position, vitesse] pour l'oscillateur

h_list = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
# Pas de temps décroissants pour étudier la convergence
# On descend jusqu'à 0.002 pour bien voir l'ordre des méthodes

# ============================================================
# 1. DÉCROISSANCE RADIOACTIVE - VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("1. DECROISSANCE RADIOACTIVE - VALIDATION")
print("=" * 80)

# Dictionnaire associant chaque nom de méthode à sa fonction
# On utilise des lambdas pour euler_implicite car sa signature
# demande un paramètre 'methode' supplémentaire qu'on fige ici à 'newton'
methodes_radio = {
    'Euler explicite': euler_explicite,
    'Euler implicite': lambda f, t0, y0, T, N: euler_implicite(f, t0, y0, T, N, methode='newton'),
    'RK2': rk2,
    'RK4': rk4,
}

# Trace le graphique de convergence (erreur en fonction de h) en échelle log-log
# La pente donne l'ordre de la méthode
trace_convergence(
    methodes_radio,
    h_list,
    f_radioactive,
    t0, y0_radio, T,
    sol_exacte=lambda t: sol_radioactive(t, y0_radio),
    # lambda qui appelle la solution exacte en passant y0_radio
    titre="Convergence - Decroissance radioactive"
)

# Affiche un tableau numérique des erreurs pour chaque pas et chaque méthode
tableau_erreurs(
    methodes_radio,
    h_list,
    f_radioactive,
    t0, y0_radio, T,
    sol_exacte=lambda t: sol_radioactive(t, y0_radio)
)

# ============================================================
# 2. PROBLÈME RAIDE - COMPARAISON DE STABILITÉ
# ============================================================

print("\n" + "=" * 80)
print("2. PROBLEME RAIDE - COMPARAISON DE STABILITE")
print("=" * 80)

# Pour k = 500, la condition de convergence du point fixe (h*k < 1) impose
# h < 0.002. Avec un pas "modéré" comme h = 0.01, le point fixe diverge.
# On utilise donc Newton pour Euler implicite, qui n'a pas cette restriction.
# Cela permet de vraiment illustrer l'A-stabilité d'Euler implicite face à
# l'instabilité d'Euler explicite sur ce même pas.

h_modere = 0.01                      # pas volontairement grand pour montrer l'instabilité
N_modere = int(round((T - t0) / h_modere))  # nombre de pas (arrondi pour éviter les erreurs)

# Référence très précise avec RK45 (via SciPy) sur 20000 pas
t_ref, y_ref = solution_reference(f_raide, t0, y0_raide, T, N_ref=20000)

plt.figure(figsize=(10, 6))          # figure avec taille adaptée

# 1) Euler explicite → instable pour ce pas (oscillations divergentes)
t_expl, y_expl = euler_explicite(f_raide, t0, y0_raide, T, N_modere)
plt.plot(t_expl, y_expl, 'o-', label=f"Euler explicite (h={h_modere})",
         markersize=4, linewidth=2)

# 2) Euler implicite avec Newton (stable même avec grand pas)
try:
    t_impl, y_impl = euler_implicite(f_raide, t0, y0_raide, T, N_modere, methode='newton')
    plt.plot(t_impl, y_impl, 'o-', label=f"Euler implicite - Newton (h={h_modere})",
             markersize=4, linewidth=2)
except DivergenceError as e:
    print(f"[ATTENTION] Euler implicite (Newton) : {e}")  # normalement pas levée

# 3) Courbe de référence
plt.plot(t_ref, y_ref, 'k-', label='Reference RK45', linewidth=2, alpha=0.7)

# Mise en forme du graphe
plt.xlabel('t (temps)', fontsize=12)
plt.ylabel('y(t)', fontsize=12)
plt.title("Comparaison Euler explicite vs implicite - Probleme raide", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)            # grille légère
plt.tight_layout()                   # ajuste les marges
plt.show()                           # affiche la figure (bloque selon l'environnement)

# ----------------------------------------------
# Petit test pédagogique sur la méthode du point fixe
# ----------------------------------------------
print("\nVerification : le point fixe diverge-t-il pour h=0.01, k=500 (h*k=5) ?")
try:
    euler_implicite(f_raide, t0, y0_raide, T, N_modere, methode='point_fixe')
    print("  -> Le point fixe a converge (inattendu).")
except DivergenceError as e:
    print(f"  -> Confirme : {e}")    # normalement on entre ici

h_petit = 0.001                      # h*k = 0.5 < 1 → condition suffisante de convergence
N_petit = int(round((T - t0) / h_petit))
print(f"\nAvec h={h_petit} (h*k={h_petit*500}), le point fixe devrait converger :")
try:
    euler_implicite(f_raide, t0, y0_raide, T, N_petit, methode='point_fixe')
    print("  -> Convergence confirmee.")
except DivergenceError as e:
    print(f"  -> {e}")               # ne devrait pas arriver

# ============================================================
# 3. OSCILLATEUR HARMONIQUE - CONSERVATION D'ÉNERGIE
# ============================================================

print("\n" + "=" * 80)
print("3. OSCILLATEUR HARMONIQUE - CONSERVATION D'ENERGIE")
print("=" * 80)

T_long = 50.0          # temps long pour voir la dérive d'énergie
h_osc = 0.05           # pas relativement grossier
N_osc = int(round((T_long - t0) / h_osc))

# Mêmes méthodes que pour la radio, mais cette fois sur l'oscillateur
methodes_osc = {
    'Euler explicite': euler_explicite,
    'Euler implicite': lambda f, t0, y0, T, N: euler_implicite(f, t0, y0, T, N, methode='newton'),
    'RK2': rk2,
    'RK4': rk4,
}

# Référence très précise pour calculer l'énergie "exacte" initiale
t_ref_osc, y_ref_osc = solution_reference(f_oscillateur, t0, y0_osc, T_long, N_ref=50000)
E_ref = energie_oscillateur(t_ref_osc, y_ref_osc.T)  # énergie de la référence
# Note : y_ref_osc.T car energie_oscillateur attend (temps, positions, vitesses)
# Mais ici elle est vectorisée, on passe tout le tableau d'un coup

plt.figure(figsize=(10, 6))

# Pour chaque méthode, on calcule la solution puis l'énergie
for nom, methode in methodes_osc.items():
    t, y = methode(f_oscillateur, t0, y0_osc, T_long, N_osc)
    # y est un tableau (N+1, 2) : colonne 0 = position, colonne 1 = vitesse
    E = energie_oscillateur(t, y.T)   # on transpose pour avoir (2, N+1)
    # On trace l'écart à l'énergie initiale (qui devrait être constante)
    plt.plot(t, E - E_ref[0], label=f"{nom} (h={h_osc})", linewidth=2)

plt.xlabel('t (temps)', fontsize=12)
plt.ylabel("E(t) - E(0) (difference d'energie)", fontsize=12)
plt.title("Conservation d'energie - Oscillateur harmonique", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Energie initiale : {E_ref[0]:.6f}")  # affiche la valeur de référence

# ============================================================
# FIN
# ============================================================

print("\n" + "=" * 80)
print("TOUS LES TESTS SONT TERMINES.")
print("=" * 80)