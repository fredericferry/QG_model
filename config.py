# -*- coding: utf-8 -*-
"""
Modèle 4.0 quasi-géostrophique à tourbillon potentiel uniforme
Portage du modèle Scilab de Matthieu Plu (GMAP/RECYF, Nov 2005)
Septembre 2014 - Janvier 2015
Créé le 24 septembre 2014
@author: Clément Blot (IENM3)
@author: Guillem Coquelet (IENM3)

Configuration du modèle.
"""
from numpy import dot, pi, transpose, zeros
import sys

#=========================================================================
# Constantes de calcul
#=========================================================================
# Troncature
M = 32  # Suivant x
N = 64  # Et y

# Taille du domaine suivant x et y
MUTIL = 2 * M
NUTIL = 2 * N

# Dimension de la matrice pour traiter l'aliasing en x et y
LM = 108
LN = 243

# Test sur LM et LN
if ((LM < 3 * M + 1) | (LN < 3 * N + 1)):
    sys.exit("Config : problème d'aliasing.\n"
             "LM doit être inférieur à 3 * M + 1 et LN doit être inférieur à "
             "3 * N + 1.")
else:
    DT = 9e2  # Pas de temps en secondes, tenir compte du critere CFL
    EPSILON = 1e-2  # Coefficient du filtre d'Asselin
    NU = 0. # Viscosité

#=========================================================================
# Constantes physiques
#=========================================================================
# Grandeurs physiques
G = 9.80665  # Constante gravitationnelle

# Grandeurs caractéristiques
A = 2e6  # Longueur horizontale caractéristique
U = 4.8e1  # Vitesse caractéristique du vent

# Dimensions du domaine (en metres)
LX = 8e6  # x : Perpendiculaire au courant-jet
LY = 16e6  # y : Parallèle au courant-jet
H = 9e3  # Hauteur de la tropopause

# Constantes de l'état de base
TH0 = 2.86e2  # Température potentielle
NO = 1e-2  # Fréquence de Brunt-Vaisala
F = 1e-4  # Paramètre de Coriolis

# Paramètre d'ajustement de la partie état de base de la température pour
# avoir un vent nul aux bords, il faut 1.6 * 0.12, alors que pendant le dea,
# on avait 1 * 0.12
ALPHA = 1.6 * 0.12

# Vent de la partie Eady de l'état de base (sert à éviter les problèmes de
# périodisation de theta)
VMAX = 1.8 * ALPHA * NO * H * U / (A * F)

#=========================================================================
# Matrices utiles
#=========================================================================
# Masque à appliquer pour la troncature elliptique
masque = zeros((2 * M, 2 * N), float)
for m in range(1, M + 2):
    for n in range(1, N + 2):
        if ((m - 1) / float(M)) ** 2 + ((n - 1) / float(N)) ** 2 <= 1:
            masque[m - 1, n - 1] = 1
masque[0:M + 1, N + 1:2 * N] = masque[0:M + 1, N - 1:0:-1]  # Symetries
masque[M + 1:2 * M, :] = masque[M - 1:0:-1,:]  # Symetries

# Matrices de 1
one_m = zeros((MUTIL, 1), float) + 1
one_n = zeros((1, NUTIL), float) + 1
# Vecteur colonne décrivant le rangement des coefficients spectraux selon x
mat_m = transpose([list(range(0, M + 1)) + list(range(-M + 1, 0))])
# et selon y
mat_n = [list(range(0, N + 1)) + list(range(-N + 1, 0))]

# Matrice de 1/hnm
# hnm hauteur de Rossby
inv_hnm = 2 * pi * NO / F * ((dot(one_m, mat_n) *
                              dot(one_m, mat_n)) /
                             LY ** 2 +
                             (dot(mat_m, one_n) *
                              dot(mat_m, one_n)) /
                             LX ** 2) ** (0.5)
inv_hnm[0, 0] = 1 / H  # On évite la division par 0
