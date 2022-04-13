# -*- coding: utf-8 -*-
"""
Modèle 4.0 quasi-géostrophique à tourbillon potentiel uniforme
Portage du modèle Scilab de Matthieu Plu (GMAP/RECYF, Nov 2005)
Septembre 2014 - Janvier 2015
Créé le 24 septembre 2014
@author: Clément Blot (IENM3)
@author: Guillem Coquelet (IENM3)

Calcul des diagnostics.
"""
from config import F, G, H, inv_hnm, LX, LY, mat_n, mat_m, MUTIL, NO, NUTIL, \
    one_m, one_n, TH0, VMAX
from model.etat_initial import EtatInitial
from model.modele import Modele
from numpy import cosh, dot, sinh, shape, zeros, pi, real, sqrt, transpose
#from numpy.fft.fftpack import ifftn
from numpy.fft import ifftn


def diag_ecin(thh, thb):
    """
    Retourne le diagnostic de l'énergie cinétique totale de la perturbation.

    Parameters
    ----------
    thh : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale à la
        tropopause.
    thb : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale au
        sol.

    Returns
    -------
    epot : float
        L'énergie potentielle utilisable de la perturbation.
    """
    n_z = 101  # Nombre de points pour la discrétisation verticale
    ecin = 0.

    for alt in range(0, n_z):
        ugz, vgz = diag_h_wind(thh, thb, alt * H / (n_z - 1) - H / 2)
        ecin += 0.5 * H / n_z \
            * (sum(sum(ugz ** 2)) + sum(sum(vgz ** 2))) \
            * LX / MUTIL * LY / NUTIL

    ecin = ecin / (LX * LY)

    return ecin


def diag_epot(thh, thb):
    """
    Retourne le diagnostic de l'énergie potentielle utilisable de la
    perturbation. Elle est définie par l'approximation de Lorentz. Elle tient
    compte de la stratification uniforme en température potentielle de la
    troposphère.

    Parameters
    ----------
    thh : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale à la
        tropopause.
    thb : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale au
        sol.

    Returns
    -------
    epot : float
        L'énergie potentielle utilisable de la perturbation.
    """
    n_z = 101  # Nombre de points pour la discrétisation verticale
    epot = 0.

    for alt in range(0, n_z):
        z = alt * H / (n_z - 1) - H / 2  # Le sol est situé à -H/2
        thz = diag_h_theta(thh, thb, z) - TH0 * \
            (1 + (NO ** 2) * (z + H / 2) / G)
        epot += H / n_z \
            * (sum(sum(thz ** 2))
               / ((TH0 * (1 + NO ** 2 * (z + H / 2) / G)) ** 2)
               * 0.5 * (G ** 2) / (NO ** 2)) \
            * LX / MUTIL * LY / NUTIL

    epot = epot / (LX * LY)  # Expression surfacique

    return epot


def diag_extrem_field(field):
    """
    Retourne le diagnostic des extremas locaux d'un champ horizontal 2D.

    Parameters
    ----------
    field : ndarray, complex
        Tableau à 2 dimensions du champ à diagnostiquer.

    Returns
    -------
    value : ndarray, float
        Valeures maximales et minimales du champ.
    x : ndarray, float
        Coordonnées méridiennes des extremas.
    y : ndarray, float
        Coordonnées zonales des extremas.
    typ : ndarray, float
        Type, vrai si minimum, faux si maximum.
    """
    # Dimension du champ
    size_x = shape(field)[0]
    size_y = shape(field)[1]

    # Si la dérivée change de signe alors True, sinon False
    derx = field[1:size_x, :] > field[0:size_x - 1,:]
    dery = field[:, 1:size_y] > field[:, 0:size_y - 1]

    derx3 = derx[1:size_x - 1, :] != derx[0:size_x - 2,:]
    dery3 = dery[:, 1:size_y - 1] != dery[:, 0:size_y - 2]

    derx2 = derx3[:, 1:size_y - 1]
    dery2 = dery3[1:size_x - 1, :]

    # On cherche les points ou les deux dérivées partielles changent de signe
    dertot = derx2 & dery2

    # Pour enlever les points-selles
    dertot = dertot & (derx[1:size_x - 1, 1:size_y - 1]
                       == dery[1:size_x - 1, 1:size_y - 1])
    x, y = dertot.nonzero()
    # On ajoute 1 pour bien tomber sur l'extremum
    x = x + 1
    y = y + 1

    # Extraction et caractérisation des extremas
    size_x = shape(x)[0]
    value = zeros(size_x, float)
    typ = zeros(size_x, float)
    for i in range(0, size_x):
        value[i] = field[x[i], y[i]]
        typ[i] = derx[x[i], y[i]]

    return value, x, y, typ


def diag_h_deform(thh, thb, z):
    """
    Retourne le diagnostic du champ de déformation sur une coupe horizontale
    à l'altitude z.

    Parameters
    ----------
    thh : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale à la
        tropopause.
    thb : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale au
        sol.
    z : scalar
        Altitude de la coupe horizontale, doit être compris entre -H/2 (sol)
        et +H/2 (tropopause).

    Returns
    -------
    def_x : ndarray, float
        Tableau à 2 dimensions de la déformation méridienne à l'altitude z.
    def_y : ndarray, float
        Tableau à 2 dimensions de la déformation zonale à l'altitude z.
    """

    dugx = (2 * 1j * pi) * (dot(mat_m, one_n) / LX) \
        * ((-2 * pi * G) / (LY * F * TH0) * 1j
           * (dot(one_m, mat_n) / (inv_hnm * sinh(inv_hnm * H)))
           * (cosh((z + H / 2) * inv_hnm) * thh
              - cosh((z - H / 2) * inv_hnm) * thb))

    dugy = (2 * 1j * pi) * (dot(one_m, mat_n) / LY) \
        * ((-2 * pi * G) / (LY * F * TH0) * 1j
           * (dot(one_m, mat_n) / (inv_hnm * sinh(inv_hnm * H)))
           * (cosh((z + H / 2) * inv_hnm) * thh
              - cosh((z - H / 2) * inv_hnm) * thb))

    dvgx = (2 * 1j * pi) * (dot(mat_m, one_n) / LX) \
        * ((2 * pi * G) / (LX * F * TH0) * 1j
           * (dot(mat_m, one_n) / (inv_hnm * sinh(inv_hnm * H)))
           * (cosh((z + H / 2) * inv_hnm) * thh
              - cosh((z - H / 2) * inv_hnm) * thb))

    dvgy = (2 * 1j * pi) * (dot(one_m, mat_n) / LY) \
        * ((2 * pi * G) / (LX * F * TH0) * 1j
           * (dot(mat_m, one_n) / (inv_hnm * sinh(inv_hnm * H)))
           * (cosh((z + H / 2) * inv_hnm) * thh
              - cosh((z - H / 2) * inv_hnm) * thb))

    def1 = dvgx + dugy  # Cisaillement
    def2 = dugx - dvgy  # Etirement

    def1 = real(ifftn(def1))
    def2 = real(ifftn(def2))

    # Taux de déformation
    deform = sqrt(def1 ** 2 + def2 ** 2)

    # Directions de la contraction
    def_x = dot(one_m, one_n) * def1
    def_y = dot(one_m, one_n) * (- def2 + deform)

    # Gradients de vent géostrophique (si besoin)
    # real(ifftn(dugx)), real(ifftn(dugy)),
    # real(ifftn(dvgx)), real(ifftn(dvgy))

    return def_x, def_y


def diag_h_geop(thh, thb, z):
    """
    Retourne le diagnostic du géopotentiel sur une coupe horizontale à
    l'altitude z.

    Parameters
    ----------
    thh : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale à la
        tropopause.
    thb : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale au
        sol.
    z : scalar
        Altitude de la coupe horizontale, doit être compris entre -H/2 (sol)
        et +H/2 (tropopause).

    Returns
    -------
    geop : ndarray, float
        Tableau à 2 dimensions du géopotentiel horizontal à l'altitude z.
    """
    # Calcul du géopotentiel en spectral
    geop = (G / TH0) * (dot(one_m, one_n) / (inv_hnm * sinh(inv_hnm * H))) * \
        (cosh((z + H / 2) * inv_hnm) * thh -
         cosh((z - H / 2) * inv_hnm) * thb)

    # Passage dans l'espace physique
    geop = real(ifftn(geop))

    # Ajout de l'état de base Eady
    if EtatInitial.eady:
        geop += dot(transpose([range(1, MUTIL + 1)]), one_n) * \
            LX / MUTIL * F * \
            (VMAX / H) * (z + H / 2)

    geop /= G

    return geop


def diag_h_gradtheta(thh, thb, z):
    """
    Retourne le diagnostic du gradient de température potentielle sur une
    coupe horizontale à l'altitude z.

    Parameters
    ----------
    thh : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale à la
        tropopause.
    thb : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale au
        sol.
    z : scalar
        Altitude de la coupe horizontale, doit être compris entre -H/2 (sol)
        et +H/2 (tropopause).

    Returns
    -------
    grad_theta : ndarray, float
        Tableau à 2 dimensions du gradient de température potentielle à
        l'altitude z.
    """
    # Calcul du gradient de theta
    dthx = (2 * 1j * pi) * (dot(mat_m, one_n) / LX) \
        * ((dot(one_m, one_n) / (sinh(inv_hnm * H)))
           * (sinh((z + H / 2) * inv_hnm) * thh
              - sinh((z - H / 2) * inv_hnm) * thb))

    dthy = (2 * 1j * pi) * (dot(one_m, mat_n) / LY) \
        * ((dot(one_m, one_n) / (sinh(inv_hnm * H)))
           * (sinh((z + H / 2) * inv_hnm) * thh
              - sinh((z - H / 2) * inv_hnm) * thb))

    grad_theta = sqrt(real(ifftn(dthx)) * real(ifftn(dthx)) +
                      real(ifftn(dthy)) * real(ifftn(dthy)))

    return grad_theta


def diag_h_q(thh, thb, z):
    """
    Retourne le diagnostic du vecteur Q sur une coupe horizontale à
    l'altitude z.

    Parameters
    ----------
    thh : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale à la
        tropopause.
    thb : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale au
        sol.
    z : scalar
        Altitude de la coupe horizontale, doit être compris entre -H/2 (sol)
        et +H/2 (tropopause).

    Returns
    -------
    q1 : ndarray, float
        Tableau à 2 dimensions de la composante Q1 du vecteur Q à l'altitude z.
    q2 : ndarray, float
        Tableau à 2 dimensions de la composante Q2 du vecteur Q à l'altitude z.
    divq : ndarray, float
        Tableau à 2 dimensions de la divergence du vecteur Q à l'altitude z.
    """
    dugx = (2 * 1j * pi) * (dot(mat_m, one_n) / LX) \
        * ((-2 * pi * G) / (LY * F * TH0) * 1j
           * (dot(one_m, mat_n) / (inv_hnm * sinh(inv_hnm * H)))
           * (cosh((z + H / 2) * inv_hnm) * thh
              - cosh((z - H / 2) * inv_hnm) * thb))

    dugy = (2 * 1j * pi) * (dot(one_m, mat_n) / LY) \
        * ((-2 * pi * G) / (LY * F * TH0) * 1j
           * (dot(one_m, mat_n) / (inv_hnm * sinh(inv_hnm * H)))
           * (cosh((z + H / 2) * inv_hnm) * thh
              - cosh((z - H / 2) * inv_hnm) * thb))

    dvgx = (2 * 1j * pi) * (dot(mat_m, one_n) / LX) \
        * ((2 * pi * G) / (LX * F * TH0) * 1j
           * (dot(mat_m, one_n) / (inv_hnm * sinh(inv_hnm * H)))
           * (cosh((z + H / 2) * inv_hnm) * thh
              - cosh((z - H / 2) * inv_hnm) * thb))

    dvgy = (2 * 1j * pi) * (dot(one_m, mat_n) / LY) \
        * ((2 * pi * G) / (LX * F * TH0) * 1j
           * (dot(mat_m, one_n) / (inv_hnm * sinh(inv_hnm * H)))
           * (cosh((z + H / 2) * inv_hnm) * thh
              - cosh((z - H / 2) * inv_hnm) * thb))

    dthx = (2 * 1j * pi) * (dot(mat_m, one_n) / LX) \
        * ((dot(one_m, one_n) / (sinh(inv_hnm * H)))
           * (sinh((z + H / 2) * inv_hnm) * thh
              - sinh((z - H / 2) * inv_hnm) * thb))

    dthy = (2 * 1j * pi) * (dot(one_m, mat_n) / LY) \
        * ((dot(one_m, one_n) / (sinh(inv_hnm * H)))
           * (sinh((z + H / 2) * inv_hnm) * thh
              - sinh((z - H / 2) * inv_hnm) * thb))

    q1 = -G / TH0 * \
        (Modele().prod_tf(dugx, dthx) + Modele().prod_tf(dvgx, dthy))
    q2 = -G / TH0 * \
        (Modele().prod_tf(dugy, dthx) + Modele().prod_tf(dvgy, dthy))
    divq = (2 * 1j * pi) * (dot(mat_m, one_n) / LX) * q1 + \
        (2 * 1j * pi) * (dot(one_m, mat_n) / LY) * q2

    q1 = real(ifftn(q1))
    q2 = real(ifftn(q2))
    divq = real(ifftn(divq))

    return q1, q2, divq


def diag_h_theta(thh, thb, z):
    """
    Retourne le diagnostic de la température potentielle sur une coupe
    horizontale à l'altitude z.

    Parameters
    ----------
    thh : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale à la
        tropopause.
    thb : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale au
        sol.
    z : scalar
        Altitude de la coupe horizontale, doit être compris entre -H/2 (sol)
        et +H/2 (tropopause).

    Returns
    -------
    theta : ndarray, float
        Tableau à 2 dimensions de la température potentielle horizontale à
        l'altitude z.
    """
    # La stratification verticale est uniforme (en TH0 * NO ** 2 / g)
    # Calcul de theta en spectral
    theta = (dot(one_m, one_n) / (sinh(inv_hnm * H))) * \
        (sinh((z + H / 2) * inv_hnm) * thh -
         sinh((z - H / 2) * inv_hnm) * thb)

    # Passage dans l'espace physique
    theta = real(ifftn(theta))

    # Ajout de l'état de base Eady
    if EtatInitial.eady:
        theta += TH0 * (1 + (NO ** 2) / G * (z + H / 2)) + \
            TH0 * F * VMAX / (G * H) * \
            (dot(transpose([range(1, MUTIL + 1)]), one_n) /
             MUTIL * LX)

    return theta


def diag_h_vorti(thh, thb, z):
    """
    Retourne le diagnostic du tourbillon vertical sur une coupe horizontale
    à l'altitude z.

    Parameters
    ----------
    thh : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale à la
        tropopause.
    thb : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale au
        sol.
    z : scalar
        Altitude de la coupe horizontale, doit être compris entre -H/2 (sol)
        et +H/2 (tropopause).

    Returns
    -------
    vorti : ndarray, float
        Tableau à 2 dimensions du tourbillon horizontal à l'altitude z.
    """
    # Calcul du tourbillon en spectral
    vorti = -(G * F / (TH0 * NO ** 2)) * (inv_hnm / (sinh(inv_hnm * H))) * \
        (cosh((z + H / 2) * inv_hnm) * thh -
         cosh((z - H / 2) * inv_hnm) * thb)
    vorti[0, 0] = 0  # La partie constante est nulle

    # Passage dans l'espace physique
    vorti = real(ifftn(vorti))

    return vorti


def diag_h_vv(thh, thb, z):
    """
    Retourne le diagnostic de la vitesse verticale sur une coupe horizontale

    Parameters
    ----------
    thh : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale à la
        tropopause.
    thb : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale au
        sol.
    z : scalar
        Altitude de la coupe horizontale, doit être compris entre -H/2 (sol)
        et +H/2 (tropopause).

    Returns
    -------
    vv : ndarray, float
        Tableau à 2 dimensions de la vitesse verticale à l'altitude z.
    """
    # Calcul de la vitesse verticale sur plusieurs niveaux verticaux.
    # w est representée en 3 dimensions : spectralement sur l'horizontale et
    # spatialement sur la verticale.

    # Calcul du champ spectral 2div(Q) sur chaque niveau vertical
    niv = 13  # Nombre de niveaux verticaux
    divq = zeros((MUTIL, NUTIL, niv), complex)
    for n in range(0, int(niv)):
        # Derivées spatiales des champs
        z_tmp = n * H / (niv - 1) - H / 2

        dugx = (2 * 1j * pi) * (dot(mat_m, one_n) / LX) \
            * ((-2 * pi * G) / (LY * F * TH0) * 1j
               * (dot(one_m, mat_n) / (inv_hnm * sinh(inv_hnm * H)))
               * (cosh((z_tmp + H / 2) * inv_hnm) * thh
                  - cosh((z_tmp - H / 2) * inv_hnm) * thb))

        dugy = (2 * 1j * pi) * (dot(one_m, mat_n) / LY) \
            * ((-2 * pi * G) / (LY * F * TH0) * 1j
               * (dot(one_m, mat_n) / (inv_hnm * sinh(inv_hnm * H)))
               * (cosh((z_tmp + H / 2) * inv_hnm) * thh
                  - cosh((z_tmp - H / 2) * inv_hnm) * thb))

        dvgx = (2 * 1j * pi) * (dot(mat_m, one_n) / LX) \
            * ((2 * pi * G) / (LX * F * TH0) * 1j
               * (dot(mat_m, one_n) / (inv_hnm * sinh(inv_hnm * H)))
               * (cosh((z_tmp + H / 2) * inv_hnm) * thh
                  - cosh((z_tmp - H / 2) * inv_hnm) * thb))

        dvgy = (2 * 1j * pi) * (dot(one_m, mat_n) / LY) \
            * ((2 * pi * G) / (LX * F * TH0) * 1j
               * (dot(mat_m, one_n) / (inv_hnm * sinh(inv_hnm * H)))
               * (cosh((z_tmp + H / 2) * inv_hnm) * thh
                  - cosh((z_tmp - H / 2) * inv_hnm) * thb))

        dthx = (2 * 1j * pi) * (dot(mat_m, one_n) / LX) \
            * ((dot(one_m, one_n) / (sinh(inv_hnm * H)))
               * (sinh((z_tmp + H / 2) * inv_hnm) * thh
                  - sinh((z_tmp - H / 2) * inv_hnm) * thb))

        dthy = (2 * 1j * pi) * (dot(one_m, mat_n) / LY) \
            * ((dot(one_m, one_n) / (sinh(inv_hnm * H)))
               * (sinh((z_tmp + H / 2) * inv_hnm) * thh
                  - sinh((z_tmp - H / 2) * inv_hnm) * thb))

        # Div Q
        divq[0:MUTIL, 0:NUTIL, n] = (G / TH0) * (2 * 1j * pi) \
            * ((dot(mat_m, one_n) / LX)
               * (Modele().prod_tf(dthx, dvgy) - Modele().prod_tf(dthy, dvgx))
               + (dot(one_m, mat_n) / LY)
               * (Modele().prod_tf(dthy, dugx) - Modele().prod_tf(dthx, dugy)))

    # Intégration de l'équation en w
    Cdiag = -(2 * pi * NO / F) ** 2 \
        * (dot(mat_m, one_n) ** 2 / LX ** 2
           + dot(one_m, mat_n) ** 2 / LY ** 2) \
        * (H / niv) ** 2 - 2

    e = zeros((MUTIL, NUTIL, niv), complex)
    f = zeros((MUTIL, NUTIL, niv), complex)

    for n in range(1, int(niv)):
        e[:, :, n] = -dot(one_m, one_n) / (Cdiag + e[:,:, n - 1])
        f[:, :, n] = ((1 / F) ** 2 * 2 * divq[:,:, n] * (H / niv) ** 2
                      - f[:, :, n - 1]) / (Cdiag + e[:,:, n - 1])

    vv = zeros((MUTIL, NUTIL, niv), complex)

    for n in range(int(niv) - 2, -1, -1):
        vv[:, :, n] = e[:,:, n] * vv[:,:, n + 1] + f[:,:, n]

    # On règle le problème des vitesses verticales nulles aux limites en se
    # plaçant à H / 2 - 1 pour H / 2 et à -H / 2 + 1 pour -H / 2.
    if H / 2 <= z:
        z = H / 2 - 1
    elif -H / 2 >= z:
        z = 1

    # On récupère les niveaux qui nous intéresse
    vv = vv[0:MUTIL, 0:NUTIL, int((H / 2 + z) / H * 10) + 1]

    # Passage dans l'espace physique
    vv = real(ifftn(vv))

    return vv


def diag_h_wind(thh, thb, z):
    """
    Retourne le diagnostic du vent géostrophique sur une coupe horizontale
    à l'altitude z.

    Parameters
    ----------
    thh : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale à la
        tropopause.
    thb : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale au
        sol.
    z : scalar
        Altitude de la coupe horizontale, doit être compris entre -H/2 (sol)
        et +H/2 (tropopause).

    Returns
    -------
    ug : ndarray, float
        Tableau à 2 dimensions du vent géostrophique méridien à l'altitude z.
    vg : ndarray, float
        Tableau à 2 dimensions du vent géostrophique zonal à l'altitude z.
    """
    # Calcul de ug et vg en spectral
    ug = (-2 * pi * G) / (LY * F * TH0) * 1j * \
        (dot(one_m, mat_n) / (inv_hnm * sinh(inv_hnm * H))) * \
        (cosh((z + H / 2) * inv_hnm) * thh -
         cosh((z - H / 2) * inv_hnm) * thb)

    vg = (2 * pi * G) / (LX * F * TH0) * 1j * \
        (dot(mat_m, one_n) / (inv_hnm * sinh(inv_hnm * H))) * \
        (cosh((z + H / 2) * inv_hnm) * thh -
         cosh((z - H / 2) * inv_hnm) * thb)

    # Passage dans l'espace physique
    ug = real(ifftn(ug))
    vg = real(ifftn(vg))

    # Ajout de l'etat de base Eady
    if EtatInitial.eady:
        vg += (VMAX / H) * (z + H / 2)

    return ug, vg


def diag_m_geop(thh, thb, y, n_z):
    """
    Retourne le diagnostic du géopotentiel sur une coupe méridienne à
    la longitude y.

    Parameters
    ----------
    thh : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale à la
        tropopause.
    thb : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale au
        sol.
    y : scalar
        Longitude de la coupe méridienne, doit être comprise entre 0 et Ly.
    n_z : scalar
        Nombre de points verticaux, entier impair de préférence.

    Returns
    -------
    geop : ndarray, float
        Tableau à 2 dimensions du géopotentiel à la longitude y.
    """
    geop = zeros((MUTIL, n_z), float)

    # Utilisation de la fonction diag_h_wind aux différentes altitudes z
    for alt in range(0, n_z):
        geop_tmp = diag_h_geop(thh, thb, alt * H / (n_z - 1) - H / 2)
        geop[:, alt] = geop_tmp[:, round((NUTIL) * y / LY) - 1]

    return geop


def diag_m_theta(thh, thb, y, n_z):
    """
    Retourne le diagnostic de la température potentielle sur une coupe
    méridienne à la longitude y.

    Parameters
    ----------
    thh : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale à la
        tropopause.
    thb : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale au
        sol.
    y : scalar
        Longitude de la coupe méridienne, doit être comprise entre 0 et Ly.
    n_z : scalar
        Nombre de points verticaux, entier impair de préférence.

    Returns
    -------
    theta : ndarray, float
        Tableau à 2 dimensions de la température potentielle à la longitude y.
    """
    theta = zeros((MUTIL, n_z), float)

    # Utilisation de la fonction diag_h_theta aux différentes altitudes z
    for alt in range(0, n_z):
        theta_tmp = diag_h_theta(thh, thb, alt * H / (n_z - 1) - H / 2)
        theta[:, alt] = theta_tmp[:, round((NUTIL) * y / LY) - 1]

    return theta


def diag_m_vorti(thh, thb, y, n_z):
    """
    Retourne le diagnostic du tourbillon vertical sur une coupe méridienne à
    la longitude y.

    Parameters
    ----------
    thh : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale à la
        tropopause.
    thb : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale au
        sol.
    y : scalar
        Longitude de la coupe méridienne, doit être comprise entre 0 et Ly.
    n_z : scalar
        Nombre de points verticaux, entier impair de préférence.

    Returns
    -------
    vorti : ndarray, float
        Tableau à 2 dimensions du tourbillon vertical à la longitude y.
    """
    vorti = zeros((MUTIL, n_z), float)

    # Utilisation de la fonction thetaz aux différentes altitudes z
    for alt in range(0, n_z):
        vorti_tmp = diag_h_vorti(thh, thb, alt * H / (n_z - 1) - H / 2)
        vorti[:, alt] = vorti_tmp[:, round((NUTIL) * y / LY) - 1]

    return vorti


def diag_m_wind(thh, thb, y, n_z):
    """
    Retourne le diagnostic du vent géostrophique sur une coupe méridienne à
    la longitude y.

    Parameters
    ----------
    thh : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale à la
        tropopause.
    thb : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale au
        sol.
    y : scalar
        Longitude de la coupe méridienne, doit être comprise entre 0 et Ly.
    n_z : scalar
        Nombre de points verticaux, entier impair de préférence.

    Returns
    -------
    ug : ndarray, float
        Tableau à 2 dimensions du vent géostrophique méridien à la longitude y.
    vg : ndarray, float
        Tableau à 2 dimensions du vent géostrophique zonal à la longitude y.
    """
    ug = zeros((MUTIL, n_z), float)
    vg = zeros((MUTIL, n_z), float)

    # Utilisation de la fonction diag_h_wind aux différentes altitudes z
    for alt in range(0, n_z):
        ug_tmp, vg_tmp = diag_h_wind(thh, thb, alt * H / (n_z - 1) - H / 2)
        ug[:, alt] = ug_tmp[:, round(NUTIL * y / LY) - 1]
        vg[:, alt] = vg_tmp[:, round(NUTIL * y / LY) - 1]

    return ug, vg


def diag_z_geop(thh, thb, x, n_z):
    """
    Retourne le diagnostic du géopotentiel sur une coupe zonale à la latitude
    x.

    Parameters
    ----------
    thh : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale à la
        tropopause.
    thb : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale au
        sol.
    x : scalar
        latitude de la coupe zonale, doit être comprise entre 0 et Lx.
    n_z : scalar
        Nombre de points verticaux, entier impair de préférence.

    Returns
    -------
    geop : ndarray, float
        Tableau à 2 dimensions du geopotentiel à la latitude x.
    """
    geop = zeros((NUTIL, n_z), float)

    # Utilisation de la fonction diag_h_wind aux différentes altitudes z
    for alt in range(0, n_z):
        geop_tmp = diag_h_geop(thh, thb, alt * H / (n_z - 1) - H / 2)
        geop[:, alt] = transpose(geop_tmp[round((MUTIL) * x / LX) - 1, :])

    return geop


def diag_z_theta(thh, thb, x, n_z):
    """
    Retourne le diagnostic de la température potentielle sur une coupe zonale
    à la latitude x.

    Parameters
    ----------
    thh : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale à la
        tropopause.
    thb : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale au
        sol.
    x : scalar
        latitude de la coupe zonale, doit être comprise entre 0 et Lx.
    n_z : scalar
        Nombre de points verticaux, entier impair de préférence.

    Returns
    -------
    theta : ndarray, float
        Tableau à 2 dimensions de la température potentielle à la latitude x.
    """
    theta = zeros((NUTIL, n_z), float)

    # Utilisation de la fonction diag_h_theta aux différentes altitudes z
    for alt in range(0, n_z):
        theta_tmp = diag_h_theta(thh, thb, alt * H / (n_z - 1) - H / 2)
        theta[:, alt] = transpose(theta_tmp[round((MUTIL) * x / LX) - 1, :])

    return theta


def diag_z_vorti(thh, thb, x, n_z):
    """
    Retourne le diagnostic du tourbillon vertical sur une coupe zonale à la
    latitude x.

    Parameters
    ----------
    thh : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale à la
        tropopause.
    thb : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale au
        sol.
    x : scalar
        latitude de la coupe zonale, doit être comprise entre 0 et Lx.
    n_z : scalar
        Nombre de points verticaux, entier impair de préférence.

    Returns
    -------
    vorti : ndarray, float
        Tableau à 2 dimensions du tourbillon vertical à la latitude x.
    """
    vorti = zeros((NUTIL, n_z), float)

    # Utilisation de la fonction diag_h_vorti aux differentes altitudes z
    for alt in range(0, n_z):
        vorti_tmp = diag_h_vorti(thh, thb, alt * H / (n_z - 1) - H / 2)
        vorti[:, alt] = transpose(vorti_tmp[round((MUTIL) * x / LX) - 1, :])

    return vorti


def diag_z_vv(thh, thb, x, n_z):
    """
    Retourne le diagnostic de la vitesse verticale sur une coupe zonale à la
    latitude x.

    Parameters
    ----------
    thh : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale à la
        tropopause.
    thb : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale au
        sol.
    x : scalar
        latitude de la coupe zonale, doit être comprise entre 0 et Lx.
    n_z : scalar
        Nombre de points verticaux, entier impair de préférence.

    Returns
    -------
    vv : ndarray, float
        Tableau à 2 dimensions de la vitesse verticale à la latitude x.
    """
    divq = zeros((MUTIL, NUTIL, n_z), complex)
    for n in range(0, int(n_z)):
        # Derivees spatiales des champs
        z_tmp = n * H / (n_z - 1) - H / 2
    
        dugx = (2 * 1j * pi) * (dot(mat_m, one_n) / LX) \
            * ((-2 * pi * G) / (LY * F * TH0) * 1j
                * (dot(one_m, mat_n) / (inv_hnm * sinh(inv_hnm * H)))
                * (cosh((z_tmp + H / 2) * inv_hnm) * thh
                    - cosh((z_tmp - H / 2) * inv_hnm) * thb))
    
        dugy = (2 * 1j * pi) * (dot(one_m, mat_n) / LY) \
            * ((-2 * pi * G) / (LY * F * TH0) * 1j
                * (dot(one_m, mat_n) / (inv_hnm * sinh(inv_hnm * H)))
                * (cosh((z_tmp + H / 2) * inv_hnm) * thh
                    - cosh((z_tmp - H / 2) * inv_hnm) * thb))
    
        dvgx = (2 * 1j * pi) * (dot(mat_m, one_n) / LX) \
            * ((2 * pi * G) / (LX * F * TH0) * 1j
                * (dot(mat_m, one_n) / (inv_hnm * sinh(inv_hnm * H)))
                * (cosh((z_tmp + H / 2) * inv_hnm) * thh
                    - cosh((z_tmp - H / 2) * inv_hnm) * thb))
    
        dvgy = (2 * 1j * pi) * (dot(one_m, mat_n) / LY) \
            * ((2 * pi * G) / (LX * F * TH0) * 1j
                * (dot(mat_m, one_n) / (inv_hnm * sinh(inv_hnm * H)))
                * (cosh((z_tmp + H / 2) * inv_hnm) * thh
                    - cosh((z_tmp - H / 2) * inv_hnm) * thb))
    
        dthx = (2 * 1j * pi) * (dot(mat_m, one_n) / LX) \
            * ((dot(one_m, one_n) / (sinh(inv_hnm * H)))
                * (sinh((z_tmp + H / 2) * inv_hnm) * thh
                    - sinh((z_tmp - H / 2) * inv_hnm) * thb))
    
        dthy = (2 * 1j * pi) * (dot(one_m, mat_n) / LY) \
            * ((dot(one_m, one_n) / (sinh(inv_hnm * H)))
                * (sinh((z_tmp + H / 2) * inv_hnm) * thh
                    - sinh((z_tmp - H / 2) * inv_hnm) * thb))
    
        # Div Q
        divq[0:MUTIL, 0:NUTIL, n] = (G / TH0) * (2 * 1j * pi) \
            * ((dot(mat_m, one_n) / LX)
                * (Modele().prod_tf(dthx, dvgy) - Modele().prod_tf(dthy, dvgx))
                + (dot(one_m, mat_n) / LY)
                * (Modele().prod_tf(dthy, dugx) - Modele().prod_tf(dthx, dugy)))

    vv = zeros((NUTIL, n_z), float)

    for alt in range(0, n_z):
    
        # Intégration de l'équation en w
        Cdiag = -(2 * pi * NO / F) ** 2 \
            * (dot(mat_m, one_n) ** 2 / LX ** 2
               + dot(one_m, mat_n) ** 2 / LY ** 2) \
            * (H / n_z) ** 2 - 2
    
        e = zeros((MUTIL, NUTIL, n_z), complex)
        f = zeros((MUTIL, NUTIL, n_z), complex)
    
        for n in range(1, int(n_z)):
            e[:, :, n] = -dot(one_m, one_n) / (Cdiag + e[:,:, n - 1])
            f[:, :, n] = ((1 / F) ** 2 * 2 * divq[:,:, n] * (H / n_z) ** 2
                          - f[:, :, n - 1]) / (Cdiag + e[:,:, n - 1])
    
        vv_tmp = zeros((MUTIL, NUTIL, n_z), complex)
    
        for n in range(int(n_z) - 2, -1, -1):
            vv_tmp[:, :, n] = e[:,:, n] * vv_tmp[:,:, n + 1] + f[:,:, n]
    
        # On récupère les niveaux qui nous intéresse
        vv_tmp = vv_tmp[0:MUTIL, 0:NUTIL, int((alt * H / (n_z - 1)) / H * 10) + 1]
    
        # Passage dans l'espace physique
        vv_tmp = real(ifftn(vv_tmp))
        
        vv[:, alt] = transpose(vv_tmp[round((MUTIL) * x / LX) - 1, :])

    return vv


def diag_z_wind(thh, thb, x, n_z):
    """
    Retourne le diagnostic du vent géostrophique sur une coupe zonale à la
    latitude x.

    Parameters
    ----------
    thh : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale à la
        tropopause.
    thb : ndarray, complex
        Tableau à 2 dimensions de température potentielle horizontale au
        sol.
    x : scalar
        latitude de la coupe zonale, doit être comprise entre 0 et Lx.
    n_z : scalar
        Nombre de points verticaux, entier impair de préférence.

    Returns
    -------
    ug : ndarray, float
        Tableau à 2 dimensions du vent géostrophique méridien à la latitude x.
    vg : ndarray, float
        Tableau à 2 dimensions du vent géostrophique zonal à la latitude x.
    """
    ug = zeros((NUTIL, n_z), float)
    vg = zeros((NUTIL, n_z), float)

    # Utilisation de la fonction diag_h_wind aux différentes altitudes z
    for alt in range(0, n_z):
        ug_tmp, vg_tmp = diag_h_wind(thh, thb, (alt) * H / (n_z - 1) - H / 2)
        ug[:, alt] = transpose(ug_tmp[round(MUTIL * x / LX) - 1, :])
        vg[:, alt] = transpose(vg_tmp[round(MUTIL * x / LX) - 1, :])

    return ug, vg
