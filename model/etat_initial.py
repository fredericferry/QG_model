# -*- coding: utf-8 -*-
"""
Modèle 4.0 quasi-géostrophique à tourbillon potentiel uniforme
Portage du modèle Scilab de Matthieu Plu (GMAP/RECYF, Nov 2005)
Septembre 2014 - Janvier 2015
Créé le 24 septembre 2014
@author: Clément Blot (IENM3)
@author: Guillem Coquelet (IENM3)

Calcul des champs initiaux.
"""
from config import A, ALPHA, F, G, H, LX, LY, M, MUTIL, N, NO, NUTIL, one_m, \
    one_n, TH0, U
from numpy import array, cos, dot, sin, sqrt, transpose, zeros

class EtatInitial:

    """
    Retourne les champs initiaux pour le modèle quasi-géostrophique à
    tourbillon potentiel uniforme.
    """
    # Variable statique qui permet de savoir si on ajoute le gradient d'Eady
    # après les diagnostics
    eady = True

    def gradeady(self):
        """
        Retourne le champ initial avec le gradient d'Eady.

        Returns
        -------
        thhi : ndarray, complex
            Tableau à 2 dimensions de température potentielle horizontale à la
            tropopause.
        thbi : ndarray, complex
            Tableau à 2 dimensions de température potentielle horizontale au
            sol.

        Examples
        --------
        >>> thhi, thbi = EtatInitial().gradeady()
         """
        EtatInitial.eady = True
        thhi = zeros((MUTIL, NUTIL), float)
        thbi = zeros((MUTIL, NUTIL), float)
        return thhi, thbi

    def gradeady_gradwernli(self):
        """
        Retourne le champ initial avec le gradient d'Eady et le gradient de
        Wernli.

        Returns
        -------
        thhi : ndarray, complex
            Tableau à 2 dimensions de température potentielle horizontale à la
            tropopause.
        thbi : ndarray, complex
            Tableau à 2 dimensions de température potentielle horizontale au
            sol.

        Examples
        --------
        >>> thhi, thbi = EtatInitial().gradeady_gradwernli()
         """
        EtatInitial.eady = True
        thhi, thbi = self._grad_th_gauss()
        return thhi, thbi

    def gradeady_gradwernli_anotripo(self, tilt):
        """
        Retourne le champ initial avec le gradient d'Eady , le gradient de
        Wernli et l'nomalie tripolaire.

        Parameters
        ----------
        tilt : scalar
            Déphasage (en km) entre la perturbation à la tropopause et la
            perturbation au sol. Déphasage positif si la perturbation à la
            tropopause et en aval de la perturbation au sol. De préférence
            inférieur à 1400 km.

        Returns
        -------
        thhi : ndarray, complex
            Tableau à 2 dimensions de température potentielle horizontale à la
            tropopause.
        thbi : ndarray, complex
            Tableau à 2 dimensions de température potentielle horizontale au
            sol.

        Examples
        --------
        >>> thhi, thbi = EtatInitial().gradeady_gradwernli_anotripo(tilt = 1400)
         """
        EtatInitial.eady = True

        thhi, thbi = self._init_ano_tripo(tilt)

        # Initialisation des champs Theta dans l'espace physique
        # Ajout du gradient barocline de temperature potentielle
        gradh, gradb = self._grad_th_gauss()
        thhi += gradh
        thbi += gradb

        return thhi, thbi

    def gradwernli(self):
        """
        Retourne le champ initial avec le gradient de Wernli.

        Returns
        -------
        thhi : ndarray, complex
            Tableau à 2 dimensions de température potentielle horizontale à la
            tropopause.
        thbi : ndarray, complex
            Tableau à 2 dimensions de température potentielle horizontale au
            sol.

        Examples
        --------
        >>> thhi, thbi = EtatInitial().gradwernli()
         """
        EtatInitial.eady = False
        thhi, thbi = self._grad_th_gauss()
        return thhi, thbi

    def anotripo(self, tilt):
        """
        Retourne le champ initial avec l'nomalie tripolaire.

        Parameters
        ----------
        tilt : scalar
            Déphasage (en km) entre la perturbation à la tropopause et la
            perturbation au sol. Déphasage positif si la perturbation à la
            tropopause et en aval de la perturbation au sol. De préférence
            inférieur à 1400 km.

        Returns
        -------
        thhi : ndarray, complex
            Tableau à 2 dimensions de température potentielle horizontale à la
            tropopause.
        thbi : ndarray, complex
            Tableau à 2 dimensions de température potentielle horizontale au
            sol.

        Examples
        --------
        >>> thhi, thbi = EtatInitial().anotripo(tilt = 1400)
         """
        EtatInitial.eady = False
        thhi, thbi = self._init_ano_tripo(tilt)
        return thhi, thbi

    def _init_ano_tripo(self, tilt):
        """
        Retourne l'état de base perturbé selon Wernli. Création d'une
        anomalie tripolaire.

        Parameters
        ----------
        tilt : scalar
            Déphasage (en km) entre la perturbation à la tropopause et la
            perturbation au sol. Déphasage positif si la perturbation à la
            tropopause et en aval de la perturbation au sol. De préférence
            inférieur à 1400 km.

        Returns
        -------
        thh : ndarray, complex
            Tableau à 2 dimensions de température potentielle horizontale à la
            tropopause.
        thb : ndarray, complex
            Tableau à 2 dimensions de température potentielle horizontale au
            sol.
        """
        # Paramètre de l'anomalie tripolaire. Valeurs de faible prévisibilité
        # par défaut.
        dzon = tilt / 2e3
        exh=0
        exb=0
        phih=0
        phib=0
        A1=0.25 * 14
        A2=-0.25 * 14
        d1=0.5
        d2=0.5
        lmerh=0
        lmerb=0
        Ys=1
        Y1 = 0
        d1b = d1 * A * NUTIL / LY
        d2b = d2 * A * NUTIL / LY
        Ys2 = Ys * A * NUTIL / LY
        Y2 = -dzon * A * NUTIL / LY
        lmerh2 = A * lmerh * MUTIL / LX
        lmerb2 = A * lmerb * MUTIL / LX

        # Tableaux temporaires utiles
        x = dot(transpose([range(1, MUTIL + 1)]), one_n) - M
        y = dot(one_m, [range(1, NUTIL + 1)]) - N
        
        # Ajout de l'anomalie tripolaire
        thh = A2 * self._fquad(x - lmerh2, y - Y2, d2b, exh, phih)
        thh += - 0.5 * A2 * \
            self._fquad(x - lmerh2, y - Y2 - Ys2, d2b, exh, phih)
        thh += - 0.5 * A2 * \
            self._fquad(x - lmerh2, y - Y2 + Ys2, d2b, exh, phih)

        thb = A1 * self._fquad(x - lmerb2, y - Y1, d1b, exb, phib)
        thb += - 0.5 * A1 * \
            self._fquad(x - lmerb2, y - Y1 - Ys2, d1b, exb, phib)
        thb += - 0.5 * A1 * \
            self._fquad(x - lmerb2, y - Y1 + Ys2, d1b, exb, phib)

        return thh, thb

    def _fquad(self, x, y, d, ex, ph):
        """
        Retourne la perturbation initiale en température potentielle. Fonction
        de base pour les anomalies tripolaires de Wernli. Le champ créé est 
        physique.

        Parameters
        ----------
        x : ndarray, float
            Tableau à 2 dimensions des composantes en x.
        y : ndarray, float
            Tableau à 2 dimensions des composantes en y.
        d : scalar
            Paramètre de la perturbation.
        ex : scalar
            Paramètre de la perturbation.
        ph : scalar
            Paramètre de la perturbation.
            
        Returns
        -------
        perturb : ndarray, complex
            Tableau à 2 dimensions de la perturbation en température.
        """
        perturb = (((x * cos(ph) + y * sin(ph)) / d) ** 2) / (1 - ex ** 2)
        perturb += 1 + ((-x * sin(ph) + y * cos(ph)) / d) ** 2
        perturb = perturb ** (-1.5)

        return perturb

    def _grad_th_gauss(self):
        """
        Retourne l'état de base barocline de forme Gaussienne. Les fonctions
        sont antisymétriques par construction, centrées sur
        xm = M + 1 / 2, et périodiques jusqu'à la derivée troisième.

        Returns
        -------
        thh : ndarray, complex
            Tableau à 2 dimensions de température potentielle horizontale à la
            tropopause.
        thb : ndarray, complex
            Tableau à 2 dimensions de température potentielle horizontale au
            sol.
        """
        # Variables locales
        x = array(range(1 - M, M + 1), float)
        x = A / M * sqrt(1 / ALPHA - 1) * x  # Centré sur M

        # Instructions
        # Caractéristiques du courant-jet
        thh = (U * TH0 * NO / G) * \
            ((0.5 * x / ((1 + NO / (A * F) * H) ** 2 + (x / A) ** 2) +
              0.5 * x / ((1 - NO / (A * F) * H) ** 2 + (x / A) ** 2) -
              ALPHA * x) / A)
        thh = transpose([thh])
        thh = dot(thh, one_n)

        thb = (U * TH0 * NO / G) * \
            ((0.5 * x / (1 + (x / A) ** 2) + 0.5 *
              x / (1 + (x / A) ** 2) - ALPHA * x) / A)
        thb = transpose([thb])
        thb = dot(thb, one_n)

        return thh, thb
