# -*- coding: utf-8 -*-
"""
Modèle 4.0 quasi-géostrophique à tourbillon potentiel uniforme
Portage du modèle Scilab de Matthieu Plu (GMAP/RECYF, Nov 2005)
Septembre 2014 - Janvier 2015
Créé le 24 septembre 2014
@author: Clément Blot (IENM3)
@author: Guillem Coquelet (IENM3)

Le modèle. Calcul l'évolution des champs initiaux.
"""
from config import DT, EPSILON, F, G, H, inv_hnm, LX, LY, LM, LN, M, masque, \
    mat_m, mat_n, MUTIL, N, NU, NUTIL, one_m, one_n, TH0, VMAX
from numpy import cosh, conjugate, dot, pi, sinh, tanh, zeros
#from numpy.fft.fftpack import fftn, ifftn
from numpy.fft import fftn, ifftn
import sys


class Modele:

    """
    Calcul l'évolution des champs initiaux.
    """
    nrun = 0

    def __init__(self):
        """
        Initialisation du modèle.

        Examples
        --------
        >>> ma_variable_modele = Modele()
        """        
        # Initialisation des variables utiles aux calculs. Ces calculs
        # préliminaires permettant d'optimiser le temps de calcul.

        # Variables pour "run".
        self.var_init = NU * (dot(mat_m, one_n) ** 2 + dot(one_m, mat_n) ** 2)

        # Variables pour "_linh" et "_linb".
        self.var_lin1 = ((2 * pi * 1j * VMAX) / LY) * dot(one_m, mat_n)
        self.var_lin2 = (1 / H) / inv_hnm
        self.var_lin3 = 1 / tanh(inv_hnm * H)
        self.var_lin4 = dot(one_m, one_n) / sinh(inv_hnm * H)

        # Variables pour "_nlinh" et "_nlinb".
        self.var_nlin1 = (-2 * pi * G) / (LY * F * TH0) * 1j * \
            (dot(one_m, mat_n) / (inv_hnm * sinh(inv_hnm * H)))
        self.var_nlin2 = (2 * pi * G) / (LX * F * TH0) * 1j * \
            (dot(mat_m, one_n) / (inv_hnm * sinh(inv_hnm * H)))
        self.var_nlin3 = cosh(H * inv_hnm)
        self.var_nlin4 = 2 * pi * 1j / LX * dot(mat_m, one_n)
        self.var_nlin5 = 2 * pi * 1j / LY * dot(one_m, mat_n)

        # Variables pour "prod_tf". On doit multiplier les coefficients de la
        # matrice spectrale par r quand on augmente la taille de la matrice.
        self.r = (float(LM) / float(MUTIL)) * (float(LN) / float(NUTIL))

    def run(self, thhi, thbi, steps):
        """
        Fonction principale, évolution des champs spectraux de température
        potentielle après n pas de temps DT.

        Parameters
        ----------
        thhi : ndarray, complex
            Tableau à 2 dimensions de températures potentielles horizontales
            physiques à la tropopause, initiale.
        thbi : ndarray, complex
            Tableau à 2 dimensions de températures potentielles horizontales
            physiques au sol, initiale.
        steps : ndarray, scalar
            Pas de temps des sorties.

        Returns
        -------
        thh : ndarray, complex
            Tableau à 2 dimensions de températures potentielles horizontales
            spectrales à la tropopause, apres n pas de temps DT.
        thbi : ndarray, complex
            Tableau à 2 dimensions de températures potentielles horizontales
            spectrales au sol, apres n pas de temps DT.

        Examples
        --------
        >>> thhi, thbi = EtatInitial().gradeady()
        >>> thh, thb = Modele().run(thhi, thbi, steps = [0])
        """
        #  Variables locales
        Modele.nrun += 1 # Numéro du run en cours
        nsteps = len(steps)
        if nsteps > 10:
            sys.exit("Modele : Impossible de garder en mémoire plus de 10 sorties.\n")
        n = steps[nsteps - 1]
        thhf, thbf = [], []

        # Message de départ
        sys.stdout.write("Run " + str(Modele.nrun) + " (" + str(n) + " pas de temps) :\n"
                         "[%-40s] %d%%" % (" ", 0) + "\r")
        sys.stdout.flush()

        # Passage en champs spectraux
        thh2 = fftn(thhi) * masque
        thb2 = fftn(thbi) * masque

        # Premier pas de temps : schema d'Euler
        thh3 = thh2 + DT * \
            (self._linh(thh2, thb2) +
             self._nlinh(thh2, thb2) -
             self.var_init * thh2)
        thb3 = thb2 + DT * \
            (self._linb(thh2, thb2) +
             self._nlinb(thh2, thb2) -
             self.var_init * thb2)

        # Champs spectraux de température potentielle en haut
        thh1 = thh2
        thh2 = thh3
        # Et en en bas
        thb1 = thb2
        thb2 = thb3

        nout = 0
        if nout < nsteps and steps[nout] < 2:
            thhf.append(thh2)
            thbf.append(thb2)
            sys.stdout.write("Sortie " + str(nout + 1) + "/" + str(nsteps) +
                             ", pas de temps " + str(steps[nout]) +
                             "                       \n")
            nout += 1

        # Pas de temps suivants : schéma leap-frog
        t = 2
        while t <= n:
            thh3 = thh1 + 2 * DT * \
                (self._linh(thh2, thb2) +
                 self._nlinh(thh2, thb2) -
                    NU * self.var_init * thh2)
            thb3 = thb1 + 2 * DT * \
                (self._linb(thh2, thb2) +
                 self._nlinb(thh2, thb2) -
                    NU * self.var_init * thb2)

            # Filtre d'Asselin
            thh2 = (1 - 2 * EPSILON) * thh2 + EPSILON * (thh3 + thh1)
            thb2 = (1 - 2 * EPSILON) * thb2 + EPSILON * (thb3 + thb1)

            thh1 = thh2
            thh2 = thh3
            thb1 = thb2
            thb2 = thb3

            # Barre de progression
            k = float(t - 1) / float(n)
            sys.stdout.write("[%-40s] %d%%" %
                             ("-" * int(k * 40), k * 100) + "\r")
            sys.stdout.flush()

            # Sorties intermédiaires
            if nout < nsteps and steps[nout] == t:
                thhf.append(thh2)
                thbf.append(thb2)
                sys.stdout.write("Sortie " + str(nout + 1) + "/" +
                                 str(nsteps) + ", pas de temps " +
                                 str(steps[nout]) +
                                 "                       \n")
                nout += 1

            # Incrément
            t += 1

        # Message de fin
        sys.stdout.write("[%-40s] %d%%" % ("-" * 40, 100) + "\r"
                         "Fin du run.                                    \n")

        return thhf, thbf

    def _linh(self, thh, thb):
        """
        Fonction de calcul des termes linéaires à la tropopause.

        Parameters
        ----------
        thh : ndarray, complex
            Tableau à 2 dimensions de températures potentielles horizontales
            spectrales à la tropopause.
        thb : ndarray, complex
            Tableau à 2 dimensions de températures potentielles horizontales
            spectrales au sol.

        Returns
        -------
        thlin : ndarray, complex
            Tableau à 2 dimensions de la partie linéaire du second membre de
            l'équation aux dérivées partielles.

        """
        # En sortie : La partie linéaire du second membre de l'EDP
        thlin = self.var_lin1 * \
            (self.var_lin2 * (self.var_lin3 * thh - self.var_lin4 * thb) - thh)

        # Prise en compte du coefficient spectral [0, 0]
        thlin[0, 0] = 0

        return thlin

    def _linb(self, thh, thb):
        """
        Fonction de calcul des termes linéaires au sol.

        Parameters
        ----------
        thh : ndarray, complex
            Tableau à 2 dimensions de températures potentielles horizontales
            spectrales à la tropopause.
        thb : ndarray, complex
            Tableau à 2 dimensions de températures potentielles horizontales
            spectrales au sol.

        Returns
        -------
        thlin : ndarray, complex
            Tableau à 2 dimensions de la partie linéaire du second membre de
            l'équation aux dérivées partielles.

        """
        thlin = -self.var_lin1 * \
            (self.var_lin2 * (self.var_lin3 * thb - self.var_lin4 * thh))

        # Prise en compte du coefficient spectral [0, 0]
        thlin[0, 0] = 0

        return thlin

    def _nlinh(self, thh, thb):
        """
        Fonction de calcul des termes non lineaires à la tropopause.

        Parameters
        ----------
        thh : ndarray, complex
            Tableau à 2 dimensions de températures potentielles horizontales
            spectrales à la tropopause.
        thb : ndarray, complex
            Tableau à 2 dimensions de températures potentielles horizontales
            spectrales au sol.

        Returns
        -------
        thnlin : ndarray, complex
            Tableau à 2 dimensions de la partie non linéaire du second membre de
            l'équation aux dérivées partielles.

        """
        # Variables locales
        # Composantes du vent géostrophique
        ug = self.var_nlin1 * (self.var_nlin3 * thh - thb)
        vg = self.var_nlin2 * (self.var_nlin3 * thh - thb)

        # Derivées spatiales de la température potentielle
        dthx = self.var_nlin4 * thh
        dthy = self.var_nlin5 * thh

        # Instructions
        thnlin = -self.prod_tf(ug, dthx) - self.prod_tf(vg, dthy)

        return thnlin

    def _nlinb(self, thh, thb):
        """
        Fonction de calcul des termes non linéaires au sol.

        Parameters
        ----------
        thh : ndarray, complex
            Tableau à 2 dimensions de températures potentielles horizontales
            spectrales à la tropopause.
        thb : ndarray, complex
            Tableau à 2 dimensions de températures potentielles horizontales
            spectrales au sol.

        Returns
        -------
        thnlin : ndarray, complex
            Tableau à 2 dimensions de la partie non linéaire du second membre de
            l'équation aux dérivées partielles.

        """
        # Variables locales
        # Composantes du vent géostrophique
        ug = self.var_nlin1 * (thh - self.var_nlin3 * thb)
        vg = self.var_nlin2 * (thh - self.var_nlin3 * thb)

        # Derivées spatiales de la température potentielle
        dthx = self.var_nlin4 * thb
        dthy = self.var_nlin5 * thb

        # Instructions
        thnlin = -self.prod_tf(ug, dthx) - self.prod_tf(vg, dthy)

        return thnlin

    def prod_tf(self, ch1, ch2):
        """
        Fonction multiplicative de 2 champs spectraux. Supprime l'aliasing.

        Parameters
        ----------
        ch1 : ndarray, complex
            Tableau à 2 dimensions d'un champ spectral.
        ch2 : ndarray, complex
            Tableau à 2 dimensions d'un champ spectral.

        Returns
        -------
        out : ndarray, complex
            Tableau à 2 dimensions d'un champ spectral.

        """
        # Variables locales
        # Tableaux de stockage des champs initiaux
        cha1 = zeros((LM, LN), complex)
        cha2 = zeros((LM, LN), complex)

        # Instructions
        # Insertion de ch1 dans une grande matrice cha1
        cha1[0:M + 1, 0:N + 1] = ch1[0:M + 1, 0:N + 1] * self.r
        cha1[0:M + 1, LN - N + 1:LN] = ch1[0:M + 1, N + 1:2 * N] * self.r
        cha1[LM - M + 1:LM, 0:N + 1] = ch1[M + 1:2 * M, 0:N + 1] * self.r
        cha1[LM - M + 1:LM,
             LN - N + 1:LN] = ch1[M + 1:2 * M,
                                  N + 1:2 * N] * self.r
        cha1[0, LN - N] = conjugate(cha1[0, N])
        cha1[LM - M, 0] = conjugate(cha1[M, 0])

        # Et ch2 dans cha2
        cha2[0:M + 1, 0:N + 1] = ch2[0:M + 1, 0:N + 1] * self.r
        cha2[0:M + 1, LN - N + 1:LN] = ch2[0:M + 1, N + 1:2 * N] * self.r
        cha2[LM - M + 1:LM, 0:N + 1] = ch2[M + 1:2 * M, 0:N + 1] * self.r
        cha2[LM - M + 1:LM,
             LN - N + 1:LN] = ch2[M + 1:2 * M,
                                  N + 1:2 * N] * self.r
        cha2[0, LN - N] = conjugate(cha2[0, N])
        cha2[LM - M, 0] = conjugate(cha2[M, 0])

        # On fait le produit en repassant en espace physique
        produit_ch = ifftn(cha1) * ifftn(cha2)
        produit_ch = fftn(produit_ch)

        # On redimensionne la matrice
        out = zeros((MUTIL, NUTIL), complex)
        out[0:M + 1, 0:N + 1] = produit_ch[0:M + 1, 0:N + 1] / self.r
        out[0:M + 1, N + 1:2 * N] = produit_ch[0:M + 1, LN - N + 1:LN] / self.r
        out[M + 1:2 * M, 0:N + 1] = produit_ch[LM - M + 1:LM, 0:N + 1] / self.r
        out[M + 1:2 * M,
            N + 1:2 * N] = produit_ch[LM - M + 1:LM,
                                      LN - N + 1:LN] / self.r
        out = masque * out

        return out
