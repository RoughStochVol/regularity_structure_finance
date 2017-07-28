#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ------------------------------------------------------------------------------
# ROUGH VOL FRAMEWORK

# ------------------------------------------------------------------------------
# IMPORTS

import numpy as np
from math import sqrt, log, e, pi, isnan
import blackscholes as BS
    
# ------------------------------------------------------------------------------
# CLASSES

class IV:
    """
    This class computes Monte Carlo approximations to the two-dimensional
    random variable (I,V), the crucial ingredient to pricing European options
    under rough volatility.

    Parameters
    ----------
    
    :param haar_level: Level of fineness of Haar grid (Haar terms: 2^N)
    :param time_steps: number of discretization steps (for trapezoidal rule)
    :param hurst_index: hurst index of the fractional brownian motion
    :param f0: functional applied to integrand (must be broadcastable)
    :param f1: derivative of functional (broadcastable)

    :type haar_level: float
    :type time_steps: int
    :type hurst_index: float
    :type f0: function object
    :type f1: function object

    See docstrings of methods for usage.
    """

    def __init__(self, haar_level, time_steps, hurst_index, f0, f1):

        # Setting parameters

        self.N = haar_level
        self.D = time_steps
        self.H = hurst_index
        self.f0 = f0
        self.f1 = f1

        # Computes the total number of KL style terms.
        self.terms = 2**self.N

        # Computes the Haar grid for [0,1].
        self.haargrid = [k * 2**(-self.N) for k in range(self.terms + 1)]

        # Initialisation of discretization grid.
        self.discgrid = np.linspace(0, 1, self.D + 1)

        # Store basis evaluations of White Noise and fBM on grid.
        self.eval_wn_basis()
        self.eval_fbm_basis()

        self.eval_renormalisation()

        self.const_renorm = sqrt(2*self.H)/((self.H+0.5)*(self.H+1.5)) * \
                            2**(-self.N*(self.H-0.5))    

    def eval_wn_basis(self):
        """
        Creates White Noise basis vectors and then evaluates all basis
        functions on discretization grid.

        Output:
        -------
        self.basis_wn_values:     numpy array(self.terms, D+1)
        """

        # Creates list of White Noise basis functions e.

        e = []
        norm_factor = 2**(self.N/2)

        # Attention: Late binding function closure.
        for k in range(self.terms):

            def fct(t, k = k):

                indicator = (t >= self.haargrid[k]) * \
                            (t < self.haargrid[k + 1])

                return norm_factor * indicator

            e.append(fct)

        # Evaluate all basis functions on discretisation grid.

        self.wn_basis_values = np.array([e[i](self.discgrid) 
                                        for i in range(self.terms)])

    def eval_fbm_basis(self):
        """
        Creates fBM basis vectors and then evaluates all basis
        functions on discretization grid.

        Output:
        -------
        self.fbm_basis_values:     numpy array(self.terms, D+1)
        """

        # Creates list of fBM basis functions e_hat.

        C = 2**(self.N/2) * sqrt(2 * self.H)/(self.H + 1/2)
      
        self.e_hat = []

        # Attention: Late binding function closure.
        for k in range(self.terms):

            def fct(t, k = k):

                # Attention: If t < self.haargrid[k], then term1 dtype complex.
                # Thus value = 0 of dtype complex. To prevent that,
                # set term1 = 0 in case diff is < 0.

                term1 = np.maximum((t - self.haargrid[k]), 0)**(self.H + 1/2)             

                term2 = (t - np.minimum(self.haargrid[k+1], t))**(self.H + 1/2)

                value = C * (t >= self.haargrid[k]) * (term1 - term2)

                return value

            self.e_hat.append(fct)

        # Evaluate all basis functions on discretization grid.

        self.fbm_basis_values = np.array([self.e_hat[i](self.discgrid) 
                                         for i in range(self.terms)])

    def eval_renormalisation(self):

        kappa = 2**(self.N) * sqrt(2*self.H)/(self.H + 1/2)

        def get_renormalisation(t):

            result = kappa * abs(t - np.floor(t * 2**(self.N)) * 
                                2**(-self.N))**(self.H + 1/2)

            return result

        self.renormalisation = get_renormalisation(self.discgrid)

    def compute_I(self, normals):
        """
        Computes approximations of $I$ via the composite trapezoidal 
        rule on a grid with $D$ steps.

        Input
        -----

        :param normals: iid normals
        :type normals: numpy array of dim (nb_samples, # Haar-terms)

        :return: Monte Carlo samples of approximations to $I$
        :rtype: numpy array of dim (nb_samples,)

        """

        nb_samples = normals.shape[0]

        # (1) Compute values of fBM on discretized grid.

        fbm_values = np.dot(normals, self.fbm_basis_values)

        # (2) Compute values of White Noise on discretized grid.

        wn_values = np.dot(normals, self.wn_basis_values)

        # (3) Compute approximation to $I$ via trapezoidal rule.

        f_values = self.f0(fbm_values)

        integrand = f_values * wn_values

        del(f_values)
        del(wn_values)

        int1 = np.trapz(integrand, self.discgrid, axis=1)

        del(integrand)

        # Computation of second integral via trapezoidal rule.

        fprime_values = self.f1(fbm_values)

        del(fbm_values)

        # renormalisation = np.tile(self.renormalisation, (nb_samples,1))

        # integrand = fprime_values * renormalisation

        integrand = self.const_renorm * fprime_values

        del(fprime_values)
        # del(renormalisation)

        int2 = np.trapz(integrand, self.discgrid, axis=1)

        del(integrand)

        # Putting things together to get approximations to $I$.

        result = int1 - int2

        return result


    def compute_IV(self, normals):
        """
        Computes approximations of $(I,V)$ via the composite trapezoidal 
        rule on a grid with $D$ steps.

        Input
        -----

        :param normals: iid normals
        :type normals: numpy array of dim (nb_samples, # Haar-terms)

        :return: Monte Carlo samples of approximations to $I$
        :rtype: numpy array of dim (nb_samples,)

        """

        nb_samples = normals.shape[0]

        # (1) Compute values of fBM on discretized grid.

        fbm_values = np.dot(normals, self.fbm_basis_values)

        # (2) Compute values of White Noise on discretized grid.

        wn_values = np.dot(normals, self.wn_basis_values)

        # (3) Compute approximation to $(I,V)$ via trapezoidal rule.

        f_values = self.f0(fbm_values)

        f_values_sq = f_values**2

        integrand = f_values * wn_values

        del(f_values)
        del(wn_values)

        int1 = np.trapz(integrand, self.discgrid, axis=1)

        del(integrand)

        fprime_values = self.f1(fbm_values)

        del(fbm_values)

        renormalisation = np.tile(self.renormalisation, (nb_samples,1))

        integrand = fprime_values * renormalisation

        del(fprime_values)
        del(renormalisation)

        int2 = np.trapz(integrand, self.discgrid, axis=1)

        del(integrand)

        I = int1 - int2

        del(int1)
        del(int2)

        # (4) Compute approximation to $V$ via trapezoidal rule.

        V = np.trapz(f_values_sq, self.discgrid, axis=1)

        return I, V


# ------------------------------------------------------------------------------
# DEFINITIONS

class Pricer:
    """
    Computes the price of a European call option under a rough volatility model 
    of the form

    !!INSERT!!

    *NOTE* 
    For convenenience, 'time_to_maturity' and 'risk-free rate' have been fixed
    to $T=1$ and $r=0$ respectively (as discussed in paper).

    Parameters
    ----------

    :param spot_price: spot price of the underlying
    :param strike: strike price
    :param hurst_index: hurst parameter of the fractional brownian motion
    :param spot_vol: spot volatility
    :param vvol: volatility of volatility
    :param correlation: correlation between driving noises
    :param nb_paths: number of Monte Carlo paths
    :param haar_level: Haar parameter N such that epsilon = 2**(-N)
    :param time_steps: number of timesteps used for trapezoidal rule
    
    :type spot_price: float
    :type strike: float
    :type hurst_index: float
    :type spot_vol: float
    :type vvol: float
    :type correlation: float
    :type nb_paths: int
    :type haar_level: int
    :type time_steps: int
    """

    def __init__(self, spot_price, strike, hurst_index, spot_vol, vvol, 
                 correlation, haar_level, time_steps):

        self.S = spot_price
        self.K = strike
        self.H = hurst_index
        self.v0 = spot_vol
        self.eta = vvol
        self.rho = correlation
        self.N = haar_level
        self.time_steps = time_steps

        self.rho_bar = sqrt(1 - self.rho**2)

        # Define model specific function $f0$ and derivative $f1$

        f0 = lambda x: self.v0 * np.exp(x)
        f1 = lambda x: self.v0 * np.exp(x)
   
        # Compute Monte Carlo realisations of (I,V) object

        self.IVobject = IV(self.N, self.time_steps, self.H, f0, f1)

    def compute(self, nb_paths, normals=None):
        """
        :return: prices
        :rtype: numpy array

        :return: mean
        :rtype: float

        :return: std_mean
        :rtype: float
        """

        if normals is None:

            normals = np.random.randn(nb_paths,2**self.N)

        I, V = self.IVobject.compute_IV(normals)

        # Compute European option price using Romano-Touzi trick

        new_spot = self.S * np.exp(self.rho * I - 0.5 * self.rho**2 * V)

        new_vol = self.rho_bar**2 * V

        prices = BS.pricer('c', new_spot, self.K, 1, new_vol, risk_free_rate=0)

        mean, std_mean = np.mean(prices), np.std(prices)/sqrt(nb_paths)

        return prices, mean, std_mean

