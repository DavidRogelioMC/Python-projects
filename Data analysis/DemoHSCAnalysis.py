import numpy as np 
import pickle
import pandas as pd
from astropy.table import Table
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import multivariate_normal
from scipy.stats import rv_histogram
from scipy.stats import lognorm
from scipy.optimize import minimize
from typing import Callable
#from matplotlib import pyplot as plt

class EllipticalSliceSampler:       
    def __init__(self, prior_mean: np.ndarray,
                 prior_cov: np.ndarray,
                 loglik: Callable):

        self.prior_mean = prior_mean
        self.prior_cov = prior_cov

        self.loglik = loglik

        self._n = len(prior_mean)  # dimensionality
        self._chol = np.linalg.cholesky(prior_cov)  # cache cholesky

        # init state; cache prev states
        self._state_f = self._chol @ np.random.randn(self._n) + prior_mean

    def _indiv_sample(self):
        """main algo for indiv samples"""
        f = self._state_f  # previous cached state
        nu = self._chol @ np.random.randn(self._n)  # choose ellipse using prior
        log_y = self.loglik(f) + np.log(np.random.uniform())  # ll threshold
        
        theta = np.random.uniform(0., 2*np.pi)  # initial proposal
        theta_min, theta_max = theta-2*np.pi, theta  # define bracket

        # main loop:  accept sample on bracket, else shrink bracket and try again
        while True:  
            assert theta != 0
            f_prime = (f - self.prior_mean)*np.cos(theta) + nu*np.sin(theta)
            f_prime += self.prior_mean
            if self.loglik(f_prime) > log_y:  # accept
                self._state_f = f_prime
                return
            
            else:  # shrink bracket and try new point
                if theta < 0:
                    theta_min = theta
                else:
                    theta_max = theta
                theta = np.random.uniform(theta_min, theta_max)

    def sample(self,
               n_samples: int,
               n_burn: int = 500) -> np.ndarray:
        """Returns n_samples samples"""
        
        samples = []
        for i in range(n_samples):
            self._indiv_sample()
            if i > n_burn:
                samples.append(self._state_f.copy())

        return np.stack(samples)