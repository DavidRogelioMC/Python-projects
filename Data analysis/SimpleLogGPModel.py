import numpy as np
from matplotlib import pyplot as plt 
from scipy.stats import norm
from sklearn.gaussian_process.kernels import RBF
from scipy.optimize import minimize
from scipy.stats import qmc
from scipy.special import erfinv
from scipy.stats.qmc import MultivariateNormalQMC
from scipy.stats import skew
from scipy.stats import bernoulli


class PZErrorRBFModel(object): 
    from sklearn.gaussian_process.kernels import RBF
    def __init__(self, nz, z_grid, std_mean_goal, num_samp=2**15): 
        """
        nz: numpy array, Sample Redshift Distribution histogram heights
        z_grid: numpy array, Midpoints corresponding to the Sample Redshift Distribuiton historam heights
        num_samp: integer, Number of samples to generate during method of moments fitting, should be power of 2
        std_mean_goal: double, Standard deviation on the mean of the sample redshift distribution log-gp process to be fitted
        
        """
        self.fid_pz = nz
        self.midpoints = z_grid 
        self.num_samp = num_samp
        self.std_mean_goal = std_mean_goal

    def fit_log_gp_model(self, par_ini=np.sqrt(np.array([0.02, 0.05]))):
    
        res = minimize(self._cost, par_ini, method='Nelder-Mead')
        return res
        
    def quasi_random_sample(self, par, num_samp): 
        """ Using Quasi Random Sampler --> num_samp should be power of 2
        It's QUASI RANDOM sampling using Sobol squences, good for fitting method of moments but under no 
        circumstances to be used in an MCMC!!! 
        
        """
        kernel = par[0]**2 * RBF(par[1]**2)
        cov = kernel(np.column_stack((self.midpoints, self.midpoints)))
        log_gp = MultivariateNormalQMC(mean = np.log(self.fid_pz+np.nextafter(np.float32(0), np.float32(1))), cov=cov)
        samples = np.exp(log_gp.random(num_samp))
        samples = np.array([el/np.trapz(el, self.midpoints) for el in samples])
        return samples
    
    def _cost(self, par):
        samples = self.quasi_random_sample(par, self.num_samp)
        mean_list = np.array([np.trapz(el*self.midpoints, self.midpoints) for el in samples])
        res = (np.std(mean_list) - self.std_mean_goal)**2 
    
        return res


#data = np.random.normal(1.0, 0.5, size=10000) 

#grid = np.linspace(0.1, 3.0, 50)
#midpoints = grid[:-1] + (grid[1] - grid[0])/0.5
#pdf_true = np.histogram(data, grid)[0]
#pz = pdf_true/np.trapz(pdf_true, midpoints)

#model = PZErrorRBFModel(pz, midpoints, 0.05, num_samp=2**10)

#res = model.fit_log_gp_model()

#print('Result: ')
#print(res)
#print(sum(pz))


#qrand_sample = model.quasi_random_sample(res.x, 2**10)

#plt.fill_between(midpoints, np.percentile(qrand_sample, 10, axis=0), np.percentile(qrand_sample, 90, axis=0))
#print(sum(qrand_sample))