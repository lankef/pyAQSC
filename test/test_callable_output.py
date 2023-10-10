import unittest
from aqsc import *
import jax.numpy as jnp
import numpy as np

# Size of the chi and phi grid used for evaluation tests
points = jnp.linspace(0, 2*np.pi*(1-1/n_grid_phi), n_grid_phi)
chi = points
phi = points
psi = jnp.linspace(0,5,100)

# Generating test objects
content_single_nfp = jnp.array([
    jnp.sin(4*points), # sin component
    jnp.cos(4*points) # cos component
])
content1 = jnp.array([
    jnp.sin(points), # sin component
    jnp.cos(points) # cos component
])
single_period = ChiPhiFunc(content_single_nfp, 1, trig_mode=True)
four_period = ChiPhiFunc(content1, nfp=4, trig_mode=True)

content0 = jnp.array([
    jnp.sin(9*points)
])
content1 = jnp.array([
    jnp.sin(3*points), # sin component
    jnp.cos(4*points) # cos component
])
content2 = jnp.array([
    jnp.cos(6*points), # sin component
    jnp.sin(3*points), # 0 component
    jnp.cos(4*points) # cos component
])
content3 = jnp.array([
    jnp.sin(3*points), # sin component
    jnp.cos(4*points), # sin component
    jnp.sin(3*points), # cos component
    jnp.cos(4*points) # cos component
])
chiphifunc0 = ChiPhiFunc(content0, nfp=4, trig_mode=True)
chiphifunc1 = ChiPhiFunc(content1, nfp=4, trig_mode=True)
chiphifunc2 = ChiPhiFunc(content2, nfp=4, trig_mode=True)
chiphifunc3 = ChiPhiFunc(content3, nfp=4, trig_mode=True)
test_chiphiepsfunc = ChiPhiEpsFunc([
    chiphifunc0, 
    chiphifunc1, 
    chiphifunc2, 
    chiphifunc3
], nfp=4)


# np.gradient is not very accurate, but it allows us to benchmark our
# different implementation
solver_tolerance = 5e-4
def is_roughly_close(array_a, array_b):
    return(jnp.abs(array_a-array_b)<solver_tolerance)

class TestCallables(unittest.TestCase):
    

    def test_chiphifunc_eval(self):
        '''
        Testing function evaluation.
        '''
        lambda_guess = four_period.eval(chi[:, None], phi[None, :])
        lambda_guess2 = single_period.eval(chi[:, None], phi[None, :])
        lambda_ans = jnp.sin(chi[:, None])*jnp.sin(4*points) + jnp.cos(chi[:, None])*jnp.cos(4*points)
        self.assertTrue(jnp.all(jnp.isclose(lambda_guess, lambda_ans)))
        self.assertTrue(jnp.all(jnp.isclose(lambda_guess2, lambda_ans)))

    def test_chiphiepsfunc_square_series_lambda(self):
        '''
        Testing lambda function output for power series like:
        Sum(iota[n]*eps**(2*i), 0, n)
        = Sum(iota[n]*psi**i, 0, n)
        '''
        test_iota = ChiPhiEpsFunc([1.2, 3.6, 0, 5.2], 1)
        lambda_psi_ans = lambda psi: 1.2+3.6*psi+5.2*psi**3

        self.assertTrue(jnp.all(jnp.isclose(
            test_iota.eval(psi, sq_eps_series=True), 
            lambda_psi_ans(psi)
        )))
    
    def test_chiphiepsfunc_lambda(self):
        def lambda_ans(psi, chi, phi):
            phi = phi*4
            eps = jnp.sqrt(psi)
            out = jnp.sin(9*phi)
            
            out += jnp.sin(3*phi)*jnp.sin(chi)*eps
            out += jnp.cos(4*phi)*jnp.cos(chi)*eps
            
            out += jnp.cos(6*phi)*jnp.sin(2*chi)*eps**2
            out += jnp.sin(3*phi)*eps**2
            out += jnp.cos(4*phi)*jnp.cos(2*chi)*eps**2

            out += jnp.sin(3*phi)*jnp.sin(3*chi)*eps**3
            out += jnp.cos(4*phi)*jnp.sin(1*chi)*eps**3
            out += jnp.sin(3*phi)*jnp.cos(1*chi)*eps**3
            out += jnp.cos(4*phi)*jnp.cos(3*chi)*eps**3

            return(out)

        self.assertTrue(jnp.all(jnp.isclose(
            test_chiphiepsfunc.eval(psi[:,None,None], chi[None,:,None], phi[None,None,:]),
            lambda_ans(psi[:,None,None], chi[None,:,None], phi[None,None,:])
        )))

    def test_chiphiepsfunc_diff(self):
        epsilons = jnp.linspace(0,10,1000)
        phis = jnp.linspace(0,2*jnp.pi*0.999,1000)
        X_epsilons = test_chiphiepsfunc.eval(epsilons**2, 0, 0)
        X_chis = test_chiphiepsfunc.eval(1, phis, 0)
        X_phis = test_chiphiepsfunc.eval(1, 0, phis)
        gradient_eps_X = np.gradient(X_epsilons, epsilons)
        gradient_chi_X = np.gradient(X_chis, phis)
        gradient_phi_X = np.gradient(X_phis, phis)
        # First and last elements of jnp.gradient are incorrect.
        jnp.all(is_roughly_close(test_chiphiepsfunc.deps().eval(epsilons**2, 0, 0), gradient_eps_X)[1:-1])
        jnp.all(is_roughly_close(test_chiphiepsfunc.dchi().eval(1, phis, 0), gradient_chi_X)[1:-1])
        jnp.all(is_roughly_close(test_chiphiepsfunc.dphi().eval(1, 0, phis), gradient_phi_X)[1:-1]) # First and last elements of np.gradient are unreliable


if __name__ == '__main__':
    unittest.main()