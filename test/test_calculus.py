import unittest
import numpy as np
from aqsc import *

import jax.numpy as jnp

# The phi derivative tests are done with splines, for which the 
# specral method is not very well-suited.
diff_tolerance = 5e-4

# Size of the chi and phi grid used for evaluation tests
points = np.linspace(0, 2*np.pi*(1-1/n_grid_phi), n_grid_phi)
chi = np.linspace(0, 2*np.pi*(1-1/n_grid_chi), n_grid_chi)
phi = points
psi = np.linspace(0,5,100)

def is_roughly_close(array_a, array_b):
    return(jnp.abs(array_a-array_b)<diff_tolerance)

class TestCalculus(unittest.TestCase):

    def test_dchi_ichi(self):
        ''' 
        chi derivatives and antiderivatives 
        '''
        # max chi mode number is even
        nfp = np.random.randint(4)+2
        rands_i = np.random.randint(1,5, size=12)
        amp=0.2
        test1 = amp*ChiPhiFunc(np.array([
                np.sin(rands_i[0]*points)+np.cos(rands_i[1]*points), 
                np.sin(rands_i[0]*points)+np.cos(rands_i[1]*points),
                np.sin(rands_i[2]*points)+np.cos(rands_i[3]*points)
        ]), nfp, trig_mode = True)

        test1_int = amp*ChiPhiFunc(np.array([
                np.sin(rands_i[0]*points)+np.cos(rands_i[1]*points), 
                np.zeros_like(points, dtype=np.complex128),
                np.sin(rands_i[2]*points)+np.cos(rands_i[3]*points)
        ]), nfp, trig_mode = True)

        dchi_test1 = amp*ChiPhiFunc(np.array([
                -2*(np.sin(rands_i[2]*points)+np.cos(rands_i[3]*points)), 
                np.zeros_like(points, dtype=np.complex128),
                2*(np.sin(rands_i[0]*points)+np.cos(rands_i[1]*points))
        ]), nfp, trig_mode = True)

        guess_dchi = test1.dchi()
        guess_ichi = dchi_test1.antid_chi()
        # chi derivative
        self.assertTrue(jnp.all(jnp.isclose(guess_dchi.content, dchi_test1.content)))
        # chi integral
        self.assertTrue(jnp.all(jnp.isclose(guess_ichi.content, test1_int.content)))
        
        # max chi mode number is odd
        nfp = np.random.randint(4)+2
        rands_i = np.random.randint(1,5, size=12)

        test1 = ChiPhiFunc(np.array([
                np.sin(rands_i[0]*points)+np.cos(rands_i[1]*points), 
                np.sin(rands_i[2]*points)+np.cos(rands_i[3]*points), 
                np.sin(rands_i[4]*points)+np.cos(rands_i[5]*points), 
                np.sin(rands_i[6]*points)+np.cos(rands_i[7]*points)
        ]), nfp, trig_mode = True)

        dchi_test1 = ChiPhiFunc(np.array([
                -3*(np.sin(rands_i[6]*points)+np.cos(rands_i[7]*points)), 
                -(np.sin(rands_i[4]*points)+np.cos(rands_i[5]*points)), 
                np.sin(rands_i[2]*points)+np.cos(rands_i[3]*points), 
                3*(np.sin(rands_i[0]*points)+np.cos(rands_i[1]*points))
        ]), nfp, trig_mode = True)

        guess_dchi = test1.dchi()
        guess_ichi = dchi_test1.antid_chi()
        self.assertTrue(jnp.all(jnp.isclose(guess_dchi.content, dchi_test1.content)))
        # chi integral
        self.assertTrue(jnp.all(jnp.isclose(guess_ichi.content, test1.content)))
        
        print('Testing chi derivative')
        print_fractional_error(guess_dchi.content, dchi_test1.content)
        print('Testing chi integral')
        print_fractional_error(guess_ichi.content, test1.content)

    def test_dphi_fft(self):
        '''
        phi derivatives
        '''
        nfp = np.random.randint(4)+2
        test_splines = rand_splines(10)

        test_diff = ChiPhiFunc_from_splines(splines=test_splines, nfp=nfp)
        ans = ChiPhiFunc_from_splines(splines=test_splines, nfp=nfp, dphi_order=1)

        print('Testing dphi based on FFT (tolerance:', diff_tolerance, ')')
        guess_fft = test_diff.dphi(mode=1)
        print_fractional_error(guess_fft.content, ans.content)
        self.assertTrue(jnp.all(is_roughly_close(guess_fft.content, ans.content)))
    
#     def test_dphi_pseudo_spectral(self):
#         '''
#         phi derivatives
#         '''
#         nfp = np.random.randint(4)+2
#         test_splines = rand_splines(10)

#         test_diff = ChiPhiFunc_from_splines(splines=test_splines, nfp=nfp)
#         ans = ChiPhiFunc_from_splines(splines=test_splines, nfp=nfp, dphi_order=1)
#         print('Testing dphi based on pseudo-spectral (tolerance:', diff_tolerance, ')')
#         guess_pseudo_spectral = test_diff.dphi(mode=2)
#         print_fractional_error(guess_pseudo_spectral.content, ans.content)
#         self.assertTrue(jnp.all(is_roughly_close(guess_pseudo_spectral.content, ans.content)))
    
    def test_phi_spectral_integral(self):
        ''' 
        phi integral 
        '''
        print('Testing spectral phi integral')
        amp=0.2
        nfp = np.random.randint(4)+2

        rands_i = np.random.randint(1,10, size=12)
        test_integral = amp*ChiPhiFunc(np.array([
                np.sin(rands_i[0]*points)+np.cos(rands_i[1]*points), 
                np.sin(rands_i[2]*points)+np.cos(rands_i[3]*points), 
                np.sin(rands_i[4]*points)+np.cos(rands_i[5]*points)
        ]),nfp, trig_mode = True)

        ans = amp*ChiPhiFunc(np.array([
                -1/rands_i[0]*np.cos(rands_i[0]*points) + 1/rands_i[1]*np.sin(rands_i[1]*points) +1/rands_i[0],
                -1/rands_i[2]*np.cos(rands_i[2]*points) + 1/rands_i[3]*np.sin(rands_i[3]*points) +1/rands_i[2],
                -1/rands_i[4]*np.cos(rands_i[4]*points) + 1/rands_i[5]*np.sin(rands_i[5]*points) +1/rands_i[4]
        ]), nfp, trig_mode = True)/nfp

        guess_fft = test_integral.integrate_phi_fft(zero_avg=False)
        print_fractional_error(guess_fft.content, ans.content)
        self.assertTrue(jnp.all(jnp.isclose(guess_fft.content, ans.content)))

    def test_diff_function(self):
        '''
        Testing the function diff() used for parsing
        '''
        # dphi mode
        nfp = np.random.randint(4)+2
        rands_i = np.random.randint(20,22, size=12)
        test_diff = ChiPhiFunc(np.array([
                np.cos(rands_i[0]*points) + np.sin(rands_i[1]*points) +1/rands_i[0]*np.cos(0),
                np.cos(rands_i[2]*points) + np.sin(rands_i[3]*points) +1/rands_i[2]*np.cos(0)
        ]), nfp, trig_mode = True)

        ans = ChiPhiFunc(np.array([
                -rands_i[0]*np.sin(rands_i[0]*points)+rands_i[1]*np.cos(rands_i[1]*points), 
                -rands_i[2]*np.sin(rands_i[2]*points)+rands_i[3]*np.cos(rands_i[3]*points)
        ]), nfp, trig_mode = True)*nfp

        ans2 = ChiPhiFunc(np.array([
                -rands_i[0]*rands_i[0]*np.cos(rands_i[0]*points)-rands_i[1]*rands_i[1]*np.sin(rands_i[1]*points), 
                -rands_i[2]*rands_i[2]*np.cos(rands_i[2]*points)-rands_i[3]*rands_i[3]*np.sin(rands_i[3]*points)
        ]), nfp, trig_mode = True)*nfp*nfp

        guess = diff(test_diff,False,1) # Dphi. The boolean is is_dchi
        guess2 = diff(test_diff,False,2) # Dphi. The boolean is is_dchi
        
        # Testing the function diff(). It is dphi() and dchi() in the backend
        # so the accuracy is not displayed again.
        self.assertTrue(jnp.all(is_roughly_close(guess.content, ans.content)))
        self.assertTrue(jnp.all(is_roughly_close(guess2.content, ans2.content)))

        # dchi mode
        rands_i = np.random.randint(1,5, size=12)
        a = np.sin(rands_i[0]*points)+np.cos(rands_i[1]*points) 
        b = np.sin(rands_i[2]*points)+np.cos(rands_i[3]*points)
        c = np.sin(rands_i[4]*points)+np.cos(rands_i[5]*points) 
        d = np.sin(rands_i[6]*points)+np.cos(rands_i[7]*points) 
        e = np.sin(rands_i[8]*points)+np.cos(rands_i[9]*points)
        test_diff = ChiPhiFunc(np.array([a,b,c,d,e]), nfp, trig_mode = True)

        ans = ChiPhiFunc(np.array([-4*e,-2*d,np.zeros_like(c),2*b,4*a]), nfp, trig_mode = True)
        # guess = test_diff.dchi(1)
        guess = diff_backend(test_diff,True,1) # Dphi. The boolean is is_dchi
        # guess = diff(test_diff,True,1) # Dphi. The boolean is is_dchi
        print('dchi')
        self.assertTrue(jnp.all(is_roughly_close(guess.content, ans.content)))
        
if __name__ == '__main__':
    unittest.main()