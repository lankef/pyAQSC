import unittest
import numpy as np
from aqsc import *

import jax.numpy as jnp

# The numerical derivatives usually aren't as accurate 
# as to go below the threshold of np.isclose().
solver_tolerance = 5e-4
def is_roughly_close(array_a, array_b):
    return(jnp.abs(array_a-array_b)<solver_tolerance)

class TestSolver(unittest.TestCase):

    def test_solve_1d(self):
        rand_i = np.random.randint(1,5, size=12)
        test_splines = rand_splines(10)

        test_p = -1 
        test_y = ChiPhiFunc_from_splines(splines=test_splines, nfp=1).content[0]
        test_yp = ChiPhiFunc_from_splines(splines=test_splines, nfp=1, dphi_order=1).content[0]
        test_f = test_yp+test_p*test_y
        test_y_guess = solve_1d_fft(p_eff=test_p, f_eff=test_f, static_max_freq=15)
        # Depreciated
        # test_y_guess_asym = solve_1d_asym(p_eff=test_p, f_eff=test_f)
        print('Testing spectral ODE solver (tolerance:', solver_tolerance, ')')
        print_fractional_error(test_y, test_y_guess)
        # print_fractional_error(test_y.content, test_y_guess_asym.content)
        if not jnp.all(is_roughly_close(test_y, test_y_guess)):
            plt.plot(test_y, label='answer')
            plt.plot(test_y_guess, label = 'spectral solution')
            plt.show()
            plt.plot(test_y-test_y_guess, label='error')
            plt.show()
        self.assertTrue(jnp.all(is_roughly_close(test_y, test_y_guess)))
        
    def test_batch_solve(self):
        
        test_splines = rand_splines(10)
        iota = np.random.random()+1

        test_y = ChiPhiFunc_from_splines(splines=test_splines, nfp=1)
        test_RHS = test_y.dphi() + iota*test_y.dchi()
        guess_y_content = solve_dphi_iota_dchi(iota, test_RHS.content, 100)

        print('Testing PDE solver for dphi y + iota dchi y = f (tolerance:', solver_tolerance, ')')
        print_fractional_error(test_y.content, guess_y_content)
        self.assertTrue(jnp.all(is_roughly_close(test_y.content, guess_y_content)))
        
if __name__ == '__main__':
    unittest.main()