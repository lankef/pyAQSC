import unittest
import numpy as np
from aqsc import *

import jax.numpy as jnp

orig_list = [0, 1, 3, jnp.array([666,667])] # legal, legal, legal, illegal
a = ChiPhiEpsFunc(orig_list, 2, square_eps_series=False, check_consistency=True) 
a = a.append(ChiPhiFuncSpecial(0)) # legal, legal
a = a.append(ChiPhiFuncSpecial(0)) # legal, legal
a = a.append(66) # legal
a = a.append(ChiPhiFuncSpecial(-10)) # legal
a = a.append(ChiPhiFunc(jnp.array([[1]]),2)) # legal
a = a.append(ChiPhiFunc(jnp.array([[1]]),10)) # illegal

class TestChiPhiEpsFunc(unittest.TestCase):

    def test_logic(self):
        print('Testing append(), zero_append(), __init__(), mask() and consistency check')
        a_final = a.mask(10)
        self.assertTrue(a_final[0]==0)
        self.assertTrue(a_final[1]==1)
        self.assertTrue(a_final[2]==3)
        # Illegal element, not a scalar or ChiPhiFunc
        self.assertTrue(a_final[3].is_special() and a_final[3].nfp==-14) 
        self.assertTrue(a_final[4].is_special() and a_final[4].nfp==0)
        self.assertTrue(a_final[5].is_special() and a_final[5].nfp==0)
        self.assertTrue(a_final[6]==66)
        self.assertTrue(a_final[7].is_special() and a_final[7].nfp==-10)
        self.assertTrue(
            (not a_final[8].is_special()) 
            and a_final[8].content.shape==(1, 1) 
            and a_final[8].nfp==2 )
        # illegal element, mismatched nfp
        self.assertTrue(a_final[9].is_special() and a_final[9].nfp==-14)
        # Masking order larger than available order
        self.assertTrue(a_final[10].is_special() and a_final[10].nfp==0)
        self.assertTrue(a_final.nfp==2)

    def test_to_from_content_list(self):
        print('Testing list conversion used in saving and loading')
        content_list = a.to_content_list()
        b = ChiPhiEpsFunc.from_content_list(content_list, 2)
        b_final = b.mask(10)
        self.assertTrue(b_final[0].is_special() and b_final[0].nfp==0)
        self.assertTrue(b_final[1]==1)
        self.assertTrue(b_final[2]==3)
        # Illegal element, not a scalar or ChiPhiFunc
        self.assertTrue(b_final[3].is_special() and b_final[3].nfp==-14) 
        self.assertTrue(b_final[4].is_special() and b_final[4].nfp==0)
        self.assertTrue(b_final[5].is_special() and b_final[5].nfp==0)
        self.assertTrue(b_final[6]==66)
        self.assertTrue(b_final[7].is_special() and b_final[7].nfp==-10)
        self.assertTrue(
            (not b_final[8].is_special()) 
            and b_final[8].content.shape==(1, 1) 
            and b_final[8].nfp==2 )
        # illegal element, mismatched nfp
        self.assertTrue(b_final[9].is_special() and b_final[9].nfp==-14)
        # Masking order larger than available order
        self.assertTrue(b_final[10].is_special() and b_final[10].nfp==0)
        self.assertTrue(b_final.nfp==2)
    
    def test_eval(self):
        coeff_list = np.random.random(4)
        psi, chi, phi = np.random.random(3)
        series1 = ChiPhiEpsFunc(coeff_list, 2, square_eps_series=False) 
        series2 = ChiPhiEpsFunc(coeff_list, 2, square_eps_series=True) 
        self.assertTrue(jnp.isclose(series1.eval(psi, chi, phi, n_max=0), coeff_list[0]))
        self.assertTrue(jnp.isclose(series2.eval(psi, chi, phi, n_max=0), coeff_list[0]))
        self.assertTrue(jnp.isclose(series1.eval(psi, chi, phi, n_max=1), coeff_list[0] + jnp.sqrt(psi) * coeff_list[1]))
        self.assertTrue(jnp.isclose(
            series1.eval(psi, chi, phi, n_max=2),
            coeff_list[0] + psi**0.5 * coeff_list[1] + psi * coeff_list[2]
        ))
        self.assertTrue(jnp.isclose(
            series1.eval(psi, chi, phi, n_max=3), 
            coeff_list[0] + psi**0.5  * coeff_list[1] + psi * coeff_list[2] + psi**1.5 * coeff_list[3]
        ))
        self.assertTrue(jnp.isclose(
            series1.eval(psi, chi, phi, n_max=jnp.inf), 
            coeff_list[0] + psi**0.5  * coeff_list[1] + psi * coeff_list[2] + psi**1.5 * coeff_list[3]
        ))
        self.assertTrue(jnp.isclose(series2.eval(psi, chi, phi, n_max=1), coeff_list[0] + psi * coeff_list[1]))
        self.assertTrue(jnp.isclose(
            series2.eval(psi, chi, phi, n_max=2), 
            coeff_list[0] + psi * coeff_list[1] + psi**2 * coeff_list[2]
        ))
        self.assertTrue(jnp.isclose(
            series2.eval(psi, chi, phi, n_max=3), 
            coeff_list[0] + psi * coeff_list[1] + psi**2 * coeff_list[2] + psi**3 * coeff_list[3]
        ))
        self.assertTrue(jnp.isclose(
            series2.eval(psi, chi, phi, n_max=jnp.inf), 
            coeff_list[0] + psi * coeff_list[1] + psi**2 * coeff_list[2] + psi**3 * coeff_list[3]
        ))
if __name__ == '__main__':
    unittest.main()