from aqsc import * 
import unittest
import pathlib
import numpy as np
import jax.numpy as jnp
import jax.config
jax.config.update("jax_enable_x64", True)



class TestChiPhiEpsFunc(unittest.TestCase):

    def test_op(self):

        # Loading test data
        test_dir = str(pathlib.Path(__file__).parent.resolve())
        B_psi_coef_cp, B_theta_coef_cp, \
            Delta_coef_cp, p_perp_coef_cp,\
            X_coef_cp, Y_coef_cp, Z_coef_cp, \
            iota_coef, dl_p,\
            nfp, Xi_0, eta, \
            B_denom_coef_c, B_alpha_coef, \
            kap_p, tau_p = read_first_three_orders(
                test_dir+'/circ/', 
                R_array=[2,0,1,2,0.0001,0],
                Z_array=[1,2,0,0.001]
            )

        random_psi = np.random.random(10)
        random_chi = np.random.random(10)
        random_phi = np.random.random(10)

        dict_self = {
            'a ChiPhiEpsFunc': B_psi_coef_cp,
            'an even ChiPhiEpsFunc': iota_coef
        }
        dict_other = {
            'another scalar': np.random.random(),
            'zero': ChiPhiFuncSpecial(0),
            'another ChiPhiFunc': Y_coef_cp[2],
            'another ChiPhiEpsFunc': Y_coef_cp,
            'another even ChiPhiEpsFunc': iota_coef,
        }

        tol = 1e-8
        def iscloseenough(a, b):
            diff = jnp.abs(a-b)
            print('Max error:', jnp.max(diff))
            return(jnp.where(diff<=tol, True, False))
        for key_a in dict_self.keys():
            for key_b in dict_other.keys():
                print('Testing + - * between', key_a, 'and', key_b)
                self_test = dict_self[key_a]
                other = dict_other[key_b]
                if np.isscalar(other): 
                    other_eval = other
                elif type(other) is ChiPhiFunc:
                    other_eval = other.eval(random_chi, random_phi)
                else:
                    other_eval = other.eval(random_psi, random_chi, random_phi)
                self_eval = self_test.eval(random_psi, random_chi, random_phi)
                self.assertTrue(jnp.all(iscloseenough(self_eval + other_eval, (self_test + other).eval(random_psi, random_chi, random_phi))))
                self.assertTrue(jnp.all(iscloseenough(self_eval + other_eval, (other + self_test).eval(random_psi, random_chi, random_phi))))
                self.assertTrue(jnp.all(iscloseenough(self_eval - other_eval, (self_test - other).eval(random_psi, random_chi, random_phi))))
                self.assertTrue(jnp.all(iscloseenough(other_eval - self_eval, (other - self_test).eval(random_psi, random_chi, random_phi))))
                self.assertTrue(jnp.all(iscloseenough(self_eval * other_eval, (self_test * other).eval(random_psi, random_chi, random_phi))))
                self.assertTrue(jnp.all(iscloseenough(self_eval * other_eval, (other * self_test).eval(random_psi, random_chi, random_phi))))

if __name__ == '__main__':
    unittest.main()