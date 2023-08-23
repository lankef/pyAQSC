import unittest
import numpy as np
import aqsc

import jax.numpy as jnp

n_2_tolerance = 5e-8
n_4_tolerance = 8e-5

class TestCircularAxis(unittest.TestCase):

    def test_governing(self):
        
        equilibrium_init = aqsc.circular_axis()
        equilibrium_new = aqsc.iterate_2(
            equilibrium_init,
            n_eval=4,
            B_alpha_nb2=aqsc.ChiPhiFuncSpecial(0),
            B_denom_nm1=aqsc.ChiPhiFuncSpecial(0), 
            B_denom_n=aqsc.ChiPhiFuncSpecial(0), # B_denom_coef_c[3]
            iota_new=-6.6367278e-01,
            static_max_freq=(20,20),
            traced_max_freq=(-1,-1),
            # Traced.
            # -1 represents no filtering (default). This value is chosen so that
            # turning on or off off-diagonal filtering does not require recompiles.
            max_k_diff_pre_inv=(-1,-1),
        )
        print('Testing order 2, tolerance:', n_2_tolerance)
        (J, Cb, Ck, Ct, I, II, III) = equilibrium_new.check_governing_equations(2)
        print('J residue:')
        aqsc.print_fractional_error(J)
        self.assertTrue(J.filter(20).get_amplitude()<n_2_tolerance)
        print('Cb residue:')
        aqsc.print_fractional_error(Cb)
        self.assertTrue(Cb.filter(20).get_amplitude()<n_2_tolerance)
        print('Ck residue:')
        aqsc.print_fractional_error(Ck)
        self.assertTrue(Ck.filter(20).get_amplitude()<n_2_tolerance)
        print('Ct residue:')
        aqsc.print_fractional_error(Ct)
        self.assertTrue(Ct.filter(20).get_amplitude()<n_2_tolerance)
        print('I residue:')
        aqsc.print_fractional_error(I)
        self.assertTrue(I.filter(20).get_amplitude()<n_2_tolerance)
        print('II residue:')
        aqsc.print_fractional_error(II)
        self.assertTrue(II.filter(20).get_amplitude()<n_2_tolerance)
        print('III residue:')
        aqsc.print_fractional_error(III)
        self.assertTrue(III.filter(20).get_amplitude()<n_2_tolerance)
        print('Testing order 4, tolerance:', n_4_tolerance)
        (J, Cb, Ck, Ct, I, II, III) = equilibrium_new.check_governing_equations(4)
        print('J residue:')
        aqsc.print_fractional_error(J)
        self.assertTrue(J.filter(20).get_amplitude()<n_4_tolerance)
        print('Cb residue:')
        aqsc.print_fractional_error(Cb)
        self.assertTrue(Cb.filter(20).get_amplitude()<n_4_tolerance)
        print('Ck residue:')
        aqsc.print_fractional_error(Ck)
        self.assertTrue(Ck.filter(20).get_amplitude()<n_4_tolerance)
        print('Ct residue:')
        aqsc.print_fractional_error(Ct)
        self.assertTrue(Ct.filter(20).get_amplitude()<n_4_tolerance)
        print('I residue:')
        aqsc.print_fractional_error(I)
        self.assertTrue(I.filter(20).get_amplitude()<n_4_tolerance)
        print('II residue:')
        aqsc.print_fractional_error(II)
        self.assertTrue(II.filter(20).get_amplitude()<n_4_tolerance)
        print('III residue:')
        aqsc.print_fractional_error(III)
        self.assertTrue(III.filter(20).get_amplitude()<n_4_tolerance)
unittest.main()