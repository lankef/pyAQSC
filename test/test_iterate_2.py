import unittest
import aqsc
import os 

os.environ['XLA_FLAGS'] = '--xla_dump_to=./tmp/foo'

n_2_tolerance = 5e-7
n_4_tolerance = 1e-4

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
        print('Testing full equilibrium order 2, tolerance:', n_2_tolerance)
        (J, Cb, Ck, Ct, I, II, III) = equilibrium_new.check_governing_equations(2)
        print('J residue:')
        aqsc.print_fractional_error(J.filter(20).content, 0)
        self.assertTrue(J.filter(20).get_amplitude()<n_2_tolerance)
        print('Cb residue:')
        aqsc.print_fractional_error(Cb.filter(20).content, 0)
        self.assertTrue(Cb.filter(20).get_amplitude()<n_2_tolerance)
        print('Ck residue:')
        aqsc.print_fractional_error(Ck.filter(20).content, 0)
        self.assertTrue(Ck.filter(20).get_amplitude()<n_2_tolerance)
        print('Ct residue:')
        aqsc.print_fractional_error(Ct.filter(20).content, 0)
        self.assertTrue(Ct.filter(20).get_amplitude()<n_2_tolerance)
        print('I residue:')
        aqsc.print_fractional_error(I.filter(20).content, 0)
        self.assertTrue(I.filter(20).get_amplitude()<n_2_tolerance)
        print('II residue:')
        aqsc.print_fractional_error(II.filter(20).content, 0)
        self.assertTrue(II.filter(20).get_amplitude()<n_2_tolerance)
        print('III residue:')
        aqsc.print_fractional_error(III.filter(20).content, 0)
        self.assertTrue(III.filter(20).get_amplitude()<n_2_tolerance)
        print('Testing full equilibrium order 4, tolerance:', n_4_tolerance)
        (J, Cb, Ck, Ct, I, II, III) = equilibrium_new.check_governing_equations(4)
        print('J residue:')
        aqsc.print_fractional_error(J.filter(20).content, 0)
        self.assertTrue(J.filter(20).get_amplitude()<n_4_tolerance)
        print('Cb residue:')
        aqsc.print_fractional_error(Cb.filter(20).content, 0)
        self.assertTrue(Cb.filter(20).get_amplitude()<n_4_tolerance)
        print('Ck residue:')
        aqsc.print_fractional_error(Ck.filter(20).content, 0)
        self.assertTrue(Ck.filter(20).get_amplitude()<n_4_tolerance)
        print('Ct residue:')
        aqsc.print_fractional_error(Ct.filter(20).content, 0)
        self.assertTrue(Ct.filter(20).get_amplitude()<n_4_tolerance)
        print('I residue:')
        aqsc.print_fractional_error(I.filter(20).content, 0)
        self.assertTrue(I.filter(20).get_amplitude()<n_4_tolerance)
        print('II residue:')
        aqsc.print_fractional_error(II.filter(20).content, 0)
        self.assertTrue(II.filter(20).get_amplitude()<n_4_tolerance)
        print('III residue:')
        aqsc.print_fractional_error(III.filter(20).content, 0)
        self.assertTrue(III.filter(20).get_amplitude()<n_4_tolerance)
        
if __name__ == '__main__':
    unittest.main()