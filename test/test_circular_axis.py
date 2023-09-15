import unittest
from aqsc import *
import pathlib

import jax.numpy as jnp

circ_equilibrium = circular_axis()
circ_axis_tolerance = 5e-6
def is_roughly_close_chiphifunc(chiphifunc_a, chiphifunc_b):
    return(
        jnp.all(
            jnp.abs(chiphifunc_a.content-chiphifunc_b.content)<circ_axis_tolerance
        )
    )

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

class TestCircularAxis(unittest.TestCase):

    def test_helicity(self):
        self.assertTrue(circ_equilibrium.get_helicity()==0)

    def test_B_psi(self):
        print('Testing leading order: B_psi[0]')
        print_fractional_error(
            circ_equilibrium.unknown['B_psi_coef_cp'][0].content,
            B_psi_coef_cp[0].content
        )
        self.assertTrue(is_roughly_close_chiphifunc(
            circ_equilibrium.unknown['B_psi_coef_cp'][0],
            B_psi_coef_cp[0]
        ))

    def test_X(self):
        print('Testing leading order: X[1]')
        print_fractional_error(
            circ_equilibrium.unknown['X_coef_cp'][1].content,
            X_coef_cp[1].content
        )
        print('Testing leading order: X[2]')
        print_fractional_error(
            circ_equilibrium.unknown['X_coef_cp'][2].content,
            X_coef_cp[2].content
        )
        self.assertTrue(is_roughly_close_chiphifunc(
            circ_equilibrium.unknown['X_coef_cp'][1],
            X_coef_cp[1]
        ))
        self.assertTrue(is_roughly_close_chiphifunc(
            circ_equilibrium.unknown['X_coef_cp'][2],
            X_coef_cp[2]
        ))

    def test_Z(self):
        print('Testing leading order: Z[2]')
        print_fractional_error(
            circ_equilibrium.unknown['Z_coef_cp'][2].content,
            Z_coef_cp[2].content
        )
        self.assertTrue(is_roughly_close_chiphifunc(
            circ_equilibrium.unknown['Z_coef_cp'][2],
            Z_coef_cp[2]
        ))

    def test_Y(self):
        print('Testing leading order: Y[1]')
        print_fractional_error(
            circ_equilibrium.unknown['Y_coef_cp'][1].content,
            Y_coef_cp[1].content
        )
        print('Testing leading order: Y[2]')
        print_fractional_error(
            circ_equilibrium.unknown['Y_coef_cp'][2].content,
            Y_coef_cp[2].content
        )
        self.assertTrue(is_roughly_close_chiphifunc(
            circ_equilibrium.unknown['Y_coef_cp'][1],
            Y_coef_cp[1]
        ))
        self.assertTrue(is_roughly_close_chiphifunc(
            circ_equilibrium.unknown['Y_coef_cp'][2],
            Y_coef_cp[2]
        ))

    def test_p(self):
        print('Testing leading order: p_perp[1]')
        print_fractional_error(
            circ_equilibrium.unknown['p_perp_coef_cp'][1].content,
            p_perp_coef_cp[1].content
        )
        print('Testing leading order: p_perp[2]')
        print_fractional_error(
            circ_equilibrium.unknown['p_perp_coef_cp'][2].content,
            p_perp_coef_cp[2].content
        )
        self.assertTrue(is_roughly_close_chiphifunc(
            circ_equilibrium.unknown['p_perp_coef_cp'][1],
            p_perp_coef_cp[1]
        ))
        self.assertTrue(is_roughly_close_chiphifunc(
            circ_equilibrium.unknown['p_perp_coef_cp'][2],
            p_perp_coef_cp[2]
        ))

    def test_Delta(self):
        print('Testing leading order: Delta[1]')
        print_fractional_error(
            circ_equilibrium.unknown['Delta_coef_cp'][1].content,
            Delta_coef_cp[1].content
        )
        print('Testing leading order: Delta[2]')
        print_fractional_error(
            circ_equilibrium.unknown['Delta_coef_cp'][2].content,
            Delta_coef_cp[2].content
        )
        self.assertTrue(is_roughly_close_chiphifunc(
            circ_equilibrium.unknown['Delta_coef_cp'][1],
            Delta_coef_cp[1]
        ))
        self.assertTrue(is_roughly_close_chiphifunc(
            circ_equilibrium.unknown['Delta_coef_cp'][2],
            Delta_coef_cp[2]
        ))

    
    def test_B_theta(self):
        print('Testing leading order: B_theta_coef_cp[2]')
        print_fractional_error(
            circ_equilibrium.unknown['B_theta_coef_cp'][2].content,
            B_theta_coef_cp[2].content
        )
        self.assertTrue(
            circ_equilibrium.unknown['B_theta_coef_cp'][1].is_special()
            and circ_equilibrium.unknown['B_theta_coef_cp'][1].nfp==0
        )
        self.assertTrue(is_roughly_close_chiphifunc(
            circ_equilibrium.unknown['B_theta_coef_cp'][2],
            B_theta_coef_cp[2]
        ))
        
if __name__ == '__main__':
    unittest.main()