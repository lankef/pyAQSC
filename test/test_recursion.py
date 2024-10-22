import unittest
from aqsc import *
import pathlib

import jax.numpy as jnp

# The numerical derivatives usually aren't as accurate 
# as to go below the threshold of np.isclose().
print('Testing all parsed recursion relations '\
      '(except the looped equation) against '\
      'Rodriguez\'s circular axis calculation')


circ_axis_tolerance = 1e-4
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

equilibrium = Equilibrium.from_known(
    X_coef_cp=X_coef_cp.mask(2),
    Y_coef_cp=Y_coef_cp.mask(2),
    Z_coef_cp=Z_coef_cp.mask(2),
    B_psi_coef_cp=B_psi_coef_cp.mask(0),
    B_theta_coef_cp=B_theta_coef_cp.mask(2),
    B_denom_coef_c=B_denom_coef_c.mask(2),
    B_alpha_coef=B_alpha_coef.mask(1),
    iota_coef=iota_coef.mask(0), 
    kap_p=kap_p, 
    dl_p=dl_p, 
    tau_p=tau_p,
    p_perp_coef_cp = p_perp_coef_cp.mask(2), 
    Delta_coef_cp = Delta_coef_cp.mask(2))

class TestCircularAxis(unittest.TestCase):

    def test_B_psi(self):
        print('Testing B_psi iteration' )
        B_psi_nm2 = iterate_dc_B_psi_nm2(n_eval=3,
            X_coef_cp=X_coef_cp,
            Y_coef_cp=Y_coef_cp,
            Z_coef_cp=Z_coef_cp,
            B_theta_coef_cp=B_theta_coef_cp.zero_append(),
            B_psi_coef_cp=B_psi_coef_cp,
            B_alpha_coef=B_alpha_coef,
            B_denom_coef_c=B_denom_coef_c,
            kap_p=kap_p,
            dl_p=dl_p,
            tau_p=tau_p,
            iota_coef=iota_coef
            ).antid_chi()
        print_fractional_error(B_psi_nm2.content,B_psi_coef_cp[1].content)
        self.assertTrue(is_roughly_close_chiphifunc(B_psi_nm2, B_psi_coef_cp[1]))

    def test_X(self):
        print('Testing X iteration' )
        X2 = iterate_Xn_cp(2,
            X_coef_cp,
            Y_coef_cp,
            Z_coef_cp,
            B_denom_coef_c,
            B_alpha_coef,
            kap_p, dl_p, tau_p,
            iota_coef)
        print_fractional_error(X2.content, X_coef_cp[2].content)
        self.assertTrue(is_roughly_close_chiphifunc(X2, X_coef_cp[2]))

    def test_Z(self):
        print('Testing Z iteration' )
        Z2 = iterate_Zn_cp(
            2,
            X_coef_cp, Y_coef_cp, Z_coef_cp,
            B_theta_coef_cp, B_psi_coef_cp,
            B_alpha_coef,
            kap_p, dl_p, tau_p,
            iota_coef)
        print_fractional_error(Z2.content, Z_coef_cp[2].content)
        self.assertTrue(is_roughly_close_chiphifunc(Z2, Z_coef_cp[2]))

    def test_Y(self):
        print('Testing Y iteration' )
        Yn = iterate_Yn_cp_magnetic(
            2,
            X_coef_cp,
            Y_coef_cp,
            Z_coef_cp,
            B_psi_coef_cp,
            B_theta_coef_cp,
            B_alpha_coef,
            B_denom_coef_c,
            kap_p, dl_p, tau_p,
            iota_coef,
            static_max_freq=50, 
            Yn0=Y_coef_cp[2][0].content)
        print_fractional_error(Yn.content, Y_coef_cp[2].content)
        self.assertTrue(is_roughly_close_chiphifunc(Yn, Y_coef_cp[2]))

    def test_p(self):
        print('Testing p iteration' )
        pn = iterate_p_perp_n(2,
            B_theta_coef_cp,
            B_psi_coef_cp,
            B_alpha_coef,
            B_denom_coef_c,
            p_perp_coef_cp,
            Delta_coef_cp,
            iota_coef)
        print_fractional_error(pn.content, p_perp_coef_cp[2].content)
        self.assertTrue(is_roughly_close_chiphifunc(pn, p_perp_coef_cp[2]))

    def test_Delta(self):
        print('Testing Delta iteration' )
        Delta_n = iterate_delta_n_0_offset(2,
            B_denom_coef_c,
            p_perp_coef_cp,
            Delta_coef_cp,
            iota_coef,
            static_max_freq=None,
            no_iota_masking = False)
        # Delta_n[0]'s average need to be solved at the following order. 
        # Here we compare two Deltas after setting both of their m=0
        # components to have zero average.
        Delta_n_no_avg = Delta_n - jnp.average(Delta_n[0].content)
        Delta_ans_no_avg = Delta_coef_cp[2] - jnp.average(Delta_coef_cp[2][0].content)
        print_fractional_error(Delta_n_no_avg.content, Delta_ans_no_avg.content)
        self.assertTrue(is_roughly_close_chiphifunc(Delta_n_no_avg, Delta_ans_no_avg))
        
if __name__ == '__main__':
    unittest.main()