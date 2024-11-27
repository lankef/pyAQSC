from aqsc import (
    get_axis_info,
    ChiPhiEpsFunc,
    ChiPhiFunc,
    ChiPhiFuncSpecial,
    Equilibrium
)
import unittest
import pathlib
import numpy as np
import jax.numpy as jnp
import jax.config
jax.config.update("jax_enable_x64", True)

# Create a torus with elliptical cross section
# for testing volume integral and coordinate transform.
Rc, Rs = ([1., 0, 0], [0, 0, 0])
Zc, Zs = ([0, 0, 0], [0, 0, 0])
axis_info = get_axis_info(Rc, Rs, Zc, Zs, 1, 1000)
a = (1 + np.random.random())/2
b = (1 + np.random.random())/2
X1 = ChiPhiFunc(jnp.array([jnp.zeros(1000), a * jnp.ones(1000)]), 1, trig_mode=True)
Y1 = ChiPhiFunc(jnp.array([b * jnp.ones(1000), jnp.zeros(1000)]), 1, trig_mode=True)
X_coef_cp = ChiPhiEpsFunc([ChiPhiFuncSpecial(0), X1], 1, False)
Y_coef_cp = ChiPhiEpsFunc([ChiPhiFuncSpecial(0), Y1], 1, False)
Z_coef_cp = ChiPhiEpsFunc([ChiPhiFuncSpecial(0), ChiPhiFuncSpecial(0)], 1, False)

equil_test = Equilibrium.from_known(
    X_coef_cp=X_coef_cp,
    Y_coef_cp=Y_coef_cp,
    Z_coef_cp=Z_coef_cp,
    B_psi_coef_cp=None,
    B_theta_coef_cp=None,
    B_denom_coef_c=None,
    B_alpha_coef=None,
    kap_p=axis_info['kap_p'],
    dl_p=axis_info['dl_p'],
    tau_p=axis_info['tau_p'],
    iota_coef=None,
    p_perp_coef_cp=None,
    Delta_coef_cp=None,
    axis_info=axis_info,
    magnetic_only=False
)

eps = np.random.random(5)
chi = np.random.random(5)
phi = np.random.random(5)
# Test cases for coordinate transform
R = 1 - a * eps * jnp.cos(chi)
Phi = phi
Z = eps * b * jnp.sin(chi)
X = R*jnp.cos(Phi)
Y = R*jnp.sin(Phi)

class TestCoords(unittest.TestCase):

    def test_coord_transform(self):
        ''' 
        Testing the coordinate transfroms from GBC to cylindrical/xyz 
        '''
        print('Testing coordinate transforms using an elliptical torus.')
        R_test, Phi_test, Z_test = equil_test.flux_to_cylindrical(eps**2,chi,phi)
        X_test, Y_test, Z_test2 = equil_test.flux_to_xyz(eps**2,chi,phi)
        self.assertTrue(jnp.all(jnp.isclose(R, R_test)))
        self.assertTrue(jnp.all(jnp.isclose(Phi, Phi_test)))
        self.assertTrue(jnp.all(jnp.isclose(Z, Z_test)))
        self.assertTrue(jnp.all(jnp.isclose(X, X_test)))
        self.assertTrue(jnp.all(jnp.isclose(Y, Y_test)))
        self.assertTrue(jnp.all(jnp.isclose(Z, Z_test2)))

    def test_covariant_basis(self):
        '''
        Testing the covariant basis calculation for the GBC
        '''
        print('Testing covariant basis calculations using an elliptical torus.')
        (
            deps_r_x,
            deps_r_y,
            deps_r_z,
            dchi_r_x,
            dchi_r_y,
            dchi_r_z,
            dphi_r_x,
            dphi_r_y,
            dphi_r_z,
        ) = equil_test.covariant_basis()
        self.assertTrue(jnp.all(jnp.isclose(deps_r_x.eval_eps(eps, chi, phi), -a * jnp.cos(chi) * jnp.cos(phi))))
        self.assertTrue(jnp.all(jnp.isclose(deps_r_y.eval_eps(eps, chi, phi), -a * jnp.cos(chi) * jnp.sin(phi))))
        self.assertTrue(jnp.all(jnp.isclose(deps_r_z.eval_eps(eps, chi, phi), b * jnp.sin(chi))))
        self.assertTrue(jnp.all(jnp.isclose(dchi_r_x.eval_eps(eps, chi, phi), a * eps * jnp.sin(chi) * jnp.cos(phi))))
        self.assertTrue(jnp.all(jnp.isclose(dchi_r_y.eval_eps(eps, chi, phi), a * eps * jnp.sin(chi) * jnp.sin(phi))))
        self.assertTrue(jnp.all(jnp.isclose(dchi_r_z.eval_eps(eps, chi, phi), b * eps * jnp.cos(chi))))
        self.assertTrue(jnp.all(jnp.isclose(dphi_r_x.eval_eps(eps, chi, phi), -(1 - a * eps * jnp.cos(chi)) * jnp.sin(phi))))
        self.assertTrue(jnp.all(jnp.isclose(dphi_r_y.eval_eps(eps, chi, phi), (1 - a * eps * jnp.cos(chi)) * jnp.cos(phi))))
        self.assertTrue(jnp.all(jnp.isclose(dphi_r_z.eval_eps(eps, chi, phi), 0)))
    
    def test_jacobian_vol_int(self):
        # Testing Jacobian and volume integral calculation 
        # by integrating 1 throughout the elliptic torus and 
        # comparing the result with the volume formula.

        # Volume as a function of ChiPhiEpsFunc
        # Its value should be:
        # 0, 0, 2pi^2 * a * b * 1
        volume_chiphiepsfunc = equil_test.volume_integral(1)
        # Testing volume integral
        print('Testing volume integral using an elliptical torus.')
        print('The volume of the torus to epsilon is:', volume_chiphiepsfunc)
        print('The second order coefficient is:         ', volume_chiphiepsfunc[2].content[0, 0])
        print('It should be (2 * jnp.pi**2 * a * b * 1):', 2 * jnp.pi**2 * a * b * 1)
        for i in range(len(volume_chiphiepsfunc.chiphifunc_list)):
            item = volume_chiphiepsfunc[i]
            if i == 2:
                # The volume of this elliptic torus is 
                # 2pi^2 a b eps^2 c, c=1
                # The second order coefficient of volume should be 
                # 2pi^2 a b
                self.assertTrue(jnp.all(jnp.isclose(item.content, 2 * jnp.pi**2 * a * b * 1)))
            else:
                self.assertTrue(
                    (item.is_special() and item.nfp==0)
                    or 
                    jnp.all(jnp.isclose(item.content, 0))
                )
    

if __name__ == '__main__':
    unittest.main()