# Wrapped/completed recursion relations based on translated expressions
# in parsed/. Necessary masking and/or n-substitution are included. All iterate_*
# methods returns ChiPhiFunc's.
import jax.numpy as jnp
# import numpy as np # used in save_plain and get_helicity
# from jax import jit, vmap, tree_util
from jax import tree_util, jit, vmap, grad
from jax.lax import fori_loop, while_loop
from functools import partial # for JAX jit with static params
from interpax import interp1d
# from matplotlib import pyplot as plt

# ChiPhiFunc and ChiPhiEpsFunc
from .chiphifunc import *
from .chiphiepsfunc import *
from .math_utilities import *
from .recursion_relations import *
from .looped_solver import iterate_looped
from .config import interp1d_method
# parsed relations
from .MHD_parsed import validate_J, validate_Cb, validate_Ck, \
    validate_Ct, validate_I, validate_II, validate_III

# Plotting
import matplotlib.pyplot as plt
from matplotlib import cm, colors

''' Equilibrium manager and Iterate '''

def interp1d_shape_memory(xq, x, f, **kwarg):
    '''
    An utility function used to call interp1d on nd/scalar xq, 1d x, and 1d f.
    '''
    ndarr = jnp.ndim(xq) > 1
    if ndarr:
        shape_memory = xq.shape
        xq = xq.flatten()
    fq = interp1d(
        xq, 
        x, 
        f, 
        **kwarg
    )
    if ndarr:
        fq = fq.reshape(shape_memory)
    return(fq)

# A container for all equilibrium quantities.
# All coef inputs must be ChiPhiEpsFunc's.
class Equilibrium:
    # nfp-dependent!!
    def __init__(self, unknown, constant, axis_info, nfp, magnetic_only):
        self.unknown = unknown
        self.constant = constant
        self.nfp = nfp
        self.magnetic_only = magnetic_only
        self.axis_info = axis_info
        # Check if every term is on the same order
        # self.check_order_consistency()

    ''' For JAX use '''
    def _tree_flatten(self):
        children = (
            self.unknown,
            self.constant,
            self.axis_info,
        )  # arrays / dynamic values
        aux_data = {
            'nfp': self.nfp,
            'magnetic_only': self.magnetic_only
        }  # static values
        return (children, aux_data)
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def from_known(
        X_coef_cp,
        Y_coef_cp,
        Z_coef_cp,
        B_psi_coef_cp,
        B_theta_coef_cp,
        B_denom_coef_c,
        B_alpha_coef,
        kap_p, dl_p, tau_p,
        iota_coef,
        p_perp_coef_cp,
        Delta_coef_cp,
        axis_info={},
        magnetic_only=False
        ):
        # Variables being solved for are stored in dicts for
        # convenience of plotting and saving
        unknown = {}
        constant = {}
        nfp = X_coef_cp.nfp
        unknown['X_coef_cp'] = X_coef_cp
        unknown['Y_coef_cp'] = Y_coef_cp
        unknown['Z_coef_cp'] = Z_coef_cp
        unknown['B_psi_coef_cp'] = B_psi_coef_cp
        unknown['B_theta_coef_cp'] = B_theta_coef_cp
        unknown['p_perp_coef_cp'] = p_perp_coef_cp
        unknown['Delta_coef_cp'] = Delta_coef_cp
        constant['iota_coef'] = iota_coef
        constant['B_denom_coef_c'] = B_denom_coef_c
        constant['B_alpha_coef'] = B_alpha_coef
        constant['kap_p'] = kap_p
        constant['dl_p'] = dl_p
        constant['tau_p'] = tau_p
        # Pressure can be trivial
        if not unknown['p_perp_coef_cp']:
            unknown['p_perp_coef_cp'] = ChiPhiEpsFunc.zeros_like(X_coef_cp)
        if not unknown['Delta_coef_cp']:
            unknown['Delta_coef_cp'] = ChiPhiEpsFunc.zeros_like(X_coef_cp)

        return(Equilibrium(
            unknown=unknown,
            constant=constant,
            axis_info=axis_info,
            nfp=nfp,
            magnetic_only=magnetic_only,
        ))
    
    def phi_gbc_to_Phi_cylindrical(self, phi):
        period = 2*jnp.pi/self.nfp
        phi_incomplete_period = phi % period
        phi_full_periods = phi - phi_incomplete_period
        phi_gbc_padded = jnp.append(self.axis_info['phi_gbc'], period)
        Phi0_padded = jnp.append(self.axis_info['Phi0'], period)

        Phi_cylindrical_incomplete_period = interp1d_shape_memory(
            phi_incomplete_period, 
            phi_gbc_padded, 
            Phi0_padded, 
            method=interp1d_method,
        )
        return(Phi_cylindrical_incomplete_period + phi_full_periods)

    ''' Coordinate transformations '''
    def frenet_basis_phi(self, phi=None):
        ''' 
        Calculates frenet basis vectors (in R, Phi, Z) from the flux coordinat phi. 
        To avoid interpolation, use phi=None.
        '''
        phi_gbc = self.axis_info['phi_gbc']
        R0 = self.axis_info['R0']
        Phi0 = self.axis_info['Phi0']
        Z0 = self.axis_info['Z0']
        tangent_cylindrical = self.axis_info['tangent_cylindrical']
        normal_cylindrical = self.axis_info['normal_cylindrical']
        binormal_cylindrical = self.axis_info['binormal_cylindrical']
        nfp=self.nfp
        # axis location and basis in term of Boozer phi
        period = 2*jnp.pi/nfp
        if phi is None:
            phi = phi_gbc
            axis_r0_Phi = Phi0
            axis_r0_R = R0
            axis_r0_z = Z0
            tangent_R = tangent_cylindrical[:, 0]
            tangent_Phi = tangent_cylindrical[:, 1]
            tangent_z = tangent_cylindrical[:, 2]
            normal_R = normal_cylindrical[:, 0]
            normal_Phi = normal_cylindrical[:, 1]
            normal_z = normal_cylindrical[:, 2]
            binormal_R = binormal_cylindrical[:, 0]
            binormal_Phi = binormal_cylindrical[:, 1]
            binormal_z = binormal_cylindrical[:, 2]
        else:
            axis_r0_Phi = self.phi_gbc_to_Phi_cylindrical(phi)
            # Less accurate than interpax
            # axis_r0_R = jnp.interp(phi, phi_gbc, R0, period=period)
            # axis_r0_Z = jnp.interp(phi, phi_gbc, Z0, period=period)
            # tangent_R = jnp.interp(phi, phi_gbc, tangent_cylindrical[:, 0], period=period)
            # tangent_Phi = jnp.interp(phi, phi_gbc, tangent_cylindrical[:, 1], period=period)
            # tangent_Z = jnp.interp(phi, phi_gbc, tangent_cylindrical[:, 2], period=period)
            # normal_R = jnp.interp(phi, phi_gbc, normal_cylindrical[:, 0], period=period)
            # normal_Phi = jnp.interp(phi, phi_gbc, normal_cylindrical[:, 1], period=period)
            # normal_Z = jnp.interp(phi, phi_gbc, normal_cylindrical[:, 2], period=period)
            # binormal_R = jnp.interp(phi, phi_gbc, binormal_cylindrical[:, 0], period=period)
            # binormal_Phi = jnp.interp(phi, phi_gbc, binormal_cylindrical[:, 1], period=period)
            # binormal_Z = jnp.interp(phi, phi_gbc, binormal_cylindrical[:, 2], period=period)
            axis_r0_R = interp1d_shape_memory(phi, phi_gbc, R0, period=period, method=interp1d_method)
            axis_r0_z = interp1d_shape_memory(phi, phi_gbc, Z0, period=period, method=interp1d_method)
            tangent_R = interp1d_shape_memory(phi, phi_gbc, tangent_cylindrical[:, 0], period=period, method=interp1d_method)
            tangent_Phi = interp1d_shape_memory(phi, phi_gbc, tangent_cylindrical[:, 1], period=period, method=interp1d_method)
            tangent_z = interp1d_shape_memory(phi, phi_gbc, tangent_cylindrical[:, 2], period=period, method=interp1d_method)
            normal_R = interp1d_shape_memory(phi, phi_gbc, normal_cylindrical[:, 0], period=period, method=interp1d_method)
            normal_Phi = interp1d_shape_memory(phi, phi_gbc, normal_cylindrical[:, 1], period=period, method=interp1d_method)
            normal_z = interp1d_shape_memory(phi, phi_gbc, normal_cylindrical[:, 2], period=period, method=interp1d_method)
            binormal_R = interp1d_shape_memory(phi, phi_gbc, binormal_cylindrical[:, 0], period=period, method=interp1d_method)
            binormal_Phi = interp1d_shape_memory(phi, phi_gbc, binormal_cylindrical[:, 1], period=period, method=interp1d_method)
            binormal_z = interp1d_shape_memory(phi, phi_gbc, binormal_cylindrical[:, 2], period=period, method=interp1d_method)

        axis_r0_x = axis_r0_R * jnp.cos(axis_r0_Phi)
        axis_r0_y = axis_r0_R * jnp.sin(axis_r0_Phi)

        R_hat_x = axis_r0_x / axis_r0_R
        R_hat_y = axis_r0_y / axis_r0_R

        Phi_hat_x = -jnp.sin(axis_r0_Phi)
        Phi_hat_y = jnp.cos(axis_r0_Phi)

        tangent_x = (
            tangent_R * R_hat_x
            + tangent_Phi * Phi_hat_x
        )
        tangent_y = (
            tangent_R * R_hat_y
            + tangent_Phi * Phi_hat_y
        )
        normal_x = (
            normal_R * R_hat_x
            + normal_Phi * Phi_hat_x
        )
        normal_y = (
            normal_R * R_hat_y
            + normal_Phi * Phi_hat_y
        )
        binormal_x = (
            binormal_R * R_hat_x
            + binormal_Phi * Phi_hat_x
        )
        binormal_y = (
            binormal_R * R_hat_y
            + binormal_Phi * Phi_hat_y
        )

        return(
            phi,
            axis_r0_x,
            axis_r0_y,
            axis_r0_z,
            tangent_x,
            tangent_y,
            tangent_z,
            normal_x,
            normal_y,
            normal_z,
            binormal_x,
            binormal_y,
            binormal_z
        )

    def flux_to_frenet(self, psi, chi, phi, n_max=float('inf')):
        ''' 
        Transforms positions in the flux coordinate to the frenet frame.
        Returns (curvature, binormal, tangent) in the Frenet frame.
        '''
        return(
            np.real(self.unknown['X_coef_cp'].eval(psi=psi, chi=chi, phi=phi, n_max=n_max)), 
            np.real(self.unknown['Y_coef_cp'].eval(psi=psi, chi=chi, phi=phi, n_max=n_max)), 
            np.real(self.unknown['Z_coef_cp'].eval(psi=psi, chi=chi, phi=phi, n_max=n_max))
        )

    def flux_to_xyz(self, psi, chi, phi, n_max=float('inf')):
        ''' 
        Transforms positions in the flux coordinate to the 
        Cartesian (xyz) coordinate.
        '''
        phi_gbc,\
        axis_r0_x,\
        axis_r0_y,\
        axis_r0_z,\
        tangent_x,\
        tangent_y,\
        tangent_z,\
        normal_x,\
        normal_y,\
        normal_z,\
        binormal_x,\
        binormal_y,\
        binormal_z = self.frenet_basis_phi(phi)
        curvature, binormal, tangent = self.flux_to_frenet(psi=psi, chi=chi, phi=phi, n_max=n_max)
        components_x = axis_r0_x\
            + tangent_x * tangent\
            + normal_x * curvature\
            + binormal_x * binormal
        
        components_y = axis_r0_y\
            + tangent_y * tangent\
            + normal_y * curvature\
            + binormal_y * binormal
        
        components_z = axis_r0_z\
            + tangent_z * tangent\
            + normal_z * curvature\
            + binormal_z * binormal
        return(
            components_x,
            components_y,
            components_z
        )
    
    def flux_to_cylindrical(self, psi, chi, phi, n_max=float('inf')):
        ''' 
        Transforms positions in the flux coordinate to the 
        Cartesian (xyz) coordinate.
        '''
        (
            components_X,
            components_Y,
            components_Z
        ) = self.flux_to_xyz(psi, chi, phi, n_max)
        components_Phi = jnp.arctan2(components_Y, components_X)
        if not(jnp.isscalar(psi) and jnp.isscalar(chi) and jnp.isscalar(phi)):
            components_Phi = jnp.unwrap(components_Phi, discont=jnp.pi)
        components_R = jnp.sqrt(components_X**2 + components_Y**2)
        return(
            components_R,
            components_Phi,
            components_Z
        )

    ''' Display and output'''

    def contravariant_basis_eps(self):
        '''
        Calculates dr/deps, dr/dchi, dr/dphi.
        '''
        len_phi = self.constant['kap_p'].content.shape[1]
        phi_grid = jnp.linspace(0, 2 * jnp.pi / self.nfp, len_phi, endpoint=False)

        # Calculate the coordinate basis on the same Phi grid as 
        # the ChiPhiFunc's.
        _,\
        _,\
        _,\
        _,\
        tangent_x_arr,\
        tangent_y_arr,\
        tangent_z_arr,\
        normal_x_arr,\
        normal_y_arr,\
        normal_z_arr,\
        binormal_x_arr,\
        binormal_y_arr,\
        binormal_z_arr = self.frenet_basis_phi(phi_grid)
        tangent_x = ChiPhiFunc(tangent_x_arr[None, :], self.nfp)
        tangent_y = ChiPhiFunc(tangent_y_arr[None, :], self.nfp)
        tangent_z = ChiPhiFunc(tangent_z_arr[None, :], self.nfp)
        normal_x = ChiPhiFunc(normal_x_arr[None, :], self.nfp)
        normal_y = ChiPhiFunc(normal_y_arr[None, :], self.nfp)
        normal_z = ChiPhiFunc(normal_z_arr[None, :], self.nfp)
        binormal_x = ChiPhiFunc(binormal_x_arr[None, :], self.nfp)
        binormal_y = ChiPhiFunc(binormal_y_arr[None, :], self.nfp)
        binormal_z = ChiPhiFunc(binormal_z_arr[None, :], self.nfp)

        kap_p = self.constant['kap_p'] # .eval(0, phi)
        tau_p = self.constant['tau_p'] # .eval(0, phi)
        dl_p = self.constant['dl_p']
        tangent_dphi_x = kap_p * normal_x * dl_p
        tangent_dphi_y = kap_p * normal_y * dl_p
        tangent_dphi_z = kap_p * normal_z * dl_p
        normal_dphi_x = (-kap_p * tangent_x - tau_p * binormal_x) * dl_p
        normal_dphi_y = (-kap_p * tangent_y - tau_p * binormal_y) * dl_p
        normal_dphi_z = (-kap_p * tangent_z - tau_p * binormal_z) * dl_p
        binormal_dphi_x = (tau_p * normal_x) * dl_p
        binormal_dphi_y = (tau_p * normal_y) * dl_p
        binormal_dphi_z = (tau_p * normal_z) * dl_p
        axis_r0_dphi_x = tangent_x * dl_p
        axis_r0_dphi_y = tangent_y * dl_p
        axis_r0_dphi_z = tangent_z * dl_p

        # Returns (n_grid, n_grid) arrays. Axis=1 is Phi.
        X_coef_cp = self.unknown['X_coef_cp']
        Y_coef_cp = self.unknown['Y_coef_cp']
        Z_coef_cp = self.unknown['Z_coef_cp']
        deps_X_coef_cp = X_coef_cp.deps()
        deps_Y_coef_cp = Y_coef_cp.deps()
        deps_Z_coef_cp = Z_coef_cp.deps()
        dchi_X_coef_cp = X_coef_cp.dchi()
        dchi_Y_coef_cp = Y_coef_cp.dchi()
        dchi_Z_coef_cp = Z_coef_cp.dchi()
        dphi_X_coef_cp = X_coef_cp.dphi()
        dphi_Y_coef_cp = Y_coef_cp.dphi()
        dphi_Z_coef_cp = Z_coef_cp.dphi()

        deps_r_x = (deps_X_coef_cp * normal_x + deps_Y_coef_cp * binormal_x + deps_Z_coef_cp * tangent_x)
        deps_r_y = (deps_X_coef_cp * normal_y + deps_Y_coef_cp * binormal_y + deps_Z_coef_cp * tangent_y)
        deps_r_z = (deps_X_coef_cp * normal_z + deps_Y_coef_cp * binormal_z + deps_Z_coef_cp * tangent_z)
        dchi_r_x = (dchi_X_coef_cp * normal_x + dchi_Y_coef_cp * binormal_x + dchi_Z_coef_cp * tangent_x)
        dchi_r_y = (dchi_X_coef_cp * normal_y + dchi_Y_coef_cp * binormal_y + dchi_Z_coef_cp * tangent_y)
        dchi_r_z = (dchi_X_coef_cp * normal_z + dchi_Y_coef_cp * binormal_z + dchi_Z_coef_cp * tangent_z)
        dphi_r_x = (axis_r0_dphi_x + dphi_X_coef_cp * normal_x + dphi_Y_coef_cp * binormal_x + dphi_Z_coef_cp * tangent_x + X_coef_cp * normal_dphi_x + Y_coef_cp * binormal_dphi_x + Z_coef_cp * tangent_dphi_x)
        dphi_r_y = (axis_r0_dphi_y + dphi_X_coef_cp * normal_y + dphi_Y_coef_cp * binormal_y + dphi_Z_coef_cp * tangent_y + X_coef_cp * normal_dphi_y + Y_coef_cp * binormal_dphi_y + Z_coef_cp * tangent_dphi_y)
        dphi_r_z = (axis_r0_dphi_z + dphi_X_coef_cp * normal_z + dphi_Y_coef_cp * binormal_z + dphi_Z_coef_cp * tangent_z + X_coef_cp * normal_dphi_z + Y_coef_cp * binormal_dphi_z + Z_coef_cp * tangent_dphi_z)
        return(
            deps_r_x,
            deps_r_y,
            deps_r_z,
            dchi_r_x,
            dchi_r_y,
            dchi_r_z,
            dphi_r_x,
            dphi_r_y,
            dphi_r_z,
        )

    def covariant_basis_eps_j_eps(self):
        '''
        Calculates J^eps_coord grad eps, J^eps_coord grad chi, J^eps_coord grad phi.
        '''
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
        ) = self.contravariant_basis_eps()
        # Here, again, the cross product has to be done explicitly because all arguments
        # are ChiPhiEpsFunc's.
        # Compute the components of j_grad_psi = dchi_r cross dphi_r
        # Compute the components of j_grad_chi = dphi_r cross deps_r
        # Compute the components of j_grad_phi = deps_r cross dchi_r
        j_eps_grad_eps_x = dchi_r_y * dphi_r_z - dchi_r_z * dphi_r_y
        j_eps_grad_eps_y = dchi_r_z * dphi_r_x - dchi_r_x * dphi_r_z
        j_eps_grad_eps_z = dchi_r_x * dphi_r_y - dchi_r_y * dphi_r_x
        j_eps_grad_chi_x = dphi_r_y * deps_r_z - dphi_r_z * deps_r_y
        j_eps_grad_chi_y = dphi_r_z * deps_r_x - dphi_r_x * deps_r_z
        j_eps_grad_chi_z = dphi_r_x * deps_r_y - dphi_r_y * deps_r_x
        j_eps_grad_phi_x = deps_r_y * dchi_r_z - deps_r_z * dchi_r_y
        j_eps_grad_phi_y = deps_r_z * dchi_r_x - deps_r_x * dchi_r_z
        j_eps_grad_phi_z = deps_r_x * dchi_r_y - deps_r_y * dchi_r_x

        return(
            j_eps_grad_eps_x,
            j_eps_grad_eps_y,
            j_eps_grad_eps_z,
            j_eps_grad_chi_x,
            j_eps_grad_chi_y,
            j_eps_grad_chi_z,
            j_eps_grad_phi_x,
            j_eps_grad_phi_y,
            j_eps_grad_phi_z,
        )

    def covariant_basis_j_eps(self):
        (
            j_grad_psi_x, # J_eps * grad eps = J * grad psi
            j_grad_psi_y,
            j_grad_psi_z,
            j_eps_grad_chi_x,
            j_eps_grad_chi_y,
            j_eps_grad_chi_z,
            j_eps_grad_phi_x,
            j_eps_grad_phi_y,
            j_eps_grad_phi_z,
        ) = self.covariant_basis_eps_j_eps()

        # J_eps = 2 eps * J
        j_eps_grad_psi_x = 2 * ChiPhiEpsFunc(
            list=[ChiPhiFuncSpecial(0)] + j_grad_psi_x.chiphifunc_list,
            nfp=j_grad_psi_x.nfp,
            square_eps_series=False
        )
        j_eps_grad_psi_y = 2 * ChiPhiEpsFunc(
            list=[ChiPhiFuncSpecial(0)] + j_grad_psi_y.chiphifunc_list,
            nfp=j_grad_psi_y.nfp,
            square_eps_series=False
        )
        j_eps_grad_psi_z = 2 * ChiPhiEpsFunc(
            list=[ChiPhiFuncSpecial(0)] + j_grad_psi_z.chiphifunc_list,
            nfp=j_grad_psi_z.nfp,
            square_eps_series=False
        )
        return(
            j_eps_grad_psi_x,
            j_eps_grad_psi_y,
            j_eps_grad_psi_z,
            j_eps_grad_chi_x,
            j_eps_grad_chi_y,
            j_eps_grad_chi_z,
            j_eps_grad_phi_x,
            j_eps_grad_phi_y,
            j_eps_grad_phi_z,
        )

    def jacobian_eps(self):
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
        ) = self.contravariant_basis_eps()
        triple_product = (
            deps_r_x * (dchi_r_y * dphi_r_z - dchi_r_z * dphi_r_y) +
            deps_r_y * (dchi_r_z * dphi_r_x - dchi_r_x * dphi_r_z) +
            deps_r_z * (dchi_r_x * dphi_r_y - dchi_r_y * dphi_r_x)
        )
        return(triple_product)
    
    def jacobian(self):
        '''
        Calculates the Jacobian.
          dr/dpsi x (dr/dchi . dr/dphi)
        = dr/d(eps^2)) x (dr/dchi . dr/dphi)
        = (1/2eps) dr/deps x (dr/dchi . dr/dphi)
        We know that J_e = dr/deps x (dr/dchi . dr/dphi) = 0 at eps=0.
        So J can be simply calculated by dropping the first element in 
        the power series, and then dividing by 2.
        '''
        jacobian_e = self.jacobian_eps()
        new_list = []
        for i in range(len(jacobian_e.chiphifunc_list)-1):
            new_list.append(jacobian_e.chiphifunc_list[i+1]/2)
        jacobian = ChiPhiEpsFunc(new_list, jacobian_e.nfp, False)
        return(jacobian)

    def jacobian_nae(self):
        return(self.constant['B_alpha_coef'] * self.constant['B_denom_coef_c'])

    def get_psi_crit(
        self, n_max=float('inf'), 
        n_grid_chi=100,
        n_grid_phi_skip=1,
        psi_init=None,
        fix_maxiter=False,
        max_iter=20,
        tol=1e-8,
        jacobian_eps_callable=None):
        '''

        Estimates the critical epsilon where flux surface self-intersects.
        by finding the zero of $min_{\chi, \phi}[\sqrt{J}(\epsilon, \chi, \phi)]=0$
        using binary search. \sqrt{J}(\epsilon, \chi, \phi) At each search step 
        is evaluated on a grid of given size.
        Can be diffed but is slow to compile, because it relies on newton iteration
        for root finding. Reducing `n_newton_iter` substantially speeds up jit.

        Parameters;

        - `n_max` - maximum order to evaluate coordinate transformation to.

        - `n_grid_chi, n_grid_phi : int`- Grid size to evaluate Jacobian $\sqrt{g}$ on. 
        The critical point occurs when $min(\sqrt{g}\leq0)$.

        - `bad_eps_init` - An initial guess of an epsilon>epsilon_crit used in Newton's 
        method. Need to be beyond t

        - `n_newton_iter:int` (static) - Maximum number of steps in Newton's method.
        higher number gives better acuracy but is slower to jit.

        Returns: 
        
        - (eps_crit, jacobian_residue): eps_crit and the flux surface min of the Jacobian at eps_crit.
        '''
        if psi_init is None:
            effective_major_radius = self.axis_info['axis_length']/jnp.pi/2
            B0 = self.constant['B_denom_coef_c'][0]
            psi_init = jnp.sqrt(effective_major_radius**2 * B0)
        phi_gbc = self.axis_info['phi_gbc'][::n_grid_phi_skip]
        points_chi = jnp.linspace(0, 2*jnp.pi, n_grid_chi, endpoint=False)

        if jacobian_eps_callable is None:
            jacobian_eps_callable = self.jacobian_eps().eval
        # Finding jacobian_min=0 using Newton's method.
        jacobian_grid = lambda psi: jnp.real(jacobian_eps_callable(psi, points_chi[:, None], phi_gbc[None, :], n_max = n_max))
        jacobian_min = lambda psi: jnp.min(jacobian_grid(psi))
        jacobian_min_prime = grad(jacobian_min)
        if fix_maxiter:
            def q(i, x):
                return(x - jacobian_min(x) / jacobian_min_prime(x))
            psi_sln = fori_loop(0, max_iter, q, psi_init)
            n_iter = max_iter
        else:
            def conv(dict_in):
                # conv = dict_in['conv']
                x = dict_in['x']
                return(
                    # This is the convergence condition (True when not converged yet)
                    jnp.logical_and(
                        dict_in['i'] <= max_iter,
                        jacobian_min(x)**2 >= tol**2,
                    )
                )
            def q_while(dict_in):
                i = dict_in['i']
                x = dict_in['x']
                return({
                    'i': i + 1,
                    'x': x - jacobian_min(x) / jacobian_min_prime(x)
                })
            dict_out = while_loop(
                cond_fun=conv,
                body_fun=q_while,
                init_val={'i': 0, 'x': psi_init},
            )
            psi_sln = dict_out['x']
            n_iter = dict_out['i']
        return(psi_sln, jacobian_min(psi_sln), n_iter)
        '''
        # Zero-finding with binary search is faster but cannot be jitted.
        # Binary search for jacobian_min=0,
        # knowing jacobian_min(0)=0 and assuming jacobian_min is concave.
        x_left = 0
        x_right = bad_eps_init
        # g_tol is the tolerance and supposed to
        error = g_tol + 1
        n_iter = 0
        n_bindary_iter_max = 200
        # Tolerance-based stopping is disabled for now
        # while error > g_tol and n_iter < n_bindary_iter_max:
        # while error > g_tol and n_iter < n_bindary_iter_max:
            # We assume that jacobian_min is concave and is 0 at 0.
            x_mid = .5*x_left + .5*x_right
            y_mid = jacobian_min(x_mid)
            if y_mid>0:
                x_left = x_mid
            else: 
                x_right = x_mid
            error = jnp.abs(y_mid)
            n_iter += 1
        '''      

    def B_vec_j_eps(self):
        '''
        Calculates vector B components.
        '''
        # The covariant bases, multiplied with J^eps_coord
        (
            j_eps_grad_psi_x,
            j_eps_grad_psi_y,
            j_eps_grad_psi_z,
            j_eps_grad_chi_x,
            j_eps_grad_chi_y,
            j_eps_grad_chi_z,
            j_eps_grad_phi_x,
            j_eps_grad_phi_y,
            j_eps_grad_phi_z,
        ) = self.covariant_basis_j_eps()
        B_theta = self.unknown['B_theta_coef_cp']
        B_alpha = self.constant['B_alpha_coef']
        B_psi = self.unknown['B_psi_coef_cp']
        iota_bar = self.constant['iota_coef']

        # Equation 6 in 
        # Solving the problem of ... I. Generalized force balance
        j_eps_B_x = (
            B_theta * j_eps_grad_chi_x
            + (B_alpha - iota_bar * B_theta) * j_eps_grad_phi_x
            + B_psi * j_eps_grad_psi_x
        )
        j_eps_B_y = (
            B_theta * j_eps_grad_chi_y
            + (B_alpha - iota_bar * B_theta) * j_eps_grad_phi_y
            + B_psi * j_eps_grad_psi_y
        )
        j_eps_B_z = (
            B_theta * j_eps_grad_chi_z
            + (B_alpha - iota_bar * B_theta) * j_eps_grad_phi_z
            + B_psi * j_eps_grad_psi_z
        )
        return(
            j_eps_B_x,
            j_eps_B_y,
            j_eps_B_z
        )

    def volume_integral(self, y):
        '''
        Calculates the volume integral of a quantity as a ChiPhiEpsFunc
        with no Chi or Phi dependence.
        '''
        jac = self.jacobian_eps()
        integrand = jac * y
        # Eps, phi and chi integral
        new_chiphifunc_list = [ChiPhiFuncSpecial(0)]
        len_phi = self.constant['kap_p'].content.shape[1]
        for i in range(len(integrand.chiphifunc_list)):
            item = integrand.chiphifunc_list[i]
            # Calculate the chi and phi integral
            if isinstance(item, ChiPhiFunc):
                # Handling of errors and zeros
                if item.is_special():
                    new_chiphifunc = item
                else:
                    # odd fourier series
                    if item.content.shape[0]%2==0:
                        new_chiphifunc = ChiPhiFuncSpecial(0)
                    else:
                        # The constant component in chi
                        item_0 = item[0]
                        len_phi = item_0.content.shape[1]
                        content_0_fft = jnp.fft.fft(item_0.content, axis=1)
                        # Phi integral
                        integral_i = 2 * jnp.pi * content_0_fft[0,0] / len_phi
                        # Chi integral
                        integral_i *= 2 * jnp.pi 
                        # Eps integral
                        integral_i /= (i+1)
                        new_chiphifunc = ChiPhiFunc(jnp.full((1, len_phi), integral_i), self.nfp)
            else: 
                integral_i = item * jnp.pi * 2 * jnp.pi * 2 / (i+1)
                new_chiphifunc = ChiPhiFunc(jnp.full((1, len_phi), integral_i), self.nfp)
            # calculate the epsilon integral
            new_chiphifunc_list.append(new_chiphifunc)
        return(ChiPhiEpsFunc(new_chiphifunc_list, self.nfp, False))

    def get_helicity(self):
        ''' 
        Returns the helicity (the normal vector, kappa's 
        # rotation around the origin)
        '''
        return(helicity_from_axis(self.axis_info, self.nfp))

    def display(self, psi_max=None, n_max=float('inf'), n_fs=5):
        # n_fs: total number of flux surfaces plotted
        if not psi_max:
            psi_max = self.get_psi_crit()[0]
            print('Plotting to critical psi:', psi_max)
            psi_max*=0.8
        fig = plt.figure()
        fig.set_dpi(400)
        ax = fig.add_subplot(projection='3d')
        phis = jnp.linspace(0, jnp.pi*2*0.9, 1000)
        chis = jnp.linspace(0, jnp.pi*2, 1000)
        x_surf, y_surf, z_surf = self.flux_to_xyz(psi=psi_max, chi=chis[:, None], phi=phis[None, :], n_max=n_max)
        # Coloring by magnitude of B
        B_denom = self.constant['B_denom_coef_c'].eval(
            psi=psi_max, chi=chis[:, None], phi=phis[None, :]
        )
        B_magnitude = 1/jnp.sqrt(jnp.real(B_denom)).astype(jnp.float32)

        norm = colors.Normalize(vmin=jnp.min(B_magnitude), vmax=jnp.max(B_magnitude), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.plasma)
        facecolors = mapper.to_rgba(B_magnitude)

        ax.plot_surface(x_surf, y_surf, z_surf, zorder=1, facecolors=facecolors)
        ax.axis('equal')

        # Plotting flux surfaces.
        # Plot the first separately 
        x_cross, y_cross, z_cross = self.flux_to_xyz(psi=0, chi=chis, phi=0, n_max=n_max)
        ax.plot(x_cross, y_cross, z_cross, zorder=2.5, linewidth=0.5, color='lightgrey', label=r'$(\epsilon=\sqrt{\psi}, \chi)$')
        eps_max = jnp.sqrt(psi_max)
        for psi_i in jnp.linspace(eps_max/n_fs, eps_max, n_fs)**2:
            x_cross, y_cross, z_cross = self.flux_to_xyz(psi=psi_i, chi=chis, phi=0, n_max=n_max)
            ax.plot(x_cross, y_cross, z_cross, zorder=2.5, linewidth=0.5, color='lightgrey')
        x_axis, y_axis, z_axis = self.flux_to_xyz(psi=0, chi=0, phi=chis, n_max=n_max)
        # Plotting the chi surfaces
        psis_dense = jnp.linspace(0, psi_max, 50)
        for chi_i in jnp.linspace(0.125, 1, 8)*jnp.pi*2:
            x_cross, y_cross, z_cross = self.flux_to_xyz(psi=psis_dense, chi=chi_i, phi=0, n_max=n_max)
            ax.plot(x_cross, y_cross, z_cross, zorder=2.5, linewidth=0.5, color='lightgrey')

        # Plotting the axis
        ax.plot(x_axis, y_axis, z_axis, zorder=3.5, linewidth=0.5, linestyle='dashed', color='lightgrey', label='Magnetic axis')
        fig.legend()
        fig.colorbar(mapper, label=r'Field strength $|B|, (T)$', shrink=0.5, ax=ax)
        return(fig, ax)

    def display_wireframe(self, psi_max=None, n_max=float('inf')):
        # n_fs: total number of flux surfaces plotted
        if not psi_max:
            psi_max = self.get_psi_crit()[0]
            print('Plotting to critical psi:', psi_max)
            psi_max*=0.8
        fig = plt.figure()
        fig.set_dpi(200)
        ax = fig.add_subplot(projection='3d')
        phis = jnp.linspace(0, jnp.pi*2*0.9, 1000)
        chis = jnp.linspace(0, jnp.pi*2, 1000)
        x_surf, y_surf, z_surf = self.flux_to_xyz(psi=psi_max, chi=chis[None, :], phi=phis[:, None], n_max=n_max)

        ax.plot_wireframe(x_surf, y_surf, z_surf, zorder=1, linewidths=0.1)
        ax.axis('equal')
        return(fig, ax)

    # not nfp-dependent
    def display_order(self, n:int):
        if n<2:
            raise ValueError('Can only display order n>1')
        for name in self.unknown.keys():
            if name!='B_psi_coef_cp':
                print(name, 'n =', n)
                n_display = n
            else:
                print(name, 'n =', n-2)
                n_display = n-2

            if type(self.unknown[name][n_display]) is ChiPhiFunc:
                self.unknown[name][n_display].display_content()
            else:
                print(self.unknown[name][n_display])

    ''' Saving and loading '''
    def save(self, file_name):
        jnp.save(file_name, self)

    def load(file_name):
        return(jnp.load(file_name, allow_pickle=True).item())

    def save_plain(self, file_name):
        ''' Saves to a npy file that can be read without aqsc or jax. '''
        unknown_dict = {}
        for key in self.unknown.keys():
            unknown_dict[key] = self.unknown[key].to_content_list()
        constant_dict={}
        constant_dict['B_denom_coef_c']\
            = self.constant['B_denom_coef_c'].to_content_list()
        constant_dict['B_alpha_coef']\
            = self.constant['B_alpha_coef'].to_content_list()
        constant_dict['iota_coef']\
            = self.constant['iota_coef'].to_content_list()
        constant_dict['kap_p']\
            = self.constant['kap_p'].content
        constant_dict['dl_p']\
            = self.constant['dl_p']
        constant_dict['tau_p']\
            = self.constant['tau_p'].content
        
        numpy_axis_info = {}
        for key in self.axis_info.keys():
            numpy_axis_info[key] = jnp.asarray(self.axis_info[key])

        big_dict = {
            'unknown':unknown_dict,
            'constant':constant_dict,
            'nfp':self.nfp,
            'magnetic_only':self.magnetic_only,
            'axis_info':numpy_axis_info,
        }
        jnp.save(file_name, big_dict)

    # nfp-dependent!!
    def load_plain(filename):
        npyfile = jnp.load(filename, allow_pickle=True)
        big_dict = npyfile.item()
        raw_unknown = big_dict['unknown']
        raw_constant = big_dict['constant']
        nfp = big_dict['nfp']
        magnetic_only = big_dict['magnetic_only']
        axis_info = big_dict['axis_info']
        print('nfp', nfp)

        unknown = {}
        for key in raw_unknown.keys():
            unknown[key] = ChiPhiEpsFunc.from_content_list(raw_unknown[key], nfp)

        constant={}
        constant['B_denom_coef_c']\
            = ChiPhiEpsFunc.from_content_list(raw_constant['B_denom_coef_c'], nfp)
        constant['B_alpha_coef']\
            = ChiPhiEpsFunc.from_content_list(raw_constant['B_alpha_coef'], nfp)
        constant['kap_p']\
            = ChiPhiFunc(raw_constant['kap_p'], nfp)
        constant['dl_p']\
            = raw_constant['dl_p']
        constant['tau_p']\
            = ChiPhiFunc(raw_constant['tau_p'], nfp)
        constant['iota_coef']\
            = ChiPhiFunc(raw_constant['iota_coef'], nfp)

        return(Equilibrium(
            unknown=unknown,
            constant=constant,
            axis_info=axis_info,
            nfp=nfp,
            magnetic_only=magnetic_only,
        ))

    # Order consistency check --------------------------------------------------
    # Get the current order of an equilibrium
    # not nfp-dependent
    def get_order(self):
        self.check_order_consistency()
        return(self.unknown['X_coef_cp'].get_order())

    # Check of all terms are on consistent orders
    # nfp=dependent!!
    def check_order_consistency(self):

        # An internal method for error throwing. Checks if all variables are
        # iterated to the correct order.
        def check_order_individial(ChiPhiEpsFunc, name_in_error_msg, n_req):
            n_current = ChiPhiEpsFunc.get_order()
            if n_current!=n_req:
                raise AttributeError(name_in_error_msg
                    + ' has order: '
                    + str(n_current)
                    + ', while required order is: '
                    + str(n_req))
            if ChiPhiEpsFunc.nfp!=0 and ChiPhiEpsFunc.nfp!=self.nfp:
                raise AttributeError(name_in_error_msg
                    + ' has nfp: '
                    + str(ChiPhiEpsFunc.nfp)
                    + ', while required nfp is: '
                    + str(self.nfp))


        n = self.unknown['X_coef_cp'].get_order()
        if n%2==1:
            raise AttributeError('X_coef_cp has odd order. Equilibrium is managed'\
            ' and updated every 2 orders and stops at even orders')

        check_order_individial(self.unknown['Y_coef_cp'], 'Y_coef_cp', n)
        check_order_individial(self.unknown['Z_coef_cp'], 'Z_coef_cp', n)
        check_order_individial(self.unknown['B_psi_coef_cp'], 'B_psi_coef_cp', n-2)
        check_order_individial(self.unknown['B_theta_coef_cp'], 'B_theta_coef_cp', n)
        check_order_individial(self.constant['B_denom_coef_c'], 'B_denom_coef_c', n)
        check_order_individial(self.constant['B_alpha_coef'], 'B_alpha_coef', n//2)
        check_order_individial(self.constant['iota_coef'], 'iota_coef', (n-2)//2)
        check_order_individial(self.unknown['p_perp_coef_cp'], 'p_perp_coef_cp', n)
        check_order_individial(self.unknown['Delta_coef_cp'], 'Delta_coef_cp', n)

    # Checks the accuracy of iteration at order n_unknown by substituting
    # results into the original form of the governing equations.
    # not nfp-dependent
    def check_governing_equations(self, n_unknown:int, normalize:bool=True):
        X_coef_cp = self.unknown['X_coef_cp']
        Y_coef_cp = self.unknown['Y_coef_cp']
        Z_coef_cp = self.unknown['Z_coef_cp']
        B_theta_coef_cp = self.unknown['B_theta_coef_cp']
        B_psi_coef_cp = self.unknown['B_psi_coef_cp']
        iota_coef = self.constant['iota_coef']
        p_perp_coef_cp = self.unknown['p_perp_coef_cp']
        Delta_coef_cp = self.unknown['Delta_coef_cp']
        B_denom_coef_c = self.constant['B_denom_coef_c']
        B_alpha_coef = self.constant['B_alpha_coef']
        kap_p = self.constant['kap_p']
        dl_p = self.constant['dl_p']
        tau_p = self.constant['tau_p']

        J = validate_J(n_unknown,
            X_coef_cp,
            Y_coef_cp,
            Z_coef_cp,
            B_denom_coef_c, B_alpha_coef,
            kap_p, dl_p, tau_p, iota_coef)
        Cb = validate_Cb(n_unknown-1,
            X_coef_cp,
            Y_coef_cp, Z_coef_cp,
            B_denom_coef_c, B_alpha_coef,
            B_psi_coef_cp, B_theta_coef_cp,
            kap_p, dl_p, tau_p, iota_coef)
        # The general expression does not work for the zeroth order, because it
        # performs order-matching with symbolic order n and then sub in n at
        # evaluation time, rather than actually performing order-matching at each
        # order. As a result, it IGNORES constant terms in the equation.
        if n_unknown==1:
            Cb-=dl_p
        Ck = validate_Ck(n_unknown-1, X_coef_cp, Y_coef_cp, Z_coef_cp,
            B_denom_coef_c, B_alpha_coef,
            B_psi_coef_cp, B_theta_coef_cp,
            kap_p, dl_p, tau_p, iota_coef)
        Ct = validate_Ct(n_unknown-1, X_coef_cp, Y_coef_cp, Z_coef_cp,
            B_denom_coef_c, B_alpha_coef,
            B_psi_coef_cp, B_theta_coef_cp,
            kap_p, dl_p, tau_p, iota_coef)
        if self.magnetic_only:
            I = ChiPhiFuncSpecial(0)
            II = ChiPhiFuncSpecial(0)
            III = ChiPhiFuncSpecial(0)
        else:
            I = validate_I(n_unknown, B_denom_coef_c,
                p_perp_coef_cp, Delta_coef_cp,
                iota_coef)
            II = validate_II(n_unknown,
                B_theta_coef_cp, B_alpha_coef, B_denom_coef_c,
                p_perp_coef_cp, Delta_coef_cp, iota_coef)
            III = validate_III(n_unknown-2,
            B_theta_coef_cp, B_psi_coef_cp,
            B_alpha_coef, B_denom_coef_c,
            p_perp_coef_cp, Delta_coef_cp,
            iota_coef)
        if normalize:
            J_norm = (X_coef_cp[n_unknown] * (2 * dl_p ** 2 * kap_p)).get_max()
            Cb_norm = (
                - n_unknown * B_alpha_coef[0]*X_coef_cp[1].dchi() / 2 * Y_coef_cp[n_unknown]
                + (B_alpha_coef[0] * X_coef_cp[1]) / 2 * Y_coef_cp[n_unknown].dchi()
            ).get_max()
            Ck_norm = (
                Z_coef_cp[n_unknown] * (-B_alpha_coef[0] * Y_coef_cp[1].dchi()) * (n_unknown+1)
                + Z_coef_cp[n_unknown].dchi() * (B_alpha_coef[0] * Y_coef_cp[1]) / 2
            ).get_max()
            Ct_norm = (
                Z_coef_cp[n_unknown] * (B_alpha_coef[0]*X_coef_cp[1].dchi()) * (n_unknown+1)
                - Z_coef_cp[n_unknown].dchi() * (B_alpha_coef[0]*X_coef_cp[1]) / 2
            ).get_max()
            I_norm = (
                (
                    Delta_coef_cp[n_unknown].dphi() 
                    + iota_coef[0] * Delta_coef_cp[n_unknown].dchi()
                )*B_denom_coef_c[0]
            ).get_max()
            II_norm = (
                - (B_denom_coef_c[0]**2 * (p_perp_coef_cp[0].dphi())) * B_theta_coef_cp[n_unknown]
                + (B_denom_coef_c[0] * Delta_coef_cp[0] - B_denom_coef_c[0]) * iota_coef[0] * B_theta_coef_cp[n_unknown].dchi()
                + (B_denom_coef_c[0] * Delta_coef_cp[0] - B_denom_coef_c[0]) * B_theta_coef_cp[n_unknown].dphi()
            ).get_max()
            III_norm = (
                n_unknown*B_alpha_coef[0]*B_denom_coef_c[0] ** 2/2 * p_perp_coef_cp[n_unknown]
            ).get_max()
            return(
                J/J_norm,
                Cb/Cb_norm,
                Ck/Ck_norm, 
                Ct/Ct_norm,
                I/I_norm, 
                II/II_norm,
                III/III_norm
            )
        else:
            return(
                J, # Unit is X*(2*dl_p^2*kap_p)
                Cb, # Unit is X*dl_p
                Ck, # Unit is X*dl_p
                Ct, # Unit is X*dl_p
                I, # Unit is (dphi + iota_coef[0] * dchi) Delta*B_denom_coef_c[0]
                # or p*B_alpha_coef[0]·B_denom_coef_c[0]^2. All components of 
                # the force balance should have the same unit.
                II, # Unit is p*B_alpha_coef[0]·B_denom_coef_c[0]^2
                III # Unit is p*B_alpha_coef[0]*B_denom_coef_c[0]^2
            )

tree_util.register_pytree_node(
Equilibrium,
Equilibrium._tree_flatten,
Equilibrium._tree_unflatten)

''' Iteration '''
# Evaluates 2 entire orders. Note that no masking is needed for any of the methods
# defined in this file. Copies the equilibrium. STOPS AT EVEN ORDERS.

# Iterates the magnetic equations only.
# Calculates Xn, Yn, Zn, B_psin-2 for 2 orders from lower order values.
# B_theta, B_psi_nm30, Y_free_nm1 are all free.
# n_eval must be even.
# not nfp-dependent
# Yn0, B_psi_nm20 must both be consts or 1d arrays.
def iterate_2_magnetic_only(equilibrium,
    B_theta_nm1, B_theta_n,
    Yn0,
    Yn1c_avg,
    B_psi_nm20,
    B_alpha_nb2,
    B_denom_nm1, B_denom_n,
    iota_nm2b2,
    static_max_freq=(-1,-1),
    traced_max_freq=(-1,-1),
    n_eval=None,
):

    if not equilibrium.magnetic_only:
        return()
    if static_max_freq == None:
        len_phi = equilibrium.unknown['X_coef_cp'][1].content.shape[1]
        static_max_freq = (len_phi//2, len_phi//2)

    # If no order is supplied, then iterate to the next order. the equilibrium
    # will be edited directly.
    if n_eval == None:
        n_eval = equilibrium.get_order() + 2 # getting order and checking consistency
    if n_eval%2 != 0:
        raise ValueError("n must be even to evaluate iota_{(n-1)/2}")

    print("Evaluating order",n_eval-1, n_eval)

    # Creating new ChiPhiEpsFunc's for the resulting equilibrium
    X_coef_cp = equilibrium.unknown['X_coef_cp'].mask(n_eval-2)
    Y_coef_cp = equilibrium.unknown['Y_coef_cp'].mask(n_eval-2)
    Z_coef_cp = equilibrium.unknown['Z_coef_cp'].mask(n_eval-2)
    B_theta_coef_cp = equilibrium.unknown['B_theta_coef_cp'].mask(n_eval-2)
    B_psi_coef_cp = equilibrium.unknown['B_psi_coef_cp'].mask(n_eval-4)
    iota_coef = equilibrium.constant['iota_coef'].mask((n_eval-4)//2)
    p_perp_coef_cp = equilibrium.unknown['p_perp_coef_cp'].mask(n_eval-2)
    Delta_coef_cp = equilibrium.unknown['Delta_coef_cp'].mask(n_eval-2)

    # Masking all init conds.
    B_denom_coef_c = equilibrium.constant['B_denom_coef_c'].mask(n_eval-2)
    B_alpha_coef = equilibrium.constant['B_alpha_coef'].mask((n_eval)//2-1)
    kap_p = equilibrium.constant['kap_p']
    dl_p = equilibrium.constant['dl_p']
    tau_p = equilibrium.constant['tau_p']
    # Appending free functions and initial conditions
    B_theta_coef_cp = B_theta_coef_cp.append(B_theta_nm1)
    B_theta_coef_cp = B_theta_coef_cp.append(B_theta_n)
    B_alpha_coef = B_alpha_coef.append(B_alpha_nb2)
    iota_coef = iota_coef.append(iota_nm2b2)
    B_denom_coef_c = B_denom_coef_c.append(B_denom_nm1)
    B_denom_coef_c = B_denom_coef_c.append(B_denom_n)
    p_perp_coef_cp = p_perp_coef_cp.append(ChiPhiFuncSpecial(0))
    p_perp_coef_cp = p_perp_coef_cp.append(ChiPhiFuncSpecial(0))
    Delta_coef_cp = Delta_coef_cp.append(ChiPhiFuncSpecial(0))
    Delta_coef_cp = Delta_coef_cp.append(ChiPhiFuncSpecial(0))

    # Evaluating order n_eval-1
    # Requires:
    # X_{n-1}, Y_{n-1}, Z_{n-1},
    # B_{\theta n-1}, B_0,
    # B_{\alpha 0}, \bar{\iota}_{(n-2)/2 or (n-3)/2}$
    B_psi_nm3 = iterate_dc_B_psi_nm2(n_eval=n_eval-1,
        X_coef_cp=X_coef_cp,
        Y_coef_cp=Y_coef_cp,
        Z_coef_cp=Z_coef_cp,
        B_theta_coef_cp=B_theta_coef_cp,
        B_psi_coef_cp=B_psi_coef_cp,
        B_alpha_coef=B_alpha_coef,
        B_denom_coef_c=B_denom_coef_c,
        kap_p=kap_p,
        dl_p=dl_p,
        tau_p=tau_p,
        iota_coef=iota_coef
        ).antid_chi()
    B_psi_coef_cp = B_psi_coef_cp.append(B_psi_nm3.filter(traced_max_freq[0]))

    # Requires:
    # X_{n-1}, Y_{n-1}, Z_{n-1},
    # B_{\theta n}, B_{\psi n-2},
    # B_{\alpha (n-2)/2 or (n-3)/2},
    # \iota_{(n-2)/2 or (n-3)/2}
    # \kappa, \frac{dl}{d\phi}, \tau
    Znm1 = iterate_Zn_cp(n_eval=n_eval-1,
        X_coef_cp=X_coef_cp,
        Y_coef_cp=Y_coef_cp,
        Z_coef_cp=Z_coef_cp,
        B_theta_coef_cp=B_theta_coef_cp,
        B_psi_coef_cp=B_psi_coef_cp,
        B_alpha_coef=B_alpha_coef,
        kap_p=kap_p,
        dl_p=dl_p,
        tau_p=tau_p,
        iota_coef=iota_coef
        )
    Z_coef_cp = Z_coef_cp.append(Znm1.filter(traced_max_freq[0]))

    # Requires:
    # X_{n-1}, Y_{n-1}, Z_n,
    # \iota_{(n-3)/2 or (n-2)/2},
    # B_{\alpha (n-1)/2 or n/2}.
    Xnm1 = iterate_Xn_cp(n_eval=n_eval-1,
        X_coef_cp=X_coef_cp,
        Y_coef_cp=Y_coef_cp,
        Z_coef_cp=Z_coef_cp,
        B_denom_coef_c=B_denom_coef_c,
        B_alpha_coef=B_alpha_coef,
        kap_p=kap_p,
        dl_p=dl_p,
        tau_p=tau_p,
        iota_coef=iota_coef
        )
    X_coef_cp = X_coef_cp.append(Xnm1.filter(traced_max_freq[0]))

    # Requires:
    # X_{n}, Y_{n-1}, Z_{n-1},
    # B_{\theta n-1}, B_{\psi n-3},
    # \iota_{(n-3)/2 or (n-4)/2}, B_{\alpha (n-1)/2 or (n-2)/2}
    Ynm1 = iterate_Yn_cp_magnetic(n_unknown=n_eval-1,
        X_coef_cp=X_coef_cp,
        Y_coef_cp=Y_coef_cp,
        Z_coef_cp=Z_coef_cp,
        B_psi_coef_cp=B_psi_coef_cp,
        B_theta_coef_cp=B_theta_coef_cp,
        B_alpha_coef=B_alpha_coef,
        B_denom_coef_c=B_denom_coef_c,
        kap_p=kap_p,
        dl_p=dl_p,
        tau_p=tau_p,
        iota_coef=iota_coef,
        static_max_freq=static_max_freq[0],
        Yn1c_avg=Yn1c_avg
    )
    Y_coef_cp = Y_coef_cp.append(Ynm1.filter(traced_max_freq[0]))

    # Order n_eval ---------------------------------------------------
    # no need to ignore_mode_0 for chi integral. This is an odd order.
    B_psi_nm2 = iterate_dc_B_psi_nm2(n_eval=n_eval,
        X_coef_cp=X_coef_cp,
        Y_coef_cp=Y_coef_cp,
        Z_coef_cp=Z_coef_cp,
        B_theta_coef_cp=B_theta_coef_cp,
        B_psi_coef_cp=B_psi_coef_cp,
        B_alpha_coef=B_alpha_coef,
        B_denom_coef_c=B_denom_coef_c,
        kap_p=kap_p,
        dl_p=dl_p,
        tau_p=tau_p,
        iota_coef=iota_coef
        ).antid_chi()
    B_psi_nm2_content_new = B_psi_nm2.content.at[B_psi_nm2.content.shape[0]//2].set(B_psi_nm20)
    B_psi_nm2 = ChiPhiFunc(B_psi_nm2_content_new, B_psi_nm2.nfp)
    B_psi_coef_cp = B_psi_coef_cp.append(B_psi_nm2.filter(traced_max_freq[1]))

    Zn = iterate_Zn_cp(n_eval=n_eval,
        X_coef_cp=X_coef_cp,
        Y_coef_cp=Y_coef_cp,
        Z_coef_cp=Z_coef_cp,
        B_theta_coef_cp=B_theta_coef_cp,
        B_psi_coef_cp=B_psi_coef_cp,
        B_alpha_coef=B_alpha_coef,
        kap_p=kap_p,
        dl_p=dl_p,
        tau_p=tau_p,
        iota_coef=iota_coef
        )
    Z_coef_cp = Z_coef_cp.append(Zn.filter(traced_max_freq[1]))

    Xn = iterate_Xn_cp(n_eval=n_eval,
        X_coef_cp=X_coef_cp,
        Y_coef_cp=Y_coef_cp,
        Z_coef_cp=Z_coef_cp,
        B_denom_coef_c=B_denom_coef_c,
        B_alpha_coef=B_alpha_coef,
        kap_p=kap_p,
        dl_p=dl_p,
        tau_p=tau_p,
        iota_coef=iota_coef
        )
    X_coef_cp = X_coef_cp.append(Xn.filter(traced_max_freq[1]))

    Yn = iterate_Yn_cp_magnetic(n_unknown=n_eval,
        X_coef_cp=X_coef_cp,
        Y_coef_cp=Y_coef_cp,
        Z_coef_cp=Z_coef_cp,
        B_psi_coef_cp=B_psi_coef_cp,
        B_theta_coef_cp=B_theta_coef_cp,
        B_alpha_coef=B_alpha_coef,
        B_denom_coef_c=B_denom_coef_c,
        kap_p=kap_p,
        dl_p=dl_p,
        tau_p=tau_p,
        iota_coef=iota_coef,
        static_max_freq=static_max_freq[1],
        Yn0=Yn0
    )
    Y_coef_cp = Y_coef_cp.append(Yn.filter(traced_max_freq[1]))

    # return(X_coef_cp,
    #     Y_coef_cp,
    #     Z_coef_cp,
    #     B_psi_coef_cp,
    #     B_theta_coef_cp,
    #     B_denom_coef_c,
    #     B_alpha_coef,
    #     kap_p,
    #     dl_p,
    #     tau_p,
    #     iota_coef,
    #     p_perp_coef_cp,
    #     Delta_coef_cp,
    # )

    return(Equilibrium.from_known(
        X_coef_cp=X_coef_cp,
        Y_coef_cp=Y_coef_cp,
        Z_coef_cp=Z_coef_cp,
        B_psi_coef_cp=B_psi_coef_cp,
        B_theta_coef_cp=B_theta_coef_cp,
        B_denom_coef_c=B_denom_coef_c,
        B_alpha_coef=B_alpha_coef,
        kap_p=kap_p,
        dl_p=dl_p,
        tau_p=tau_p,
        iota_coef=iota_coef,
        p_perp_coef_cp=p_perp_coef_cp,
        Delta_coef_cp=Delta_coef_cp,
        magnetic_only=True,
        axis_info=equilibrium.axis_info
    ))

# @partial(jit, static_argnums=(5, ))
def iterate_2(equilibrium,
    B_alpha_nb2=0,
    B_denom_nm1=None, B_denom_n=None,
    # Not implemented yet. At odd orders, the periodic BC
    # will constrain one of sigma_n(0), iota_n and avg(B_theta_n0)
    # given value for the other 2.
    # free_param_values need to be a 2-element tuple.
    # Now only implemented avg(B_theta_n0)=0 and given iota.
    iota_new=None, # arg 4
    B_theta_n0_avg=None,
    static_max_freq=(-1,-1),
    traced_max_freq=(-1,-1),
    # Traced.
    # -1 represents no filtering (default). This value is chosen so that
    # turning on or off off-diagonal filtering does not require recompiles.
    max_k_diff_pre_inv=(-1,-1),
    n_eval=None,
    ):
    if equilibrium.magnetic_only:
        return()
    if B_denom_nm1 is None:
        B_denom_nm1 = ChiPhiFunc(
            jnp.array([[0],[0]]),
            equilibrium.nfp
        )
    if B_denom_n is None:
        B_denom_n = ChiPhiFunc(
            jnp.array([[0]]),
            equilibrium.nfp
        )
    if n_eval == None:
        n_eval = equilibrium.get_order() + 2 # getting order and checking consistency
    if n_eval%2 != 0:
        raise ValueError("n must be even to evaluate iota_{(n-1)/2}")

    print("Evaluating order",n_eval-1, n_eval)

    # Resetting unknowns
    X_coef_cp = equilibrium.unknown['X_coef_cp'].mask(n_eval-2)
    Y_coef_cp = equilibrium.unknown['Y_coef_cp'].mask(n_eval-2)
    Z_coef_cp = equilibrium.unknown['Z_coef_cp'].mask(n_eval-2)
    B_theta_coef_cp = equilibrium.unknown['B_theta_coef_cp'].mask(n_eval-2)
    B_psi_coef_cp = equilibrium.unknown['B_psi_coef_cp'].mask(n_eval-4)
    iota_coef = equilibrium.constant['iota_coef'].mask((n_eval-4)//2)
    p_perp_coef_cp = equilibrium.unknown['p_perp_coef_cp'].mask(n_eval-2)
    Delta_coef_cp = equilibrium.unknown['Delta_coef_cp'].mask(n_eval-2)

    # Masking all init conds.
    B_denom_coef_c = equilibrium.constant['B_denom_coef_c'].mask(n_eval-2)
    B_alpha_coef = equilibrium.constant['B_alpha_coef'].mask((n_eval)//2-1)
    kap_p = equilibrium.constant['kap_p']
    dl_p = equilibrium.constant['dl_p']
    tau_p = equilibrium.constant['tau_p']

    B_denom_coef_c = B_denom_coef_c.append(B_denom_nm1)
    B_denom_coef_c = B_denom_coef_c.append(B_denom_n)
    B_alpha_coef = B_alpha_coef.append(B_alpha_nb2)
    if (iota_new is not None) and (B_theta_n0_avg is not None):
        raise ValueError('Only one of iota[n/2-1] and B_theta[n,0] avg should be provided.')
    # B_theta average is zero by default
    # for current-free configs.
    if (iota_new is None):
        if (B_theta_n0_avg is None):
            B_theta_n0_avg = 0.0
    else:
        # Append iota only if it's provided.
        # Otherwise, iterate_looped will work with
        # iota that's one element short automatically.
        iota_coef = iota_coef.append(iota_new)
    # print('iota 1 right before loop',iota_coef[1])
    solution_nm1_known_iota = iterate_looped(
        n_unknown=n_eval-1,
        nfp=equilibrium.nfp,
        target_len_phi=X_coef_cp[n_eval-2].content.shape[1],
        X_coef_cp=X_coef_cp,
        Y_coef_cp=Y_coef_cp,
        Z_coef_cp=Z_coef_cp,
        p_perp_coef_cp=p_perp_coef_cp,
        Delta_coef_cp=Delta_coef_cp,
        B_psi_coef_cp=B_psi_coef_cp,
        B_theta_coef_cp=B_theta_coef_cp,
        B_alpha_coef=B_alpha_coef,
        B_denom_coef_c=B_denom_coef_c,
        kap_p=kap_p,
        tau_p=tau_p,
        dl_p=dl_p,
        iota_coef=iota_coef,
        static_max_freq=static_max_freq[0],
        traced_max_freq=traced_max_freq[0],
        max_k_diff_pre_inv=max_k_diff_pre_inv[0],
        solve_iota=iota_new is None,
        B_theta_np10_avg=B_theta_n0_avg
    )
    if (iota_new is None):
        iota_coef = iota_coef.append(solution_nm1_known_iota['iota_nm1b2'])
    B_theta_coef_cp = B_theta_coef_cp.append(solution_nm1_known_iota['B_theta_n']) 
    B_psi_coef_cp = B_psi_coef_cp.append(solution_nm1_known_iota['B_psi_nm2']) 
    X_coef_cp = X_coef_cp.append(solution_nm1_known_iota['Xn']) 
    Y_coef_cp = Y_coef_cp.append(solution_nm1_known_iota['Yn']) 
    Z_coef_cp = Z_coef_cp.append(solution_nm1_known_iota['Zn']) 
    p_perp_coef_cp = p_perp_coef_cp.append(solution_nm1_known_iota['pn']) 
    Delta_coef_cp = Delta_coef_cp.append(solution_nm1_known_iota['Deltan']) 

    # This "partial" solution will be fed into
    # iterate_looped. This is already filtered.
    B_theta_coef_cp = B_theta_coef_cp.append(solution_nm1_known_iota['B_theta_np10'])

    # B_psi[n-2] without the zeroth component.
    # MAY NOT BE NECESSARY
    B_psi_nm2 = iterate_dc_B_psi_nm2(n_eval=n_eval,
        X_coef_cp=X_coef_cp,
        Y_coef_cp=Y_coef_cp,
        Z_coef_cp=Z_coef_cp,
        B_theta_coef_cp=B_theta_coef_cp,
        B_psi_coef_cp=B_psi_coef_cp,
        B_alpha_coef=B_alpha_coef,
        B_denom_coef_c=B_denom_coef_c,
        kap_p=kap_p,
        dl_p=dl_p,
        tau_p=tau_p,
        iota_coef=iota_coef
        ).antid_chi()
    B_psi_coef_cp = B_psi_coef_cp.append(B_psi_nm2)

    solution_n = iterate_looped(
        n_unknown=n_eval,
        nfp=equilibrium.nfp,
        target_len_phi=X_coef_cp[n_eval-2].content.shape[1],
        X_coef_cp=X_coef_cp,
        Y_coef_cp=Y_coef_cp,
        Z_coef_cp=Z_coef_cp,
        p_perp_coef_cp=p_perp_coef_cp,
        Delta_coef_cp=Delta_coef_cp,
        B_psi_coef_cp=B_psi_coef_cp,
        B_theta_coef_cp=B_theta_coef_cp,
        B_alpha_coef=B_alpha_coef,
        B_denom_coef_c=B_denom_coef_c,
        kap_p=kap_p,
        tau_p=tau_p,
        dl_p=dl_p,
        iota_coef=iota_coef,
        static_max_freq=static_max_freq[1],
        traced_max_freq=traced_max_freq[1],
        max_k_diff_pre_inv=max_k_diff_pre_inv[1],
    )
    # Partial solutions for these variables were appended to their
    # ChiPhiEpsFunc's for iterate_looped. Now remove them and re-append
    # This only reassigns the pointer B_psi. Need to re-assign equilibrium.unknown[]
    # too.
    B_psi_coef_cp = B_psi_coef_cp.mask(n_eval-3)
    B_psi_coef_cp = B_psi_coef_cp.append(solution_n['B_psi_nm2'].filter(traced_max_freq[1]))
    # This only reassigns the pointer B_theta. Need to re-assign equilibrium.unknown[]
    # too.
    B_theta_coef_cp = B_theta_coef_cp.mask(n_eval-1)
    B_theta_coef_cp = B_theta_coef_cp.append(solution_n['B_theta_n'].filter(traced_max_freq[1]))

    X_coef_cp = X_coef_cp.append(solution_n['Xn']) 
    Z_coef_cp = Z_coef_cp.append(solution_n['Zn']) 
    p_perp_coef_cp = p_perp_coef_cp.append(solution_n['pn']) 
    Delta_coef_cp = Delta_coef_cp.append(solution_n['Deltan']) 
    Y_coef_cp = Y_coef_cp.append(solution_n['Yn']) 

    return(Equilibrium.from_known(
        X_coef_cp=X_coef_cp,
        Y_coef_cp=Y_coef_cp,
        Z_coef_cp=Z_coef_cp,
        B_psi_coef_cp=B_psi_coef_cp,
        B_theta_coef_cp=B_theta_coef_cp,
        B_denom_coef_c=B_denom_coef_c,
        B_alpha_coef=B_alpha_coef,
        kap_p=kap_p,
        dl_p=dl_p,
        tau_p=tau_p,
        iota_coef=iota_coef,
        p_perp_coef_cp=p_perp_coef_cp,
        Delta_coef_cp=Delta_coef_cp,
        magnetic_only=False,
        axis_info=equilibrium.axis_info
    ))

''' Utilities '''
def helicity_from_axis(axis_info, nfp):
    ''' 
    Returns the helicity (the normal vector, kappa's 
    # rotation around the origin)
    '''
    # R, phi, Z of the normal vector
    norm_R = jnp.pad(axis_info['normal_cylindrical'][:, 0], (0, 1), 'wrap')
    norm_Z = jnp.pad(axis_info['normal_cylindrical'][:, -1], (0, 1), 'wrap')
    norm_R = norm_R-jnp.average(norm_R) # zero-centering
    norm_Z = norm_Z-jnp.average(norm_Z) # zero-centering
    # To make arctan accumulate over or under +- pi,
    # we need to detect crosses between quadrant 2 and 3
    arctan = jnp.arctan2(norm_Z,norm_R)
    arctan_unwrap = jnp.unwrap(arctan)
    return((arctan_unwrap[-1]-arctan_unwrap[0])/(2*jnp.pi)*nfp)
