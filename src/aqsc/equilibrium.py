# Wrapped/completed recursion relations based on translated expressions
# in parsed/. Necessary masking and/or n-substitution are included. All iterate_*
# methods returns ChiPhiFunc's.
import jax.numpy as jnp
import numpy as np # used in save_plain and get_helicity
# from jax import jit, vmap, tree_util
from jax import tree_util
# from functools import partial # for JAX jit with static params
# from matplotlib import pyplot as plt

# ChiPhiFunc and ChiPhiEpsFunc
from .chiphifunc import *
from .chiphiepsfunc import *
from .math_utilities import *
from .recursion_relations import *
from .looped_solver import iterate_looped
# parsed relations
from .MHD_parsed import validate_J, validate_Cb, validate_Ck, \
    validate_Ct, validate_I, validate_II, validate_III

# Plotting
import matplotlib.pyplot as plt
from matplotlib import cm, colors

''' Equilibrium manager and Iterate '''

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
    
    ''' Coordinate transformations '''
    def frenet_basis_phi(self, phi):
        ''' 
        Calculates frenet basis vectors (in R, Phi, Z) from the flux coordinat phi. 
        '''
        varphi = self.axis_info['varphi']
        R0 = self.axis_info['R0']
        phi_grids = self.axis_info['phi']
        Z0 = self.axis_info['Z0']
        tangent_cylindrical = self.axis_info['tangent_cylindrical']
        normal_cylindrical = self.axis_info['normal_cylindrical']
        binormal_cylindrical = self.axis_info['binormal_cylindrical']
        nfp=self.nfp
        # axis location and basis in term of Boozer phi
        period = 2*jnp.pi/nfp
        axis_r0_phi_R = jnp.interp(phi, varphi, R0, period=period)
        axis_r0_phi_Phi = jnp.interp(phi, varphi, phi_grids, period=period) + (phi//period)*period
        axis_r0_phi_Z = jnp.interp(phi, varphi, Z0, period=period)
        tangent_phi_R = jnp.interp(phi, varphi, tangent_cylindrical[:, 0], period=period)
        tangent_phi_Phi = jnp.interp(phi, varphi, tangent_cylindrical[:, 1], period=period)
        tangent_phi_Z = jnp.interp(phi, varphi, tangent_cylindrical[:, 2], period=period)
        normal_phi_R = jnp.interp(phi, varphi, normal_cylindrical[:, 0], period=period)
        normal_phi_Phi = jnp.interp(phi, varphi, normal_cylindrical[:, 1], period=period)
        normal_phi_Z = jnp.interp(phi, varphi, normal_cylindrical[:, 2], period=period)
        binormal_phi_R = jnp.interp(phi, varphi, binormal_cylindrical[:, 0], period=period)
        binormal_phi_Phi = jnp.interp(phi, varphi, binormal_cylindrical[:, 1], period=period)
        binormal_phi_Z = jnp.interp(phi, varphi, binormal_cylindrical[:, 2], period=period)
        return(
            axis_r0_phi_R,
            axis_r0_phi_Phi,
            axis_r0_phi_Z,
            tangent_phi_R,
            tangent_phi_Phi,
            tangent_phi_Z,
            normal_phi_R,
            normal_phi_Phi,
            normal_phi_Z,
            binormal_phi_R,
            binormal_phi_Phi,
            binormal_phi_Z
        )

    def flux_to_frenet(self, psi, chi, phi, n_max=float('inf')):
        ''' 
        Transforms positions in the flux coordinate to the frenet frame.
        Returns (curvature, binormal, tangent) in the Frenet frame.
        '''
        if double_precision:
            target_type=jnp.float64
        else:
            target_type=jnp.float32
        return(
            self.unknown['X_coef_cp'].eval(psi=psi, chi=chi, phi=phi, n_max=n_max), 
            self.unknown['Y_coef_cp'].eval(psi=psi, chi=chi, phi=phi, n_max=n_max), 
            self.unknown['Z_coef_cp'].eval(psi=psi, chi=chi, phi=phi, n_max=n_max)
        )

    def flux_to_cylindrical(self, psi, chi, phi, n_max=float('inf')):
        ''' 
        Transforms positions in the flux coordinate to the 
        cylindrical coordinate.
        (R, Phi, Z) in the cylindrical coordinate.
        '''
        axis_r0_phi_R,\
        axis_r0_phi_Phi,\
        axis_r0_phi_Z,\
        tangent_phi_R,\
        tangent_phi_Phi,\
        tangent_phi_Z,\
        normal_phi_R,\
        normal_phi_Phi,\
        normal_phi_Z,\
        binormal_phi_R,\
        binormal_phi_Phi,\
        binormal_phi_Z = self.frenet_basis_phi(phi)
        curvature, binormal, tangent = self.flux_to_frenet(psi=psi, chi=chi, phi=phi, n_max=n_max)
        components_R = axis_r0_phi_R\
            + tangent_phi_R * tangent\
            + normal_phi_R * curvature\
            + binormal_phi_R * binormal
        
        components_Phi = axis_r0_phi_Phi\
            + tangent_phi_Phi * tangent\
            + normal_phi_Phi * curvature\
            + binormal_phi_Phi * binormal
        
        components_Z = axis_r0_phi_Z\
            + tangent_phi_Z * tangent\
            + normal_phi_Z * curvature\
            + binormal_phi_Z * binormal
        return(
            components_R,
            components_Phi,
            components_Z
        )

    def flux_to_xyz(self, psi, chi, phi, n_max=float('inf')):
        ''' 
        Transforms positions in the flux coordinate to the 
        XYZ coordinate.
        Returns (X, Y, Z) in the cylindrical coordinate.
        '''
        R, Phi, Z = self.flux_to_cylindrical(psi=psi, chi=chi, phi=phi, n_max=n_max)
        X = R*jnp.cos(Phi)
        Y = R*jnp.sin(Phi)
        return(X, Y, Z)

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
            numpy_axis_info[key] = np.asarray(self.axis_info[key])

        big_dict = {
            'unknown':unknown_dict,
            'constant':constant_dict,
            'nfp':self.nfp,
            'magnetic_only':self.magnetic_only,
            'axis_info':numpy_axis_info,
        }
        np.save(file_name, big_dict)

    # nfp-dependent!!
    def load_plain(filename):
        npyfile = np.load(filename, allow_pickle=True)
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
    def check_governing_equations(self, n_unknown:int):
        # if n_unknown is None:
        #     n_unknown = self.get_order()
        # elif n_unknown>self.get_order():
        #     raise ValueError('Validation order should be <= than the current order')

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
        if n_unknown==0:
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
        return(J, Cb, Ck, Ct, I, II, III)

    ''' Display and output'''

    def get_psi_crit(
        self, n_max=float('inf'), 
        n_grid_chi=100, n_grid_phi=100, 
        eps_cap = 3.0,
        n_newton_iter = 20):
        eps_crit, jacobian_residue = self.get_eps_crit(
            n_max=n_max, 
            n_grid_chi=n_grid_chi, 
            n_grid_phi=n_grid_phi, 
            eps_cap=eps_cap,
            n_newton_iter=n_newton_iter
        )
        return(eps_crit**2, jacobian_residue)
        

    def get_eps_crit(
        self, n_max=float('inf'), 
        n_grid_chi=100, n_grid_phi=100, 
        eps_cap = 2.0,
        n_newton_iter = 10):
        '''
        Estimates the critical epsilon where flux surface self-intersects.
        by finding the zero of $min_{\chi, \phi}[\sqrt{g}(\epsilon, \chi, \phi)]=0$
        using binary search. \sqrt{g}(\epsilon, \chi, \phi) At each search step 
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
        points = jnp.linspace(0, 2*np.pi*(1-1/n_grid_phi), n_grid_phi)
        points_chi = jnp.linspace(0, 2*np.pi*(1-1/n_grid_chi), n_grid_chi)
        axis_r0_phi_R,\
        axis_r0_phi_Phi,\
        axis_r0_phi_Z,\
        tangent_phi_R,\
        tangent_phi_Phi,\
        tangent_phi_Z,\
        normal_phi_R,\
        normal_phi_Phi,\
        normal_phi_Z,\
        binormal_phi_R,\
        binormal_phi_Phi,\
        binormal_phi_Z = self.frenet_basis_phi(points)
        axis_r0 = jnp.real(jnp.array([
            axis_r0_phi_R,
            axis_r0_phi_Phi,
            axis_r0_phi_Z
        ]))
        tangent = jnp.real(jnp.array([
            tangent_phi_R,
            tangent_phi_Phi,
            tangent_phi_Z
        ]))
        normal = jnp.real(jnp.array([
            normal_phi_R,
            normal_phi_Phi,
            normal_phi_Z
        ]))
        binormal = jnp.real(jnp.array([
            binormal_phi_R,
            binormal_phi_Phi,
            binormal_phi_Z
        ]))

        # These quantities are periodic, and we apply
        # spectral derivatives, taking advantage of implementation
        # in aqsc.
        tangent_dphi = jnp.real(ChiPhiFunc(tangent, self.nfp).dphi().content)
        normal_dphi = jnp.real(ChiPhiFunc(normal, self.nfp).dphi().content)
        binormal_dphi = jnp.real(ChiPhiFunc(binormal, self.nfp).dphi().content)
        axis_r0_dphi = jnp.real(jnp.gradient(axis_r0, points[1]-points[0], axis=1)) # First and last 2 elems untrustworthy
        X_coef_cp = self.unknown['X_coef_cp']
        Y_coef_cp = self.unknown['Y_coef_cp']
        Z_coef_cp = self.unknown['Z_coef_cp']

        # Returns (n_grid, n_grid) arrays. Axis=1 is Phi.
        X_grid = lambda eps: jnp.real(X_coef_cp.eval(eps**2, points_chi[:, None], points[None, :], n_max=n_max))
        Y_grid = lambda eps: jnp.real(Y_coef_cp.eval(eps**2, points_chi[:, None], points[None, :], n_max=n_max))
        Z_grid = lambda eps: jnp.real(Z_coef_cp.eval(eps**2, points_chi[:, None], points[None, :], n_max=n_max))
        deps_X_grid = lambda eps: jnp.real(X_coef_cp.deps().eval(eps**2, points_chi[:, None], points[None, :], n_max=n_max))
        deps_Y_grid = lambda eps: jnp.real(Y_coef_cp.deps().eval(eps**2, points_chi[:, None], points[None, :], n_max=n_max))
        deps_Z_grid = lambda eps: jnp.real(Z_coef_cp.deps().eval(eps**2, points_chi[:, None], points[None, :], n_max=n_max))
        dchi_X_grid = lambda eps: jnp.real(X_coef_cp.dchi().eval(eps**2, points_chi[:, None], points[None, :], n_max=n_max))
        dchi_Y_grid = lambda eps: jnp.real(Y_coef_cp.dchi().eval(eps**2, points_chi[:, None], points[None, :], n_max=n_max))
        dchi_Z_grid = lambda eps: jnp.real(Z_coef_cp.dchi().eval(eps**2, points_chi[:, None], points[None, :], n_max=n_max))
        dphi_X_grid = lambda eps: jnp.real(X_coef_cp.dphi().eval(eps**2, points_chi[:, None], points[None, :], n_max=n_max))
        dphi_Y_grid = lambda eps: jnp.real(Y_coef_cp.dphi().eval(eps**2, points_chi[:, None], points[None, :], n_max=n_max))
        dphi_Z_grid = lambda eps: jnp.real(Z_coef_cp.dphi().eval(eps**2, points_chi[:, None], points[None, :], n_max=n_max))

        deps_r = lambda eps:(
            deps_X_grid(eps)[None, :, :]*normal[:, None, :]
            +deps_Y_grid(eps)[None, :, :]*binormal[:, None, :]
            +deps_Z_grid(eps)[None, :, :]*tangent[:, None, :]
        )

        dchi_r = lambda eps:(
            dchi_X_grid(eps)[None, :, :]*normal[:, None, :]
            +dchi_Y_grid(eps)[None, :, :]*binormal[:, None, :]
            +dchi_Z_grid(eps)[None, :, :]*tangent[:, None, :]
        )

        dphi_r = lambda eps:(
            axis_r0_dphi[:, None, :]
            +dphi_X_grid(eps)[None, :, :]*normal[:, None, :]
            +dphi_Y_grid(eps)[None, :, :]*binormal[:, None, :]
            +dphi_Z_grid(eps)[None, :, :]*tangent[:, None, :]
            +X_grid(eps)[None, :, :]*normal_dphi[:, None, :]
            +Y_grid(eps)[None, :, :]*binormal_dphi[:, None, :]
            +Z_grid(eps)[None, :, :]*tangent_dphi[:, None, :]
        )
        # Sqrt g on grid
        # HACKY: np.gradient seem unreliable for the leading and trailing 2 elements/
        # Because g is a smooth function, we remove these two cols and assume errors 
        # from neglecting 2 cols of grid points in argmin{min[g(r)]<=0} is negligible.
        jacobian_grid = lambda eps: jnp.sum(
            jnp.cross(
                deps_r(eps), 
                dchi_r(eps), 
                axisa=0, axisb=0, axisc=0
            )*dphi_r(eps), 
            axis=0
        )[:, 2:-2] 

        # Finding jacobian_min=0 using Newtom's method.
        jacobian_min = lambda eps: jnp.min(jacobian_grid(eps))
        jacobian_min_prime = jax.grad(jacobian_min)
        def q(x):
            return x - jacobian_min(x) / jacobian_min_prime(x)
        eps_i = eps_cap
        for i in range(n_newton_iter):
            y = q(eps_i)
            eps_i = y
        return(eps_i, jacobian_min(eps_i))
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

    def get_helicity(self):
        ''' 
        Returns the helicity (the normal vector, kappa's 
        # rotation around the origin)
        '''
        return(helicity_from_axis(self.axis_info, self.nfp))

    def display(self, n_max=float('inf'), psi_max=None):
        if not psi_max:
            psi_max = self.get_psi_crit()[0]
            print('Plotting to 0.8(critical psi):', psi_max)
            psi_max*=0.8
        fig = plt.figure()
        fig.set_dpi(400)
        ax = fig.add_subplot(projection='3d')
        phis = jnp.linspace(0, np.pi*2*0.9, 100)
        chis = jnp.linspace(0, np.pi*2, 100)
        x_surf, y_surf, z_surf = self.flux_to_xyz(psi=psi_max, chi=chis[None, :], phi=phis[:, None], n_max=n_max)
        # Coloring by magnitude of B
        B_denom = self.constant['B_denom_coef_c'].eval(
            psi=psi_max, chi=chis[None, :], phi=phis[:, None]
        )
        B_magnitude = 1/jnp.real(B_denom).astype(jnp.float32)

        norm = colors.Normalize(vmin=jnp.min(B_magnitude), vmax=jnp.max(B_magnitude), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.plasma)
        facecolors = mapper.to_rgba(B_magnitude)

        ax.plot_surface(x_surf, y_surf, z_surf, zorder=1, facecolors=facecolors)
        ax.axis('equal')

        # Plotting flux surfaces.
        # Plot the first separately 
        x_cross, y_cross, z_cross = self.flux_to_xyz(psi=0, chi=chis, phi=0, n_max=n_max)
        ax.plot(x_cross, y_cross, z_cross, zorder=2.5, linewidth=0.5, color='lightgrey', label=r'$(\epsilon=\sqrt{\psi}, \chi)$')
        eps_max = np.sqrt(psi_max)
        for psi_i in np.linspace(eps_max/8, eps_max, 8)**2:
            x_cross, y_cross, z_cross = self.flux_to_xyz(psi=psi_i, chi=chis, phi=0, n_max=n_max)
            ax.plot(x_cross, y_cross, z_cross, zorder=2.5, linewidth=0.5, color='lightgrey')
        x_axis, y_axis, z_axis = self.flux_to_xyz(psi=0, chi=0, phi=chis, n_max=n_max)
        # Plotting the chi surfaces
        psis_dense = np.linspace(0, psi_max, 50)
        for chi_i in np.linspace(0.125, 1, 8)*np.pi*2:
            x_cross, y_cross, z_cross = self.flux_to_xyz(psi=psis_dense, chi=chi_i, phi=0, n_max=n_max)
            ax.plot(x_cross, y_cross, z_cross, zorder=2.5, linewidth=0.5, color='lightgrey')

        # Plotting the axis
        ax.plot(x_axis, y_axis, z_axis, zorder=3.5, linewidth=0.5, linestyle='dashed', color='lightgrey', label='Magnetic axis')
        fig.legend()
        fig.colorbar(mapper, label=r'$|B|^2$', shrink=0.5, ax=ax)
        fig.show()

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

tree_util.register_pytree_node(
Equilibrium,
Equilibrium._tree_flatten,
Equilibrium._tree_unflatten)

''' Iteration '''
# # Evaluates 2 entire orders. Note that no masking is needed for any of the methods
# # defined in this file. Copies the equilibrium. STOPS AT EVEN ORDERS.

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
    norm_R = np.pad(axis_info['normal_cylindrical'][:, 0], (0, 1), 'wrap')
    norm_Z = np.pad(axis_info['normal_cylindrical'][:, -1], (0, 1), 'wrap')
    norm_R = norm_R-np.average(norm_R) # zero-centering
    norm_Z = norm_Z-np.average(norm_Z) # zero-centering
    # To make arctan accumulate over or under +- pi,
    # we need to detect crosses between quadrant 2 and 3
    arctan = np.arctan2(norm_Z,norm_R)
    arctan_unwrap = np.unwrap(arctan)
    return((arctan_unwrap[-1]-arctan_unwrap[0])/(2*np.pi)*nfp)
