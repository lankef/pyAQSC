# Wrapped/completed recursion relations based on translated expressions
# in parsed/. Necessary masking and/or n-substitution are included. All iterate_*
# methods returns ChiPhiFunc's.
import jax.numpy as jnp
from jax import jit, vmap, tree_util
from functools import partial # for JAX jit with static params
from matplotlib import pyplot as plt

# ChiPhiFunc and ChiPhiEpsFunc
from .chiphifunc import *
from .chiphiepsfunc import *
from .math_utilities import *
import aqsc.looped_solver as looped_solver

# parsed relations
import aqsc.parsed as parsed
import aqsc.MHD_parsed as MHD_parsed
import aqsc.looped_coefs as looped_coefs

''' I. Magnetic equations '''
# The magnetic equations alone can serve as a set of recursion relations,
# solving for $X, Y, Z, B_{\psi}, \iota$ from $B_{\theta}, B_{alpha}$ and $B$.
''' I.1 Recursion relations for individual variables '''
# All iterate_<variable name> functions evaluate said variable at order n_eval.
# See overleaf (will be offline later) document which variables are needed and
# which orders are needed.
# not nfp-dependent
def iterate_Xn_cp(n_eval,
    X_coef_cp,
    Y_coef_cp,
    Z_coef_cp,
    B_denom_coef_c,
    B_alpha_coef,
    kap_p, dl_p, tau_p,
    iota_coef):
    return(parsed.eval_Xn_cp(
        n=n_eval,
        X_coef_cp=X_coef_cp.mask(n_eval-1).zero_append(),
        Y_coef_cp=Y_coef_cp.mask(n_eval-1),
        Z_coef_cp=Z_coef_cp.mask(n_eval),
        B_denom_coef_c=B_denom_coef_c,
        B_alpha_coef=B_alpha_coef,
        kap_p=kap_p, dl_p=dl_p, tau_p=tau_p,
        iota_coef=iota_coef).cap_m(n_eval))

# O_matrices, O_einv, vector_free_coef only uses B_alpha_coef and X_coef_cp
def iterate_Yn_cp_operators(n_unknown, X_coef_cp, B_alpha_coef, Y1c_mode=False): # nfp-dependent only in output
    '''
    Input: -----
    '''
    # Getting coeffs
    # Both uses B_alpha0 and X1 only
    chiphifunc_A = parsed.coef_a(n_unknown-1, B_alpha_coef, X_coef_cp)
    chiphifunc_B = parsed.coef_b(B_alpha_coef, X_coef_cp)

    # Calculating the inverted matrices
    O_matrices, O_einv, vector_free_coef = get_O_O_einv_from_A_B(
        chiphifunc_A=chiphifunc_A,
        chiphifunc_B=chiphifunc_B,
        rank_rhs=n_unknown,
        Y1c_mode=Y1c_mode)
    return(O_matrices, O_einv, vector_free_coef)

# O_matrices, O_einv, vector_free_coef only uses B_alpha_coef and X_coef_cp
# nfp-dependent!!
def iterate_Yn_cp_RHS(n_unknown,
    X_coef_cp,
    Y_coef_cp,
    Z_coef_cp,
    B_psi_coef_cp,
    B_theta_coef_cp,
    B_alpha_coef,
    B_denom_coef_c,
    kap_p, dl_p, tau_p,
    iota_coef): # nfp-dependent!!
    # This result is known due to indexing.
    if n_unknown == 1:
        return(ChiPhiFunc(jnp.zeros((3,X_coef_cp[1].content.shape[1])), X_coef_cp.nfp))
    # Getting rhs-lhs for the Yn+1 equation
    # for Yn to work, "n" must be subbed with n-1 here
    chiphifunc_rhs = parsed.rhs_minus_lhs(n_unknown-1,
        X_coef_cp,
        Y_coef_cp.mask(n_unknown-1).zero_append(),
        Z_coef_cp,
        B_psi_coef_cp,
        B_theta_coef_cp,
        B_alpha_coef,
        kap_p, dl_p, tau_p,
        iota_coef)

    # Making sure RHS isn't null
    if not isinstance(chiphifunc_rhs, ChiPhiFunc):
        return()

    len_chi = chiphifunc_rhs.content.shape[0]
    # Center-pad chiphifunc_rhs if it's too short
    # ChiPhiFunc_A always have 2 chi components.
    #          [0  ]
    # [RHS] => [RHS]
    #          [0  ]
    if n_unknown+2 != len_chi:
        chiphifunc_rhs = chiphifunc_rhs+\
            ChiPhiFunc(jnp.zeros(
                (
                    n_unknown+2,
                    chiphifunc_rhs.content.shape[1]
                ), dtype=jnp.complex128
            ), chiphifunc_rhs.nfp)

    return(chiphifunc_rhs)

# Evaluates Yn using Yn+1 general formula. The free component is either Yn0 or
# Yn1s.
# Requires:
# X_{n}, Y_{n-1}, Z_{n-1},
# B_{\theta n-1}, B_{\psi  n-3},
# \iota_{(n-3)/2 or (n-4)/2}, B_{\alpha  (n-1)/2 or (n-2)/2}
# nfp-dependent!!
# Cannot be jitted beause of the lambda funcs
def iterate_Yn_cp_magnetic(n_unknown,
    X_coef_cp,
    Y_coef_cp,
    Z_coef_cp,
    B_psi_coef_cp,
    B_theta_coef_cp,
    B_alpha_coef,
    B_denom_coef_c,
    kap_p, dl_p, tau_p,
    iota_coef,
    max_freq,
    Yn0=None):

    nfp = X_coef_cp.nfp
    n_eval = n_unknown+1

    _, O_einv, vector_free_coef = \
        iterate_Yn_cp_operators(n_unknown,
            X_coef_cp=X_coef_cp,
            B_alpha_coef=B_alpha_coef)

    Yn_rhs = iterate_Yn_cp_RHS(n_unknown=n_unknown,
        X_coef_cp=X_coef_cp,
        Y_coef_cp=Y_coef_cp,
        Z_coef_cp=Z_coef_cp,
        B_psi_coef_cp=B_psi_coef_cp,
        B_theta_coef_cp=B_theta_coef_cp.zero_append(),
        B_alpha_coef=B_alpha_coef,
        B_denom_coef_c=B_denom_coef_c,
        kap_p=kap_p,
        dl_p=dl_p,
        tau_p=tau_p,
        iota_coef=iota_coef)
    Yn_rhs_content = Yn_rhs.content
    new_Y_n_no_unknown = ChiPhiFunc(jnp.einsum('ijk,jk->ik',O_einv,Yn_rhs_content), Yn_rhs.nfp)
    Y_coef_cp_no_unknown = Y_coef_cp.mask(n_eval-2)
    Y_coef_cp_no_unknown.append(new_Y_n_no_unknown)

    # Calculating D3 to solve for Yn1p
    if n_unknown%2==0:
        if Yn0 is None:
            return(ChiPhiFuncSpecial(-15))
        Yn_free_content = Yn0
    else:
        D3_RHS_no_unknown = -parsed.eval_D3_RHS_m_LHS(
            n = n_eval,
            X_coef_cp = X_coef_cp.mask(n_unknown).zero_append(),
            # Only dep on Y[+-1]
            Y_coef_cp = Y_coef_cp_no_unknown.zero_append(),
            # The m=0 component is actually indep of Z[n+1]
            Z_coef_cp = Z_coef_cp.mask(n_unknown).zero_append(),
            # This equation may contain both B_theta[n,+-1] and B_theta[n+1,0].
            B_theta_coef_cp = B_theta_coef_cp.mask(n_unknown+1).zero_append(),
            B_denom_coef_c = B_denom_coef_c,
            B_alpha_coef = B_alpha_coef,
            iota_coef = iota_coef, #.mask((n_unknown-3)//2).zero_append(), # iota is also masked
            dl_p = dl_p,
            tau_p = tau_p,
        kap_p = kap_p)[0]

        coef_Yn1p_in_D3 = looped_coefs.lambda_coef_Yn1p_in_D3(
            vector_free_coef,
            X_coef_cp, Y_coef_cp,
            iota_coef,
            dl_p, tau_p, nfp
        )
        coef_dp_Yn1p_in_D3 = looped_coefs.lambda_coef_dp_Yn1p_in_D3(
            vector_free_coef,
            X_coef_cp, Y_coef_cp,
            iota_coef,
            dl_p, tau_p, nfp
        )

        Yn_free_content = solve_integration_factor(
            coeff_arr=coef_Yn1p_in_D3.content,
            coeff_dp_arr=coef_dp_Yn1p_in_D3.content*nfp,
            f_arr=D3_RHS_no_unknown.content,
            fft_max_freq=max_freq)

    Yn = new_Y_n_no_unknown + ChiPhiFunc(vector_free_coef*Yn_free_content, nfp)
    return(Yn)

# Evaluates Zn,
# Requires:
# X_{n-1}, Y_{n-1}, Z_{n-1},
# B_{\theta n}, B_{\psi n-1},
# B_{\alpha (n-2)/2 \text{ or } (n-3)/2},
# \iota_{(n-2)/2 \text{ or } (n-3)/2}
# \kappa, \frac{dl}{d\phi}, \tau
# not nfp-dependent
def iterate_Zn_cp(
    n_eval,
    X_coef_cp, Y_coef_cp, Z_coef_cp,
    B_theta_coef_cp, B_psi_coef_cp,
    B_alpha_coef,
    kap_p, dl_p, tau_p,
    iota_coef):
    return(parsed.eval_Zn_cp(
        n=n_eval,
        X_coef_cp=X_coef_cp.mask(n_eval-1),
        Y_coef_cp=Y_coef_cp.mask(n_eval-1),
        Z_coef_cp=Z_coef_cp.mask(n_eval-1).zero_append(),
        # B_theta n isn't required but need to be provided (cancellation?)
        B_theta_coef_cp=B_theta_coef_cp.mask(n_eval-1).zero_append(),
        B_psi_coef_cp=B_psi_coef_cp.mask(n_eval-2),
        B_alpha_coef=B_alpha_coef,
        kap_p=kap_p, dl_p=dl_p, tau_p=tau_p,
        iota_coef = iota_coef
    ).cap_m(n_eval))

# Evaluates B_{\psi n-2}
# Requires:
# X_{n-1}, Y_{n-1}, Z_{n-1},
# B_{\theta n-1}, B_0,
# B_{\alpha 0}, \bar{\iota}_{(n-2)/2 or (n-3)/2}$
# not nfp-dependent
def iterate_dc_B_psi_nm2(
    n_eval,
    X_coef_cp, Y_coef_cp, Z_coef_cp,
    B_theta_coef_cp, B_psi_coef_cp,
    B_alpha_coef, B_denom_coef_c,
    kap_p, dl_p, tau_p,
    iota_coef):
    # Ignore B_theta_n
    dchi_b_psi_nm2 = parsed.eval_dchi_B_psi_cp_nm2(n_eval, \
            X_coef_cp.mask(n_eval-1).zero_append(), \
            Y_coef_cp.mask(n_eval-1).zero_append(), \
            Z_coef_cp.mask(n_eval-1).zero_append(), \
            B_theta_coef_cp.mask(n_eval), B_psi_coef_cp.mask(n_eval-3).zero_append(), B_alpha_coef, B_denom_coef_c, \
            kap_p, dl_p, tau_p, iota_coef)
    # The evaluation gives an extra component.
    if isinstance(dchi_b_psi_nm2, ChiPhiFunc):
        return(dchi_b_psi_nm2.cap_m(n_eval-2))
    else:
        return(dchi_b_psi_nm2)

''' II. Magnetic and MHD equations '''
''' II.1. Recursion relations for individual variables '''

# Uses B_theta [n-2], B_[psi n-2], B_alpha first-order terms, B_denom[n],
# p_[perp n-1], Delta_[n-1],iota_[(n-3)/2 (n-2)/2]
# not nfp-dependent
def iterate_p_perp_n(n_eval,
    B_theta_coef_cp,
    B_psi_coef_cp,
    B_alpha_coef,
    B_denom_coef_c,
    p_perp_coef_cp,
    Delta_coef_cp,
    iota_coef):
    return(
        MHD_parsed.eval_p_perp_n_cp(n_eval,
        B_theta_coef_cp.mask(n_eval-2).zero_append().zero_append(), # cancellation for 2 orders
        B_psi_coef_cp,
        B_alpha_coef,
        B_denom_coef_c,
        p_perp_coef_cp.mask(n_eval-1).zero_append(),
        Delta_coef_cp,
        iota_coef).cap_m(n_eval)
    )

# Uses B_denom[n], p_[\perp n]$, $\Delta_[n-1]$, $\iota_[(n-1)/2, n/2]$
# Delta always contain one free, scalar parameter that's determined through
# the looped equation. At even orders, this is avg(Delta[n,0]), the constant
# component of the FFT of Delta[n,0]. At odd orders, this is iota[(n-1)/2].
# This method always ensures zero avg(Delta[n,0]), and will by default
# set iota[(n-1)/2] to zero, but has the option to use known iota for debugging.
# nfp-dependent!!
def iterate_delta_n_0_offset(n_eval,
    B_denom_coef_c,
    p_perp_coef_cp,
    Delta_coef_cp,
    iota_coef,
    max_freq=None,
    no_iota_masking = False): # nfp-dependent!!

    # At even orders, the free parameter is Delta_offset (the average of Delta n0)
    if n_eval%2==0:
        Delta_n_inhomog_component = MHD_parsed.eval_inhomogenous_Delta_n_cp(n_eval,
        B_denom_coef_c,
        p_perp_coef_cp,
        Delta_coef_cp.mask(n_eval-1).zero_append(),
        iota_coef)
    # At odd orders, the free parameter is iota (n-1)/2. Note the masking on iota_coef.
    # no_iota_masking is for debugging
    else:
        if no_iota_masking:
            Delta_n_inhomog_component = MHD_parsed.eval_inhomogenous_Delta_n_cp(n_eval,
            B_denom_coef_c,
            p_perp_coef_cp,
            Delta_coef_cp.mask(n_eval-1).zero_append(),
            iota_coef)
        else:
            Delta_n_inhomog_component = MHD_parsed.eval_inhomogenous_Delta_n_cp(n_eval,
            B_denom_coef_c,
            p_perp_coef_cp,
            Delta_coef_cp.mask(n_eval-1).zero_append(),
            iota_coef.mask((n_eval-3)//2).zero_append())

    # At even orders, setting Delta[even, 0] to have zero average.
    # This average is a free parameter, because the center component of
    # the ODE is dphi x = f.
    content = solve_dphi_iota_dchi(
        iota=iota_coef[0]/Delta_n_inhomog_component.nfp,
        f=Delta_n_inhomog_component.content/Delta_n_inhomog_component.nfp,
        fft_max_freq=max_freq
    )
    Delta_out = ChiPhiFunc(content, Delta_n_inhomog_component.nfp).cap_m(n_eval)
    if n_eval%2==0:
        Delta_out -= jnp.average(Delta_out[0].content)
    return(Delta_out)

''' III. Equilibrium manager and Iterate '''

# A container for all equilibrium quantities.
# All coef inputs must be ChiPhiEpsFunc's.
class Equilibrium:
    # nfp-dependent!!
    def __init__(self, unknown, constant, nfp, magnetic_only, axis_info={}):
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
            nfp=nfp,
            magnetic_only=magnetic_only,
            axis_info=axis_info,
        ))

    def save(self, file_name):
        jnp.save(file_name, self)

    def load(file_name):
        return(jnp.load(file_name, allow_pickle=True).item())

    def save_plain(self, file_name):
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

        big_dict = {\
            'unknown':unknown_dict,
            'constant':constant_dict,
            'nfp':self.nfp,
            'magnetic_only': self.magnetic_only,
            'axis_info': self.axis_info,
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
        constant['iota_coef']\
            = ChiPhiEpsFunc.from_content_list(raw_constant['iota_coef'], nfp)
        constant['B_alpha_coef']\
            = ChiPhiEpsFunc.from_content_list(raw_constant['B_alpha_coef'], nfp)
        constant['kap_p']\
            = ChiPhiFunc(raw_constant['kap_p'], nfp)
        constant['dl_p']\
            = raw_constant['dl_p']
        constant['tau_p']\
            = ChiPhiFunc(raw_constant['tau_p'], nfp)

        return(Equilibrium(
            unknown=unknown,
            constant=constant,
            nfp=nfp,
        magnetic_only=magnetic_only,
            axis_info=axis_info
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
    def check_governing_equations(self, n_unknown):
        if n_unknown is None:
            n_unknown = self.get_order()
        elif n_unknown>self.get_order():
            raise ValueError('Validation order should be <= than the current order')

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

        J = MHD_parsed.validate_J(n_unknown,
            X_coef_cp,
            Y_coef_cp,
            Z_coef_cp,
            B_denom_coef_c, B_alpha_coef,
            kap_p, dl_p, tau_p, iota_coef)
        Cb = MHD_parsed.validate_Cb(n_unknown-1,
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
        Ck = MHD_parsed.validate_Ck(n_unknown-1, X_coef_cp, Y_coef_cp, Z_coef_cp,
            B_denom_coef_c, B_alpha_coef,
            B_psi_coef_cp, B_theta_coef_cp,
            kap_p, dl_p, tau_p, iota_coef)
        Ct = MHD_parsed.validate_Ct(n_unknown-1, X_coef_cp, Y_coef_cp, Z_coef_cp,
            B_denom_coef_c, B_alpha_coef,
            B_psi_coef_cp, B_theta_coef_cp,
            kap_p, dl_p, tau_p, iota_coef)
        I = MHD_parsed.validate_I(n_unknown, B_denom_coef_c,
            p_perp_coef_cp, Delta_coef_cp,
            iota_coef)
        II = MHD_parsed.validate_II(n_unknown,
            B_theta_coef_cp, B_alpha_coef, B_denom_coef_c,
            p_perp_coef_cp, Delta_coef_cp, iota_coef)
        III = MHD_parsed.validate_III(n_unknown-2,
            B_theta_coef_cp, B_psi_coef_cp,
            B_alpha_coef, B_denom_coef_c,
            p_perp_coef_cp, Delta_coef_cp,
            iota_coef)
        return(J, Cb, Ck, Ct, I, II, III)

    ''' Display '''
    # not nfp-dependent
    def display_order(self, n):
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

    # The looped equation uses some very long coefficients with simple n dependence.
    # the below methods calculates these coefficients using n-independent parts
    # calculated in prepare
    def iterate_2_magnetic_only():
        raise NotImplementedError('Clip the order of the equilibrium in a \
        non-jitted portion before running the jitted portion to avoid recompile.\
        running to ')

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
    B_psi_nm20,
    B_alpha_nb2,
    B_denom_nm1, B_denom_n,
    iota_nm2b2,
    max_freq=None,
    n_eval=None,
):

    if not equilibrium.magnetic_only:
        return()
    if max_freq == None:
        len_phi = equilibrium.unknown['X_coef_cp'][1].content.shape[1]
        max_freq = (len_phi//2, len_phi//2)

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
    p_perp_coef_cp = p_perp_coef_cp.zero_append()
    p_perp_coef_cp = p_perp_coef_cp.zero_append()
    Delta_coef_cp = Delta_coef_cp.zero_append()
    Delta_coef_cp = Delta_coef_cp.zero_append()

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
    B_psi_coef_cp = B_psi_coef_cp.append(B_psi_nm3.filter(max_freq[0]))

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
    Z_coef_cp = Z_coef_cp.append(Znm1.filter(max_freq[0]))

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
    X_coef_cp = X_coef_cp.append(Xnm1.filter(max_freq[0]))

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
        max_freq=max_freq[0]
    )
    Y_coef_cp = Y_coef_cp.append(Ynm1.filter(max_freq[0]))



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
    B_psi_coef_cp = B_psi_coef_cp.append(B_psi_nm2.filter(max_freq[1]))

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
    Z_coef_cp = Z_coef_cp.append(Zn.filter(max_freq[1]))

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
    X_coef_cp = X_coef_cp.append(Xn.filter(max_freq[1]))

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
        max_freq=max_freq[1],
        Yn0=Yn0
    )
    Y_coef_cp = Y_coef_cp.append(Yn.filter(max_freq[1]))

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

# @partial(jit, static_argnums=(5, 6, ))
def iterate_2(equilibrium,
    B_alpha_nb2,
    B_denom_nm1, B_denom_n,
    # Not implemented yet. At odd orders, the periodic BC
    # will constrain one of sigma_n(0), iota_n and avg(B_theta_n0)
    # given value for the other 2.
    # free_param_values need to be a 2-element tuple.
    # Now only implemented avg(B_theta_n0)=0 and given iota.
    iota_new, # arg 4
    n_eval=None,
    max_freq=None,
    # -1 represents no filtering (default). This value is chosen so that
    # turning on or off off-diagonal filtering does not require recompiles.
    max_k_diff_pre_inv=(-1, -1),
    max_k_diff_post_inv=(-1, -1)
    ):
    if equilibrium.magnetic_only:
        return()
    len_phi = equilibrium.unknown['X_coef_cp'][1].content.shape[1]
    if max_freq==None:
        max_freq = (len_phi//2, len_phi//2)
    # if max_k_diff_pre_inv==None:
    #     max_k_diff_pre_inv = (len_phi, len_phi)
    # if max_k_diff_post_inv==None:
    #     max_k_diff_post_inv = (len_phi, len_phi)
    # If no order is supplied, then iterate to the next order. the Equilibrium
    # will be edited directly.
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

    iota_coef = iota_coef.append(iota_new)
    B_denom_coef_c = B_denom_coef_c.append(B_denom_nm1)
    B_denom_coef_c = B_denom_coef_c.append(B_denom_n)
    B_alpha_coef = B_alpha_coef.append(B_alpha_nb2)

    # Evaluating order n_eval-1

    # print('iota 1 right before loop',iota_coef[1])
    solution_nm1_known_iota = looped_solver.iterate_looped(
        n_unknown = n_eval-1,
        nfp = equilibrium.nfp,
        target_len_phi = 1000,
        X_coef_cp = X_coef_cp,
        Y_coef_cp = Y_coef_cp,
        Z_coef_cp = Z_coef_cp,
        p_perp_coef_cp = p_perp_coef_cp,
        Delta_coef_cp = Delta_coef_cp,
        B_psi_coef_cp = B_psi_coef_cp,
        B_theta_coef_cp = B_theta_coef_cp,
        B_alpha_coef = B_alpha_coef,
        B_denom_coef_c = B_denom_coef_c,
        kap_p = kap_p,
        tau_p = tau_p,
        dl_p = dl_p,
        iota_coef = iota_coef,
        max_freq = max_freq[0],
        max_k_diff_pre_inv = max_k_diff_pre_inv[0],
        max_k_diff_post_inv = max_k_diff_post_inv[0],
    )
    B_theta_coef_cp = B_theta_coef_cp.append(solution_nm1_known_iota['B_theta_n']) # .filter(max_freq[0]))
    B_psi_coef_cp = B_psi_coef_cp.append(solution_nm1_known_iota['B_psi_nm2']) # .filter(max_freq[0]))
    X_coef_cp = X_coef_cp.append(solution_nm1_known_iota['Xn']) # .filter(max_freq[0]))
    Y_coef_cp = Y_coef_cp.append(solution_nm1_known_iota['Yn']) # .filter(max_freq[0]))
    Z_coef_cp = Z_coef_cp.append(solution_nm1_known_iota['Zn']) # .filter(max_freq[0]))
    p_perp_coef_cp = p_perp_coef_cp.append(solution_nm1_known_iota['pn']) # .filter(max_freq[0]))
    Delta_coef_cp = Delta_coef_cp.append(solution_nm1_known_iota['Deltan']) # .filter(max_freq[0]))

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

    solution_n = looped_solver.iterate_looped(
        n_unknown = n_eval,
        nfp=equilibrium.nfp,
        target_len_phi = 1000,
        X_coef_cp = X_coef_cp,
        Y_coef_cp = Y_coef_cp,
        Z_coef_cp = Z_coef_cp,
        p_perp_coef_cp = p_perp_coef_cp,
        Delta_coef_cp = Delta_coef_cp,
        B_psi_coef_cp = B_psi_coef_cp,
        B_theta_coef_cp = B_theta_coef_cp,
        B_alpha_coef = B_alpha_coef,
        B_denom_coef_c = B_denom_coef_c,
        kap_p = kap_p,
        tau_p = tau_p,
        dl_p = dl_p,
        iota_coef = iota_coef,
        max_freq = max_freq[1],
        max_k_diff_pre_inv = max_k_diff_pre_inv[1],
        max_k_diff_post_inv = max_k_diff_post_inv[1],
    )
    # Partial solutions for these variables were appended to their
    # ChiPhiEpsFunc's for iterate_looped. Now remove them and re-append
    # This only reassigns the pointer B_psi. Need to re-assign equilibrium.unknown[]
    # too.
    B_psi_coef_cp = B_psi_coef_cp.mask(n_eval-3)
    B_psi_coef_cp = B_psi_coef_cp.append(solution_n['B_psi_nm2'].filter(max_freq[1]))
    # This only reassigns the pointer B_theta. Need to re-assign equilibrium.unknown[]
    # too.
    B_theta_coef_cp = B_theta_coef_cp.mask(n_eval-1)
    B_theta_coef_cp = B_theta_coef_cp.append(solution_n['B_theta_n'].filter(max_freq[1]))

    X_coef_cp = X_coef_cp.append(solution_n['Xn']) # .filter(max_freq[1])
    Z_coef_cp = Z_coef_cp.append(solution_n['Zn']) # .filter(max_freq[1])
    p_perp_coef_cp = p_perp_coef_cp.append(solution_n['pn']) # .filter(max_freq[1])
    Delta_coef_cp = Delta_coef_cp.append(solution_n['Deltan']) # .filter(max_freq[1])
    Y_coef_cp = Y_coef_cp.append(solution_n['Yn']) # .filter(max_freq[1])

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