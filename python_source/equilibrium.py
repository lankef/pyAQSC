# Wrapped/completed recursion relations based on translated expressions
# in parsed/. Necessary masking and/or n-substitution are included. All iterate_*
# methods returns ChiPhiFunc's.

# ChiPhiFunc and ChiPhiEpsFunc
from chiphifunc import *
from chiphiepsfunc import *
from math_utilities import *
from lambda_coefs_looped import *
from lambda_coefs_shared import *
from lambda_coefs_B_psi import *
import looped_solver
import numpy as np
from matplotlib import pyplot as plt

# parsed relations
import parsed
import MHD_parsed


# Performance
import time

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
    return(parsed.eval_xn.eval_Xn_cp(
        n=n_eval,
        X_coef_cp=X_coef_cp.mask(n_eval-1).zero_append(),
        Y_coef_cp=Y_coef_cp.mask(n_eval-1),
        Z_coef_cp=Z_coef_cp.mask(n_eval),
        B_denom_coef_c=B_denom_coef_c,
        B_alpha_coef=B_alpha_coef,
        kap_p=kap_p, dl_p=dl_p, tau_p=tau_p,
        iota_coef=iota_coef).cap_m(n_eval))

# O_matrices, O_einv, vector_free_coef only uses B_alpha_coef and X_coef_cp
def iterate_Yn_cp_operators(n_eval, X_coef_cp, B_alpha_coef): # nfp-dependent only in output

    # Getting coeffs
    # Both uses B_alpha0 and X1 only
    chiphifunc_A = parsed.eval_ynp1.coef_a(n_eval-1, B_alpha_coef, X_coef_cp)
    chiphifunc_B = parsed.eval_ynp1.coef_b(B_alpha_coef, X_coef_cp)

    # Calculating the inverted matrices
    i_free = (n_eval+1)//2 # We'll always use Yn0 or Yn1p as the free var.
    O_matrices, O_einv, vector_free_coef, nfp = ChiPhiFunc.get_O_O_einv_from_A_B(chiphifunc_A, chiphifunc_B, i_free, n_eval)
    return(O_matrices, O_einv, vector_free_coef, nfp)

# O_matrices, O_einv, vector_free_coef only uses B_alpha_coef and X_coef_cp
# nfp-dependent!!
def iterate_Yn_cp_RHS(n_eval,
    X_coef_cp,
    Y_coef_cp,
    Z_coef_cp,
    B_psi_coef_cp,
    B_theta_coef_cp,
    B_alpha_coef,
    B_denom_coef_c,
    kap_p, dl_p, tau_p, eta,
    iota_coef,
    max_freq = None): # nfp-dependent!!

    # Getting coeffs
    # Both uses B_alpha0 and X1 only
    chiphifunc_A = parsed.eval_ynp1.coef_a(n_eval-1, B_alpha_coef, X_coef_cp)

    # Getting rhs-lhs for the Yn+1 equation
    # for Yn to work, "n" must be subbed with n-1 here
    chiphifunc_rhs = parsed.eval_ynp1.rhs_minus_lhs(n_eval-1,
        X_coef_cp,
        Y_coef_cp.mask(n_eval-1).zero_append(),
        Z_coef_cp,
        B_psi_coef_cp,
        B_theta_coef_cp,
        B_alpha_coef,
        kap_p, dl_p, tau_p,
        iota_coef)

    # Making sure RHS isn't null
    if not (
        (type(chiphifunc_rhs) is ChiPhiFunc) and
        (type(chiphifunc_rhs) is not ChiPhiFuncNull)
        ):
        raise TypeError('chiphifunc_rhs should be ChiPhiFunc. The actual type is:'
                        +str(type(chiphifunc_rhs)))

    # Stretching RHS (stretching between A and B is done in get_O_O_einv_from_A_B)
    chiphifunc_A_content, chiphifunc_rhs_content = chiphifunc_A.stretch_phi_to_match(chiphifunc_rhs)
    len_chi = chiphifunc_rhs.get_shape()[0]

    # Center-pad chiphifunc_rhs if it's too short
    #          [0  ]
    # [RHS] => [RHS]
    #          [0  ]
    if chiphifunc_A.get_shape()[0] + n_eval != len_chi:
        chiphifunc_rhs_content = ChiPhiFunc.add_jit(\
            chiphifunc_rhs_content,\
            np.zeros((chiphifunc_A.get_shape()[0]+n_eval, chiphifunc_rhs.get_shape()[1]), dtype=np.complex128),\
            1 # Sign is 1
        )
    if max_freq is not None:
        chiphifunc_rhs_content = ChiPhiFunc(chiphifunc_rhs_content, chiphifunc_rhs.nfp).filter('low_pass', max_freq).content

    return(chiphifunc_rhs_content)

# Evaluates Yn using Yn+1 general formula. The free component is either Yn0 or
# Yn1s.
# Requires:
# X_{n}, Y_{n-1}, Z_{n-1},
# B_{\theta n-1}, B_{\psi  n-3},
# \iota_{(n-3)/2 or (n-4)/2}, B_{\alpha  (n-1)/2 or (n-2)/2}
# nfp-dependent!!
def iterate_Yn_cp_magnetic(n_unknown,
    X_coef_cp,
    Y_coef_cp,
    Z_coef_cp,
    B_psi_coef_cp,
    B_theta_coef_cp,
    B_alpha_coef,
    B_denom_coef_c,
    kap_p, dl_p, tau_p, eta,
    iota_coef,
    max_freq, lambda_coefs_shared,
    Yn0=None):

    len_tensor = max_freq*2
    nfp = X_coef_cp.nfp
    n_eval = n_unknown+1

    O_matrices, O_einv, vector_free_coef, Y_nfp = \
        iterate_Yn_cp_operators(n_unknown,
            X_coef_cp=X_coef_cp,
            B_alpha_coef=B_alpha_coef)

    Yn_rhs_content = iterate_Yn_cp_RHS(n_eval=n_unknown,
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
        eta=eta,
        iota_coef=iota_coef,
        max_freq=max_freq)
    new_Y_n_no_unknown = ChiPhiFunc(np.einsum('ijk,jk->ik',O_einv,Yn_rhs_content), Y_nfp)
    Y_coef_cp_no_unknown = Y_coef_cp.mask(n_eval-2)
    Y_coef_cp_no_unknown.append(new_Y_n_no_unknown)

    # Calculating D3
    if n_unknown%2==0:
        if Yn0 is None:
            raise AttributeError('Yn0 must be provided for even orders.')
        Yn_free_content = Yn0
    else:
        print('n_unknown+n_unknown%2',n_unknown+n_unknown%2)
        print('B_theta_coef_cp',B_theta_coef_cp.get_order())
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

        coef_Yn1p_in_D3 = lambda_coefs_shared['lambda_coef_Yn1p_in_D3'](
            vector_free_coef,
            nfp
        )
        coef_dp_Yn1p_in_D3 = lambda_coefs_shared['lambda_coef_dp_Yn1p_in_D3'](
            vector_free_coef,
            nfp
        )

        Yn_free_content = solve_integration_factor(
            coeff = coef_Yn1p_in_D3.content,
            coeff_dp = coef_dp_Yn1p_in_D3.content*nfp,
            f = D3_RHS_no_unknown.content,
            integral_mode='auto', asymptotic_order=asymptotic_order,
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
    return(parsed.eval_znp1.eval_Zn_cp(
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
    dchi_b_psi_nm2 = parsed.eval_dchi_b_psi_nm2.eval_dchi_B_psi_cp_nm2(n_eval, \
            X_coef_cp.mask(n_eval-1).zero_append(), \
            Y_coef_cp.mask(n_eval-1).zero_append(), \
            Z_coef_cp.mask(n_eval-1).zero_append(), \
            B_theta_coef_cp.mask(n_eval), B_psi_coef_cp.mask(n_eval-3).zero_append(), B_alpha_coef, B_denom_coef_c, \
            kap_p, dl_p, tau_p, iota_coef)
    # The evaluation gives an extra component.
    try:
        return(dchi_b_psi_nm2.cap_m(n_eval-2))
    except TypeError:
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
        parsed.eval_p_perp_n.eval_p_perp_n_cp(n_eval,
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
    integral_mode,
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
        integral_mode=integral_mode,
        fft_max_freq=max_freq
    )
    Delta_out = ChiPhiFunc(content, Delta_n_inhomog_component.nfp).cap_m(n_eval)
    if n_eval%2==0:
        Delta_out -= np.average(Delta_out.get_constant().content)
    return(Delta_out)

''' III. Equilibrium manager and Iterate '''

# A container for all equilibrium quantities.
# All coef inputs must be ChiPhiEpsFunc's.
class Equilibrium:
    # nfp-dependent!!
    def __init__(self, unknown, constant, noise, magnetic_only, nfp):
        self.noise = noise
        self.unknown = unknown
        self.constant = constant
        self.nfp = nfp
        self.magnetic_only = magnetic_only

        # Check if every term is on the same order
        self.check_order_consistency()

        # Some coeffs are really long. We only calc them once.
        # Very little speed improvement compared to eval_B_psi_coefs_full.
        self.prepare_lambdas(magnetic_only)

    # nfp-dependent!!
    def from_known(
        X_coef_cp,
        Y_coef_cp,
        Z_coef_cp,
        B_psi_coef_cp,
        B_theta_coef_cp,
        B_denom_coef_c,
        B_alpha_coef,
        kap_p, dl_p, tau_p,
        iota_coef, eta,
        p_perp_coef_cp,
        Delta_coef_cp,
        noise=None,
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
        unknown['iota_coef'] = iota_coef
        unknown['p_perp_coef_cp'] = p_perp_coef_cp
        unknown['Delta_coef_cp'] = Delta_coef_cp

        constant['B_denom_coef_c'] = B_denom_coef_c
        constant['B_alpha_coef'] = B_alpha_coef
        constant['kap_p'] = kap_p
        constant['dl_p'] = dl_p
        constant['tau_p'] = tau_p
        constant['eta'] = eta

        # Pressure can be trivial
        current_order = X_coef_cp.get_order()
        if not unknown['p_perp_coef_cp']:
            unknown['p_perp_coef_cp'] = ChiPhiEpsFunc.zeros_to_order(current_order)
        if not unknown['Delta_coef_cp']:
            unknown['Delta_coef_cp'] = ChiPhiEpsFunc.zeros_to_order(current_order)

        # Manages noises
        if not noise:
            noise = {} # dict of dict managing types of noises
            # Tracks different types of noise
            noise['filter'] = {}
            for key in noise.keys():
                noise[key]['X_coef_cp'] = ChiPhiEpsFunc.zeros_like(X_coef_cp, nfp)
                noise[key]['Y_coef_cp'] = ChiPhiEpsFunc.zeros_like(Y_coef_cp, nfp)
                noise[key]['Z_coef_cp'] = ChiPhiEpsFunc.zeros_like(Z_coef_cp, nfp)
                noise[key]['B_psi_coef_cp'] = ChiPhiEpsFunc.zeros_like(B_psi_coef_cp, nfp)
                noise[key]['B_theta_coef_cp'] = ChiPhiEpsFunc.zeros_like(B_theta_coef_cp, nfp)
                noise[key]['p_perp_coef_cp'] = ChiPhiEpsFunc.zeros_like(unknown['p_perp_coef_cp'], nfp)
                noise[key]['Delta_coef_cp'] = ChiPhiEpsFunc.zeros_like(unknown['Delta_coef_cp'], nfp)

        return(Equilibrium(unknown, constant, noise, magnetic_only, nfp))

    def save(self, file_name):

        # Both unknown and noise are entirely made of ChiPhiFuncs.
        unknown_to_content_list = {}
        for key in self.unknown.keys():
            unknown_to_content_list[key] = self.unknown[key].to_content_list()

        noise_to_content_list = {}
        for keya in self.noise.keys():
            noise_to_content_list[keya]={}
            for keyb in self.noise[keya].keys():
                noise_to_content_list[keya][keyb] = self.noise[keya][keyb].to_content_list()

        const_to_content_list={}
        const_to_content_list['B_denom_coef_c']\
            = self.constant['B_denom_coef_c'].to_content_list()
        const_to_content_list['B_alpha_coef']\
            = self.constant['B_alpha_coef'].to_content_list()
        const_to_content_list['kap_p']\
            = self.constant['kap_p'].content
        const_to_content_list['dl_p']\
            = self.constant['dl_p']
        const_to_content_list['tau_p']\
            = self.constant['tau_p'].content
        const_to_content_list['eta']\
            = self.constant['eta']

        big_dict = {\
            'unknown':unknown_to_content_list,
            'noise':noise_to_content_list,
            'constant':const_to_content_list,
            'nfp':self.nfp,
            'magnetic_only': self.magnetic_only
        }
        np.savez(file_name, big_dict)

    # nfp-dependent!!
    def load(filename):
        npzfile = np.load(filename, allow_pickle=True)
        big_dict = npzfile['arr_0'].item()
        raw_unknown = big_dict['unknown']
        raw_constant = big_dict['constant']
        raw_noise = big_dict['noise']
        nfp = big_dict['nfp']
        magnetic_only = big_dict['magnetic_only']
        print('nfp', nfp)

        unknown = {}
        for key in raw_unknown.keys():
            if key == 'iota_coef':
                unknown[key] = ChiPhiEpsFunc.from_content_list(raw_unknown[key], 0)
            else:
                unknown[key] = ChiPhiEpsFunc.from_content_list(raw_unknown[key], nfp)

        noise = {}
        for keya in raw_noise.keys():
            noise[keya]={}
            for keyb in raw_noise[keya].keys():
                noise[keya][keyb] \
                    = ChiPhiEpsFunc.from_content_list(raw_noise[keya][keyb], nfp)

        constant={}
        constant['B_denom_coef_c']\
            = ChiPhiEpsFunc.from_content_list(raw_constant['B_denom_coef_c'], 0)
        constant['B_alpha_coef']\
            = ChiPhiEpsFunc.from_content_list(raw_constant['B_alpha_coef'], 0)
        constant['kap_p']\
            = ChiPhiFunc(raw_constant['kap_p'], nfp)
        constant['dl_p']\
            = raw_constant['dl_p']
        constant['tau_p']\
            = ChiPhiFunc(raw_constant['tau_p'], nfp)
        constant['eta']\
            = raw_constant['eta']

        return(Equilibrium(unknown, constant, noise, magnetic_only, nfp))

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
        check_order_individial(self.unknown['iota_coef'], 'iota_coef', (n-2)//2)
        check_order_individial(self.unknown['p_perp_coef_cp'], 'p_perp_coef_cp', n)
        check_order_individial(self.unknown['Delta_coef_cp'], 'Delta_coef_cp', n)

        for key in self.noise.keys():
            check_order_individial(self.noise[key]['Y_coef_cp'], 'Y_coef_cp', n)
            check_order_individial(self.noise[key]['Z_coef_cp'], 'Z_coef_cp', n)
            check_order_individial(self.noise[key]['B_psi_coef_cp'], 'B_psi_coef_cp', n-2)
            check_order_individial(self.noise[key]['B_theta_coef_cp'], 'B_theta_coef_cp', n)
            check_order_individial(self.noise[key]['p_perp_coef_cp'], 'p_perp_coef_cp', n)
            check_order_individial(self.noise[key]['Delta_coef_cp'], 'Delta_coef_cp', n)

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
        iota_coef = self.unknown['iota_coef']
        p_perp_coef_cp = self.unknown['p_perp_coef_cp']
        Delta_coef_cp = self.unknown['Delta_coef_cp']
        B_denom_coef_c = self.constant['B_denom_coef_c']
        B_alpha_coef = self.constant['B_alpha_coef']
        kap_p = self.constant['kap_p']
        dl_p = self.constant['dl_p']
        tau_p = self.constant['tau_p']
        eta = self.constant['eta']

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
    ''' Looped coefficients '''
    # The looped equation contains some complicated constants with simple n_eval
    # dependence. This methods evaluates their n_eval independent components, and
    # generates a set of lambda functions that evaluates these coefficients
    # at order n. See looped_lambdas.py for more details.
    # not nfp-dependent
    def prepare_lambdas(self, magnetic_only):
        self.lambda_coefs_shared = eval_lambda_coefs_shared(self)
        if not magnetic_only:
            self.lambda_coefs_looped = eval_lambda_coefs_looped(self)
            self.lambda_coefs_B_psi = eval_B_psi_lambdas_full(
                self.unknown['X_coef_cp'],
                self.unknown['Y_coef_cp'],
                self.unknown['Delta_coef_cp'],
                self.constant['B_alpha_coef'],
                self.constant['B_denom_coef_c'],
                self.constant['dl_p'],
                self.constant['tau_p'],
                self.constant['kap_p'],
                self.unknown['iota_coef']
            )
        return()

    # The looped equation uses some very long coefficients with simple n dependence.
    # the below methods calculates these coefficients using n-independent parts
    # calculated in prepare
    ''' Iteration '''
    # # Evaluates 2 entire orders. Note that no masking is needed for any of the methods
    # # defined in this file. Copies the equilibrium. STOPS AT EVEN ORDERS.

    # Iterates the magnetic equations only.
    # Calculates Xn, Yn, Zn, B_psin-2 for 2 orders from lower order values.
    # B_theta, B_psi_nm30, Y_free_nm1 are all free.
    # n_eval must be even.
    # not nfp-dependent
    def iterate_2_magnetic_only(self,
        B_theta_nm1, B_theta_n,
        Yn0,
        B_psi_nm20,
        B_alpha_nb2,
        B_denom_nm1, B_denom_n,
        iota_nm2b2,
        max_freq=None,
        n_eval=None, filter=False,
        ):

        if max_freq == None:
            len_phi = self.unknown['X_coef_cp'][1].get_shape()[1]
            max_freq = (len_phi//2, len_phi//2)

        # If no order is supplied, then iterate to the next order. the Equilibrium
        # will be edited directly.
        if n_eval == None:
            n_eval = self.get_order() + 2 # getting order and checking consistency
        if n_eval%2 != 0:
            raise ValueError("n must be even to evaluate iota_{(n-1)/2}")

        start_time = time.time()
        print("Evaluating order",n_eval-1, n_eval)

        # Creating new ChiPhiEpsFunc's for the resulting Equilibrium
        self.unknown['X_coef_cp'] = self.unknown['X_coef_cp'].mask(n_eval-2)
        self.unknown['Y_coef_cp'] = self.unknown['Y_coef_cp'].mask(n_eval-2)
        self.unknown['Z_coef_cp'] = self.unknown['Z_coef_cp'].mask(n_eval-2)
        self.unknown['B_theta_coef_cp'] = self.unknown['B_theta_coef_cp'].mask(n_eval-2)
        self.unknown['B_psi_coef_cp'] = self.unknown['B_psi_coef_cp'].mask(n_eval-4)
        self.unknown['iota_coef'] = self.unknown['iota_coef'].mask((n_eval-4)//2)
        self.unknown['p_perp_coef_cp'] = self.unknown['p_perp_coef_cp'].mask(n_eval-2)
        self.unknown['Delta_coef_cp'] = self.unknown['Delta_coef_cp'].mask(n_eval-2)
        # For readability
        X_coef_cp = self.unknown['X_coef_cp']
        Y_coef_cp = self.unknown['Y_coef_cp']
        Z_coef_cp = self.unknown['Z_coef_cp']
        B_theta_coef_cp = self.unknown['B_theta_coef_cp']
        B_psi_coef_cp = self.unknown['B_psi_coef_cp']
        iota_coef = self.unknown['iota_coef']
        p_perp_coef_cp = self.unknown['p_perp_coef_cp']
        Delta_coef_cp = self.unknown['Delta_coef_cp']

        # Resetting all noises
        self.noise['filter']['X_coef_cp']\
            = self.noise['filter']['X_coef_cp'].mask(n_eval-2)
        self.noise['filter']['Y_coef_cp']\
            = self.noise['filter']['Y_coef_cp'].mask(n_eval-2)
        self.noise['filter']['Z_coef_cp']\
            = self.noise['filter']['Z_coef_cp'].mask(n_eval-2)
        self.noise['filter']['B_theta_coef_cp']\
            = self.noise['filter']['B_theta_coef_cp'].mask(n_eval-2)
        self.noise['filter']['B_psi_coef_cp']\
            = self.noise['filter']['B_psi_coef_cp'].mask(n_eval-4)
        self.noise['filter']['p_perp_coef_cp']\
            = self.noise['filter']['p_perp_coef_cp'].mask(n_eval-2)
        self.noise['filter']['Delta_coef_cp']\
            = self.noise['filter']['Delta_coef_cp'].mask(n_eval-2)

        # Masking all init conds.
        self.constant['B_denom_coef_c'] = self.constant['B_denom_coef_c'].mask(n_eval-2)
        self.constant['B_alpha_coef'] = self.constant['B_alpha_coef'].mask((n_eval)//2-1)
        self.constant['kap_p'] = self.constant['kap_p']
        self.constant['dl_p'] = self.constant['dl_p']
        self.constant['tau_p'] = self.constant['tau_p']
        self.constant['eta'] = self.constant['eta']

        B_denom_coef_c = self.constant['B_denom_coef_c']
        B_alpha_coef = self.constant['B_alpha_coef']
        kap_p = self.constant['kap_p']
        dl_p = self.constant['dl_p']
        tau_p = self.constant['tau_p']
        eta = self.constant['eta']
        # Appending free functions and initial conditions
        B_theta_coef_cp.append(B_theta_nm1)
        B_theta_coef_cp.append(B_theta_n)
        B_alpha_coef.append(B_alpha_nb2)
        iota_coef.append(iota_nm2b2)
        B_denom_coef_c.append(B_denom_nm1)
        B_denom_coef_c.append(B_denom_n)
        p_perp_coef_cp.append(0)
        p_perp_coef_cp.append(0)
        Delta_coef_cp.append(0)
        Delta_coef_cp.append(0)
        # Managing noise
        self.noise['filter']['B_theta_coef_cp'].append(0)
        self.noise['filter']['p_perp_coef_cp'].append(0)
        self.noise['filter']['Delta_coef_cp'].append(0)
        self.noise['filter']['B_theta_coef_cp'].append(0)
        self.noise['filter']['p_perp_coef_cp'].append(0)
        self.noise['filter']['Delta_coef_cp'].append(0)

        # For reducing the amount of duplicate code
        def filter_record_noise_and_append(name, chiphifunc, max_freq_i):
            self.noise['filter'][name].append(chiphifunc.noise_filter('low_pass', max_freq[max_freq_i]))
            if filter:
                chiphifunc = chiphifunc.filter('low_pass', max_freq[max_freq_i])
            self.unknown[name].append(chiphifunc)

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
            ).integrate_chi(ignore_mode_0=True)
        filter_record_noise_and_append('B_psi_coef_cp', B_psi_nm3,0)

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
        filter_record_noise_and_append('Z_coef_cp', Znm1,0)

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
        filter_record_noise_and_append('X_coef_cp', Xnm1,0)

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
            eta=eta,
            iota_coef=iota_coef,
            max_freq=max_freq[0],
            lambda_coefs_shared=self.lambda_coefs_shared
        )
        filter_record_noise_and_append('Y_coef_cp', Ynm1,0)

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
            ).integrate_chi(ignore_mode_0 = True)
        B_psi_nm2.content[B_psi_nm2.get_shape()[0]//2] = B_psi_nm20
        filter_record_noise_and_append('B_psi_coef_cp', B_psi_nm2,1)

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
        filter_record_noise_and_append('Z_coef_cp', Zn,1)

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
        filter_record_noise_and_append('X_coef_cp', Xn,1)

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
            eta=eta,
            iota_coef=iota_coef,
            max_freq=max_freq[1],
            lambda_coefs_shared=self.lambda_coefs_shared,
            Yn0=Yn0
        )
        filter_record_noise_and_append('Y_coef_cp', Yn,1)
        # self.noise['filter']['Y_coef_cp'].append(Yn.noise_filter('low_pass', max_freq))
        # if filter:
        #     Yn = Yn.filter('low_pass', max_freq)
        # Y_coef_cp.append(Yn)
        self.check_order_consistency()
        print("Time elapsed(s):",(time.time() - start_time))

    def iterate_2(self,
        B_alpha_nb2,
        B_denom_nm1, B_denom_n,
        # Not implemented yet. At odd orders, the periodic BC
        # will constrain one of sigma_n(0), iota_n and avg(B_theta_n0)
        # given value for the other 2.
        # free_param_values need to be a 2-element tuple.
        # Now only implemented avg(B_theta_n0)=0 and given iota.
        iota_new,
        n_eval=None, filter=False,
        max_freq=None,
        max_k_diff_pre_inv=None,
        max_k_diff_post_inv=None
        ):

        len_phi = self.unknown['X_coef_cp'][1].get_shape()[1]
        if max_freq==None:
            max_freq = (len_phi//2, len_phi//2)
        if max_k_diff_pre_inv==None:
            max_k_diff_pre_inv = (len_phi, len_phi)
        if max_k_diff_post_inv==None:
            max_k_diff_post_inv = (len_phi, len_phi)
        # If no order is supplied, then iterate to the next order. the Equilibrium
        # will be edited directly.
        if n_eval == None:
            n_eval = self.get_order() + 2 # getting order and checking consistency
        if n_eval%2 != 0:
            raise ValueError("n must be even to evaluate iota_{(n-1)/2}")

        start_time = time.time()
        print("Evaluating order",n_eval-1, n_eval)

        # Resetting unknowns
        self.unknown['X_coef_cp'] = self.unknown['X_coef_cp'].mask(n_eval-2)
        self.unknown['Y_coef_cp'] = self.unknown['Y_coef_cp'].mask(n_eval-2)
        self.unknown['Z_coef_cp'] = self.unknown['Z_coef_cp'].mask(n_eval-2)
        self.unknown['B_theta_coef_cp'] = self.unknown['B_theta_coef_cp'].mask(n_eval-2)
        self.unknown['B_psi_coef_cp'] = self.unknown['B_psi_coef_cp'].mask(n_eval-4)
        self.unknown['iota_coef'] = self.unknown['iota_coef'].mask((n_eval-4)//2)
        self.unknown['p_perp_coef_cp'] = self.unknown['p_perp_coef_cp'].mask(n_eval-2)
        self.unknown['Delta_coef_cp'] = self.unknown['Delta_coef_cp'].mask(n_eval-2)
        # For readability
        X_coef_cp = self.unknown['X_coef_cp']
        Y_coef_cp = self.unknown['Y_coef_cp']
        Z_coef_cp = self.unknown['Z_coef_cp']
        B_theta_coef_cp = self.unknown['B_theta_coef_cp']
        B_psi_coef_cp = self.unknown['B_psi_coef_cp']
        iota_coef = self.unknown['iota_coef']
        p_perp_coef_cp = self.unknown['p_perp_coef_cp']
        Delta_coef_cp = self.unknown['Delta_coef_cp']

        # Resetting noise
        self.noise['filter']['X_coef_cp']\
            = self.noise['filter']['X_coef_cp'].mask(n_eval-2)
        self.noise['filter']['Y_coef_cp']\
            = self.noise['filter']['Y_coef_cp'].mask(n_eval-2)
        self.noise['filter']['Z_coef_cp']\
            = self.noise['filter']['Z_coef_cp'].mask(n_eval-2)
        self.noise['filter']['B_theta_coef_cp']\
            = self.noise['filter']['B_theta_coef_cp'].mask(n_eval-2)
        self.noise['filter']['B_psi_coef_cp']\
            = self.noise['filter']['B_psi_coef_cp'].mask(n_eval-4)
        self.noise['filter']['p_perp_coef_cp']\
            = self.noise['filter']['p_perp_coef_cp'].mask(n_eval-2)
        self.noise['filter']['Delta_coef_cp']\
            = self.noise['filter']['Delta_coef_cp'].mask(n_eval-2)

        # Masking all init conds.
        self.constant['B_denom_coef_c'] = self.constant['B_denom_coef_c'].mask(n_eval-2)
        self.constant['B_alpha_coef'] = self.constant['B_alpha_coef'].mask((n_eval)//2-1)
        self.constant['kap_p'] = self.constant['kap_p']
        self.constant['dl_p'] = self.constant['dl_p']
        self.constant['tau_p'] = self.constant['tau_p']
        self.constant['eta'] = self.constant['eta']
        # For readability
        B_denom_coef_c = self.constant['B_denom_coef_c']
        B_alpha_coef = self.constant['B_alpha_coef']
        kap_p = self.constant['kap_p']
        dl_p = self.constant['dl_p']
        tau_p = self.constant['tau_p']
        eta = self.constant['eta']
        iota_coef.append(iota_new)
        B_denom_coef_c.append(B_denom_nm1)
        B_denom_coef_c.append(B_denom_n)
        B_alpha_coef.append(B_alpha_nb2)

        # For reducing duplicate code
        def filter_record_noise_and_append(name, chiphifunc, max_freq_i):
            self.noise['filter'][name].append(chiphifunc.noise_filter('low_pass', max_freq[max_freq_i]))
            if filter:
                chiphifunc = chiphifunc.filter('low_pass', max_freq[max_freq_i])
            self.unknown[name].append(chiphifunc)

        # Evaluating order n_eval-1

        # print('iota 1 right before loop',iota_coef[1])
        solution_nm1_known_iota = looped_solver.iterate_looped(
            n_unknown = n_eval-1,
            nfp=self.nfp,
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
            eta = eta,
            iota_coef = iota_coef,
            lambda_coefs_looped = self.lambda_coefs_looped,
            lambda_coefs_shared = self.lambda_coefs_shared,
            lambda_coefs_B_psi = self.lambda_coefs_B_psi,
            max_freq = max_freq[0],
            max_k_diff_pre_inv = max_k_diff_pre_inv[0],
            max_k_diff_post_inv = max_k_diff_post_inv[0],
        )
        filter_record_noise_and_append('B_theta_coef_cp', solution_nm1_known_iota['B_theta_n'], 0)
        filter_record_noise_and_append('B_psi_coef_cp', solution_nm1_known_iota['B_psi_nm2'], 0)
        filter_record_noise_and_append('X_coef_cp', solution_nm1_known_iota['Xn'], 0)
        filter_record_noise_and_append('Y_coef_cp', solution_nm1_known_iota['Yn'], 0)
        filter_record_noise_and_append('Z_coef_cp', solution_nm1_known_iota['Zn'], 0)
        filter_record_noise_and_append('p_perp_coef_cp', solution_nm1_known_iota['pn'], 0)
        filter_record_noise_and_append('Delta_coef_cp', solution_nm1_known_iota['Deltan'], 0)

        # We don't record noise for B_theta_n0 yet.
        # This "partial" solution will be fed into
        # iterate_looped
        B_theta_coef_cp.append(solution_nm1_known_iota['B_theta_np10'])

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
            ).integrate_chi(ignore_mode_0=True)
        # Don't record noise yet. This "partial" solution will be fed into
        # iterate_looped
        B_psi_coef_cp.append(B_psi_nm2)

        solution_n = looped_solver.iterate_looped(
            n_unknown = n_eval,
            nfp=self.nfp,
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
            eta = eta,
            iota_coef = iota_coef,
            lambda_coefs_looped = self.lambda_coefs_looped,
            lambda_coefs_shared = self.lambda_coefs_shared,
            lambda_coefs_B_psi = self.lambda_coefs_B_psi,
            max_freq = max_freq[1],
            max_k_diff_pre_inv = max_k_diff_pre_inv[1],
            max_k_diff_post_inv = max_k_diff_post_inv[1],
        )
        # Partial solutions for these variables were appended to their
        # ChiPhiEpsFunc's for iterate_looped. Now remove them, re-append
        # and record noises.
        # This only reassigns the pointer B_psi. Need to re-assign self.unknown[]
        # too.
        B_psi_coef_cp = B_psi_coef_cp.mask(n_eval-3)
        self.unknown['B_psi_coef_cp'] = B_psi_coef_cp
        filter_record_noise_and_append('B_psi_coef_cp', solution_n['B_psi_nm2'],1)
        # This only reassigns the pointer B_theta. Need to re-assign self.unknown[]
        # too.
        B_theta_coef_cp = B_theta_coef_cp.mask(n_eval-1)
        self.unknown['B_theta_coef_cp'] = B_theta_coef_cp
        filter_record_noise_and_append('B_theta_coef_cp', solution_n['B_theta_n'],1)

        filter_record_noise_and_append('X_coef_cp', solution_n['Xn'],1)
        filter_record_noise_and_append('Z_coef_cp', solution_n['Zn'],1)
        filter_record_noise_and_append('p_perp_coef_cp', solution_n['pn'],1)
        filter_record_noise_and_append('Delta_coef_cp', solution_n['Deltan'],1)
        filter_record_noise_and_append('Y_coef_cp', solution_n['Yn'],1)

        print("Time elapsed(s):",(time.time() - start_time))
        self.check_order_consistency()
        return(solution_nm1_known_iota, solution_n)
