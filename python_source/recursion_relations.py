# Wrapped/completed recursion relations based on translated expressions
# in parsed/. Necessary masking and/or n-substitution are included. All iterate_*
# methods returns ChiPhiFunc's.

# ChiPhiFunc and ChiPhiEpsFunc
from chiphifunc import *
from chiphiepsfunc import *
from math_utilities import *
from looped_coef_lambdas import *
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
def iterate_Yn_cp_operators(n_eval, X_coef_cp, B_alpha_coef):

    # Getting coeffs
    # Both uses B_alpha0 and X1 only
    chiphifunc_A = parsed.eval_ynp1.coef_a(n_eval-1, B_alpha_coef, X_coef_cp)
    chiphifunc_B = parsed.eval_ynp1.coef_b(B_alpha_coef, X_coef_cp)

    # Calculating the inverted matrices
    i_free = (n_eval+1)//2 # We'll always use Yn0 or Yn1p as the free var.
    O_matrices, O_einv, vector_free_coef = ChiPhiFunc.get_O_O_einv_from_A_B(chiphifunc_A, chiphifunc_B, i_free, n_eval)
    return(O_matrices, O_einv, vector_free_coef)


# O_matrices, O_einv, vector_free_coef only uses B_alpha_coef and X_coef_cp
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
    max_freq = None):

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
        chiphifunc_rhs_content = ChiPhiFunc(chiphifunc_rhs_content).filter('low_pass', max_freq).content

    return(chiphifunc_rhs_content)


# Evaluates Yn using Yn+1 general formula. The free component is either Yn0 or
# Yn1s.
# Requires:
# X_{n}, Y_{n-1}, Z_{n-1},
# B_{\theta n-1}, B_{\psi  n-3},
# \iota_{(n-3)/2 or (n-4)/2}, B_{\alpha  (n-1)/2 or (n-2)/2}
def iterate_Yn_cp_magnetic(n_eval,
    X_coef_cp,
    Y_coef_cp,
    Z_coef_cp,
    B_psi_coef_cp,
    B_theta_coef_cp,
    B_alpha_coef,
    B_denom_coef_c,
    kap_p, dl_p, tau_p, eta,
    iota_coef,
    Yn0=np.nan,
    Yn1s_p=np.nan):

    # Cleaning up the free component
    if n_eval%2==0:
        if Yn0==np.nan:
            raise AttributeError('Yn0 must be provided for even orders')
        else:
            Yn_free = Yn0
    else:
        if Yn1s_p==np.nan:
            raise AttributeError('Yn1s_p must be provided for odd orders')
        else:
            Yn_free = Yn1s_p

    if n_eval%2==1:
        Yn_free, _= iterate_Yn1c_p(0,
            n_eval=n_eval,
            X_coef_cp=X_coef_cp,
            Y_coef_cp=Y_coef_cp,
            Z_coef_cp=Z_coef_cp,
            iota_coef=iota_coef,
            tau_p=tau_p,
            dl_p=dl_p,
            kap_p=kap_p,
            eta=eta,
            B_denom_coef_c=B_denom_coef_c,
            B_alpha_coef=B_alpha_coef,
            B_psi_coef_cp=B_psi_coef_cp,
            B_theta_coef_cp=B_theta_coef_cp,
            Yn1s_p=Yn1s_p)

    O_matrices, O_einv, vector_free_coef = \
        iterate_Yn_cp_operators(n_eval,
            X_coef_cp=X_coef_cp,
            B_alpha_coef=B_alpha_coef)

    chiphifunc_rhs_content = \
        iterate_Yn_cp_RHS(n_eval,
            X_coef_cp=X_coef_cp,
            Y_coef_cp=Y_coef_cp,
            Z_coef_cp=Z_coef_cp,
            B_psi_coef_cp=B_psi_coef_cp,
            B_theta_coef_cp=B_theta_coef_cp,
            B_alpha_coef=B_alpha_coef,
            B_denom_coef_c=B_denom_coef_c,
            kap_p=kap_p, dl_p=dl_p, tau_p=tau_p, eta=eta,
            iota_coef=iota_coef)

    # Converting constant Yn_free to ChiPhiFunc
    if type(Yn_free) is not ChiPhiFunc:
        if np.isscalar(Yn_free):
            Yn_free_content = np.full((1,chiphifunc_rhs_content.shape[1]),Yn_free,dtype=np.complex128)
        else:
            raise TypeError('ChiPhiFunc.solve_underdetermined: '\
                            'Yn_free is not scalar ChiPhiFunc. '\
                            'The actual type is: '+str(type(Yn_free)))
    else:
        Yn_free_content = Yn_free.content

    if n_eval%2!=0: # at odd orders (ODE exists)
        # The rest of the procedure is carried out normally with
        # i_free pointing at Yn1p. The resulting Yn should be
        # Yn = (A_einv@np.ascontiguousarray(chiphifunc_rhs) + Yn1p * vector_free_coef)[:n_dim]
        # where vector_free_coef is a vector. This gives Yn1n = Yn1n and
        #
        # Yn1n = Yn[i_1n] = (A_einv@chiphifunc_rhs + Yn1p * vector_free_coef)[i_1n]
        # = A_einv[i_1n]@chiphifunc_rhs + Yn1p * vector_free_coef[i_1n]
        #
        # Therefore,
        #                                           Yn1n + Yn1p = Yn_free is equivalent to
        # A_einv[i_1n]@chiphifunc_rhs + Yn1p * vector_free_coef[i_1n] + Yn1p = Yn_free
        # A_einv[i_1n]@chiphifunc_rhs + Yn1p * (vector_free_coef[i_1n]+1) = Yn_free
        # -A_einv[i_1n]@chiphifunc_rhs + Yn_free = Yn1p * (vector_free_coef[i_1n]+1)
        # (-A_einv[i_1n]@chiphifunc_rhs + Yn_free)/(vector_free_coef[i_1n]+1) = Yn1p
        # i_1n is shifted by the padding
        i_1p = (n_eval+1)//2 # location of Yn0 or Yn1p
        i_1n = i_1p-1 # Index of Yn1n (or if n is even, Yn2_n)
        # in chiphifunc:
        # vector_free_coef[i_free] = -np.ones((vector_free_coef.shape[1]))
        vec_free = (-np.einsum('ik,ik->k',O_einv[i_1n],chiphifunc_rhs_content) + Yn_free_content)/(vector_free_coef[i_1n]+1)

    else:
        vec_free = Yn_free_content
    # print('vec_free IS NOT WORKING')
    # print(vec_free)
    # print('Yn_free_content')
    # print(Yn_free_content)
    # print('vector_free_coef[i_1n]-1')
    # print(vector_free_coef[i_1n]-1)
    Yn = np.einsum('ijk,jk->ik',O_einv,chiphifunc_rhs_content) + vec_free * vector_free_coef
    # print('Yn:')
    # print(Yn)
    return(ChiPhiFunc(Yn))


# Evaluates iota_{(n-1)/2}.
# Requires:
# \Lambda_n (X_{n-1}, Y_{n-1}, Z_{n}, \iota_{(n-3)/2}),
# B_{\theta n}, B_0, B_{\alpha 0}$
def iterate_iota_nm1b2(sigma_tilde_n0, n_eval,
    X_coef_cp, Y_coef_cp, Z_coef_cp,\
               tau_p, dl_p, kap_p,\
               iota_coef, eta,\
               B_denom_coef_c, B_alpha_coef, B_theta_coef_cp, Yn1s_p):

    if n_eval%2!=1:
        raise ValueError("n must be even to evaluate iota_{(n-1)/2}")

    Y11s_p, Y11c_p = Y_coef_cp[1].get_Yn1s_Yn1c()
    _, X11c_p = X_coef_cp[1].get_Yn1s_Yn1c()

    sigma_p = Y11c_p/Y11s_p # Definition. On pg 4 below (6)

    # Note: mask n leaves nth-order as the last order in.
    # Xi requires Yn, Zn+1=0. This means mask(n-1) and mask(n)
    Xi_n_p = parsed.eval_full_Xi_n_p(
        n_eval, X_coef_cp, Y_coef_cp.mask(n_eval-1).zero_append(), Z_coef_cp.mask(n_eval).zero_append(), \
        kap_p, dl_p, tau_p, iota_coef.mask((n_eval-1)//2-1).zero_append()).get_constant()
    iota_0 = iota_coef[0]

    # Evaluates exp(2*iota_bar_0*integral_of_sigma_to_phi').
    exponent = 2*iota_0*sigma_p.integrate_phi(periodic = False)
    exp_factor = exponent.exp()
    exponent_2pi = 2*iota_0*sigma_p.integrate_phi(periodic = True)
    if isinstance(exponent_2pi,ChiPhiFunc):
        exp_factor_2pi = exponent_2pi.exp() # the result was a float
    else:
        exp_factor_2pi = np.e**exponent_2pi # the result was a float

    # Yn1s_p MUST be evaluated with iota_nm1b2=0!
    Lambda_n_p_tilde = Lambda_n_p(Yn1s_p, Y11s_p, Y11c_p, X11c_p, iota_0, tau_p, dl_p, sigma_p, Xi_n_p)

    # B_0
    B0 = B_denom_coef_c[0]

    # B_{\alpha 0}
    B_alpha0 = B_alpha_coef[0]

    # B_{\theta n 0}. The whole term might be 0 depending on init conds (current free?)
    try:
        B_theta_n0 = B_theta_coef_cp[n_eval].get_constant()
    except AttributeError:
        B_theta_n0 = B_theta_coef_cp[n_eval]

    # The denominator,
    denom = (exp_factor*(1+sigma_p**2+(eta/kap_p)**4/4/B0)).integrate_phi(periodic = True)

    return(
        ((exp_factor*(Lambda_n_p_tilde+2*B_alpha0*B0*B_theta_n0/(Y11s_p**2))).integrate_phi(periodic = True)
        + sigma_tilde_n0*(1-exp_factor_2pi))/denom
    )


# Evaluates Zn,
# Requires:
# X_{n-1}, Y_{n-1}, Z_{n-1},
# B_{\theta n}, B_{\psi n-1},
# B_{\alpha (n-2)/2 \text{ or } (n-3)/2},
# \iota_{(n-2)/2 \text{ or } (n-3)/2}
# \kappa, \frac{dl}{d\phi}, \tau
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

def iterate_Yn1s_p(n_eval,
    X_coef_cp, Y_coef_cp, Z_coef_cp,
    B_psi_coef_cp, B_theta_coef_cp,
    B_alpha_coef,
    kap_p, dl_p, tau_p, eta,
    iota_coef):
    return(parsed.evaluate_ynp1s1_full(n_eval-1,
        X_coef_cp,
        # Y_coef_cp.mask(n_eval) also works
        Y_coef_cp.mask(n_eval-1).zero_append(),
        Z_coef_cp,
        B_psi_coef_cp,
        B_theta_coef_cp,
        B_alpha_coef,
        kap_p, dl_p, tau_p, eta,
        iota_coef).get_constant())

# Solving for Yn1c for odd orders
def iterate_Yn1c_p(sigma_tilde_n0,\
                  n_eval, X_coef_cp, Y_coef_cp, Z_coef_cp,\
                  iota_coef,\
                  tau_p, dl_p, kap_p, eta,\
                  B_denom_coef_c, B_alpha_coef,
                  B_psi_coef_cp, B_theta_coef_cp, Yn1s_p):

    Y11s_p, Y11c_p = Y_coef_cp[1].get_Yn1s_Yn1c()
    _, X11c_p = X_coef_cp[1].get_Yn1s_Yn1c()

    # Note the difference with iota_nm1b2: iota(n-1)//2=0 is no longer applied here.
    Xi_n_p_no_iota_mask = parsed.eval_full_Xi_n_p(
        n_eval, X_coef_cp, Y_coef_cp.mask(n_eval-1).zero_append(), Z_coef_cp.mask(n_eval).zero_append(), \
        kap_p, dl_p, tau_p, iota_coef).get_constant()
    Xi_n_p = parsed.eval_full_Xi_n_p(
        n_eval, X_coef_cp, Y_coef_cp.mask(n_eval-1).zero_append(), Z_coef_cp.mask(n_eval).zero_append(), \
        kap_p, dl_p, tau_p, iota_coef.mask((n_eval-1)//2-1).zero_append()).get_constant()

    iota_0 = iota_coef[0]

    sigma_p = Y11c_p/Y11s_p # Definition. On pg 4 below (6)
    Lambda_n_p_eval = Lambda_n_p(Yn1s_p, Y11s_p, Y11c_p, X11c_p, iota_0, tau_p, dl_p, sigma_p, Xi_n_p_no_iota_mask)

    exponent = 2*iota_0*sigma_p.integrate_phi(periodic = False)
    exp_factor = exponent.exp()
    exp_factor_neg = (-exponent).exp()

    # B_0
    B0 = B_denom_coef_c[0]

    # B_{\alpha 0}
    B_alpha0 = B_alpha_coef[0]

    # B_{\theta n 0}. The whole term might be 0 depending on init conds (current free?)
    try:
        B_theta_np10 = B_theta_coef_cp[n_eval+1].get_constant()
    except AttributeError:
        B_theta_np10 = B_theta_coef_cp[n_eval+1]

    sigma_tilde_n = exp_factor_neg*(\
        sigma_tilde_n0\
        +(\
            exp_factor\
            *(Lambda_n_p_eval+2*B_alpha0*B0*B_theta_np10/(Y11s_p**2))\
        ).integrate_phi(periodic=False)\
    )

    return((sigma_tilde_n*Y11s_p), Lambda_n_p_eval)

# Evaluates Yn1s for odd n's.
# X_{n}, Y_{n-1}, Z_{n-1},
# B_{\theta n-1}, B_{\psi  n-3},
# \iota_{(n-3)/2 or (n-4)/2}, B_{\alpha  (n-1)/2 or (n-2)/2}


# Evaluates \Lambda_n. Used in Yn and iota_(n-1)/2
# Must be evaluated with iota_{(n-1)/2} = 0 to get lambda_tilde
def Lambda_n_p(Yn1s_p, Y11s_p, Y11c_p, X11c_p, iota_0, tau_p, dl_p, sigma_p, Xi_n_p):
    return(
        (diff(Yn1s_p,'phi',1)*sigma_p/Y11s_p
        -2*iota_0*Yn1s_p/Y11s_p
        -Yn1s_p*diff(Y11c_p,'phi',1)/(Y11s_p**2)
        +2*tau_p*dl_p*Yn1s_p*X11c_p/(Y11s_p**2)
        -2*Xi_n_p/(Y11s_p**2))
    )

# Evaluates B_{\psi n-2}
# Requires:
# X_{n-1}, Y_{n-1}, Z_{n-1},
# B_{\theta n-1}, B_0,
# B_{\alpha 0}, \bar{\iota}_{(n-2)/2 or (n-3)/2}$
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

def iterate_delta_n_0_offset(n_eval,
    B_denom_coef_c,
    p_perp_coef_cp,
    Delta_coef_cp,
    iota_coef,
    integral_mode,
    fft_max_freq=None,
    no_iota_masking = False):

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
        iota=iota_coef[0],
        f=Delta_n_inhomog_component.content,
        integral_mode=integral_mode,
        fft_max_freq=fft_max_freq
    )
    Delta_out = ChiPhiFunc(content).cap_m(n_eval)
    if n_eval%2==0:
        Delta_out -= np.average(Delta_out.get_constant().content)
    return(Delta_out)

''' III. Equilibrium manager and Iterate '''

# A container for all equilibrium quantities.
# All coef inputs must be ChiPhiEpsFunc's.
class Equilibrium:
    def __init__(self, unknown, constant, noise, magnetic_only):
        self.noise=noise
        self.unknown=unknown
        self.constant=constant

        # Check if every term is on the same order
        self.check_order_consistency()

        # Some coeffs are really long. We only calc them once.
        # Very little speed improvement compared to eval_B_psi_coefs_full.
        self.prepare_lambdas(magnetic_only)

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
                noise[key]['X_coef_cp'] = ChiPhiEpsFunc.zeros_like(X_coef_cp)
                noise[key]['Y_coef_cp'] = ChiPhiEpsFunc.zeros_like(Y_coef_cp)
                noise[key]['Z_coef_cp'] = ChiPhiEpsFunc.zeros_like(Z_coef_cp)
                noise[key]['B_psi_coef_cp'] = ChiPhiEpsFunc.zeros_like(B_psi_coef_cp)
                noise[key]['B_theta_coef_cp'] = ChiPhiEpsFunc.zeros_like(B_theta_coef_cp)
                noise[key]['p_perp_coef_cp'] = ChiPhiEpsFunc.zeros_like(unknown['p_perp_coef_cp'])
                noise[key]['Delta_coef_cp'] = ChiPhiEpsFunc.zeros_like(unknown['Delta_coef_cp'])

        return(Equilibrium(unknown, constant, noise, magnetic_only))

    def save_self(self, file_name):

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
            'unknown':unknown_to_content_list,\
            'noise':noise_to_content_list,\
            'constant':const_to_content_list,\
        }
        np.savez(file_name, big_dict)

    def load(filename):
        npzfile = np.load(filename, allow_pickle=True)
        big_dict = npzfile['arr_0'].item()
        raw_unknown = big_dict['unknown']
        raw_constant = big_dict['constant']
        raw_noise = big_dict['noise']

        unknown = {}
        for key in raw_unknown.keys():
            unknown[key] = ChiPhiEpsFunc.from_content_list(raw_unknown[key])

        noise = {}
        for keya in raw_noise.keys():
            noise[keya]={}
            for keyb in raw_noise[keya].keys():
                noise[keya][keyb] \
                    = ChiPhiEpsFunc.from_content_list(raw_noise[keya][keyb])

        constant={}
        constant['B_denom_coef_c']\
            = ChiPhiEpsFunc.from_content_list(raw_constant['B_denom_coef_c'])
        constant['B_alpha_coef']\
            = ChiPhiEpsFunc.from_content_list(raw_constant['B_alpha_coef'])
        constant['kap_p']\
            = ChiPhiFunc(raw_constant['kap_p'])
        constant['dl_p']\
            = raw_constant['dl_p']
        constant['tau_p']\
            = ChiPhiFunc(raw_constant['tau_p'])
        constant['eta']\
            = raw_constant['eta']

        return(Equilibrium(unknown, constant, noise))

    # Order consistency check --------------------------------------------------
    # Get the current order of an equilibrium
    def get_order(self):
        self.check_order_consistency()
        return(self.unknown['X_coef_cp'].get_order())

    # Check of all terms are on consistent orders
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

    def check_governing_equations(self, n_eval=None):
        if n_eval is None:
            n_eval = self.get_order()
        elif n_eval>self.get_order():
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

        n=n_eval
        n_unknown = n_eval-1

        J = MHD_parsed.validate_J(n,
            X_coef_cp,
            Y_coef_cp, Z_coef_cp,
            B_denom_coef_c, B_alpha_coef,
            kap_p, dl_p, tau_p, iota_coef)
        Cb = MHD_parsed.validate_Cb(n,
            X_coef_cp,
            Y_coef_cp, Z_coef_cp,
            B_denom_coef_c, B_alpha_coef,
            B_psi_coef_cp, B_theta_coef_cp,
            kap_p, dl_p, tau_p, iota_coef)
        Ck = MHD_parsed.validate_Ck(n, X_coef_cp, Y_coef_cp, Z_coef_cp,
            B_denom_coef_c, B_alpha_coef,
            B_psi_coef_cp, B_theta_coef_cp,
            kap_p, dl_p, tau_p, iota_coef)
        Ct = MHD_parsed.validate_Ct(n, X_coef_cp, Y_coef_cp, Z_coef_cp,
            B_denom_coef_c, B_alpha_coef,
            B_psi_coef_cp, B_theta_coef_cp,
            kap_p, dl_p, tau_p, iota_coef)
        I = MHD_parsed.validate_I(n, B_denom_coef_c,
            p_perp_coef_cp, Delta_coef_cp,
            iota_coef)
        II = MHD_parsed.validate_II(n,
            B_theta_coef_cp, B_alpha_coef, B_denom_coef_c,
            p_perp_coef_cp, Delta_coef_cp, iota_coef)
        III = MHD_parsed.validate_III(n,
            B_theta_coef_cp, B_psi_coef_cp,
            B_alpha_coef, B_denom_coef_c,
            p_perp_coef_cp, Delta_coef_cp,
            iota_coef)
        E6 = MHD_parsed.validate_E6(n,
            B_theta_coef_cp, B_psi_coef_cp,
            B_alpha_coef, B_denom_coef_c,
            p_perp_coef_cp, Delta_coef_cp,
            iota_coef)
        D2 = MHD_parsed.validate_D2(
            n, X_coef_cp, Y_coef_cp, Z_coef_cp,
            B_denom_coef_c, B_alpha_coef,
            B_psi_coef_cp, B_theta_coef_cp,
            kap_p, dl_p, tau_p, iota_coef
        )
        D3 = MHD_parsed.validate_D3(
            n, X_coef_cp, Y_coef_cp, Z_coef_cp,
            B_denom_coef_c, B_alpha_coef,
            B_psi_coef_cp, B_theta_coef_cp,
            kap_p, dl_p, tau_p, iota_coef
        )
        D23 = MHD_parsed.validate_D23(
            n, X_coef_cp, Y_coef_cp, Z_coef_cp,
            B_denom_coef_c, B_alpha_coef,
            B_psi_coef_cp, B_theta_coef_cp,
            kap_p, dl_p, tau_p, iota_coef
        )
        kt = MHD_parsed.validate_kt(
            n, X_coef_cp, Y_coef_cp, Z_coef_cp,
            B_denom_coef_c, B_alpha_coef,
            B_psi_coef_cp, B_theta_coef_cp,
            kap_p, dl_p, tau_p, iota_coef
        )

        return(J, Cb, Ck, Ct, I, II, III, E6, D2, D3, D23, kt)
    ''' Display '''
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
    def prepare_lambdas(self, magnetic_only):
        if not magnetic_only:
            self.looped_coef_lambdas = eval_looped_coef_lambdas(self)
            self.looped_B_psi_lambdas = MHD_parsed.eval_B_psi_lambdas_full(
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
    def iterate_2_magnetic_only(self,
        B_theta_nm1, B_theta_n,
        B_psi_nm20,
        B_alpha_nb2,
        B_denom_nm1, B_denom_n,
        Yn0,
        p_perp_nm1=0, p_perp_n=0,
        Delta_nm1=0, Delta_n=0,
        n_eval=None, filter=False, filter_mode='low_pass', filter_arg=100):


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
        B_denom_coef_c.append(B_denom_nm1)
        B_denom_coef_c.append(B_denom_n)
        p_perp_coef_cp.append(p_perp_nm1)
        p_perp_coef_cp.append(p_perp_n)
        Delta_coef_cp.append(Delta_nm1)
        Delta_coef_cp.append(Delta_n)
        # Managing noise
        self.noise['filter']['B_theta_coef_cp'].append(0)
        self.noise['filter']['p_perp_coef_cp'].append(0)
        self.noise['filter']['Delta_coef_cp'].append(0)
        self.noise['filter']['B_theta_coef_cp'].append(0)
        self.noise['filter']['p_perp_coef_cp'].append(0)
        self.noise['filter']['Delta_coef_cp'].append(0)

        # For reducing the amount of duplicate code
        def filter_record_noise_and_append(name, chiphifunc):
            self.noise['filter'][name].append(chiphifunc.noise_filter(filter_mode, filter_arg))
            if filter:
                chiphifunc = chiphifunc.filter(filter_mode, filter_arg)
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
        filter_record_noise_and_append('B_psi_coef_cp', B_psi_nm3)
        # self.noise['filter']['B_psi_coef_cp'].append(
        #     B_psi_nm3.noise_filter(filter_mode, filter_arg)
        # )
        # if filter:
        #     B_psi_nm3 = B_psi_nm3.filter(filter_mode, filter_arg)
        # B_psi_coef_cp.append(B_psi_nm3)

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
        filter_record_noise_and_append('Z_coef_cp', Znm1)
        # self.noise['filter']['Z_coef_cp'].append(
        #     Znm1.noise_filter(filter_mode, filter_arg)
        # )
        # if filter:
        #     Znm1 = Znm1.filter(filter_mode, filter_arg)
        # Z_coef_cp.append(Znm1)

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
        self.noise['filter']['X_coef_cp'].append(Xnm1.noise_filter(filter_mode, filter_arg))
        filter_record_noise_and_append('X_coef_cp', Xnm1)
        # if filter:
        #     Xnm1 = Xnm1.filter(filter_mode, filter_arg)
        # X_coef_cp.append(Xnm1)

        Ynm11s_p = iterate_Yn1s_p(n_eval=n_eval-1,
            X_coef_cp=X_coef_cp,
            Y_coef_cp=Y_coef_cp,
            Z_coef_cp=Z_coef_cp,
            B_psi_coef_cp=B_psi_coef_cp,
            B_theta_coef_cp=B_theta_coef_cp,
            B_alpha_coef=B_alpha_coef,
            kap_p=kap_p,
            dl_p=dl_p,
            tau_p=tau_p,
            eta=eta,
            iota_coef=iota_coef
            )

        iota_nm2b2 = iterate_iota_nm1b2(sigma_tilde_n0=0,
            n_eval=n_eval-1,
            X_coef_cp=X_coef_cp,
            Y_coef_cp=Y_coef_cp,
            Z_coef_cp=Z_coef_cp,
            tau_p=tau_p,
            dl_p=dl_p,
            kap_p=kap_p,
            iota_coef=iota_coef,
            eta=eta,
            B_denom_coef_c=B_denom_coef_c,
            B_alpha_coef=B_alpha_coef,
            B_theta_coef_cp=B_theta_coef_cp,
            Yn1s_p=Ynm11s_p
            )
        iota_coef.append(iota_nm2b2)

        # Requires:
        # X_{n}, Y_{n-1}, Z_{n-1},
        # B_{\theta n-1}, B_{\psi n-3},
        # \iota_{(n-3)/2 or (n-4)/2}, B_{\alpha (n-1)/2 or (n-2)/2}
        Ynm1 = iterate_Yn_cp_magnetic(n_eval=n_eval-1,
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
            # Yn_free=Y_free_nm1, Evaluated in Ynm1
            Yn1s_p=Ynm11s_p
            )
        filter_record_noise_and_append('Y_coef_cp', Ynm1)
        # self.noise['filter']['Y_coef_cp'].append(Ynm1.noise_filter(filter_mode, filter_arg))
        # if filter:
        #     Ynm1 = Ynm1.filter(filter_mode, filter_arg)
        # Y_coef_cp.append(Ynm1)

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
        filter_record_noise_and_append('B_psi_coef_cp', B_psi_nm2)
        # self.noise['filter']['B_psi_coef_cp'].append(B_psi_nm2.noise_filter(filter_mode, filter_arg))
        # if filter:
        #     B_psi_nm2 = B_psi_nm2.filter(filter_mode, filter_arg)
        # B_psi_coef_cp.append(B_psi_nm2)

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
        filter_record_noise_and_append('Z_coef_cp', Zn)
        # self.noise['filter']['Z_coef_cp'].append(Zn.noise_filter(filter_mode, filter_arg))
        # if filter:
        #     Zn = Zn.filter(filter_mode, filter_arg)
        # Z_coef_cp.append(Zn)

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
        filter_record_noise_and_append('X_coef_cp', Xn)
        # self.noise['filter']['X_coef_cp'].append(Xn.noise_filter(filter_mode, filter_arg))
        # if filter:
        #     Xn = Xn.filter(filter_mode, filter_arg)
        # X_coef_cp.append(Xn)

        Yn = iterate_Yn_cp_magnetic(n_eval=n_eval,
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
            # Yn1s_p=Yn1s_p, Not needed
            Yn0=Yn0
            )
        filter_record_noise_and_append('Y_coef_cp', Yn)
        # self.noise['filter']['Y_coef_cp'].append(Yn.noise_filter(filter_mode, filter_arg))
        # if filter:
        #     Yn = Yn.filter(filter_mode, filter_arg)
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
        filter_mode='low_pass', filter_arg=100,
        loop_max_freq=(500,500),
        max_k_diff_pre_inv=(1000,1000),
        max_k_diff_post_inv=(1000,1000)
        ):

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
        def filter_record_noise_and_append(name, chiphifunc):
            self.noise['filter'][name].append(chiphifunc.noise_filter(filter_mode, filter_arg))
            if filter:
                chiphifunc = chiphifunc.filter(filter_mode, filter_arg)
            self.unknown[name].append(chiphifunc)

        # Evaluating order n_eval-1

        # print('iota 1 right before loop',iota_coef[1])
        solution_nm1_known_iota = looped_solver.iterate_looped(
            n_unknown = n_eval-1,
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
            looped_coef_lambdas = self.looped_coef_lambdas,
            looped_B_psi_lambdas = self.looped_B_psi_lambdas,
            max_freq = loop_max_freq[0],
            max_k_diff_pre_inv = max_k_diff_pre_inv[0],
            max_k_diff_post_inv = max_k_diff_post_inv[0],
        )
        filter_record_noise_and_append('B_theta_coef_cp', solution_nm1_known_iota['B_theta_n'])
        filter_record_noise_and_append('B_psi_coef_cp', solution_nm1_known_iota['B_psi_nm2'])
        filter_record_noise_and_append('X_coef_cp', solution_nm1_known_iota['Xn'])
        filter_record_noise_and_append('Y_coef_cp', solution_nm1_known_iota['Yn'])
        filter_record_noise_and_append('Z_coef_cp', solution_nm1_known_iota['Zn'])
        filter_record_noise_and_append('p_perp_coef_cp', solution_nm1_known_iota['pn'])
        filter_record_noise_and_append('Delta_coef_cp', solution_nm1_known_iota['Deltan'])
        # Don't record noise yet. This "partial" solution will be fed into
        # iterate_looped

        B_theta_coef_cp.append(solution_nm1_known_iota['B_theta_np10'])

        # Calculating Yn from the looped equation
        # Ynm1_RHS = iterate_Yn_cp_RHS(n_eval-1,
        #     X_coef_cp=X_coef_cp,
        #     Y_coef_cp=Y_coef_cp,
        #     Z_coef_cp=Z_coef_cp,
        #     B_psi_coef_cp=B_psi_coef_cp,
        #     B_theta_coef_cp=B_theta_coef_cp,
        #     B_alpha_coef=B_alpha_coef,
        #     B_denom_coef_c=B_denom_coef_c,
        #     kap_p=kap_p, dl_p=dl_p, tau_p=tau_p, eta=eta,
        #     iota_coef=iota_coef)
        # O_einvnm1 = solution_nm1_known_iota['O_einv']
        # vector_free_coefnm1 = solution_nm1_known_iota['vector_free_coef']
        # Ynm11p = solution_nm1_known_iota['vec_free']
        # Ynm1 = ChiPhiFunc((np.einsum('ijk,jk->ik',O_einvnm1,Ynm1_RHS) + Ynm11p * vector_free_coefnm1))

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
            looped_coef_lambdas = self.looped_coef_lambdas,
            looped_B_psi_lambdas = self.looped_B_psi_lambdas,
            max_freq = loop_max_freq[1],
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
        filter_record_noise_and_append('B_psi_coef_cp', solution_n['B_psi_nm2'])
        # This only reassigns the pointer B_theta. Need to re-assign self.unknown[]
        # too.
        B_theta_coef_cp = B_theta_coef_cp.mask(n_eval-1)
        self.unknown['B_theta_coef_cp'] = B_theta_coef_cp
        filter_record_noise_and_append('B_theta_coef_cp', solution_n['B_theta_n'])

        filter_record_noise_and_append('X_coef_cp', solution_n['Xn'])
        filter_record_noise_and_append('Z_coef_cp', solution_n['Zn'])
        filter_record_noise_and_append('p_perp_coef_cp', solution_n['pn'])
        filter_record_noise_and_append('Delta_coef_cp', solution_n['Deltan'])
        filter_record_noise_and_append('Y_coef_cp', solution_n['Yn'])

        print("Time elapsed(s):",(time.time() - start_time))
        self.check_order_consistency()
        return(solution_nm1_known_iota, solution_n)
