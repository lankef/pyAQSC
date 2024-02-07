# Wrapped/completed recursion relations based on translated expressions
# in parsed/. Necessary masking and/or n-substitution are included. All iterate_*
# methods returns ChiPhiFunc's.
import jax.numpy as jnp

# ChiPhiFunc and ChiPhiEpsFunc
from .chiphifunc import *
from .chiphiepsfunc import *
from .math_utilities import *

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
def iterate_Yn_cp_magnetic(
    n_unknown,
    X_coef_cp,
    Y_coef_cp,
    Z_coef_cp,
    B_psi_coef_cp,
    B_theta_coef_cp,
    B_alpha_coef,
    B_denom_coef_c,
    kap_p, dl_p, tau_p,
    iota_coef,
    static_max_freq,
    Yn0=None,
    Yn1c_avg=None):

    nfp = X_coef_cp.nfp
    n_eval = n_unknown+1

    _, O_einv, vector_free_coef = \
        iterate_Yn_cp_operators(
            n_unknown,
            X_coef_cp=X_coef_cp,
            B_alpha_coef=B_alpha_coef, 
            Y1c_mode=True
        )

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
    Y_coef_cp_no_unknown = Y_coef_cp_no_unknown.append(new_Y_n_no_unknown)

    # Calculating D3 to solve for Yn1p
    if n_unknown%2==0:
        if Yn0 is None:
            return(ChiPhiFuncSpecial(-15))
        Yn_free_content = Yn0
    else:
        if Yn1c_avg is None:
            Yn1c_avg = 0
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

        coef_Yn1c_in_D3, coef_dp_Yn1c_in_D3 = looped_coefs.lambda_coef_Yn1c_in_D3(
            vector_free_coef, 
            X_coef_cp, Y_coef_cp, 
            iota_coef, 
            dl_p, tau_p, 
            nfp
        )
        # Solving y'+py=f for Yn1+. This equation has no unique solution,
        # and a initial condition is provided.
        p_eff = (coef_Yn1c_in_D3.content/coef_dp_Yn1c_in_D3.content)[0]
        f_eff = (D3_RHS_no_unknown.content/coef_dp_Yn1c_in_D3.content)[0]
        p_fft = fft_filter(jnp.fft.fft(p_eff), static_max_freq, axis=0)
        f_fft = fft_filter(jnp.fft.fft(f_eff), static_max_freq, axis=0)
        # Creating differential operator and convolution operator
        # as in solve_ODE
        diff_matrix = fft_dphi_op(static_max_freq)
        conv_matrix = fft_conv_op(p_fft)
        tot_matrix = diff_matrix + conv_matrix
        
        tot_matrix_normalization = jnp.max(jnp.abs(tot_matrix))
        tot_matrix = tot_matrix.at[:, 0].set(
            tot_matrix[:, 0]+tot_matrix_normalization # was +1
        )
        f_fft = f_fft+Yn1c_avg*static_max_freq*tot_matrix_normalization
        sln_fft = jnp.linalg.solve(tot_matrix, f_fft)

        
        len_phi = Yn_rhs.content.shape[1]
        Yn_free_content = jnp.fft.ifft(fft_pad(sln_fft, len_phi, axis=0), axis=0)[None, :]
        
        len_phi = Yn_rhs.content.shape[1]
        Yn_free_content = jnp.fft.ifft(fft_pad(sln_fft, len_phi, axis=0), axis=0)[None, :]
        # Yn_free_content = solve_ODE(
        #     coeff_arr=coef_Yn1c_in_D3.content,
        #     coeff_dp_arr=coef_dp_Yn1c_in_D3.content*nfp,
        #     f_arr=D3_RHS_no_unknown.content,
        #     static_max_freq=static_max_freq
        # ) # Seems underdetermined. Blows up.

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
    static_max_freq=-1,
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
        static_max_freq=static_max_freq
    )
    Delta_out = ChiPhiFunc(content, Delta_n_inhomog_component.nfp).cap_m(n_eval)
    if n_eval%2==0:
        Delta_out -= jnp.average(Delta_out[0].content)
    return(Delta_out)
