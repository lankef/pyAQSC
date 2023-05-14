# Solves the looped equations.
import jax.numpy as jnp
from functools import partial
from jax import jit

# ChiPhiFunc and ChiPhiEpsFunc
from chiphifunc import *
from chiphiepsfunc import *
from math_utilities import *

# parsed relations
import MHD_parsed
import looped_coefs

# TODO: make delta and B_theta coefficients constants carried by the equilibrium.

def lambda_coef_delta(n_eval, B_alpha_coef, B_denom_coef_c):
     return(phi_avg(-(B_alpha_coef[0]*B_denom_coef_c[1].dchi()))*(n_eval-1)/(2*n_eval))

# This will not be jitted, equilibrium will not be jitted, but iterate will be jitted.
''' I. RHS '''
# RHS is run BEFORE operators because B_psi, X, Z, p, Delta are calculated during
# RHS calculation. Uses
# Even order: B_psi[n_unknown-2-1] and X, Z, p, Delta[n_unknown-1] for even orders,
# Odd order:  B_psi[n_unknown-2-1], Delta[n-1] and X, Z, p, Delta, Y[n_unknown]
# Outputs a dictionary containing:
# B_psi_nm2_no_unknown (even order only)
# O_einv,
# vector_free_coef,
# RHS_0_offset,
def generate_RHS(
    n_unknown,
    X_coef_cp, Y_coef_cp, Z_coef_cp,
    p_perp_coef_cp, Delta_coef_cp,
    B_psi_coef_cp, B_theta_coef_cp,
    B_alpha_coef, B_denom_coef_c,
    kap_p, tau_p, dl_p,
    iota_coef, hard_cutoff_freq, cutoff_freq,
):
# n_eval is the order at which the "looped" equations are evaluated at
    n_eval = n_unknown+1
    len_phi = X_coef_cp[n_unknown-1].content.shape[1]
    out_dict_RHS = {}
    ''' O_einv and vec_free for Y. Y is unknown at all orders '''
    # but solved from different equations (II tilde/D3) at each (even/odd) order.
    O_matrices, O_einv, vector_free_coef = \
        equilibrium.iterate_Yn_cp_operators(
            n_unknown=n_unknown,
            X_coef_cp=X_coef_cp,
            B_alpha_coef=B_alpha_coef
        )
    out_dict_RHS['O_einv'] = O_einv
    out_dict_RHS['vector_free_coef'] = vector_free_coef
    ''' B_psi without B_theta_n and (if even order) no m=0 component '''
    B_psi_nm2_no_unknown = equilibrium.iterate_dc_B_psi_nm2(
        n_eval=n_unknown,
        X_coef_cp=X_coef_cp,
        Y_coef_cp=Y_coef_cp,
        Z_coef_cp=Z_coef_cp,
        B_theta_coef_cp=B_theta_coef_cp.mask(n_unknown-1).zero_append(),
        B_psi_coef_cp=B_psi_coef_cp,
        B_alpha_coef=B_alpha_coef,
        B_denom_coef_c=B_denom_coef_c,
        kap_p=kap_p,
        dl_p=dl_p,
        tau_p=tau_p,
        iota_coef=iota_coef
    ).antid_chi()

    B_psi_coef_cp_no_unknown = B_psi_coef_cp.mask(n_unknown-3)
    B_psi_coef_cp_no_unknown = B_psi_coef_cp_no_unknown.append(B_psi_nm2_no_unknown)
    out_dict_RHS['B_psi_nm2_no_unknown'] = B_psi_nm2_no_unknown
    # Even orders
    if n_unknown%2==0:
        ''' B_theta, center is known at even orders '''
        B_theta_n = B_theta_coef_cp[n_unknown]
        B_theta_n_0_only = B_theta_n[0]
        B_theta_coef_cp_0_only = B_theta_coef_cp.mask(n_unknown-1)
        B_theta_coef_cp_0_only = B_theta_coef_cp_0_only.append(B_theta_n_0_only)
        # # Used to calculate B_theta_n
        ''' Z and p, contains B_psi, unknown at even orders '''
        Zn_no_B_psi = equilibrium.iterate_Zn_cp(
            n_unknown,
            X_coef_cp, Y_coef_cp, Z_coef_cp,
            B_theta_coef_cp, B_psi_coef_cp_no_unknown,
            B_alpha_coef,
            kap_p, dl_p, tau_p,
            iota_coef)
        Z_coef_cp_no_unknown = Z_coef_cp.mask(n_eval-2)
        Z_coef_cp_no_unknown = Z_coef_cp_no_unknown.append(Zn_no_B_psi)

        pn_no_B_psi = equilibrium.iterate_p_perp_n(
            n_eval=n_unknown,
            B_theta_coef_cp=B_theta_coef_cp,
            B_psi_coef_cp=B_psi_coef_cp_no_unknown,
            B_alpha_coef=B_alpha_coef,
            B_denom_coef_c=B_denom_coef_c,
            p_perp_coef_cp=p_perp_coef_cp,
            Delta_coef_cp=Delta_coef_cp,
            iota_coef=iota_coef)
        p_perp_coef_cp_no_unknown = p_perp_coef_cp.mask(n_eval-2)
        p_perp_coef_cp_no_unknown = p_perp_coef_cp_no_unknown.append(pn_no_B_psi)

        ''' X and Delta, contains B_psi, unknown at even orders '''
        Xn_no_B_psi = equilibrium.iterate_Xn_cp(n_unknown,
            X_coef_cp,
            Y_coef_cp,
            Z_coef_cp_no_unknown,
            B_denom_coef_c,
            B_alpha_coef,
            kap_p, dl_p, tau_p,
            iota_coef)
        X_coef_cp_no_unknown = X_coef_cp.mask(n_eval-2)
        X_coef_cp_no_unknown = X_coef_cp_no_unknown.append(Xn_no_B_psi)

        Deltan_no_B_psi = equilibrium.iterate_delta_n_0_offset(n_eval=n_unknown,
            B_denom_coef_c=B_denom_coef_c,
            p_perp_coef_cp=p_perp_coef_cp_no_unknown,
            Delta_coef_cp=Delta_coef_cp,
            cutoff_freq=cutoff_freq,
            hard_cutoff_freq=hard_cutoff_freq,
            iota_coef=iota_coef)
        Delta_coef_cp_no_unknown_0_offset = Delta_coef_cp.mask(n_eval-2)
        Delta_coef_cp_no_unknown_0_offset = Delta_coef_cp_no_unknown_0_offset.append(Deltan_no_B_psi)

        out_dict_RHS['B_theta_n_0_only'] = B_theta_n_0_only
        out_dict_RHS['Zn_no_B_psi']=Zn_no_B_psi
        out_dict_RHS['pn_no_B_psi']=pn_no_B_psi
        out_dict_RHS['Xn_no_B_psi']=Xn_no_B_psi
        out_dict_RHS['Deltan_no_B_psi']=Deltan_no_B_psi
    # Odd orders
    else:
        B_theta_coef_cp_0_only = B_theta_coef_cp.mask(n_unknown-1).zero_append()

        Zn_no_B_theta = equilibrium.iterate_Zn_cp(n_eval=n_unknown,
            X_coef_cp=X_coef_cp,
            Y_coef_cp=Y_coef_cp,
            Z_coef_cp=Z_coef_cp,
            B_theta_coef_cp=B_theta_coef_cp,
            B_psi_coef_cp=B_psi_coef_cp_no_unknown,
            B_alpha_coef=B_alpha_coef,
            kap_p=kap_p,
            dl_p=dl_p,
            tau_p=tau_p,
            iota_coef=iota_coef
            )
        Z_coef_cp_no_unknown=Z_coef_cp.mask(n_unknown-1)
        Z_coef_cp_no_unknown = Z_coef_cp_no_unknown.append(Zn_no_B_theta)

        Xn_no_B_theta = equilibrium.iterate_Xn_cp(n_eval=n_unknown,
            X_coef_cp=X_coef_cp,
            Y_coef_cp=Y_coef_cp,
            Z_coef_cp=Z_coef_cp_no_unknown,
            B_denom_coef_c=B_denom_coef_c,
            B_alpha_coef=B_alpha_coef,
            kap_p=kap_p,
            dl_p=dl_p,
            tau_p=tau_p,
            iota_coef=iota_coef
            )
        X_coef_cp_no_unknown = X_coef_cp.mask(n_unknown-1)
        X_coef_cp_no_unknown = X_coef_cp_no_unknown.append(Xn_no_B_theta)
        pn_no_B_theta = equilibrium.iterate_p_perp_n(
            n_eval=n_unknown,
            B_theta_coef_cp=B_theta_coef_cp,
            B_psi_coef_cp=B_psi_coef_cp_no_unknown,
            B_alpha_coef=B_alpha_coef,
            B_denom_coef_c=B_denom_coef_c,
            p_perp_coef_cp=p_perp_coef_cp,
            Delta_coef_cp=Delta_coef_cp,
            iota_coef=iota_coef)
        p_perp_coef_cp_no_unknown = p_perp_coef_cp.mask(n_unknown-1)
        p_perp_coef_cp_no_unknown = p_perp_coef_cp_no_unknown.append(pn_no_B_theta)
        Deltan_with_iota_no_B_theta = equilibrium.iterate_delta_n_0_offset(n_eval=n_unknown,
            B_denom_coef_c=B_denom_coef_c,
            p_perp_coef_cp=p_perp_coef_cp_no_unknown,
            Delta_coef_cp=Delta_coef_cp,
            iota_coef=iota_coef,
            cutoff_freq=cutoff_freq,
            hard_cutoff_freq=hard_cutoff_freq,
            no_iota_masking = True)
        Delta_coef_cp_no_unknown_0_offset = Delta_coef_cp.mask(n_eval-2)
        Delta_coef_cp_no_unknown_0_offset = Delta_coef_cp_no_unknown_0_offset.append(Deltan_with_iota_no_B_theta)
        out_dict_RHS['Zn_no_B_theta'] = Zn_no_B_theta
        out_dict_RHS['pn_no_B_theta'] = pn_no_B_theta
        out_dict_RHS['Xn_no_B_theta'] = Xn_no_B_theta
        out_dict_RHS['Deltan_no_B_theta'] = Deltan_with_iota_no_B_theta

    ''' Y, unknown at all orders and contains B_psi. '''
    Yn_rhs_no_unknown = equilibrium.iterate_Yn_cp_RHS(n_unknown=n_unknown,
        X_coef_cp=X_coef_cp_no_unknown,
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
    Yn_rhs_content_no_unknown = Yn_rhs_no_unknown.content
    new_Y_n_no_unknown = ChiPhiFunc(jnp.einsum('ijk,jk->ik', O_einv, Yn_rhs_content_no_unknown), Yn_rhs_no_unknown.nfp)
    Y_coef_cp_no_unknown = Y_coef_cp.mask(n_eval-2)
    Y_coef_cp_no_unknown = Y_coef_cp_no_unknown.append(new_Y_n_no_unknown)
    out_dict_RHS['Yn_rhs_content_no_unknown'] = Yn_rhs_content_no_unknown

    ''' Evaluating looped RHS '''
    looped_RHS_0_offset = -MHD_parsed.eval_loop(
        n=n_eval, \
        # Modified ChiPhiEpsFunc's dont need to be masked.
        X_coef_cp=X_coef_cp_no_unknown.zero_append(), # Done
        Y_coef_cp=Y_coef_cp_no_unknown.zero_append(), # Done
        Z_coef_cp=Z_coef_cp_no_unknown.zero_append(), # Done
        B_theta_coef_cp=B_theta_coef_cp_0_only.zero_append(),
        B_psi_coef_cp=B_psi_coef_cp_no_unknown.zero_append(),
        B_alpha_coef=B_alpha_coef,
        B_denom_coef_c=B_denom_coef_c.zero_append(), # Seems B_denom n+1 is needed but cancels?
        p_perp_coef_cp=p_perp_coef_cp_no_unknown.zero_append(), # Done
        Delta_coef_cp=Delta_coef_cp_no_unknown_0_offset.zero_append(), # Done
         # this zero_append on iota_coef doesn't do anythin at even n_unknown,
         # but masks iota (n_unknown-1)/2 at odd n_unknown.
        kap_p=kap_p, dl_p=dl_p, tau_p=tau_p, iota_coef=iota_coef #iota_coef.zero_append()
    ).cap_m(n_eval-2)

    # Calculating D3. This cannot be merged into the previous
    # if statement because we need to know Y first.
    # Also, the zero-th component of the looped equation
    # is not affected by any substitution, and is identical to
    # II's zeroth component.
    if n_unknown%2==1:
        # Caluclating the center element of II
        II_RHS_no_unknown = -MHD_parsed.eval_II_center(
            n = n_eval,
            B_theta_coef_cp = B_theta_coef_cp_0_only.zero_append(),
            B_alpha_coef = B_alpha_coef,
            B_denom_coef_c = B_denom_coef_c,
            # This mask gets rid of p_perp n_unknown+1, which only appears as dchi.
            p_perp_coef_cp = p_perp_coef_cp_no_unknown.zero_append(),
            # This mask is redundant. Delta appears at order n_unknown.
            Delta_coef_cp = Delta_coef_cp_no_unknown_0_offset.zero_append(),
            iota_coef = iota_coef)[0]

        # Calculating D3
        D3_RHS_no_unknown = -MHD_parsed.eval_D3_RHS_m_LHS(
            n = n_eval,
            X_coef_cp = X_coef_cp_no_unknown.mask(n_unknown),
            # Only dep on Y[+-1]
            Y_coef_cp = Y_coef_cp_no_unknown,
            # The m=0 component is actually indep of Z[n+1]
            Z_coef_cp = Z_coef_cp_no_unknown.mask(n_unknown).zero_append(),
            # This equation may contain both B_theta[n,+-1] and B_theta[n+1,0].
            B_theta_coef_cp = B_theta_coef_cp_0_only.zero_append(),
            B_denom_coef_c = B_denom_coef_c,
            B_alpha_coef = B_alpha_coef,
            iota_coef = iota_coef, #.mask((n_unknown-3)//2).zero_append(), # iota is also masked
            dl_p = dl_p,
            tau_p = tau_p,
        kap_p = kap_p)[0]

        # Setting looped center to be II.
        # cap_m guaranteed loop_center's shape to be (n_unknown, len_phi).
        # [0] guaranteed
        # Since we'll immediately redefine looped_RHS_0_offset, set always_copy
        # to False to save a bit of time
        looped_content = looped_RHS_0_offset.content
        II_content = II_RHS_no_unknown.content
        D3_RHS_content = D3_RHS_no_unknown.content

        looped_content = looped_content.at[n_unknown//2].set(II_content[0])
        looped_content = jnp.concatenate((looped_content, D3_RHS_content), axis=0)
        looped_RHS_0_offset = ChiPhiFunc(looped_content, looped_RHS_0_offset.nfp)

    looped_RHS_0_offset = looped_RHS_0_offset.filter_reduce_length(hard_cutoff_freq)
    looped_RHS_0_offset = looped_RHS_0_offset.filter(cutoff_freq)
    out_dict_RHS['RHS_0_offset'] = looped_RHS_0_offset.fft().content
    return(out_dict_RHS)

''' II. Tensor operator '''
# Generate the differential operator for the dphi(looped equations).
# Because we only know the B_psi0 dependence of (dphi+iota dchi) Delta
# (same as the B_psi0 dependence of dphi Delta), we solve the phi derivative
# of the looped equation, instead of the looped equation itself directly.
# Outputs an (n,n,len_phi,len_phi) differential operator acting on
# the fft of an unknown vector
# Even order:
# [
#     [B_theta +n-1],
#     ...
#     (0th ELEMENT REMOVED)
#     ...
#     [B_theta -n+1],
#     [Y_n 0],
#     [dphi B_psi n-2 0], (sometimes abreviated as B_psi in this code)
# ]
# Odd order:
# [
#     [B_theta +n-1],
#     ...
#     ...
#     [B_theta -n+1],
#     [Y_n 1+],
#     [B_theta n,0]
# ]
# by jnp.tensordot(full_tensor_fft_op_reordered, fft_of_LHS, 2).
# len_phi: the number of phi modes kept in the tensor.
# For avoiding singular tensor.
# @partial(jit, static_argnums=(0,1,17))
def generate_tensor_operator(
    n_unknown,
    nfp,
    X_coef_cp, Y_coef_cp, Z_coef_cp,
    p_perp_coef_cp, Delta_coef_cp,
    B_psi_coef_cp, B_theta_coef_cp,
    B_alpha_coef, B_denom_coef_c,
    kap_p, tau_p, dl_p,
    iota_coef,
    O_einv, vector_free_coef,
    hard_cutoff_freq,
    cutoff_freq,
    # Filter off-diagonal comps of the linear diff operator before inverting
    max_k_diff_pre_inv = -1,
    # Filter off-diagonal comps of the linear diff operator after inverting
    max_k_diff_post_inv = -1,
):

    out_dict_tensor = {}
    # Tilde II has n_unknown components at order n_unknown.
    n_eval = n_unknown+1
    # len_tensor is the number of phi modes kept in a tensor.
    len_tensor = hard_cutoff_freq*2

    ''' Y coeffs '''
    # An (n, 1, 2, 2) tensor, acting on
    #     [[Y_n,free]]
    # solved at even orders only
    # Loading coefficients
    looped_y_coefs = looped_coefs.eval_y_coefs(
        n_eval=n_eval,
        X_coef_cp=X_coef_cp,
        Y_coef_cp=Y_coef_cp,
        Delta_coef_cp=Delta_coef_cp,
        B_alpha_coef=B_alpha_coef,
        dl_p=dl_p,
        tau_p=tau_p,
        iota_coef=iota_coef)
    coef_Y = looped_y_coefs['coef_Y']
    coef_dchi_Y = looped_y_coefs['coef_dchi_Y']
    coef_dphi_Y = looped_y_coefs['coef_dphi_Y']
    coef_dchi_dphi_Y = looped_y_coefs['coef_dchi_dphi_Y']
    coef_dphi_dphi_Y = looped_y_coefs['coef_dphi_dphi_Y']
    coef_dchi_dchi_Y = looped_y_coefs['coef_dchi_dchi_Y']
    coef_dchi_dchi_dphi_Y = looped_y_coefs['coef_dchi_dchi_dphi_Y']
    coef_dphi_dphi_dchi_Y = looped_y_coefs['coef_dphi_dphi_dchi_Y']
    coef_dchi_dchi_dchi_Y = looped_y_coefs['coef_dchi_dchi_dchi_Y']

    # dchi operates on only vector_free_coef and dphi operates only on the free component
    # This can be simplified to be operator manipulations, rather than chain rules,
    # but this is known to work at the moment. TODO: re-write as linear operations.
    vector_free_coef_short = jnp.fft.ifft(jnp.fft.fft(vector_free_coef, axis = 1), axis=1)
    coef_Y_n0_dphi_0 = (
        coef_Y * ChiPhiFunc(vector_free_coef_short, nfp)
        +coef_dchi_Y * ChiPhiFunc(vector_free_coef_short, nfp).dchi()
        +coef_dchi_dchi_Y * ChiPhiFunc(vector_free_coef_short, nfp).dchi().dchi()
        +coef_dchi_dchi_dchi_Y * ChiPhiFunc(vector_free_coef_short, nfp).dchi().dchi().dchi()
    ).cap_m(n_eval-2)
    coef_Y_n0_dphi_1 = (
        coef_dphi_Y * ChiPhiFunc(vector_free_coef_short, nfp)
        +coef_dchi_dphi_Y * ChiPhiFunc(vector_free_coef_short, nfp).dchi()
        +coef_dchi_dchi_dphi_Y * ChiPhiFunc(vector_free_coef_short, nfp).dchi().dchi()
    ).cap_m(n_eval-2)

    print('looped, right before the error, type of len_tensor, when its first defined:',type(len_tensor))
    coef_Y_n0_dphi_2 = (
        coef_dphi_dphi_Y * ChiPhiFunc(vector_free_coef_short, nfp)
        +coef_dphi_dphi_dchi_Y * ChiPhiFunc(vector_free_coef_short, nfp).dchi()
    ).cap_m(n_eval-2)

    print('len_tensor',len_tensor)
    tensor_fft_op_Y_n0_dphi_0 = to_tensor_fft_op(len_tensor=len_tensor,ChiPhiFunc_in=coef_Y_n0_dphi_0)
    tensor_fft_op_Y_n0_dphi_1 = to_tensor_fft_op(len_tensor=len_tensor,ChiPhiFunc_in=coef_Y_n0_dphi_1, dphi=1)
    tensor_fft_op_Y_n0_dphi_2 = to_tensor_fft_op(len_tensor=len_tensor,ChiPhiFunc_in=coef_Y_n0_dphi_2, dphi=2)
    full_tensor_fft_op_Y_n0 = (
        tensor_fft_op_Y_n0_dphi_0
        +tensor_fft_op_Y_n0_dphi_1
        +tensor_fft_op_Y_n0_dphi_2
    )
    full_tensor_fft_op = full_tensor_fft_op_Y_n0

    ''' B_theta dependence '''
    # A (n, 1, 2, 2) tensor, acting on
    #     [[B_psi0']]
    # B_psi are only solved at even orders (odd n_eval).
    # This is calculated at all orders, because
    # B_psi[n,m>0] contains B_theta[n]. This is needed at
    # even n or n>2. Since n starts at 2, this means all n.
    if n_unknown>2:
        ''' B_theta in B_psi terms of Y'''
        # coef_B_psi_dphi_1_in_Y_RHS (4, 1000) has fixed shape.
        # When we only care about B_psi0, n_unknown is always even.
        # since B_psi0 is a scalar, there was no need for a convolution matrix,
        # and we apply o_einv by .pad_chi(n_eval+1) and then einsum.

        # However, when we are tracking B_theta dependence in B_psi, these
        # coefs are convolution matrices acting on B_theta. Because
        # Y var = O_einv @ Y_RHS
        # Y var = O_einv @ (
        #     conv_matrix(coef_B_psi_dphi_0_dchi_1_in_Y_RHS) @ dchi @ B_psi
        #     + ...
        # )
        # The procedure of converting coef_B_psi_dphi_0_dchi_1_in_Y_RHS
        # to an operator representing
        # O_einv @ conv_matrix(coef_B_psi_dphi_0_dchi_1_in_Y_RHS) @ dchi
        # should then be identical to the direct B_theta coefficients, except we
        # add a matrix multiplication, O_einv @ ..., at the front.
        # Now, the presence of dphi Y and dchi Y complicates the matter:
        # dchi Y var = dchi @ O_einv @ (
        #     conv_matrix(coef_B_psi_dphi_0_dchi_1_in_Y_RHS) @ dchi @ B_psi
        #     + ...
        # )
        # dphi Y var = dphi @ O_einv @ (
        #     conv_matrix(coef_B_psi_dphi_0_dchi_1_in_Y_RHS) @ dchi @ B_psi
        #     + ...
        # )
        # It's easier to think of these terms directly as linear operators
        # acting on B_psi, rather than a linear operator representing coefficients
        # acting on B_psi and its derivatives through chain rules.
        #                     (<tilde II,  B_theta unknown, ...         )
        # These operators are
        # [n_unknown, n_unknown-1, len_tensor, len_tensor]
        # arrays.
        tensor_fft_op_B_psi_dphi_0_dchi_1_in_Y_RHS = \
            to_tensor_fft_op_multi_dim(len_tensor=len_tensor,
                num_mode=n_unknown-1, cap_axis0=n_unknown,
                nfp=nfp,
                ChiPhiFunc_in=looped_coefs.coef_B_psi_dphi_0_dchi_1_in_Y_RHS(
                    X_coef_cp=X_coef_cp,
                    Y_coef_cp=Y_coef_cp,
                    Delta_coef_cp=Delta_coef_cp,
                    B_alpha_coef=B_alpha_coef,
                    B_denom_coef_c=B_denom_coef_c,
                    dl_p=dl_p,
                    tau_p=tau_p,
                    kap_p=kap_p,
                    iota_coef=iota_coef,
                    n_eval=n_eval),
                dphi=0, dchi=1
            )
        tensor_fft_op_B_psi_dphi_0_dchi_2_in_Y_RHS = \
            to_tensor_fft_op_multi_dim(len_tensor=len_tensor,
                num_mode=n_unknown-1, cap_axis0=n_unknown,
                nfp=nfp,
                ChiPhiFunc_in=looped_coefs.coef_B_psi_dphi_0_dchi_2_in_Y_RHS(
                    X_coef_cp=X_coef_cp,
                    Y_coef_cp=Y_coef_cp,
                    Delta_coef_cp=Delta_coef_cp,
                    B_alpha_coef=B_alpha_coef,
                    B_denom_coef_c=B_denom_coef_c,
                    dl_p=dl_p,
                    tau_p=tau_p,
                    kap_p=kap_p,
                    iota_coef=iota_coef,
                    n_eval=n_eval),
                dphi=0, dchi=2
            )
        tensor_fft_op_B_psi_dphi_1_dchi_0_in_Y_RHS = \
            to_tensor_fft_op_multi_dim(len_tensor=len_tensor,
                num_mode=n_unknown-1, cap_axis0=n_unknown,
                nfp=nfp,
                ChiPhiFunc_in=looped_coefs.coef_B_psi_dphi_1_dchi_0_in_Y_RHS(
                    X_coef_cp=X_coef_cp,
                    Y_coef_cp=Y_coef_cp,
                    Delta_coef_cp=Delta_coef_cp,
                    B_alpha_coef=B_alpha_coef,
                    B_denom_coef_c=B_denom_coef_c,
                    dl_p=dl_p,
                    tau_p=tau_p,
                    kap_p=kap_p,
                    iota_coef=iota_coef,
                    n_eval=n_eval),
                dphi=1, dchi=0
            )
        tensor_fft_op_B_psi_dphi_1_dchi_1_in_Y_RHS = \
            to_tensor_fft_op_multi_dim(len_tensor=len_tensor,
                num_mode=n_unknown-1, cap_axis0=n_unknown,
                nfp=nfp,
                ChiPhiFunc_in=looped_coefs.coef_B_psi_dphi_1_dchi_1_in_Y_RHS(
                    X_coef_cp=X_coef_cp,
                    Y_coef_cp=Y_coef_cp,
                    Delta_coef_cp=Delta_coef_cp,
                    B_alpha_coef=B_alpha_coef,
                    B_denom_coef_c=B_denom_coef_c,
                    dl_p=dl_p,
                    tau_p=tau_p,
                    kap_p=kap_p,
                    iota_coef=iota_coef,
                    n_eval=n_eval),
                dphi=1, dchi=1
            )
        # This is the differential operator acting on B_psi in Y_RHS
        # At n_unknown=2, has an extra m=0 component whereas the rest=
        # to_tensor_fft_op_multi_dim(len_tensor=len_tensor,
        #     coef_B_psi_dphi_1_in_Y_RHS,
        #     dphi=1, dchi=0
        # )
        tensor_fft_op_B_psi_in_Y_RHS = (
            tensor_fft_op_B_psi_dphi_0_dchi_1_in_Y_RHS
            +tensor_fft_op_B_psi_dphi_0_dchi_2_in_Y_RHS
            +tensor_fft_op_B_psi_dphi_1_dchi_0_in_Y_RHS
            +tensor_fft_op_B_psi_dphi_1_dchi_1_in_Y_RHS
        )
        # Recall that the axis=0 shape of the tensor fft operator is determined
        # by the number of components of B_theta[n_unknown] and cannot be changed.
        # O_einv always have shape (n_unknown+1, n_unknown+2, len_tensor).
        # To apply O_einv, we remove the first and last elements from axis=1,
        # and then apply einsum to perform matrix multiplication for the first
        # 2 axes and point-wise multiplication of O_einv's axis=2 with tensor_fft_op's
        # axis=2. O_einv_clipped has shape (n_unknown+1, n_unknown, len_tensor)
        O_einv_clipped = O_einv[:, 1:-1, :]
        O_einv_clipped_fft = jnp.fft.fft(O_einv_clipped, axis = 2)
        # 'Tensor coefficients', dimension is (n_eval-1, n_eval-3, len_tensor)
        # Last 2 dimensions are for convolving phi cells.
        O_einv_clipped_fft_op = fft_conv_tensor_batch(O_einv_clipped_fft)

        ''' This is used later for the odd order equation D3, too. '''
        tensor_fft_op_B_psi_in_Y = einsum_ijkl_jmln_to_imkn(
            O_einv_clipped_fft_op,
            tensor_fft_op_B_psi_in_Y_RHS
        )
        out_dict_tensor['tensor_fft_op_B_psi_in_Y'] = tensor_fft_op_B_psi_in_Y
        # The resulting tensors should have shape
        # (n_unknown+1, n_unknown-1, len_tensor, len_tensor)
        # These are obviously, too many components. This
        # will be addressed by capping the length of
        # axis=0 in Y coefficients.
        tensor_fft_op_Y = to_tensor_fft_op_multi_dim(len_tensor=len_tensor,
            nfp=nfp,
            ChiPhiFunc_in=coef_Y,
            dphi=0, dchi=0,
            num_mode=n_unknown+1, cap_axis0=n_unknown
        )
        tensor_fft_op_Y += to_tensor_fft_op_multi_dim(len_tensor=len_tensor,
            nfp=nfp,
            ChiPhiFunc_in=coef_dchi_Y,
            dphi=0, dchi=1,
            num_mode=n_unknown+1, cap_axis0=n_unknown
        )
        tensor_fft_op_Y += to_tensor_fft_op_multi_dim(len_tensor=len_tensor,
            nfp=nfp,
            ChiPhiFunc_in=coef_dchi_dchi_Y,
            dphi=0, dchi=2,
            num_mode=n_unknown+1, cap_axis0=n_unknown
        )
        tensor_fft_op_Y += to_tensor_fft_op_multi_dim(len_tensor=len_tensor,
            nfp=nfp,
            ChiPhiFunc_in=coef_dchi_dchi_dchi_Y,
            dphi=0, dchi=3,
            num_mode=n_unknown+1, cap_axis0=n_unknown
        )
        tensor_fft_op_Y += to_tensor_fft_op_multi_dim(len_tensor=len_tensor,
            nfp=nfp,
            ChiPhiFunc_in=coef_dphi_Y,
            dphi=1, dchi=0,
            num_mode=n_unknown+1, cap_axis0=n_unknown
        )
        tensor_fft_op_Y += to_tensor_fft_op_multi_dim(len_tensor=len_tensor,
            nfp=nfp,
            ChiPhiFunc_in=coef_dchi_dphi_Y,
            dphi=1, dchi=1,
            num_mode=n_unknown+1, cap_axis0=n_unknown
        )
        tensor_fft_op_Y += to_tensor_fft_op_multi_dim(len_tensor=len_tensor,
            nfp=nfp,
            ChiPhiFunc_in=coef_dchi_dchi_dphi_Y,
            dphi=1, dchi=2,
            num_mode=n_unknown+1, cap_axis0=n_unknown
        )
        tensor_fft_op_Y += to_tensor_fft_op_multi_dim(len_tensor=len_tensor,
            nfp=nfp,
            ChiPhiFunc_in=coef_dphi_dphi_Y,
            dphi=2, dchi=0,
            num_mode=n_unknown+1, cap_axis0=n_unknown
        )
        tensor_fft_op_Y += to_tensor_fft_op_multi_dim(len_tensor=len_tensor,
            nfp=nfp,
            ChiPhiFunc_in=coef_dphi_dphi_dchi_Y,
            dphi=2, dchi=1,
            num_mode=n_unknown+1, cap_axis0=n_unknown
        )
        # Luckily these all have the same dimensions, and we can add them together
        # to form a large linear operator representing the linear diffeential
        # operator acting on Y. This large operator has shape
        # (n_unknown, n_unknown+1, 1000, 1000). Convenient!
        # This is an operator acting on Y.
        # dchi B_psi contains n_unknown/2 B_theta
        # B_psi's outer components contain n_unknown/2 ichi B_theta.
        # B_theta always have n_unknown-1 components.
        dchi_temp_B_theta = (1j*jnp.arange(-n_unknown+2,n_unknown-1,2)[None, :, None, None])
        if n_unknown%2==0:
            dchi_temp_B_theta = dchi_temp_B_theta.at[:,(n_unknown-1)//2,:,:].set(jnp.inf)

        # overall B_psi coefficients carried by Y.
        # should be of shape (n_unknown, n_unknown-1, len_tensor, len_tensor)
        ''' B_theta dependence of Y. '''
        tensor_fft_op_B_theta_through_Y = einsum_ijkl_jmln_to_imkn(
            tensor_fft_op_Y,
            tensor_fft_op_B_psi_in_Y
        )*n_unknown/2/dchi_temp_B_theta
        # This shape should be robust. No cap_m needed.
        # The center column of this operator should be identical to the
        # B_psi0 dependence calculated by hand-applying chain rules. (see below block)
        # This is validated against the correct B_psi0 coefficient calculation
        # by comparing the 2 following quantities at n_unknown=2:
        # tensor_fft_op_B_psi_through_Y = jnp.einsum(
        #     'ijlm,jkmn->ikln',
        #     tensor_fft_op_Y,
        #     tensor_fft_op_B_psi_in_Y
        # )
        # B_psi0_carried_by_Y_ans = (
        #     coef_B_psi_dphi_1_in_Y*B_psi_coef_cp[0].dphi()
        #     +coef_B_psi_dphi_2_in_Y*B_psi_coef_cp[0].dphi().dphi()
        #     +coef_B_psi_dphi_3_in_Y*B_psi_coef_cp[0].dphi().dphi().dphi()
        # )
        # B_psi0_carried_by_Y_test = ChiPhiFunc(
        #     jnp.einsum('ijkl,jl->ik',tensor_fft_op_B_psi_through_Y,fft_test_B_psi0)
        # ).ifft()
        # However, it is impossible to produce the tensor operator on
        # B_psi' from the tensor operator on B, because the operator
        # dphi is non-invertible.
        # op_B @ B = op_B' @ dphi @ B
        # op_B = op_B' @ dphi (cancel B)
        '''Direct B_theta tensors'''
        looped_B_theta_coefs = looped_coefs.eval_B_theta_coefs(
            p_perp_coef_cp=p_perp_coef_cp,
            Delta_coef_cp=Delta_coef_cp,
            B_denom_coef_c=B_denom_coef_c,
            iota_coef=iota_coef)
        coef_B_theta = looped_B_theta_coefs['coef_B_theta']
        coef_dchi_B_theta = looped_B_theta_coefs['coef_dchi_B_theta']
        coef_dphi_B_theta = looped_B_theta_coefs['coef_dphi_B_theta']

        ''' Converting to operators '''
        # Validated to the original implementation.
        tensor_fft_op_B_theta = to_tensor_fft_op_multi_dim(len_tensor=len_tensor,
            num_mode=n_unknown-1, cap_axis0=n_unknown,
            nfp=nfp,
            ChiPhiFunc_in=coef_B_theta, dphi=0, dchi=0)
        tensor_fft_op_dchi_B_theta = to_tensor_fft_op_multi_dim(len_tensor=len_tensor,
            num_mode=n_unknown-1, cap_axis0=n_unknown,
            nfp=nfp,
            ChiPhiFunc_in=coef_dchi_B_theta, dphi=0, dchi=1)
        tensor_fft_op_dphi_B_theta = to_tensor_fft_op_multi_dim(len_tensor=len_tensor,
            num_mode=n_unknown-1, cap_axis0=n_unknown,
            nfp=nfp,
            ChiPhiFunc_in=coef_dphi_B_theta, dphi=1, dchi=0)

        ''' B_theta dependence carried by B_psi in terms other than Y '''
        tensor_fft_op_B_theta_in_all_but_Y = \
            looped_coefs.tensor_fft_op_B_psi_in_all_but_Y(
                n_eval=n_eval,
                len_tensor=len_tensor,
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_alpha_coef=B_alpha_coef,
                B_denom_coef_c=B_denom_coef_c,
                dl_p=dl_p,
                tau_p=tau_p,
                kap_p=kap_p,
                iota_coef=iota_coef,
                num_mode=n_unknown-1, cap_axis0=n_unknown,
                nfp=nfp
            )*n_unknown/2/dchi_temp_B_theta


        # The shape of these tensors should be (n_unknown, n_unknown-1, len_chi, len_chi)
        # for n_unknown >2, B_theta would be an unknown.

        full_tensor_fft_op_B_theta = (
            tensor_fft_op_B_theta # Direct coefficients
            +tensor_fft_op_dchi_B_theta
            +tensor_fft_op_dphi_B_theta
            +tensor_fft_op_B_theta_through_Y
            +tensor_fft_op_B_theta_in_all_but_Y
        )

        # Removing the center "column" when n_unknown is even.
        if n_unknown%2==0:
            full_tensor_fft_op_B_theta = jnp.delete(
                full_tensor_fft_op_B_theta,
                full_tensor_fft_op_B_theta.shape[1]//2, 1
            )

        # Should be (n_unknown, n_unknown, len_tensor, len_tensor) at odd orders,
        # and (n_unknown, n_unknown-1, len_tensor, len_tensor) at even orders,
        full_tensor_fft_op = jnp.concatenate((full_tensor_fft_op_B_theta, full_tensor_fft_op), axis=1)

    if n_unknown%2==0:
        ''' B_psi m=0 coefs in Y '''
        # This padding ...?
        coef_B_psi_dphi_1_in_Y_var = ChiPhiFunc(jnp.einsum(
            'ijk,jk->ik',
            O_einv,
            looped_coefs.coef_B_psi_dphi_1_dchi_0_in_Y_RHS(
                    X_coef_cp=X_coef_cp,
                    Y_coef_cp=Y_coef_cp,
                    Delta_coef_cp=Delta_coef_cp,
                    B_alpha_coef=B_alpha_coef,
                    B_denom_coef_c=B_denom_coef_c,
                    dl_p=dl_p,
                    tau_p=tau_p,
                    kap_p=kap_p,
                    iota_coef=iota_coef,
                    n_eval=n_eval).pad_chi(n_eval+1).content
        ), nfp)
        out_dict_tensor['coef_B_psi_dphi_1_in_Y_var'] = coef_B_psi_dphi_1_in_Y_var
        coef_B_psi_dphi_1_in_Y = (
            coef_Y*coef_B_psi_dphi_1_in_Y_var
            + coef_dchi_Y*coef_B_psi_dphi_1_in_Y_var.dchi()
            + coef_dchi_dchi_Y*coef_B_psi_dphi_1_in_Y_var.dchi().dchi()
            + coef_dchi_dchi_dchi_Y*coef_B_psi_dphi_1_in_Y_var.dchi().dchi().dchi()
            + coef_dphi_Y*coef_B_psi_dphi_1_in_Y_var.dphi()
            + coef_dchi_dphi_Y*coef_B_psi_dphi_1_in_Y_var.dphi().dchi()
            + coef_dchi_dchi_dphi_Y*coef_B_psi_dphi_1_in_Y_var.dphi().dchi().dchi()
            + coef_dphi_dphi_Y*coef_B_psi_dphi_1_in_Y_var.dphi().dphi()
            + coef_dphi_dphi_dchi_Y*coef_B_psi_dphi_1_in_Y_var.dphi().dphi().dchi()
        ).cap_m(n_eval-2)
        coef_B_psi_dphi_2_in_Y = (
            coef_dphi_Y*coef_B_psi_dphi_1_in_Y_var
            + coef_dchi_dphi_Y*coef_B_psi_dphi_1_in_Y_var.dchi()
            + coef_dchi_dchi_dphi_Y*coef_B_psi_dphi_1_in_Y_var.dchi().dchi()
            + 2*coef_dphi_dphi_Y*coef_B_psi_dphi_1_in_Y_var.dphi()
            + 2*coef_dphi_dphi_dchi_Y*coef_B_psi_dphi_1_in_Y_var.dphi().dchi()
        ).cap_m(n_eval-2)
        coef_B_psi_dphi_3_in_Y = (
            coef_dphi_dphi_Y*coef_B_psi_dphi_1_in_Y_var
            + coef_dphi_dphi_dchi_Y*coef_B_psi_dphi_1_in_Y_var.dchi()
        ).cap_m(n_eval-2)

        ''' Merging B_psi coeffs '''
        coef_B_psi_dphi_1_dchi_0_all_but_Y = looped_coefs.coef_B_psi_dphi_1_dchi_0_all_but_Y(
            X_coef_cp=X_coef_cp,
            Y_coef_cp=Y_coef_cp,
            Delta_coef_cp=Delta_coef_cp,
            B_alpha_coef=B_alpha_coef,
            B_denom_coef_c=B_denom_coef_c,
            dl_p=dl_p,
            tau_p=tau_p,
            kap_p=kap_p,
            iota_coef=iota_coef,
            n_eval=n_eval)
        coef_B_psi_dphi_2_dchi_0_all_but_Y = looped_coefs.coef_B_psi_dphi_2_dchi_0_all_but_Y(
            X_coef_cp=X_coef_cp,
            Y_coef_cp=Y_coef_cp,
            Delta_coef_cp=Delta_coef_cp,
            B_alpha_coef=B_alpha_coef,
            B_denom_coef_c=B_denom_coef_c,
            dl_p=dl_p,
            tau_p=tau_p,
            kap_p=kap_p,
            iota_coef=iota_coef,
            n_eval=n_eval)
        coef_B_psi_dphi_3_dchi_0_all_but_Y = looped_coefs.coef_B_psi_dphi_3_dchi_0_all_but_Y(
            X_coef_cp=X_coef_cp,
            Y_coef_cp=Y_coef_cp,
            Delta_coef_cp=Delta_coef_cp,
            B_alpha_coef=B_alpha_coef,
            B_denom_coef_c=B_denom_coef_c,
            dl_p=dl_p,
            tau_p=tau_p,
            kap_p=kap_p,
            iota_coef=iota_coef,
            n_eval=n_eval)
        coef_B_psi_dphi_1 = (
            coef_B_psi_dphi_1_in_Y
            +coef_B_psi_dphi_1_dchi_0_all_but_Y
        ).cap_m(n_eval-2)
        coef_B_psi_dphi_2 = (
            coef_B_psi_dphi_2_in_Y
            +coef_B_psi_dphi_2_dchi_0_all_but_Y
        ).cap_m(n_eval-2)
        coef_B_psi_dphi_3 = (
            coef_B_psi_dphi_3_in_Y
            +coef_B_psi_dphi_3_dchi_0_all_but_Y
        ).cap_m(n_eval-2)
        tensor_fft_op_B_psi_dphi_1 = to_tensor_fft_op(
            len_tensor=len_tensor,
            ChiPhiFunc_in=coef_B_psi_dphi_1
        )
        tensor_fft_op_B_psi_dphi_2 = to_tensor_fft_op(
            len_tensor=len_tensor,
            ChiPhiFunc_in=coef_B_psi_dphi_2,
            dphi=1
        )
        tensor_fft_op_B_psi_dphi_3 = to_tensor_fft_op(
            len_tensor=len_tensor,
            ChiPhiFunc_in=coef_B_psi_dphi_3,
            dphi=2
        )
        full_tensor_fft_op_dphi_B_psi = (
            tensor_fft_op_B_psi_dphi_1
            +tensor_fft_op_B_psi_dphi_2
            +tensor_fft_op_B_psi_dphi_3
        )
        full_tensor_fft_op = jnp.concatenate((full_tensor_fft_op, full_tensor_fft_op_dphi_B_psi), axis=1)
    else:
        # At odd orders, we need to replace the center component.
        # Very luckily, this component of II has no B_psi dependence
        # at all (and no X, Y, Z, p, Delta), meaning we'll have no
        # lengthy tracking of B_theta components.
        ''' The center component of the looped equation (II) '''
        # The center comp of the looped equation is identical to the center
        # component of II, because the quantity substituted into II is
        # dchi p and does not have the m=0 component.
        # Inspection on II shows that it is dependent on:
        # B_theta n+1,0
        # B_theta n
        # but NOT Y n.
        # Now full_tensor_fft_op has dimension:
        # (n_unknown       , n_unknown, [phi matrix])
        # (looped RHS comps, unknowns , [phi matrix])
        # First clear all coeffs for the center elem
        full_tensor_fft_op = full_tensor_fft_op.at[n_unknown//2].set(0)
        ''' B_theta dependebce carried by B_psi terms in Delta n and p n '''
        coef_Delta_n = -(B_alpha_coef[0]*(diff(B_denom_coef_c[1],True,1)))/2
        coef_dchi_p_n = 2*B_alpha_coef[0]*B_denom_coef_c[0]*B_denom_coef_c[1]
        # dchi B_psi contains n_unknown/2 B_theta
        # Because this portion of the code is only for odd orders,
        # B_psi contains ichi n_unknown/2 B_theta.
        # These are the coefficients for B_psi in II
        # through Delta[n_unknown]
        coef_B_psi_in_II_through_Delta = -coef_Delta_n*B_denom_coef_c[0]*looped_coefs.lambda_B_psi_nm2_in_p_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)
        coef_dchi_B_psi_in_II_through_Delta = -coef_Delta_n*B_denom_coef_c[0]*looped_coefs.lambda_dchi_B_psi_nm2_in_p_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)
        coef_dphi_B_psi_in_II_through_Delta = -coef_Delta_n*B_denom_coef_c[0]*looped_coefs.lambda_dphi_B_psi_nm2_in_p_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)
        # p is present in II as a dchi p term.
        coef_B_psi_in_II_in_p = looped_coefs.lambda_B_psi_nm2_in_p_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)
        coef_dchi_B_psi_in_II_in_p = looped_coefs.lambda_dchi_B_psi_nm2_in_p_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)
        coef_dphi_B_psi_in_II_in_p = looped_coefs.lambda_dphi_B_psi_nm2_in_p_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)
        # Chain rules
        coef_B_psi_in_II_through_dchi_p = coef_dchi_p_n*(coef_B_psi_in_II_in_p.dchi())
        coef_dchi_B_psi_in_II_through_dchi_p = coef_dchi_p_n*(coef_B_psi_in_II_in_p+coef_dchi_B_psi_in_II_in_p.dchi())
        coef_dchi_dchi_B_psi_in_II_through_dchi_p = coef_dchi_p_n*(coef_dchi_B_psi_in_II_in_p)
        coef_dphi_B_psi_in_II_through_dchi_p = coef_dchi_p_n*(coef_dphi_B_psi_in_II_in_p.dchi())
        coef_dphi_dchi_B_psi_in_II_through_dchi_p = coef_dchi_p_n*(coef_dphi_B_psi_in_II_in_p)
        # Constructing operators for Delta
        tensor_fft_op_B_theta_n_in_II_through_Delta = to_tensor_fft_op_multi_dim(
            len_tensor=len_tensor,
            nfp=nfp,
            ChiPhiFunc_in=coef_B_psi_in_II_through_Delta*n_unknown/2,
            dphi=0, dchi=-1,
            num_mode=n_unknown-1, cap_axis0=1
        )
        tensor_fft_op_B_theta_n_in_II_through_Delta += to_tensor_fft_op_multi_dim(
            len_tensor=len_tensor,
            nfp=nfp,
            ChiPhiFunc_in=coef_dchi_B_psi_in_II_through_Delta*n_unknown/2,
            dphi=0, dchi=0,
            num_mode=n_unknown-1, cap_axis0=1
        )
        tensor_fft_op_B_theta_n_in_II_through_Delta += to_tensor_fft_op_multi_dim(
            len_tensor=len_tensor,
            nfp=nfp,
            ChiPhiFunc_in=coef_dphi_B_psi_in_II_through_Delta*n_unknown/2,
            dphi=1, dchi=-1,
            num_mode=n_unknown-1, cap_axis0=1
        )
        # Constructing operators for p
        tensor_fft_op_B_theta_n_in_II_through_dchi_p = to_tensor_fft_op_multi_dim(
            len_tensor=len_tensor,
            nfp=nfp,
            ChiPhiFunc_in=coef_B_psi_in_II_through_dchi_p*n_unknown/2,
            dphi=0, dchi=-1,
            num_mode=n_unknown-1, cap_axis0=1
        )
        tensor_fft_op_B_theta_n_in_II_through_dchi_p += to_tensor_fft_op_multi_dim(
            len_tensor=len_tensor,
            nfp=nfp,
            ChiPhiFunc_in=coef_dchi_B_psi_in_II_through_dchi_p*n_unknown/2,
            dphi=0, dchi=0,
            num_mode=n_unknown-1, cap_axis0=1
        )
        tensor_fft_op_B_theta_n_in_II_through_dchi_p += to_tensor_fft_op_multi_dim(
            len_tensor=len_tensor,
            nfp=nfp,
            ChiPhiFunc_in=coef_dchi_dchi_B_psi_in_II_through_dchi_p*n_unknown/2,
            dphi=0, dchi=1,
            num_mode=n_unknown-1, cap_axis0=1
        )
        tensor_fft_op_B_theta_n_in_II_through_dchi_p += to_tensor_fft_op_multi_dim(
            len_tensor=len_tensor,
            nfp=nfp,
            ChiPhiFunc_in=coef_dphi_B_psi_in_II_through_dchi_p*n_unknown/2,
            dphi=1, dchi=-1,
            num_mode=n_unknown-1, cap_axis0=1
        )
        tensor_fft_op_B_theta_n_in_II_through_dchi_p += to_tensor_fft_op_multi_dim(
            len_tensor=len_tensor,
            nfp=nfp,
            ChiPhiFunc_in=coef_dphi_dchi_B_psi_in_II_through_dchi_p*n_unknown/2,
            dphi=1, dchi=0,
            num_mode=n_unknown-1, cap_axis0=1
        )
        ''' Direct B_theta terms in II '''
        coef_B_theta_0_chiphifunc = -(
            2*B_denom_coef_c[0]**2*p_perp_coef_cp[1].dphi()
            +2*B_denom_coef_c[0]**2*iota_coef[0]*p_perp_coef_cp[1].dchi()
            -Delta_coef_cp[0]*iota_coef[0]*B_denom_coef_c[1].dchi()
            +4*B_denom_coef_c[0]*B_denom_coef_c[1]*p_perp_coef_cp[0].dphi()
        )/2
        coef_dc_B_theta_0_chiphifunc = (
            B_denom_coef_c[0]*iota_coef[0]*Delta_coef_cp[1]
            +(Delta_coef_cp[0]-1)*iota_coef[0]*B_denom_coef_c[1])
        coef_dp_B_theta_0_chiphifunc = (
            B_denom_coef_c[0]*Delta_coef_cp[1]
            +(Delta_coef_cp[0]-1)*B_denom_coef_c[1])

        tensor_fft_op_B_theta_n_in_II_direct = to_tensor_fft_op_multi_dim(len_tensor=len_tensor,
            nfp=nfp,
            ChiPhiFunc_in=coef_B_theta_0_chiphifunc,
            dphi=0, dchi=0,
            num_mode=n_unknown-1, cap_axis0=1
        )
        tensor_fft_op_B_theta_n_in_II_direct += to_tensor_fft_op_multi_dim(len_tensor=len_tensor,
            nfp=nfp,
            ChiPhiFunc_in=coef_dc_B_theta_0_chiphifunc,
            dphi=0, dchi=1,
            num_mode=n_unknown-1, cap_axis0=1
        )
        tensor_fft_op_B_theta_n_in_II_direct += to_tensor_fft_op_multi_dim(len_tensor=len_tensor,
            nfp=nfp,
            ChiPhiFunc_in=coef_dp_B_theta_0_chiphifunc,
            dphi=1, dchi=0,
            num_mode=n_unknown-1, cap_axis0=1
        )
        tensor_fft_op_B_theta_n_in_II=(
            tensor_fft_op_B_theta_n_in_II_direct
            +tensor_fft_op_B_theta_n_in_II_through_Delta
            +tensor_fft_op_B_theta_n_in_II_through_dchi_p
        )
        ''' Merging '''
        # Editing the center row of the existing tensor.
        # At the moment, the tensor does not include the B_theta n+1
        # column yet.
        full_tensor_fft_op = \
            full_tensor_fft_op.at[n_unknown//2, :n_unknown-1].set(
                tensor_fft_op_B_theta_n_in_II[0]
            )

        ''' B_theta [n+1,0] in II '''
        coef_B_theta_np10 = -B_denom_coef_c[0]**2*p_perp_coef_cp[0].dphi()
        coef_dp_B_theta_np10 = B_denom_coef_c[0]*(Delta_coef_cp[0]-1)

        tensor_fft_op_B_theta_np1_in_II = to_tensor_fft_op_multi_dim(
            len_tensor=len_tensor,
            nfp=nfp,
            ChiPhiFunc_in=coef_B_theta_np10,
            dphi=0, dchi=0,
            num_mode=1, cap_axis0=1
        )
        tensor_fft_op_B_theta_np1_in_II += to_tensor_fft_op_multi_dim(
            len_tensor=len_tensor,
            nfp=nfp,
            ChiPhiFunc_in=coef_dp_B_theta_np10,
            dphi=1, dchi=0,
            num_mode=1, cap_axis0=1
        )
        # Col for B_theta n+1
        new_column = jnp.zeros(
            (
                full_tensor_fft_op.shape[0], # n components: Looped equation
                1,                           # One unknown: B_theta[n+1]
                full_tensor_fft_op.shape[2], # phi operator
                full_tensor_fft_op.shape[2]),
            jnp.complex128)
        new_column = new_column.at[n_unknown//2].set(tensor_fft_op_B_theta_np1_in_II[0])
        full_tensor_fft_op = jnp.concatenate((full_tensor_fft_op, new_column), axis=1)

        # Sadly, D3 includes Zn and Xn, which implicitly carries B_psi
        ''' Adding D3 '''
        # Now the tensor is [n_unknown, n_unknown+1, len_tensor, len_tensor]:
        # [
        #     [B_theta[n+1,0], Y[n,1]]
        #     [B_theta[n+1,0], Y[n,1]]
        #     ... (n_unknown components for II)
        # ]
        #
        # The new tensor slice need to have shape (1, n_unknown+1, len_tensor, len_tensor)
        # D3 slice:
        # [0, ..., Bn1-,Bn1+, ..., 0, Y, Bn+1]
        ''' D3: Y[n,+1] '''

        coef_Yn1p_in_D3 = looped_coefs.lambda_coef_Yn1p_in_D3(
            vector_free_coef,
            X_coef_cp, Y_coef_cp,
            iota_coef, dl_p, tau_p, nfp
        )
        coef_dp_Yn1p_in_D3 = looped_coefs.lambda_coef_dp_Yn1p_in_D3(
            vector_free_coef,
            X_coef_cp, Y_coef_cp,
            iota_coef, dl_p, tau_p, nfp
        )
        elem_fft_op_Yn1p = to_tensor_fft_op(
            len_tensor=len_tensor,
            ChiPhiFunc_in=coef_Yn1p_in_D3
        )
        elem_fft_op_dp_Yn1p = to_tensor_fft_op(
            len_tensor=len_tensor,
            ChiPhiFunc_in=coef_dp_Yn1p_in_D3,
            dphi=1
        )
        ''' B_theta[n] and B_theta[n+1,0] coefficient in D3 '''
        ones = ChiPhiFunc(jnp.ones((1, len_tensor),jnp.complex128), nfp)
        coef_B_theta_n_1p = phi_avg((-B_alpha_coef[0]*B_denom_coef_c[1][-1]))*ones
        coef_B_theta_n_1n = phi_avg((-B_alpha_coef[0]*B_denom_coef_c[1][1]))*ones
        coef_B_theta_np1_0 = phi_avg(-B_alpha_coef[0]*B_denom_coef_c[0])*ones
        # 'Tensor element operators', dimension is (1, 1, len_tensor, len_tensor)
        # Last 2 dimensions are for convolving phi cells.
        elem_fft_op_B_theta_n_1p = to_tensor_fft_op(
            len_tensor=len_tensor,
            ChiPhiFunc_in=coef_B_theta_n_1p
        )
        elem_fft_op_B_theta_n_1n = to_tensor_fft_op(
            len_tensor=len_tensor,
            ChiPhiFunc_in=coef_B_theta_n_1n
        )
        elem_fft_op_B_theta_np1_0 = to_tensor_fft_op(
            len_tensor=len_tensor,
            ChiPhiFunc_in=coef_B_theta_np1_0
        )
        # Merge, remove redundant dims, and constructing row to append
        D3_comp_fft_op = jnp.zeros((1, n_unknown+1, len_tensor, len_tensor), dtype=jnp.complex128)
        D3_comp_fft_op = D3_comp_fft_op.at[0, n_unknown//2-1].set(elem_fft_op_B_theta_n_1n[0][0])
        D3_comp_fft_op = D3_comp_fft_op.at[0, n_unknown//2].set(elem_fft_op_B_theta_n_1p[0][0])
        D3_comp_fft_op = D3_comp_fft_op.at[0, -2].set((elem_fft_op_Yn1p + elem_fft_op_dp_Yn1p)[0][0])
        D3_comp_fft_op = D3_comp_fft_op.at[0, -1].set(elem_fft_op_B_theta_np1_0[0][0])

        tensor_fft_op_D3_B_psi_in_all_but_Y = looped_coefs.eval_B_psi_coefs_D3(
            n_eval=n_eval,
            len_tensor=len_tensor,
            X_coef_cp=X_coef_cp,
            Y_coef_cp=Y_coef_cp,
            Delta_coef_cp=Delta_coef_cp,
            B_alpha_coef=B_alpha_coef,
            B_denom_coef_c=B_denom_coef_c,
            dl_p=dl_p,
            tau_p=tau_p,
            kap_p=kap_p,
            iota_coef=iota_coef,
            num_mode=n_unknown-1,
            cap_axis0=n_unknown,
            nfp=nfp
        )*n_unknown/2/dchi_temp_B_theta

        # Full Yn coef for B_theta dependence. In theory
        D3_Yn_coef = diff(X_coef_cp[1],True,1)*dl_p*tau_p
        D3_dchi_Yn_coef = -X_coef_cp[1]*dl_p*tau_p+diff(Y_coef_cp[1],False,1)+2*iota_coef[0]*(diff(Y_coef_cp[1],True,1))
        D3_dphi_Yn_coef = diff(Y_coef_cp[1],True,1)
        # We now generate Y coefficient operators that works with
        # (n_unknown+1, n_unknown-1, len_tensor, len_tensor)
        # tensor_fft_op_B_psi_in_Y
        # Recall that the shape comes from the number of
        # components in B_theta[n_unknown], n_unknown-1.
        # cap_axis0 is set to 1 because we only care about
        # the center element of D3.
        tensor_fft_op_D3_Y = to_tensor_fft_op_multi_dim(
            len_tensor=len_tensor,
            nfp=nfp,
            ChiPhiFunc_in=D3_Yn_coef,
            dphi=0, dchi=0,
            num_mode=n_unknown+1, cap_axis0=1
        )
        tensor_fft_op_D3_Y += to_tensor_fft_op_multi_dim(
            len_tensor=len_tensor,
            nfp=nfp,
            ChiPhiFunc_in=D3_dchi_Yn_coef,
            dphi=0, dchi=1,
            num_mode=n_unknown+1, cap_axis0=1
        )
        tensor_fft_op_D3_Y += to_tensor_fft_op_multi_dim(
            len_tensor=len_tensor,
            nfp=nfp,
            ChiPhiFunc_in=D3_dphi_Yn_coef,
            dphi=1, dchi=0,
            num_mode=n_unknown+1, cap_axis0=1
        )
        tensor_fft_op_D3_B_theta_through_Y = einsum_ijkl_jmln_to_imkn(
            tensor_fft_op_D3_Y,
            tensor_fft_op_B_psi_in_Y
        )*n_unknown/2/dchi_temp_B_theta
        # This operator should have the shape:
        # (1, n_unknown-1, len_tensor, len_tensor)
        # D3 is only used at odd orders so no need to delete
        # the operator components acting on B_theta[n,m=0].
        D3_comp_fft_op = D3_comp_fft_op.at[:,:n_unknown-1,:,:].set(
            D3_comp_fft_op[:,:n_unknown-1,:,:]
            + tensor_fft_op_D3_B_theta_through_Y
            + tensor_fft_op_D3_B_psi_in_all_but_Y[n_unknown//2]
        )
        full_tensor_fft_op = jnp.concatenate((full_tensor_fft_op, D3_comp_fft_op), axis=0)

        # * Is this statement true?
        # B_theta[n+1, 0] does not actually have a unique solution, and its average
        # need to be given. To do this, we edit the [:, -1, :, 0] element of the
        # operator. This will add a constant offset of B_theta_np10_avg*target_length
        # to all elements in RHS.
        # full_tensor_fft_op = full_tensor_fft_op.at[:, -1, :, 0].set(full_tensor_fft_op[:, -1, :, 0]+1)
    transposed_tensor_fft_op = jnp.transpose(full_tensor_fft_op, (0,2,1,3))
    # Filter off-diagonal elements in the linear differential operator.
    transposed_tensor_fft_op = filter_low_pass_tensor(
        transposed_tensor_fft_op,
        cutoff_freq,
        True
    )
    transposed_tensor_fft_op = filter_off_diag_tensor(transposed_tensor_fft_op, max_k_diff_pre_inv)
    # Finding the inverse differential operator
    inv_transposed_tensor_fft_op = jnp.linalg.tensorinv(transposed_tensor_fft_op)
    inv_transposed_tensor_fft_op = filter_low_pass_tensor(
        inv_transposed_tensor_fft_op,
        cutoff_freq,
        False
    )
    inv_transposed_tensor_fft_op = filter_off_diag_tensor(inv_transposed_tensor_fft_op, max_k_diff_post_inv)
    # Filter off-diagonal elements in the inverted linear differential operator.
    # (n_unknown(+1), len_tensor, n_unknown(+1), len_tensor)
    # out_dict_tensor['transposed_tensor_fft_op']= transposed_tensor_fft_op
    out_dict_tensor['inv_transposed_tensor_fft_op'] = inv_transposed_tensor_fft_op
    return(out_dict_tensor)

''' III. Calculating Delta_offset and padding solution '''
# At even order, calculate B_psi[n-2,0], Y[n,0] and the average of Delta[n,0]
# (called 'Delta_offset' below)
# coef_iota_nm1b2_in_Delta: Coefficient of iota (n-1)/2 in Delta. Is a constant
# in the Equilibrium object. Needed for odd orders.
# @partial(jit, static_argnums=(0, 1, 2, ))
# TODO
# TODO
# TODO Calculate the iota dependence of the RHS and the unit iota contribution to the LHS
# TODO sigma 0 is a special constraint because it constrains the sum of certain
# TODO fourier mode coefficients, rather than the value of any single one?
# TODO However, the biggest trouble comes from B_theta[2, 0]. Because the solution
# TODO to it is non-unique, it's highly likely the LHS operator is nearly
# TODO singular, but only possible to invert due to rounding errors.
# TODO Now, how do we encode avg of B_theta or stellarator symmetry into the
# TODO differential operator??
# The cutoff freqs here MUST be positive and smaller than len_phi.
def solve_free_param(n_unknown, nfp,
    inv_transposed_tensor_fft_op, RHS_0_offset,
    len_phi,
    coef_Delta_offset = 0,
    # B_theta_np10_avg = 0 # Only used at odd orders, when B_theta[n+1,0] is a free param.
    ):
    out_dict_solve = {}
    print('inv_transposed_tensor_fft_op',inv_transposed_tensor_fft_op.shape)
    print('RHS_0_offset',RHS_0_offset.shape)
    # Solution with zero value for the free constant parameter
    solution = jnp.tensordot(
        inv_transposed_tensor_fft_op,
        RHS_0_offset, # + B_theta_np10_avg*RHS_0_offset.shape[1],
        2
    )
    # To have periodic B_psi, values for a constant free marameter, Delt_offset
    # must be found at even orders.
    # Integral(B_psi') = 0 is equivalent to solution[-1,0] = 0.
    # (-1: B_psi is always the second last row in the solution. 0: m=0 mode
    # in a jnp.fft.fft array.)
    # Because the looped ODE is now treated as a linear equation,
    # the Delta_offset dependence of solution[-1,0] is linear.

    # Making a blank ChiPhiFunc with the correct shape. The free parameter's
    # contribution only has 2 chi components, and will cause errors when
    # n_unknown>2.
    Delta_offset_unit_contribution = ChiPhiFunc(
        jnp.zeros((
            inv_transposed_tensor_fft_op.shape[0],
            inv_transposed_tensor_fft_op.shape[1]
        )),
        nfp
    )
    # How much a Delta_offset of 1 shift -solution[-1,0]
    # This has shape (2, len_phi)
    Delta_offset_unit_contribution += coef_Delta_offset
    # Sampling the rate of change of solution[-2,0] wrt Delta_offset.
    # The free parameter is a scalar, but its coefficient is a ChiPhiFunc. This
    # means that its contribution exists in all phi Fourier mode of the RHS.
    # It's easier to calculate the ratio this way.
    if n_unknown%2==0:
        # FFT the contribution
        fft_Delta_offset_unit_contribution = Delta_offset_unit_contribution.fft().content
        # Contribition to the LHS vector
        sln_Delta_offset_unit_contribution = jnp.tensordot(inv_transposed_tensor_fft_op, fft_Delta_offset_unit_contribution)
        # The amount of Delta_offset required
        Delta_offset = -solution[-1,0]/sln_Delta_offset_unit_contribution[-1,0]
        # Output the offset for calculating Delta
        out_dict_solve['Delta_offset'] = Delta_offset
        # Making a blank ChiPhiFunc with the correct shape. The free parameter's
        # contribution only has 2 chi components, and will cause errors when
        # n_unknown>2.
        Delta_offset_correction = ChiPhiFunc(
            jnp.zeros((
                inv_transposed_tensor_fft_op.shape[0],
                inv_transposed_tensor_fft_op.shape[1]
            )),
            nfp
        )
        Delta_offset_correction += Delta_offset_unit_contribution*Delta_offset
        # Adding the free parameter's contributions to the solution
        solution += jnp.tensordot(inv_transposed_tensor_fft_op, Delta_offset_correction.fft().content, 2)

    ifft_solution = jnp.fft.ifft(fft_pad(solution, len_phi, axis=1), axis=1)
    out_dict_solve['solution'] = ifft_solution
    return(out_dict_solve)

''' IV. Wrapper '''
# Outputs a dictionary containing
# B_theta_n
# Delta_offset (even order only)
# B_psi_nm2 (even order only)
# vec_free
# @partial(jit, static_argnums=(0, 1, 2, 16, 17, 18))
def iterate_looped(
    n_unknown,
    nfp,
    X_coef_cp, Y_coef_cp, Z_coef_cp,
    p_perp_coef_cp, Delta_coef_cp,
    B_psi_coef_cp, B_theta_coef_cp,
    B_alpha_coef, B_denom_coef_c,
    kap_p, tau_p, dl_p,
    iota_coef,
    # A static max freq
    hard_cutoff_freq=-1,
    cutoff_freq=-1,
    # B_theta_np10_avg = 0,
    max_k_diff_pre_inv=-1,
    max_k_diff_post_inv=-1
):
    len_phi = X_coef_cp[n_unknown-1].content.shape[1]
    if cutoff_freq<0 or cutoff_freq>len_phi/2:
        cutoff_freq=len_phi//2
    if hard_cutoff_freq<0 or hard_cutoff_freq>len_phi/2:
        hard_cutoff_freq=len_phi//2
    out_dict_RHS = generate_RHS(
        n_unknown=n_unknown,
        hard_cutoff_freq=hard_cutoff_freq,
        cutoff_freq=cutoff_freq, # Used exclusively in Delta ODE solving.
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
        iota_coef=iota_coef
    )

    O_einv = out_dict_RHS['O_einv']
    vector_free_coef = out_dict_RHS['vector_free_coef']
    # Then calculate the inverted differential operators
    RHS_0_offset = out_dict_RHS['RHS_0_offset']
    print('looped, before evaluating tensor, type of hard_cutoff_freq:',type(hard_cutoff_freq))
    out_dict_tensor = generate_tensor_operator(
        n_unknown=n_unknown,
        nfp=nfp,
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
        O_einv=O_einv,
        vector_free_coef=vector_free_coef,
        cutoff_freq=cutoff_freq,
        hard_cutoff_freq=hard_cutoff_freq,
        max_k_diff_pre_inv=max_k_diff_pre_inv,
        max_k_diff_post_inv=max_k_diff_post_inv,
    )
    inv_transposed_tensor_fft_op = out_dict_tensor['inv_transposed_tensor_fft_op']
    # Even order
    if n_unknown%2==0:
        # Solve for Delta n0.
        coef_Delta_offset = lambda_coef_delta(n_unknown+1, B_alpha_coef, B_denom_coef_c)
        solve_result = solve_free_param(
            n_unknown=n_unknown,
            nfp=nfp,
            inv_transposed_tensor_fft_op=inv_transposed_tensor_fft_op,
            RHS_0_offset=RHS_0_offset,
            len_phi=len_phi,
            coef_Delta_offset=coef_Delta_offset
        )
        solution = solve_result['solution']
        B_theta_n = None

        # B_psi0
        dphi_B_psi_nm2_0 = ChiPhiFunc(solution[-1][None, :], nfp)
        B_psi_nm2_0 = dphi_B_psi_nm2_0.integrate_phi_fft(zero_avg=True)
        B_psi_nm2 = out_dict_RHS['B_psi_nm2_no_unknown'] + B_psi_nm2_0

        # Calculating Yn
        coef_B_psi_dphi_1_in_Y_var = ChiPhiFunc(jnp.fft.ifft(
            jnp.fft.fft(out_dict_tensor['coef_B_psi_dphi_1_in_Y_var'].content, axis=1),
            axis=1
        ), nfp)
        vec_free = solution[-2]
        Yn = ChiPhiFunc(
            (
                jnp.einsum('ijk,jk->ik',O_einv,out_dict_RHS['Yn_rhs_content_no_unknown'])
                + vec_free * vector_free_coef
            ),
            nfp
        ) + coef_B_psi_dphi_1_in_Y_var*dphi_B_psi_nm2_0

        # Calculating Xn, Zn, pn, Delta_n
        Zn = (
            out_dict_RHS['Zn_no_B_psi']
            +looped_coefs.lambda_B_psi_nm2_in_Z_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)*B_psi_nm2_0
        )
        pn = (
            out_dict_RHS['pn_no_B_psi']
            +looped_coefs.lambda_B_psi_nm2_in_p_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)*B_psi_nm2_0
            +looped_coefs.lambda_dphi_B_psi_nm2_in_p_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)*dphi_B_psi_nm2_0
        )
        Xn = (
            out_dict_RHS['Xn_no_B_psi']
            +looped_coefs.lambda_B_psi_nm2_in_X_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)*B_psi_nm2_0
            +looped_coefs.lambda_dphi_B_psi_nm2_in_X_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)*dphi_B_psi_nm2_0
        )
        Deltan = (
            out_dict_RHS['Deltan_no_B_psi']
            +looped_coefs.lambda_B_psi_nm2_in_Delta_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)*B_psi_nm2_0
            +looped_coefs.lambda_dphi_B_psi_nm2_in_Delta_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)*dphi_B_psi_nm2_0
            -solve_result['Delta_offset']
        )

        Yn_B_theta_terms = 0
        if n_unknown>2:
            B_theta_n_no_center_content = jnp.zeros((n_unknown-1, solution.shape[1]), jnp.complex128)
            B_theta_n_no_center_content = B_theta_n_no_center_content.at[:n_unknown//2-1].set(solution[:n_unknown//2-1])
            B_theta_n_no_center_content = B_theta_n_no_center_content.at[n_unknown//2:].set(solution[n_unknown//2-1:-2])
            B_theta_n = ChiPhiFunc(B_theta_n_no_center_content, nfp) + B_theta_coef_cp[n_unknown][0]

            B_theta_in_B_psi = n_unknown/2*B_theta_n.antid_chi()
            B_theta_in_B_psi_fft_short=B_theta_in_B_psi.fft().content

            B_psi_nm2 += B_theta_in_B_psi
            Yn_B_theta_terms = ChiPhiFunc(
                jnp.einsum(
                        'ijkl, jl -> ik',
                        out_dict_tensor['tensor_fft_op_B_psi_in_Y'],
                        B_theta_in_B_psi_fft_short
                    ),
                nfp
            ).ifft()
            Yn += Yn_B_theta_terms

            Zn += (looped_coefs.lambda_B_psi_nm2_in_Z_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)*B_theta_in_B_psi
            )
            pn += (looped_coefs.lambda_B_psi_nm2_in_p_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)*B_theta_in_B_psi
                +looped_coefs.lambda_dchi_B_psi_nm2_in_p_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)*B_theta_in_B_psi.dchi()
                +looped_coefs.lambda_dphi_B_psi_nm2_in_p_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)*B_theta_in_B_psi.dphi()
            )
            Xn += (looped_coefs.lambda_B_psi_nm2_in_X_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)*B_theta_in_B_psi
                +looped_coefs.lambda_dchi_B_psi_nm2_in_X_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)*B_theta_in_B_psi.dchi()
                +looped_coefs.lambda_dphi_B_psi_nm2_in_X_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)*B_theta_in_B_psi.dphi()
            )
            Deltan += (
                -looped_coefs.lambda_B_psi_nm2_in_p_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)*B_denom_coef_c[0]*B_theta_in_B_psi
                -looped_coefs.lambda_dchi_B_psi_nm2_in_p_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)*B_denom_coef_c[0]*B_theta_in_B_psi.dchi()
                -looped_coefs.lambda_dphi_B_psi_nm2_in_p_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)*B_denom_coef_c[0]*B_theta_in_B_psi.dphi()
            )
        return({
            'B_theta_n': B_theta_n,
            'B_psi_nm2': B_psi_nm2,#!!_BTHETA!!!!!,
            'Yn':Yn,
            'Xn':Xn,
            'Zn':Zn,
            'pn':pn,
            'Deltan':Deltan,
            'Delta_offset': solve_result['Delta_offset'],
            # 'solution': solution,
            # 'out_dict_RHS':out_dict_RHS,
            # 'out_dict_tensor':out_dict_tensor,
            # 'Yn0': vec_free,
            # 'Yn_B_theta_terms': Yn_B_theta_terms,
            # 'Yn_B_psi_0_terms': coef_B_psi_dphi_1_in_Y_var*dphi_B_psi_nm2_0,
            # 'dphi_B_psi_nm2_0': dphi_B_psi_nm2_0
        })
    else:
        solve_result = solve_free_param(
            n_unknown=n_unknown,
            nfp=nfp,
            inv_transposed_tensor_fft_op=inv_transposed_tensor_fft_op,
            RHS_0_offset=RHS_0_offset,
            len_phi=len_phi,
            # B_theta_np10_avg = B_theta_np10_avg,
        )
        solution = solve_result['solution']
        B_theta_n = ChiPhiFunc(solution[:-2], nfp)
        B_theta_in_B_psi = n_unknown/2*B_theta_n.antid_chi()
        B_theta_in_B_psi_fft_short=B_theta_in_B_psi.fft().content

        B_psi_nm2 = out_dict_RHS['B_psi_nm2_no_unknown'] + B_theta_in_B_psi

        # Calculating Yn
        vec_free = solution[-2]
        Yn_rhs_content_no_unknown = out_dict_RHS['Yn_rhs_content_no_unknown']
        Yn_B_theta_terms = ChiPhiFunc(
            jnp.einsum(
                    'ijkl, jl -> ik',
                    out_dict_tensor['tensor_fft_op_B_psi_in_Y'],
                    B_theta_in_B_psi_fft_short
                ),
            nfp
        ).ifft()
        Yn = ChiPhiFunc(
            jnp.einsum('ijk,jk->ik',O_einv,out_dict_RHS['Yn_rhs_content_no_unknown'])
            + vec_free * vector_free_coef,
            nfp
        ) + Yn_B_theta_terms

        Zn = (
            out_dict_RHS['Zn_no_B_theta']
            +looped_coefs.lambda_B_psi_nm2_in_Z_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)*B_theta_in_B_psi
        )
        pn = (
            out_dict_RHS['pn_no_B_theta']
            +looped_coefs.lambda_B_psi_nm2_in_p_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)*B_theta_in_B_psi
            +looped_coefs.lambda_dchi_B_psi_nm2_in_p_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)*B_theta_in_B_psi.dchi()
            +looped_coefs.lambda_dphi_B_psi_nm2_in_p_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)*B_theta_in_B_psi.dphi()
        )
        Xn = (
            out_dict_RHS['Xn_no_B_theta']
            +looped_coefs.lambda_B_psi_nm2_in_X_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)*B_theta_in_B_psi
            +looped_coefs.lambda_dchi_B_psi_nm2_in_X_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)*B_theta_in_B_psi.dchi()
            +looped_coefs.lambda_dphi_B_psi_nm2_in_X_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)*B_theta_in_B_psi.dphi()
        )
        Deltan = (
            out_dict_RHS['Deltan_no_B_theta']
            -looped_coefs.lambda_B_psi_nm2_in_p_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)*B_denom_coef_c[0]*B_theta_in_B_psi
            -looped_coefs.lambda_dchi_B_psi_nm2_in_p_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)*B_denom_coef_c[0]*B_theta_in_B_psi.dchi()
            -looped_coefs.lambda_dphi_B_psi_nm2_in_p_n(
                X_coef_cp=X_coef_cp,
                Y_coef_cp=Y_coef_cp,
                B_theta_coef_cp=B_theta_coef_cp,
                B_psi_coef_cp=B_psi_coef_cp,
                iota_coef=iota_coef,
                p_perp_coef_cp=p_perp_coef_cp,
                Delta_coef_cp=Delta_coef_cp,
                B_denom_coef_c=B_denom_coef_c,
                B_alpha_coef=B_alpha_coef,
                kap_p=kap_p,
                dl_p=dl_p,
                tau_p=tau_p,
                n_eval=n_unknown+1)*B_denom_coef_c[0]*B_theta_in_B_psi.dphi()
        )
        return({
            'B_theta_n': B_theta_n,
            'B_theta_np10': ChiPhiFunc(jnp.array([solution[-1]]), nfp),
            # These filters are here because dphi and B_theta_in_B_psi still produce error
            'B_psi_nm2': B_psi_nm2,
            'Yn': Yn,
            'Xn': Xn,
            'Zn': Zn,
            'pn': pn,
            'Deltan': Deltan,
            # 'Yn_B_theta_terms': Yn_B_theta_terms,
            # 'Yn1p': vec_free,
            'solution': solution,
            # 'out_dict_RHS':out_dict_RHS,
            # 'out_dict_tensor':out_dict_tensor,
        })

''' V. Utilities '''
# Set all elements of a fft_op that couples two phi modes
# with mode number difference larger than crit_diff to 0.
# Edits the original array.
def filter_off_diag_tensor(operator, max_k_diff):
    len_phi = operator.shape[1]
    fft_freq_int = jit_fftfreq_int(len_phi)
    mode_diff_mat = jnp.abs(
        fft_freq_int[:,None] - fft_freq_int[None, :]
    )
    # operator = operator.at[:,:,mode_diff_mat>max_k_diff].set(0)
    operator = jnp.where(
        jnp.logical_or(
            mode_diff_mat[None, :, None, :]<max_k_diff,
            max_k_diff<0
        ),
        operator, 0
    )
    return(operator)

# Cyclic import
import equilibrium
