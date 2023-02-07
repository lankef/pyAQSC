# Solves the looped equations.
# ChiPhiFunc and ChiPhiEpsFunc
from chiphifunc import *
from chiphiepsfunc import *
from math_utilities import *
import numpy as np

# parsed relations
import MHD_parsed
from recursion_relations import *

# Performance
import time

''' I. RHS '''
# RHS is run BEFORE operators because B_psi, X, Z, p, Delta are calculated during
# RHS calculation. Uses
# Even order: B_psi[n_unknown-2-1] and X, Z, p, Delta[n_unknown-1] for even orders,
# Odd order:  B_psi[n_unknown-2-1], Delta[n-1] and X, Z, p, Delta, Y[n_unknown]
# Outputs a dictionary containing:
# B_psi_n_n0_masked (even order only)
# O_einv,
# vector_free_coef,
# filtered_RHS_0_offset,
def generate_RHS(
    n_unknown, max_freq,
    X_coef_cp, Y_coef_cp, Z_coef_cp,
    p_perp_coef_cp, Delta_coef_cp,
    B_psi_coef_cp, B_theta_coef_cp,
    B_alpha_coef, B_denom_coef_c,
    kap_p, tau_p, dl_p,
    eta, iota_coef
):
    # n_eval is the order at which the "looped" equations are evaluated at
    n_eval = n_unknown+1
    out_dict_RHS = {}
    if n_unknown%2==0: # Mask B_psi[n-2,0] at even orders
        ''' B_theta, center is known at even orders '''
        B_theta_n = B_theta_coef_cp[n_eval-1]
        B_theta_n_center_only = B_theta_n.get_constant()
        B_theta_coef_cp_center_only = B_theta_coef_cp.mask(n_eval-2)
        B_theta_coef_cp_center_only.append(B_theta_n_center_only)
        out_dict_RHS['B_theta_n_center_only'] = B_theta_n_center_only
        ''' B_psi n-2 0 '''
        B_psi_n = B_psi_coef_cp[n_eval-3]
        B_psi_n_n0_masked = B_psi_n.mask_constant()
        out_dict_RHS['B_psi_n_n0_masked'] = B_psi_n_n0_masked
        B_psi_coef_cp_n0_masked = B_psi_coef_cp.mask(n_eval-4)
        B_psi_coef_cp_n0_masked.append(B_psi_n_n0_masked)
        ''' Z and p, contains B_psi, unknown at even orders '''
        Znm1_no_B_psi = iterate_Zn_cp(
            n_eval-1,
            X_coef_cp, Y_coef_cp, Z_coef_cp,
            B_theta_coef_cp, B_psi_coef_cp_n0_masked,
            B_alpha_coef,
            kap_p, dl_p, tau_p,
            iota_coef)
        pnm1_no_B_psi = iterate_p_perp_n(
            n_eval=n_eval-1,
            B_theta_coef_cp=B_theta_coef_cp,
            B_psi_coef_cp=B_psi_coef_cp_n0_masked,
            B_alpha_coef=B_alpha_coef,
            B_denom_coef_c=B_denom_coef_c,
            p_perp_coef_cp=p_perp_coef_cp,
            Delta_coef_cp=Delta_coef_cp,
            iota_coef=iota_coef)
        p_perp_coef_cp_no_B_psi = p_perp_coef_cp.mask(n_eval-2)
        p_perp_coef_cp_no_B_psi.append(pnm1_no_B_psi)
        Z_coef_cp_no_B_psi = Z_coef_cp.mask(n_eval-2)
        Z_coef_cp_no_B_psi.append(Znm1_no_B_psi)
        ''' X and Delta, contains B_psi, unknown at even orders '''
        Xnm1_no_B_psi = iterate_Xn_cp(n_eval-1,
            X_coef_cp,
            Y_coef_cp,
            Z_coef_cp_no_B_psi,
            B_denom_coef_c,
            B_alpha_coef,
            kap_p, dl_p, tau_p,
            iota_coef)
        Deltanm1_no_B_psi = iterate_delta_n_0_offset(n_eval=n_eval-1,
            B_denom_coef_c=B_denom_coef_c,
            p_perp_coef_cp=p_perp_coef_cp_no_B_psi,
            Delta_coef_cp=Delta_coef_cp,
            iota_coef=iota_coef)
        X_coef_cp_no_B_psi = X_coef_cp.mask(n_eval-2)
        X_coef_cp_no_B_psi.append(Xnm1_no_B_psi)
        Delta_coef_cp_no_B_psi_0_offset = Delta_coef_cp.mask(n_eval-2)
        Delta_coef_cp_no_B_psi_0_offset.append(Deltanm1_no_B_psi)
        ''' First Evaluate O_einv and vec_free for use in even LHS operator '''
        O_matrices, O_einv, vector_free_coef = \
            iterate_Yn_cp_operators(
                n_eval=n_eval-1,
                X_coef_cp=X_coef_cp,
                B_alpha_coef=B_alpha_coef
            )
        out_dict_RHS['O_einv'] = O_einv
        out_dict_RHS['vector_free_coef'] = vector_free_coef
        ''' Y, unknown at even orders and contains B_psi. '''
        chiphifunc_rhs_content_no_B_psi0 = iterate_Yn_cp_RHS(n_eval=n_eval-1,
            X_coef_cp=X_coef_cp_no_B_psi,
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
            iota_coef=iota_coef)
        new_Y_n_no_unknown = ChiPhiFunc(np.einsum('ijk,jk->ik',O_einv,chiphifunc_rhs_content_no_B_psi0))
        Y_coef_cp_no_unknown = Y_coef_cp.mask(n_eval-2)
        Y_coef_cp_no_unknown.append(new_Y_n_no_unknown)
    else:
        # B_theta[n_eval, 0] need to be masked. We mask the whole next order
        # because the rest cancels out.
        B_theta_coef_cp_center_only = B_theta_coef_cp.mask(n_eval-1).zero_append()
        # No B_psi[n-2,0] or Y[n,0] masking at even orders
        B_psi_coef_cp_n0_masked = B_psi_coef_cp
        p_perp_coef_cp_no_B_psi = p_perp_coef_cp
        Z_coef_cp_no_B_psi = Z_coef_cp
        X_coef_cp_no_B_psi = X_coef_cp
        Y_coef_cp_no_unknown = Y_coef_cp
        Delta_coef_cp_no_B_psi_0_offset = Delta_coef_cp

        if ((p_perp_coef_cp_no_B_psi.get_order()<n_eval-1)
            or (p_perp_coef_cp_no_B_psi.get_order()<n_eval-1)
            or (Z_coef_cp_no_B_psi.get_order()<n_eval-1)
            or (X_coef_cp_no_B_psi.get_order()<n_eval-1)):
            raise ValueError('Z, X, p and Delta at order '+str(n_eval-1)+' must '\
            'be calculated before evaluating the looped equations at an odd order.')

    looped_RHS_0_offset = -MHD_parsed.eval_loop(
        n=n_eval, \
        X_coef_cp=X_coef_cp_no_B_psi.mask(n_eval-1).zero_append(), # Done
        Y_coef_cp=Y_coef_cp_no_unknown.mask(n_eval-1).zero_append(), # Done
        Z_coef_cp=Z_coef_cp_no_B_psi.mask(n_eval-1).zero_append(), # Done
        B_theta_coef_cp=B_theta_coef_cp_center_only.mask(n_eval-1).zero_append(),
        B_psi_coef_cp=B_psi_coef_cp_n0_masked.mask(n_eval-3).zero_append(),
        B_alpha_coef=B_alpha_coef,
        B_denom_coef_c=B_denom_coef_c.mask(n_eval-1).zero_append(),
        p_perp_coef_cp=p_perp_coef_cp_no_B_psi.mask(n_eval-1).zero_append(), # Done
        Delta_coef_cp=Delta_coef_cp_no_B_psi_0_offset.mask(n_eval-1).zero_append(), # Done
         # this zero_append on iota_coef doesn't do anythin at even n_unknown,
         # but masks iota (n_unknown-1)/2 at odd n_unknown.
        kap_p=kap_p, dl_p=dl_p, tau_p=tau_p, iota_coef=iota_coef.zero_append()
    ).cap_m(n_eval-2).filter('low_pass',max_freq)

    out_dict_RHS['filtered_RHS_0_offset'] = fft_filter(looped_RHS_0_offset.fft().content, max_freq*2, axis=1)

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
#     [dphi B_psi n-2 0], (sometimes abreviated as B_psi in this code)
#     [Y_n 0],
# ]
# Odd order:
# [
#     [B_theta +n-1],
#     ...
#     ...
#     [B_theta -n+1],
#     [B_theta n,0]
# ]
# by np.tensordot(full_tensor_fft_op_reordered, fft_of_LHS, 2).
# len_tensor: the number of phi modes kept in the tensor.
# For avoiding singular tensor.
def generate_tensor_operator(
    n_unknown, max_freq,
    X_coef_cp, Y_coef_cp, Z_coef_cp,
    p_perp_coef_cp, Delta_coef_cp,
    B_psi_coef_cp, B_theta_coef_cp,
    B_alpha_coef, B_denom_coef_c,
    kap_p, tau_p, dl_p,
    eta, iota_coef,
    O_einv, vector_free_coef
):
    n_eval = n_unknown+1
    # len_tensor is the number of phi modes kept in a tensor.
    len_tensor = max_freq*2
    # dphi_array
    # Multiply a dphi matrix to the end of source.
    # Here it's done by pointwise multiplying
    # a, a, a            a, 0, 0
    # b, b, b instead of 0, b, 0
    # c, c, c            0, 0, c.
    fft_freq = np.fft.fftfreq(len_tensor)*len_tensor
    dphi_slice = (np.full((len_tensor,len_tensor), 1) * 1j * fft_freq)
    dphi_array_single_row = np.tile(dphi_slice, [n_eval-1,1,1,1])
    dphi_array = np.tile(dphi_slice, [n_eval-1,n_eval-3,1,1])

    if n_eval%2==1:
        ''' Y coeffs '''
        # An (n, 1, 2, 2) tensor, acting on
        #     [[Y_n,free]]
        # solved at even orders only
        # Loading coefficients
        looped_y_coefs = MHD_parsed.eval_y_coefs(
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
        coef_Y_n0_dphi_0 = (
            coef_Y * ChiPhiFunc(-vector_free_coef)
            +coef_dchi_Y * ChiPhiFunc(-vector_free_coef).dchi()
            +coef_dchi_dchi_Y * ChiPhiFunc(-vector_free_coef).dchi().dchi()
            +coef_dchi_dchi_dchi_Y * ChiPhiFunc(-vector_free_coef).dchi().dchi().dchi()
        ).cap_m(n_eval-2).filter('low_pass',max_freq)
        coef_Y_n0_dphi_1 = (
            coef_dphi_Y * ChiPhiFunc(-vector_free_coef)
            +coef_dchi_dphi_Y * ChiPhiFunc(-vector_free_coef).dchi()
            +coef_dchi_dchi_dphi_Y * ChiPhiFunc(-vector_free_coef).dchi().dchi()
        ).cap_m(n_eval-2)
        coef_Y_n0_dphi_2 = (
            coef_dphi_dphi_Y * ChiPhiFunc(-vector_free_coef)
            +coef_dphi_dphi_dchi_Y * ChiPhiFunc(-vector_free_coef).dchi()
        ).cap_m(n_eval-2)
        tensor_coef_Y_n0_dphi_0 = np.expand_dims(coef_Y_n0_dphi_0.content, axis=1)
        tensor_coef_Y_n0_dphi_1 = np.expand_dims(coef_Y_n0_dphi_1.content, axis=1)
        tensor_coef_Y_n0_dphi_2 = np.expand_dims(coef_Y_n0_dphi_2.content, axis=1)
        tensor_fft_coef_Y_n0_dphi_0 = fft_filter(np.fft.fft(tensor_coef_Y_n0_dphi_0, axis = 2), len_tensor, axis=2)
        tensor_fft_coef_Y_n0_dphi_1 = fft_filter(np.fft.fft(tensor_coef_Y_n0_dphi_1, axis = 2), len_tensor, axis=2)
        tensor_fft_coef_Y_n0_dphi_2 = fft_filter(np.fft.fft(tensor_coef_Y_n0_dphi_2, axis = 2), len_tensor, axis=2)
        tensor_fft_op_Y_n0_dphi_0 = fft_conv_tensor_batch(tensor_fft_coef_Y_n0_dphi_0)
        tensor_fft_op_Y_n0_dphi_1 = fft_conv_tensor_batch(tensor_fft_coef_Y_n0_dphi_1)
        tensor_fft_op_Y_n0_dphi_2 = fft_conv_tensor_batch(tensor_fft_coef_Y_n0_dphi_2)
        tensor_fft_op_Y_n0_dphi_0 = tensor_fft_op_Y_n0_dphi_0
        tensor_fft_op_Y_n0_dphi_1 = tensor_fft_op_Y_n0_dphi_1*dphi_array_single_row
        tensor_fft_op_Y_n0_dphi_2 = tensor_fft_op_Y_n0_dphi_2*dphi_array_single_row**2
        full_tensor_fft_op_Y_n0 = (
            tensor_fft_op_Y_n0_dphi_0
            +tensor_fft_op_Y_n0_dphi_1
            +tensor_fft_op_Y_n0_dphi_2
        )
        full_tensor_fft_op = full_tensor_fft_op_Y_n0
        ''' B_psi tensors '''
        # A (n, 1, 2, 2) tensor, acting on
        #     [[B_psi0']]
        # B_psi are only solved at even orders (odd n_eval)
        looped_B_psi_coefs = MHD_parsed.eval_B_psi_coefs(
            n_eval=n_eval,
            X_coef_cp=X_coef_cp,
            Y_coef_cp=Y_coef_cp,
            Delta_coef_cp=Delta_coef_cp,
            B_alpha_coef=B_alpha_coef,
            B_denom_coef_c=B_denom_coef_c,
            dl_p=dl_p,
            tau_p=tau_p,
            kap_p=kap_p,
            iota_coef=iota_coef)
        ''' B_psi coefs in Y '''
        # Coeff of B_psi in the RHS of Y equations
        coef_B_psi_dphi_1_in_Y_RHS = looped_B_psi_coefs['coef_B_psi_dphi_1_in_Y_RHS']
        # Applying inverted operators
        coef_B_psi_dphi_1_in_Y_var = ChiPhiFunc(np.einsum('ijk,jk->ik',O_einv,coef_B_psi_dphi_1_in_Y_RHS.content))
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
        coef_B_psi_dphi_0_direct = looped_B_psi_coefs['coef_B_psi_dphi_0_direct']
        coef_B_psi_dphi_1_direct = looped_B_psi_coefs['coef_B_psi_dphi_1_direct']
        coef_B_psi_dphi_1_in_X = looped_B_psi_coefs['coef_B_psi_dphi_1_in_X']
        coef_B_psi_dphi_2_in_X = looped_B_psi_coefs['coef_B_psi_dphi_2_in_X']
        coef_B_psi_dphi_3_in_X = looped_B_psi_coefs['coef_B_psi_dphi_3_in_X']
        coef_B_psi_dphi_0_in_Z = looped_B_psi_coefs['coef_B_psi_dphi_0_in_Z']
        coef_B_psi_dphi_1_in_Z = looped_B_psi_coefs['coef_B_psi_dphi_1_in_Z']
        coef_B_psi_dphi_0_in_p = looped_B_psi_coefs['coef_B_psi_dphi_0_in_p']
        coef_B_psi_dphi_1_in_p = looped_B_psi_coefs['coef_B_psi_dphi_1_in_p']
        coef_B_psi_dphi_0_in_Delta = looped_B_psi_coefs['coef_B_psi_dphi_0_in_Delta']
        coef_B_psi_dphi_1_in_Delta = looped_B_psi_coefs['coef_B_psi_dphi_1_in_Delta']
        coef_B_psi_dphi_1 = (
            coef_B_psi_dphi_1_in_Y
            +coef_B_psi_dphi_1_direct
            +coef_B_psi_dphi_1_in_X
            +coef_B_psi_dphi_1_in_Z
            +coef_B_psi_dphi_1_in_p
            +coef_B_psi_dphi_1_in_Delta
        ).cap_m(n_eval-2)
        coef_B_psi_dphi_2 = (
            coef_B_psi_dphi_2_in_Y
            +coef_B_psi_dphi_2_in_X
        ).cap_m(n_eval-2)
        coef_B_psi_dphi_3 = (
            coef_B_psi_dphi_3_in_Y
            +coef_B_psi_dphi_3_in_X
        ).cap_m(n_eval-2)
        tensor_coef_B_psi_dphi_1 = np.expand_dims(coef_B_psi_dphi_1.content, axis=1)
        tensor_coef_B_psi_dphi_2 = np.expand_dims(coef_B_psi_dphi_2.content, axis=1)
        tensor_coef_B_psi_dphi_3 = np.expand_dims(coef_B_psi_dphi_3.content, axis=1)
        tensor_fft_coef_B_psi_dphi_1 = fft_filter(np.fft.fft(tensor_coef_B_psi_dphi_1, axis = 2), len_tensor, axis=2)
        tensor_fft_coef_B_psi_dphi_2 = fft_filter(np.fft.fft(tensor_coef_B_psi_dphi_2, axis = 2), len_tensor, axis=2)
        tensor_fft_coef_B_psi_dphi_3 = fft_filter(np.fft.fft(tensor_coef_B_psi_dphi_3, axis = 2), len_tensor, axis=2)
        tensor_fft_op_B_psi_dphi_1 = fft_conv_tensor_batch(tensor_fft_coef_B_psi_dphi_1)
        tensor_fft_op_B_psi_dphi_2 = fft_conv_tensor_batch(tensor_fft_coef_B_psi_dphi_2)
        tensor_fft_op_B_psi_dphi_3 = fft_conv_tensor_batch(tensor_fft_coef_B_psi_dphi_3)
        tensor_fft_op_B_psi_dphi_1 = tensor_fft_op_B_psi_dphi_1
        tensor_fft_op_B_psi_dphi_2 = tensor_fft_op_B_psi_dphi_2*dphi_array_single_row**1
        tensor_fft_op_B_psi_dphi_3 = tensor_fft_op_B_psi_dphi_3*dphi_array_single_row**2
        full_tensor_fft_op_dphi_B_psi = (
            tensor_fft_op_B_psi_dphi_1
            +tensor_fft_op_B_psi_dphi_2
            +tensor_fft_op_B_psi_dphi_3
        )
        full_tensor_fft_op = np.concatenate((full_tensor_fft_op_dphi_B_psi, full_tensor_fft_op), axis=1)
    else:
        ''' B_theta [n,0] '''
        coef_B_theta0 = -B_denom_coef_c[0]**2*(diff(p_perp_coef_cp[0],'phi',1))
        coef_dp_B_theta0 = B_denom_coef_c[0]*(Delta_coef_cp[0]-1)
        tensor_coef_B_theta0 = np.expand_dims(coef_B_theta0.content, axis=1)
        tensor_coef_dp_B_theta0 = np.expand_dims(coef_dp_B_theta0.content, axis=1)
        tensor_fft_coef_B_theta0 = fft_filter(np.fft.fft(tensor_coef_B_theta0, axis = 2), len_tensor, axis=2)
        tensor_fft_coef_dp_B_theta0 = fft_filter(np.fft.fft(tensor_coef_dp_B_theta0, axis = 2), len_tensor, axis=2)
        tensor_fft_op_B_theta0 = fft_conv_tensor_batch(tensor_fft_coef_B_theta0)
        tensor_fft_op_dp_B_theta0 = fft_conv_tensor_batch(tensor_fft_coef_dp_B_theta0)
        tensor_fft_op_B_theta0 = tensor_fft_op_B_theta0
        tensor_fft_op_dp_B_theta0 = tensor_fft_op_dp_B_theta0*dphi_array_single_row
        full_tensor_fft_op_B_theta0 = (
            tensor_fft_op_B_theta0
            +tensor_fft_op_dp_B_theta0
        )
        full_tensor_fft_op = full_tensor_fft_op_B_theta0
    ''' B_theta tensors '''
    # (n, n-2, len_tensor, len_tensor), acting on the FFT of
    # [
    #     [B_theta +n-1],
    #     ...
    #     [B_theta -n+1],
    # ]
    # Generating convolution tensors from B_theta coefficients.
    # These are only needed for n>2 (n_eval>3)
    # NOTE: these 'tensors' are for B_theta[n=even]
    # WITH CENTER ELEMENT REMOVED!
    if n_unknown>2:
        looped_B_theta_coefs = MHD_parsed.eval_B_theta_coefs(
            p_perp_coef_cp=p_perp_coef_cp,
            Delta_coef_cp=Delta_coef_cp,
            B_denom_coef_c=B_denom_coef_c,
            iota_coef=iota_coef)
        coef_B_theta = looped_B_theta_coefs['coef_B_theta']
        coef_dchi_B_theta = looped_B_theta_coefs['coef_dchi_B_theta']
        coef_dphi_B_theta = looped_B_theta_coefs['coef_dphi_B_theta']
        # 'Tensor coefficients', dimension is (n_eval-1, n_eval-3, len_phi)
        tensor_coef_B_theta = conv_tensor(coef_B_theta.content, n_eval-2)
        tensor_coef_dchi_B_theta = conv_tensor(coef_dchi_B_theta.content, n_eval-2)
        tensor_coef_dphi_B_theta = conv_tensor(coef_dphi_B_theta.content, n_eval-2)
        # Putting in dchi
        dchi_matrix = dchi_op(n_eval-2, False)
        tensor_coef_dchi_B_theta = np.einsum('ijk,jl->ilk',tensor_coef_dchi_B_theta, dchi_matrix)
        # Removing the 'column' acting on the B_theta=0 component.
        if n_eval%2==1:
            tensor_coef_B_theta = np.delete(tensor_coef_B_theta, (n_eval-2)//2, 1)
            tensor_coef_dchi_B_theta = np.delete(tensor_coef_dchi_B_theta, (n_eval-2)//2, 1)
            tensor_coef_dphi_B_theta = np.delete(tensor_coef_dphi_B_theta, (n_eval-2)//2, 1)
        # Applying FFT
        tensor_fft_coef_B_theta = fft_filter(np.fft.fft(tensor_coef_B_theta, axis = 2), len_tensor, axis=2)
        tensor_fft_coef_dchi_B_theta = fft_filter(np.fft.fft(tensor_coef_dchi_B_theta, axis = 2), len_tensor, axis=2)
        tensor_fft_coef_dphi_B_theta = fft_filter(np.fft.fft(tensor_coef_dphi_B_theta, axis = 2), len_tensor, axis=2)
        # 'Tensor coefficients', dimension is (n_eval-1, n_eval-3, len_phi)
        # Last 2 dimensions are for convolving phi cells.
        tensor_fft_op_B_theta = fft_conv_tensor_batch(tensor_fft_coef_B_theta)
        tensor_fft_op_dchi_B_theta = fft_conv_tensor_batch(tensor_fft_coef_dchi_B_theta)
        tensor_fft_op_dphi_B_theta = fft_conv_tensor_batch(tensor_fft_coef_dphi_B_theta)
        # Applying dphi
        tensor_fft_op_dphi_B_theta = tensor_fft_op_dphi_B_theta*dphi_array
        # Merging
        full_tensor_fft_op_B_theta = (
            tensor_fft_op_B_theta
            +tensor_fft_op_dchi_B_theta
            +tensor_fft_op_dphi_B_theta
        )
        full_tensor_fft_op = np.concatenate((full_tensor_fft_op_B_theta, full_tensor_fft_op), axis=1)
    # Reorder so that the operator applied onto an (n, len_phi) vector by
    # np.tensordot(filtered_looped_fft_operator, test_unknown_low_res, 2)
    filtered_looped_fft_operator = np.transpose(full_tensor_fft_op, (0,2,1,3))
    # Finding the inverse differential operator
    filtered_inv_looped_fft_operator = np.linalg.tensorinv(filtered_looped_fft_operator)
    return({
        # (n_unknown, len_tensor, n_unknown, len_tensor)
        'filtered_looped_fft_operator': filtered_looped_fft_operator,
        'filtered_inv_looped_fft_operator': filtered_inv_looped_fft_operator,
    })

''' III. Calculating Delta_offset '''
# At even order, calculate B_psi[n-2,0], Y[n,0] and the average of Delta[n,0]
# (called 'Delta_offset' below)
# target_len_phi: target phi length of the solution
# coef_iota_nm1b2_in_Delta: Coefficient of iota (n-1)/2 in Delta. Is a constant
# in the Equilibrium object. Needed for odd orders.
def solve(n_unknown, target_len_phi,
    filtered_inv_looped_fft_operator, filtered_RHS_0_offset,
    coef_Delta_offset = None
    ):
    out_dict_solve = {}
    # Solution with zero value for the free constant parameter
    filtered_solution = np.tensordot(filtered_inv_looped_fft_operator, filtered_RHS_0_offset, 2)
    # To have periodic B_psi, values for a constant free marameter, Delt_offset
    # must be found at even orders.
    # Integral(B_psi') = 0 is equivalent to filtered_solution[-2,0] = 0.
    # (-2: B_psi is always the second last row in the solution. 0: m=0 mode
    # in a np.fft.fft array.)
    # Because the looped ODE is now treated as a linear equation,
    # the Delta_offset dependence of filtered_solution[-2,0] is linear.

    # Making a blank ChiPhiFunc with the correct shape. The free parameter's
    # contribution only has 2 chi components, and will cause errors when
    # n_unknown>2.
    print('filtered_inv_looped_fft_operator',filtered_inv_looped_fft_operator.shape)
    Delta_offset_unit_contribution = ChiPhiFunc(np.zeros((filtered_inv_looped_fft_operator.shape[0], filtered_inv_looped_fft_operator.shape[1])))
    # How much a Delta_offset of 1 shift -filtered_solution[-2,0]
    # This has shape (2, len_tensor)
    Delta_offset_unit_contribution += coef_Delta_offset
    # Sampling the rate of change of filtered_solution[-2,0] wrt Delta_offset.
    # The free parameter is a scalar, but its coefficient is a ChiPhiFunc. This
    # means that its contribution exists in all phi Fourier mode of the RHS.
    # It's easier to calculate the ratio this way.
    if n_unknown%2==0:
        # FFT the contribution
        fft_Delta_offset_unit_contribution = Delta_offset_unit_contribution.fft().content
        # Contribition to the LHS vector
        sln_Delta_offset_unit_contribution = np.tensordot(filtered_inv_looped_fft_operator, fft_Delta_offset_unit_contribution)
        # The amount of Delta_offset required
        Delta_offset = -filtered_solution[-2,0]/sln_Delta_offset_unit_contribution[0,0]
        # Output the offset for calculating Delta
        out_dict_solve['Delta_offset'] = Delta_offset
        # Making a blank ChiPhiFunc with the correct shape. The free parameter's
        # contribution only has 2 chi components, and will cause errors when
        # n_unknown>2.
        Delta_offset_correction = ChiPhiFunc(np.zeros((filtered_inv_looped_fft_operator.shape[0], filtered_inv_looped_fft_operator.shape[1])))
        Delta_offset_correction += Delta_offset_unit_contribution*Delta_offset
        # Adding the free parameter's contributions to the solution
        filtered_solution += np.tensordot(filtered_inv_looped_fft_operator, Delta_offset_correction.fft().content, 2)

    # Padding solution to a desired len_phi
    padded_solution = fft_pad(
        filtered_solution,
        target_len_phi,
        axis=1
    )
    ifft_solution = np.fft.ifft(padded_solution, axis=1)
    out_dict_solve['solution'] = ifft_solution
    return(out_dict_solve)

''' IV. Wrapper '''
# Outputs a dictionary containing
# B_theta_n
# Delta_offset (even order only)
# B_psi_nm2 (even order only)
# vec_free
def iterate_looped(
    n_unknown, max_freq, target_len_phi,
    X_coef_cp, Y_coef_cp, Z_coef_cp,
    p_perp_coef_cp, Delta_coef_cp,
    B_psi_coef_cp, B_theta_coef_cp,
    B_alpha_coef, B_denom_coef_c,
    kap_p, tau_p, dl_p,
    eta, iota_coef, lambda_coef_Delta_offset = None,
):
    if target_len_phi<max_freq*2:
        raise ValueError('target_len_phi must >= max_freq*2.')

    # First calculate RHS
    RHS_result = generate_RHS(
        n_unknown = n_unknown,
        max_freq = max_freq,
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
        iota_coef = iota_coef
    )
    if n_unknown%2==0:
        O_einv = RHS_result['O_einv']
        vector_free_coef = RHS_result['vector_free_coef']
    else:
        O_einv = None
        vector_free_coef = None

    filtered_RHS_0_offset = RHS_result['filtered_RHS_0_offset']
    operator_result = generate_tensor_operator(
        n_unknown = n_unknown,
        max_freq = max_freq,
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
        O_einv = O_einv,
        vector_free_coef = vector_free_coef
    )
    filtered_inv_looped_fft_operator =  operator_result['filtered_inv_looped_fft_operator']


    if n_unknown%2==0:
        # Saves time.
        coef_Delta_offset = lambda_coef_Delta_offset(n_unknown+1)
        solve_result = solve(
            n_unknown=n_unknown,
            target_len_phi=target_len_phi,
            filtered_inv_looped_fft_operator=filtered_inv_looped_fft_operator,
            filtered_RHS_0_offset=filtered_RHS_0_offset,
            coef_Delta_offset=coef_Delta_offset
        )
        solution = solve_result['solution']
        # Even order
        B_theta_n = RHS_result['B_theta_n_center_only']
        if n_unknown>2:
            B_theta_n_no_center_content = np.zeros((n_unknown-1, target_len_phi))
            B_theta_n_no_center_content[:n_unknown//2-1] = solution[:n_unknown//2-1]
            B_theta_n_no_center_content[n_unknown//2:] = solution[n_unknown//2-1:-2]
            B_theta_n += ChiPhiFunc(B_theta_n_no_center_content)
        return({
            'B_theta_n': B_theta_n,
            'Delta_offset': solve_result['Delta_offset'],
            'B_psi_nm2': RHS_result['B_psi_n_n0_masked']+ChiPhiFunc(np.array([solution[-2]])).integrate_phi(periodic=False),
            'vec_free': solution[-1],
            'O_einv': O_einv,
            'vector_free_coef': vector_free_coef
        })
    else:
        solve_result = solve(
            n_unknown=n_unknown,
            target_len_phi=target_len_phi,
            filtered_inv_looped_fft_operator=filtered_inv_looped_fft_operator,
            filtered_RHS_0_offset=filtered_RHS_0_offset
        )
        # Odd order
        B_theta_n_content = solution[:-1]
        B_theta_n = ChiPhiFunc(B_theta_n_content)
        # To use: Yn = (np.einsum('ijk,jk->ik',O_einv,chiphifunc_rhs_content) - vec_free * vector_free_coef)
        vec_free = solution[-1]
        return({
            'B_theta_n': B_theta_n,
            'iota_nm1b2': solve_result['iota_nm1b2'],
            'vec_free': solution[-1],
            'O_einv': O_einv,
            'vector_free_coef': vector_free_coef
        })
