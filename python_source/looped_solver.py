# Solves the looped equations.
# ChiPhiFunc and ChiPhiEpsFunc
from chiphifunc import *
from chiphiepsfunc import *
from math_utilities import *
import numpy as np

# parsed relations
import MHD_parsed

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
    len_tensor = max_freq*2
    out_dict_RHS = {}
    ''' O_einv and vec_free for Y. Y is unknown at all orders '''
    # but solved from different equations (II tilde/D3) at each (even/odd) order.
    O_matrices, O_einv, vector_free_coef = \
        recursion_relations.iterate_Yn_cp_operators(
            n_eval=n_eval-1,
            X_coef_cp=X_coef_cp,
            B_alpha_coef=B_alpha_coef
        )
    out_dict_RHS['O_einv'] = O_einv
    out_dict_RHS['vector_free_coef'] = np.fft.ifft(fft_filter(np.fft.fft(vector_free_coef, axis = 1), len_tensor, axis=1), axis=1)
    # Even orders
    if n_unknown%2==0:
        ''' B_theta, center is known at even orders '''
        B_theta_n = B_theta_coef_cp[n_eval-1]
        B_theta_n_0_only = B_theta_n.get_constant()
        B_theta_coef_cp_0_only = B_theta_coef_cp.mask(n_eval-2)
        B_theta_coef_cp_0_only.append(B_theta_n_0_only)
        out_dict_RHS['B_theta_n_0_only'] = B_theta_n_0_only
        ''' B_psi n-2 0 '''
        B_psi_n = B_psi_coef_cp[n_eval-3]
        B_psi_n_n0_masked = B_psi_n.mask_constant()
        out_dict_RHS['B_psi_n_n0_masked'] = B_psi_n_n0_masked
        B_psi_coef_cp_n0_masked = B_psi_coef_cp.mask(n_eval-4)
        B_psi_coef_cp_n0_masked.append(B_psi_n_n0_masked)
        ''' Z and p, contains B_psi, unknown at even orders '''
        Znm1_no_B_psi = recursion_relations.iterate_Zn_cp(
            n_eval-1,
            X_coef_cp, Y_coef_cp, Z_coef_cp,
            B_theta_coef_cp, B_psi_coef_cp_n0_masked,
            B_alpha_coef,
            kap_p, dl_p, tau_p,
            iota_coef)
        pnm1_no_B_psi = recursion_relations.iterate_p_perp_n(
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
        Xnm1_no_B_psi = recursion_relations.iterate_Xn_cp(n_eval-1,
            X_coef_cp,
            Y_coef_cp,
            Z_coef_cp_no_B_psi,
            B_denom_coef_c,
            B_alpha_coef,
            kap_p, dl_p, tau_p,
            iota_coef)
        Deltanm1_no_B_psi = recursion_relations.iterate_delta_n_0_offset(n_eval=n_eval-1,
            B_denom_coef_c=B_denom_coef_c,
            p_perp_coef_cp=p_perp_coef_cp_no_B_psi,
            Delta_coef_cp=Delta_coef_cp,
            iota_coef=iota_coef)
        X_coef_cp_no_B_psi = X_coef_cp.mask(n_eval-2)
        X_coef_cp_no_B_psi.append(Xnm1_no_B_psi)
        Delta_coef_cp_no_B_psi_0_offset = Delta_coef_cp.mask(n_eval-2)
        Delta_coef_cp_no_B_psi_0_offset.append(Deltanm1_no_B_psi)
        out_dict_RHS['Znm1_no_B_psi']=Znm1_no_B_psi
        out_dict_RHS['pnm1_no_B_psi']=pnm1_no_B_psi
        out_dict_RHS['Xnm1_no_B_psi']=Xnm1_no_B_psi
        out_dict_RHS['Deltanm1_no_B_psi']=Deltanm1_no_B_psi
    # Odd orders
    else:
        # B_theta[n_unknown] and B_theta[n_unknown+1, 0] need to be masked.
        # We mask the whole next order because the rest cancels out.
        B_theta_coef_cp_0_only = B_theta_coef_cp.mask(n_unknown-1).zero_append().zero_append()
        # No B_psi[n-2,0] masking at even orders
        B_psi_coef_cp_n0_masked = B_psi_coef_cp
        p_perp_coef_cp_no_B_psi = p_perp_coef_cp
        Z_coef_cp_no_B_psi = Z_coef_cp
        X_coef_cp_no_B_psi = X_coef_cp
        Delta_coef_cp_no_B_psi_0_offset = Delta_coef_cp

        if ((p_perp_coef_cp_no_B_psi.get_order()<n_eval-1)
            or (p_perp_coef_cp_no_B_psi.get_order()<n_eval-1)
            or (Z_coef_cp_no_B_psi.get_order()<n_eval-1)
            or (X_coef_cp_no_B_psi.get_order()<n_eval-1)):
            raise ValueError('Z, X, p and Delta at order '+str(n_eval-1)+' must '\
            'be calculated before evaluating the looped equations at an odd order.')

    ''' Y, unknown at all orders and contains B_psi. '''
    Yn_rhs_content_no_unknown = recursion_relations.iterate_Yn_cp_RHS(n_eval=n_eval-1,
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
    new_Y_n_no_unknown = ChiPhiFunc(np.einsum('ijk,jk->ik',O_einv,Yn_rhs_content_no_unknown))
    Y_coef_cp_no_unknown = Y_coef_cp.mask(n_eval-2)
    Y_coef_cp_no_unknown.append(new_Y_n_no_unknown)
    out_dict_RHS['Yn_rhs_content_no_unknown']=Yn_rhs_content_no_unknown

    ''' Evaluating looped RHS '''
    looped_RHS_0_offset = -MHD_parsed.eval_loop(
        n=n_eval, \
        X_coef_cp=X_coef_cp_no_B_psi.mask(n_eval-1).zero_append(), # Done
        Y_coef_cp=Y_coef_cp_no_unknown.mask(n_eval-1).zero_append(), # Done
        Z_coef_cp=Z_coef_cp_no_B_psi.mask(n_eval-1).zero_append(), # Done
        B_theta_coef_cp=B_theta_coef_cp_0_only.mask(n_eval-1).zero_append(),
        B_psi_coef_cp=B_psi_coef_cp_n0_masked.mask(n_eval-3).zero_append(),
        B_alpha_coef=B_alpha_coef,
        B_denom_coef_c=B_denom_coef_c.mask(n_eval-1).zero_append(),
        p_perp_coef_cp=p_perp_coef_cp_no_B_psi.mask(n_eval-1).zero_append(), # Done
        Delta_coef_cp=Delta_coef_cp_no_B_psi_0_offset.mask(n_eval-1).zero_append(), # Done
         # this zero_append on iota_coef doesn't do anythin at even n_unknown,
         # but masks iota (n_unknown-1)/2 at odd n_unknown.
        kap_p=kap_p, dl_p=dl_p, tau_p=tau_p, iota_coef=iota_coef #iota_coef.zero_append()
    ).cap_m(n_eval-2).filter('low_pass',max_freq)

    # Calculating D3. This cannot be merged into the previous
    # if statement because we need to know Y first.
    # Also, the zero-th component of the looped equation
    # is not affected by any substitution, and is identical to
    # II's zeroth component.
    if n_unknown%2==1:
        # Caluclating the center element of II
        II_RHS_no_unknown = -MHD_parsed.eval_II_center(
            n = n_eval,
            B_theta_coef_cp = B_theta_coef_cp.mask(n_unknown-1).zero_append().zero_append(),
            B_alpha_coef = B_alpha_coef,
            B_denom_coef_c = B_denom_coef_c,
            # This mask gets rid of p_perp n_unknown+1, which only appears as dchi.
            p_perp_coef_cp = p_perp_coef_cp_no_B_psi.mask(n_eval-1).zero_append(),
            # This mask is redundant. Delta appears at order n_unknown.
            Delta_coef_cp = Delta_coef_cp_no_B_psi_0_offset.mask(n_eval-1).zero_append(),
            iota_coef = iota_coef)[0]

        # Calculating D3
        D3_RHS_no_unknown = -MHD_parsed.eval_D3_RHS_m_LHS(
            n = n_eval,
            X_coef_cp = X_coef_cp.mask(n_unknown),
            # Only dep on Y[+-1]
            Y_coef_cp = Y_coef_cp_no_unknown,
            # The m=0 component is actually indep of Z[n+1]
            Z_coef_cp = Z_coef_cp.mask(n_unknown).zero_append(),
            # This equation may contain both B_theta[n,+-1] and B_theta[n+1,0].
            B_theta_coef_cp = B_theta_coef_cp_0_only,
            B_denom_coef_c = B_denom_coef_c,
            B_alpha_coef = B_alpha_coef,
            iota_coef = iota_coef, #.mask((n_unknown-3)//2).zero_append(), # iota is also masked
            dl_p = dl_p,
            tau_p = tau_p,
        kap_p = kap_p)[0]

        # Type-checking D3 and the center looped component
        if type(D3_RHS_no_unknown) is ChiPhiFuncNull \
        or type(II_RHS_no_unknown) is ChiPhiFuncNull:
            raise ValueError('D3_RHS_no_unknown evaluates to null!')

        # Setting looped center to be II.
        # cap_m guaranteed loop_center's shape to be (n_unknown, len_phi).
        # [0] guaranteed
        # Since we'll immediately redefine looped_RHS_0_offset, set always_copy
        # to False to save a bit of time
        looped_content, II_content = \
            looped_RHS_0_offset.stretch_phi_to_match(II_RHS_no_unknown, always_copy=False)
        looped_content[n_unknown//2] = II_content[0]
        looped_RHS_0_offset = ChiPhiFunc(looped_content)

        looped_content, D3_RHS_content = \
            looped_RHS_0_offset.stretch_phi_to_match(D3_RHS_no_unknown, always_copy=False)
        looped_content = np.concatenate((looped_content, D3_RHS_content), axis=0)
        looped_RHS_0_offset = ChiPhiFunc(looped_content)

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
    O_einv, vector_free_coef,
    looped_coef_lambdas
):
    # An internal method taking care of some dupicant code
    def to_tensor_fft_op(ChiPhiFunc_in):
        tensor_coef = np.expand_dims(ChiPhiFunc_in.content, axis=1)
        tensor_fft_coef = fft_filter(np.fft.fft(tensor_coef, axis = 2), len_tensor, axis=2)
        tensor_fft_op = fft_conv_tensor_batch(tensor_fft_coef)
        return(tensor_fft_op)

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
    dphi_array_single_elem = np.tile(dphi_slice, [1,1,1,1])
    dphi_array_single_row = np.tile(dphi_slice, [n_unknown,1,1,1])
    # dphi_array is used exclusively for B_theta.
    # At even orders there is one less comp to solve for.
    if n_unknown%2==0:
        dphi_array_B_theta = np.tile(dphi_slice, [n_unknown, n_unknown-2,1,1])
    else:
        dphi_array_B_theta = np.tile(dphi_slice, [n_unknown, n_unknown-1,1,1])
    out_dict_tensor={}
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
        coef_Y * ChiPhiFunc(vector_free_coef)
        +coef_dchi_Y * ChiPhiFunc(vector_free_coef).dchi()
        +coef_dchi_dchi_Y * ChiPhiFunc(vector_free_coef).dchi().dchi()
        +coef_dchi_dchi_dchi_Y * ChiPhiFunc(vector_free_coef).dchi().dchi().dchi()
    ).cap_m(n_eval-2).filter('low_pass',max_freq)
    coef_Y_n0_dphi_1 = (
        coef_dphi_Y * ChiPhiFunc(vector_free_coef)
        +coef_dchi_dphi_Y * ChiPhiFunc(vector_free_coef).dchi()
        +coef_dchi_dchi_dphi_Y * ChiPhiFunc(vector_free_coef).dchi().dchi()
    ).cap_m(n_eval-2)
    coef_Y_n0_dphi_2 = (
        coef_dphi_dphi_Y * ChiPhiFunc(vector_free_coef)
        +coef_dphi_dphi_dchi_Y * ChiPhiFunc(vector_free_coef).dchi()
    ).cap_m(n_eval-2)
    tensor_fft_op_Y_n0_dphi_0 = to_tensor_fft_op(coef_Y_n0_dphi_0)
    tensor_fft_op_Y_n0_dphi_1 = to_tensor_fft_op(coef_Y_n0_dphi_1)*dphi_array_single_row
    tensor_fft_op_Y_n0_dphi_2 = to_tensor_fft_op(coef_Y_n0_dphi_2)*dphi_array_single_row**2
    full_tensor_fft_op_Y_n0 = (
        tensor_fft_op_Y_n0_dphi_0
        +tensor_fft_op_Y_n0_dphi_1
        +tensor_fft_op_Y_n0_dphi_2
    )
    full_tensor_fft_op = full_tensor_fft_op_Y_n0
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
        # Cannot be converted using to_tensor_fft_op() because
        # of dchi and known m=0 components at even orders
        tensor_coef_B_theta = conv_tensor(coef_B_theta.content, n_eval-2)
        tensor_coef_dchi_B_theta = conv_tensor(coef_dchi_B_theta.content, n_eval-2)
        tensor_coef_dphi_B_theta = conv_tensor(coef_dphi_B_theta.content, n_eval-2)
        # Putting in dchi
        dchi_matrix = dchi_op(n_eval-2, False)
        tensor_coef_dchi_B_theta = np.einsum('ijk,jl->ilk',tensor_coef_dchi_B_theta, dchi_matrix)
        # Removing the 'column' acting on the m=0 component.
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
        tensor_fft_op_dphi_B_theta = tensor_fft_op_dphi_B_theta*dphi_array_B_theta
        # Merging
        full_tensor_fft_op_B_theta = (
            tensor_fft_op_B_theta
            +tensor_fft_op_dchi_B_theta
            +tensor_fft_op_dphi_B_theta
        )
        full_tensor_fft_op = np.concatenate((full_tensor_fft_op_B_theta, full_tensor_fft_op), axis=1)
    if n_eval%2==1:
        ''' B_psi tensors '''
        # A (n, 1, 2, 2) tensor, acting on
        #     [[B_psi0']]
        # B_psi are only solved at even orders (odd n_eval)
        # looped_coef_lambdas = MHD_parsed.eval_B_psi_coefs(
        #     n_eval=n_eval,
        #     X_coef_cp=X_coef_cp,
        #     Y_coef_cp=Y_coef_cp,
        #     Delta_coef_cp=Delta_coef_cp,
        #     B_alpha_coef=B_alpha_coef,
        #     B_denom_coef_c=B_denom_coef_c,
        #     dl_p=dl_p,
        #     tau_p=tau_p,
        #     kap_p=kap_p,
        #     iota_coef=iota_coef)
        ''' B_psi coefs in Y '''
        # Coeff of B_psi in the RHS of Y equations
        coef_B_psi_dphi_1_in_Y_RHS = looped_coef_lambdas['lambda_B_psi_dphi_1_in_Y_RHS'](n_eval).pad_chi(n_eval+1)
        out_dict_tensor['coef_B_psi_dphi_1_in_Y_RHS'] = coef_B_psi_dphi_1_in_Y_RHS
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
        coef_B_psi_dphi_0_direct = looped_coef_lambdas['lambda_B_psi_dphi_0_direct'](n_eval)
        coef_B_psi_dphi_1_direct = looped_coef_lambdas['lambda_B_psi_dphi_1_direct'](n_eval)
        coef_B_psi_dphi_1_in_X = looped_coef_lambdas['lambda_B_psi_dphi_1_in_X'](n_eval)
        coef_B_psi_dphi_2_in_X = looped_coef_lambdas['lambda_B_psi_dphi_2_in_X'](n_eval)
        coef_B_psi_dphi_3_in_X = looped_coef_lambdas['lambda_B_psi_dphi_3_in_X'](n_eval)
        coef_B_psi_dphi_0_in_Z = looped_coef_lambdas['lambda_B_psi_dphi_0_in_Z'](n_eval)
        coef_B_psi_dphi_1_in_Z = looped_coef_lambdas['lambda_B_psi_dphi_1_in_Z'](n_eval)
        coef_B_psi_dphi_0_in_p = looped_coef_lambdas['lambda_B_psi_dphi_0_in_p'](n_eval)
        coef_B_psi_dphi_1_in_p = looped_coef_lambdas['lambda_B_psi_dphi_1_in_p'](n_eval)
        coef_B_psi_dphi_0_in_Delta = looped_coef_lambdas['lambda_B_psi_dphi_0_in_Delta'](n_eval)
        coef_B_psi_dphi_1_in_Delta = looped_coef_lambdas['lambda_B_psi_dphi_1_in_Delta'](n_eval)
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
        tensor_fft_op_B_psi_dphi_1 = to_tensor_fft_op(coef_B_psi_dphi_1)
        tensor_fft_op_B_psi_dphi_2 = to_tensor_fft_op(coef_B_psi_dphi_2)*dphi_array_single_row**1
        tensor_fft_op_B_psi_dphi_3 = to_tensor_fft_op(coef_B_psi_dphi_3)*dphi_array_single_row**2
        full_tensor_fft_op_dphi_B_psi = (
            tensor_fft_op_B_psi_dphi_1
            +tensor_fft_op_B_psi_dphi_2
            +tensor_fft_op_B_psi_dphi_3
        )
        full_tensor_fft_op = np.concatenate((full_tensor_fft_op, full_tensor_fft_op_dphi_B_psi), axis=1)
    else:
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
        full_tensor_fft_op[n_unknown//2] = 0
        ''' B_theta in II '''
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

        coef_B_theta_1p_0 = coef_B_theta_0_chiphifunc[-1]+coef_dc_B_theta_0_chiphifunc[-1]*1j
        coef_B_theta_1n_0 = coef_B_theta_0_chiphifunc[1]+coef_dc_B_theta_0_chiphifunc[1]*(-1j)
        coef_dp_B_theta_1p_0 = coef_dp_B_theta_0_chiphifunc[-1]
        coef_dp_B_theta_1n_0 = coef_dp_B_theta_0_chiphifunc[1]
        # 'Tensor elements', dimension is (1, 1, len_phi)
        # (as opposed to (n_unknown, 1, len_phi) for B_psi and Y_free in II)
        # This is an individual matrix element.
        # Note: the first 2 dimensions are redundant, but left in to reduce
        # duplicate code.
        elem_fft_op_B_theta_1p_0 = to_tensor_fft_op(coef_B_theta_1p_0)
        elem_fft_op_B_theta_1n_0 = to_tensor_fft_op(coef_B_theta_1n_0)
        elem_fft_op_dp_B_theta_1p_0 = to_tensor_fft_op(coef_dp_B_theta_1p_0)*dphi_array_single_elem
        elem_fft_op_dp_B_theta_1n_0 = to_tensor_fft_op(coef_dp_B_theta_1n_0)*dphi_array_single_elem
        # Merge and remove the redundant dims
        fin_elem_B_theta_1p_0 = (elem_fft_op_dp_B_theta_1p_0 + elem_fft_op_B_theta_1p_0)[0][0]
        fin_elem_B_theta_1n_0 = (elem_fft_op_dp_B_theta_1n_0 + elem_fft_op_B_theta_1n_0)[0][0]
        # The first n_unknown//2 means that the coefficients for terms in
        # the m=0 component of the looped equations is modified.
        # The second n_unknown//2(-1) modifies the coefficient
        # of the unknown at the center and right before the center,
        # which always correspond to B_theta[n_unknown,1] and B_theta[n_unknown,-1]:
        # B[5,-3], B[5,-1], B[5,+1], B[5,+3], Y[5,1]
        full_tensor_fft_op[n_unknown//2, n_unknown//2] = fin_elem_B_theta_1p_0
        full_tensor_fft_op[n_unknown//2, n_unknown//2-1] = fin_elem_B_theta_1n_0

        ''' B_theta [n+1,0] in II '''
        coef_B_theta_np10 = -B_denom_coef_c[0]**2*p_perp_coef_cp[0].dphi()
        coef_dp_B_theta_np10 = B_denom_coef_c[0]*(Delta_coef_cp[0]-1)
        elem_fft_op_B_theta_np10 = to_tensor_fft_op(coef_B_theta_np10)
        elem_fft_op_dp_B_theta_np10 = to_tensor_fft_op(coef_dp_B_theta_np10)*dphi_array_single_elem
        # Merging and removing redundant dims
        fin_elem_B_theta_np10 = (elem_fft_op_B_theta_np10 + elem_fft_op_dp_B_theta_np10)[0][0]
        new_column = np.zeros(
            (
                full_tensor_fft_op.shape[0], # n components: Looped equation
                1,                           # One unknown: B_theta[n+1]
                full_tensor_fft_op.shape[2], # phi operator
                full_tensor_fft_op.shape[2]),
            np.complex128)
        new_column[n_unknown//2] = fin_elem_B_theta_np10
        full_tensor_fft_op = np.concatenate((full_tensor_fft_op, new_column), axis=1)
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
        coef_Yn1p = (
            -2j*dl_p*tau_p*X_coef_cp[1][-1]
            +1j*Y_coef_cp[1][-1].dphi()
            +2*iota_coef[0]*Y_coef_cp[1][-1])
        coef_dp_Yn1p = (
            -1j*Y_coef_cp[1][-1])
        coef_Yn1n = (
            2j*dl_p*tau_p*X_coef_cp[1][1]
            -1j*Y_coef_cp[1][1].dphi()
            +2*iota_coef[0]*Y_coef_cp[1][1])
        coef_dp_Yn1n = (
            1j*Y_coef_cp[1][1])
        coef_Yn1p_in_Yn1n = ChiPhiFunc(vector_free_coef)[-1]
        fin_coef_Yn1p = coef_Yn1p + coef_Yn1n*coef_Yn1p_in_Yn1n + coef_dp_Yn1n*coef_Yn1p_in_Yn1n.dphi()
        fin_coef_dp_Yn1p = coef_dp_Yn1p + coef_dp_Yn1n*coef_Yn1p_in_Yn1n
        elem_fft_op_Yn1p = to_tensor_fft_op(fin_coef_Yn1p)
        elem_fft_op_dp_Yn1p = to_tensor_fft_op(fin_coef_dp_Yn1p)*dphi_array_single_row
        ''' B_theta[n] and B_theta[n+1,0] coefficient in D3 '''
        ones = ChiPhiFunc(np.ones((1, len_tensor),np.complex128))
        coef_B_theta_n_1p = phi_avg((-B_alpha_coef[0]*B_denom_coef_c[1][-1]))*ones
        coef_B_theta_n_1n = phi_avg((-B_alpha_coef[0]*B_denom_coef_c[1][1]))*ones
        coef_B_theta_np1_0 = phi_avg(-B_alpha_coef[0]*B_denom_coef_c[0])*ones
        # 'Tensor element operators', dimension is (1, 1, len_phi, len_phi)
        # Last 2 dimensions are for convolving phi cells.
        elem_fft_op_B_theta_n_1p = to_tensor_fft_op(coef_B_theta_n_1p)
        elem_fft_op_B_theta_n_1n = to_tensor_fft_op(coef_B_theta_n_1n)
        elem_fft_op_B_theta_np1_0 = to_tensor_fft_op(coef_B_theta_np1_0)
        # Merge, remove redundant dims, and constructing row to append
        D3_comp_fft_op = np.zeros((1,n_unknown+1, len_tensor, len_tensor), dtype=np.complex128)
        D3_comp_fft_op[0][n_unknown//2-1] = elem_fft_op_B_theta_n_1n[0][0]
        D3_comp_fft_op[0][n_unknown//2] = elem_fft_op_B_theta_n_1p[0][0]
        D3_comp_fft_op[0][-2] = (elem_fft_op_Yn1p + elem_fft_op_dp_Yn1p)[0][0]
        D3_comp_fft_op[0][-1] = elem_fft_op_B_theta_np1_0[0][0]
        # Axis 0 corresponds to the output
        full_tensor_fft_op = np.concatenate((full_tensor_fft_op, D3_comp_fft_op), axis=0)
    filtered_looped_fft_operator = np.transpose(full_tensor_fft_op, (0,2,1,3))
    # Finding the inverse differential operator
    filtered_inv_looped_fft_operator = np.linalg.tensorinv(filtered_looped_fft_operator)
    # (n_unknown(+1), len_tensor, n_unknown(+1), len_tensor)
    out_dict_tensor['filtered_looped_fft_operator'] = filtered_looped_fft_operator
    out_dict_tensor['filtered_inv_looped_fft_operator'] = filtered_inv_looped_fft_operator

    return(out_dict_tensor)

''' III. Calculating Delta_offset '''
# At even order, calculate B_psi[n-2,0], Y[n,0] and the average of Delta[n,0]
# (called 'Delta_offset' below)
# target_len_phi: target phi length of the solution
# coef_iota_nm1b2_in_Delta: Coefficient of iota (n-1)/2 in Delta. Is a constant
# in the Equilibrium object. Needed for odd orders.
def solve(n_unknown, target_len_phi,
    filtered_inv_looped_fft_operator, filtered_RHS_0_offset,
    coef_Delta_offset = 0
    ):
    out_dict_solve = {}
    # Solution with zero value for the free constant parameter
    filtered_solution = np.tensordot(filtered_inv_looped_fft_operator, filtered_RHS_0_offset, 2)
    # To have periodic B_psi, values for a constant free marameter, Delt_offset
    # must be found at even orders.
    # Integral(B_psi') = 0 is equivalent to filtered_solution[-1,0] = 0.
    # (-1: B_psi is always the second last row in the solution. 0: m=0 mode
    # in a np.fft.fft array.)
    # Because the looped ODE is now treated as a linear equation,
    # the Delta_offset dependence of filtered_solution[-1,0] is linear.

    # Making a blank ChiPhiFunc with the correct shape. The free parameter's
    # contribution only has 2 chi components, and will cause errors when
    # n_unknown>2.
    Delta_offset_unit_contribution = ChiPhiFunc(np.zeros((filtered_inv_looped_fft_operator.shape[0], filtered_inv_looped_fft_operator.shape[1])))
    # How much a Delta_offset of 1 shift -filtered_solution[-1,0]
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
        Delta_offset = -filtered_solution[-1,0]/sln_Delta_offset_unit_contribution[-1,0]
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
    eta, iota_coef,
    # lamnda for the coefficient of the scalar free parameter in RHS
    looped_coef_lambdas,
):
    if target_len_phi<max_freq*2:
        raise ValueError('target_len_phi must >= max_freq*2.')

    # First calculate RHS
    out_dict_RHS = generate_RHS(
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
    O_einv = out_dict_RHS['O_einv']
    vector_free_coef = out_dict_RHS['vector_free_coef']

    filtered_RHS_0_offset = out_dict_RHS['filtered_RHS_0_offset']
    out_dict_tensor = generate_tensor_operator(
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
        vector_free_coef = vector_free_coef,
        looped_coef_lambdas = looped_coef_lambdas
    )
    filtered_inv_looped_fft_operator =  out_dict_tensor['filtered_inv_looped_fft_operator']

    if n_unknown%2==0:
        # Solve for Delta n0.
        coef_Delta_offset = looped_coef_lambdas['lambda_coef_delta'](n_unknown+1)
        solve_result = solve(
            n_unknown=n_unknown,
            target_len_phi=target_len_phi,
            filtered_inv_looped_fft_operator=filtered_inv_looped_fft_operator,
            filtered_RHS_0_offset=filtered_RHS_0_offset,
            coef_Delta_offset=coef_Delta_offset
        )
        solution = solve_result['solution']
        dphi_B_psin0 = ChiPhiFunc(np.array([solution[-1]]))
        B_psin0 = dphi_B_psin0.integrate_phi(periodic=False)
        # Even order
        B_theta_n = B_theta_coef_cp[n_unknown][0]
        if n_unknown>2:
            B_theta_n_no_center_content = np.zeros((n_unknown-1, target_len_phi), np.complex128)
            B_theta_n_no_center_content[:n_unknown//2-1] = solution[:n_unknown//2-1]
            B_theta_n_no_center_content[n_unknown//2:] = solution[n_unknown//2-1:-2]
            B_theta_n += ChiPhiFunc(B_theta_n_no_center_content)

        # Calculating Yn
        coef_B_psi_dphi_1_in_Y_RHS = out_dict_tensor['coef_B_psi_dphi_1_in_Y_RHS']
        Yn_rhs_no_unknown = ChiPhiFunc(out_dict_RHS['Yn_rhs_content_no_unknown'])
        Yn_rhs = Yn_rhs_no_unknown + coef_B_psi_dphi_1_in_Y_RHS*dphi_B_psin0
        vec_free = solution[-2]
        Yn = ChiPhiFunc((np.einsum('ijk,jk->ik',O_einv,Yn_rhs.content) + vec_free * vector_free_coef))

        # Calculating Xn, Zn, pn, Delta_n
        Zn = (
            out_dict_RHS['Znm1_no_B_psi']
            +looped_coef_lambdas['lambda_B_psin0_in_Znm1'](n_unknown+1)*B_psin0
        )
        pn = (
            out_dict_RHS['pnm1_no_B_psi']
            +looped_coef_lambdas['lambda_B_psin0_in_pnm1'](n_unknown+1)*B_psin0
            +looped_coef_lambdas['lambda_dphi_B_psin0_in_pnm1'](n_unknown+1)*dphi_B_psin0
        )
        Xn = (
            out_dict_RHS['Xnm1_no_B_psi']
            +looped_coef_lambdas['lambda_B_psin0_in_Xnm1'](n_unknown+1)*B_psin0
            +looped_coef_lambdas['lambda_dphi_B_psin0_in_Xnm1'](n_unknown+1)*dphi_B_psin0
        )
        Deltan = (
            out_dict_RHS['Deltanm1_no_B_psi']
            +looped_coef_lambdas['lambda_B_psin0_in_Deltanm1'](n_unknown+1)*B_psin0
            +looped_coef_lambdas['lambda_dphi_B_psin0_in_Deltanm1'](n_unknown+1)*dphi_B_psin0
            -solve_result['Delta_offset']
        )
        return({
            'B_theta_n': B_theta_n,
            'Delta_offset': solve_result['Delta_offset'],
            'B_psi_nm2': out_dict_RHS['B_psi_n_n0_masked']+B_psin0,
            'Yn':Yn,
            'Xn':Xn,
            'Zn':Zn,
            'pn':pn,
            'Deltan':Deltan,
            'vec_free': vec_free,
            'O_einv': O_einv,
            'vector_free_coef': vector_free_coef
        })
    else:
        solve_result = solve(
            n_unknown=n_unknown,
            target_len_phi=target_len_phi,
            filtered_inv_looped_fft_operator=filtered_inv_looped_fft_operator,
            filtered_RHS_0_offset=filtered_RHS_0_offset,
        )
        solution = solve_result['solution']
        # Calculating Yn
        vec_free = solution[-2]
        Yn_rhs_content_no_unknown = out_dict_RHS['Yn_rhs_content_no_unknown']
        Yn = ChiPhiFunc(np.einsum('ijk,jk->ik',O_einv,Yn_rhs_content_no_unknown) + vec_free * vector_free_coef)
        return({
            'B_theta_n': ChiPhiFunc(solution[:-2]),
            'B_theta_np10': ChiPhiFunc(np.array([solution[-1]])),
            'vec_free': vec_free,
            'O_einv': O_einv,
            'vector_free_coef': vector_free_coef,
            'Yn': Yn,
            'filtered_RHS_0_offset': filtered_RHS_0_offset,# TODO: REMOVE!
            'filtered_inv_looped_fft_operator': filtered_inv_looped_fft_operator,# TODO: REMOVE!
        })

''' V. Utilities '''
''' V.1. Low-pass filter for simplifying tensor to invert '''
# Shorten an array in FFT representation to leave only target_length elements.
# by removing the highest frequency modes. The resulting array can be IFFT'ed.
def fft_filter(fft_in, target_length, axis):
    if target_length>fft_in.shape[axis]:
        raise ValueError('target_length should be smaller than the'\
                        'length of fft_in along axis.')
    elif target_length==fft_in.shape[axis]:
        return(fft_in)
    # FFT of an array contains mode amplitude in the order given by
    # fftfreq(length)*length. For example, for length=7,
    # [ 0.,  1.,  2.,  3., -3., -2., -1.]
    left = fft_in.take(indices=range(0, (target_length+1)//2), axis=axis)
    right = fft_in.take(indices=range(-(target_length//2), 0), axis=axis)
    return(np.concatenate((left, right), axis=axis)*target_length/fft_in.shape[axis])

# Pad an array in FFT representation to target_length elements.
# by adding zeroes as highest frequency modes.
# The resulting array can be IFFT'ed.
def fft_pad(fft_in, target_length, axis):
    if target_length<fft_in.shape[axis]:
        raise ValueError('target_length should be larger than the'\
                        'length of fft_in along axis.')
    elif target_length==fft_in.shape[axis]:
        return(fft_in)
    new_shape = list(fft_in.shape)
    original_length = new_shape[axis]
    new_shape[axis] = target_length - original_length
    center_array = np.zeros(new_shape)
    # FFT of an array contains mode amplitude in the order given by
    # fftfreq(length)*length. For example, for length=7,
    # [ 0.,  1.,  2.,  3., -3., -2., -1.]
    left = fft_in.take(indices=range(0, (original_length+1)//2), axis=axis)
    right = fft_in.take(indices=range(-(original_length//2), 0), axis=axis)
    return(np.concatenate((left, center_array, right), axis=axis)*target_length/fft_in.shape[axis])

# Cyclic import
import recursion_relations
