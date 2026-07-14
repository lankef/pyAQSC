'''
chiphifunc_solvers.py — linear-algebra and ODE-solving machinery that operates
on ChiPhiFunc content arrays.

None of the functions here are methods on ChiPhiFunc; they are algorithms that
receive ChiPhiFunc objects (or their raw content arrays) as arguments. Keeping
them here lets chiphifunc.py tell its own class story end-to-end without the
~700-line solver block interrupting it.

Sections
--------
I.   Deconvolution helpers (batch_matrix_inv_excluding_col, conv_tensor, ...)
II.  Chi-mode operator (get_O_O_einv_from_A_B)
III. Phi ODE solvers (solve_1d_fft, solve_ODE, solve_dphi_iota_dchi, ...)
IV.  Tensor constructors for looped_solver (to_tensor_fft_op, ...)
V.   Miscellaneous (linear_least_sq_2d_svd)
'''
import jax.numpy as jnp
from jax import vmap

from .chiphifunc import ChiPhiFunc
from .math_utilities import (
    dchi_op,
    trig_to_exp_op,
    exp_to_trig_op,
    jit_fftfreq_int,
    fft_filter,
    fft_pad,
)

# Maximum allowed asymptotic series order for solve_1d_asym (deprecated path).
asymptotic_order = 6


# ---------------------------------------------------------------------------
# I. Deconvolution / convolution helpers
# ---------------------------------------------------------------------------

''' I.1 Batch matrix inversion '''

def batch_matrix_inv_excluding_col(in_matrices: jnp.ndarray):
    '''
    Invert an (n,n) submatrix of a (n+2,n+1) rectangular matrix by taking the
    center n rows and excluding the rank_rhs column. "Taking the center n rows"
    is motivated by the RHS being rank n-1. Used for solving for Yn (an
    (n+1)-dimensional vector).

    Input:  in_matrices — (n+2, n+1, len_phi) array.
    Return: (n+2, n+1, len_phi) array with the center (n,n) block inverted
            and padded back to the original shape.
    '''
    if in_matrices.ndim != 3:
        raise ValueError(
            f"batch_matrix_inv_excluding_col: expected a 3d array, got ndim={in_matrices.ndim}. "
            "This used to silently return jnp.nan; converted to a loud error since a shape "
            "mismatch here usually means a ChiPhiFuncPadded operand reached this exact-rank-"
            "sensitive linear algebra without being cap_m()'d down to its true width first."
        )
    n_row = in_matrices.shape[0]
    n_col = in_matrices.shape[1]
    rank_rhs = n_col - 1
    ind_col = (rank_rhs + 1) // 2
    if n_row - 1 != n_col:
        raise ValueError(
            f"batch_matrix_inv_excluding_col: expected n_row-1==n_col, got n_row={n_row}, "
            f"n_col={n_col}. This used to silently return jnp.nan; converted to a loud error "
            "since a shape mismatch here usually means a ChiPhiFuncPadded operand reached this "
            "exact-rank-sensitive linear algebra without being cap_m()'d down to its true width "
            "first."
        )
    if n_row <= n_col:
        raise ValueError(
            f"batch_matrix_inv_excluding_col: expected n_row>n_col, got n_row={n_row}, "
            f"n_col={n_col}. Input should have more rows than cols."
        )
    rows_to_remove = (n_row - (n_col - 1)) // 2
    sub = jnp.delete(in_matrices, ind_col, axis=1)[rows_to_remove:-rows_to_remove, :, :]
    sub = jnp.moveaxis(sub, 2, 0)
    sqinv = jnp.linalg.inv(sub)
    sqinv = jnp.moveaxis(sqinv, 0, 2)
    padded = jnp.pad(sqinv, ((0, 0), (rows_to_remove, rows_to_remove), (0, 0)))
    return padded


def get_reduced_square_matrices(in_matrices: jnp.ndarray):
    '''
    Extract the square (n,n) matrices for solving linear systems by excluding
    a column and slicing the middle rows.

    Input:  in_matrices — (n+2, n+1, len_phi).
    Output: (len_phi, n, n+1) array (n_phi, n_col-1, n_col).
    '''
    n_row, n_col, n_phi = in_matrices.shape
    rank_rhs = n_col - 1
    ind_col = (rank_rhs + 1) // 2
    rows_to_remove = (n_row - (n_col - 1)) // 2
    sub = jnp.delete(in_matrices, ind_col, axis=1)[rows_to_remove:-rows_to_remove, :, :]
    sub = jnp.moveaxis(sub, 2, 0)
    return sub


''' I.2 Convolution tensor generators '''

# Vectorized jnp.roll over axis 0, batched over columns (axis 1).
roll_axis_0 = lambda a, shift: jnp.roll(a, shift, axis=0)
batch_roll_axis_0 = vmap(roll_axis_0, in_axes=1, out_axes=1)


def conv_tensor(content: jnp.ndarray, n_dim: int):
    '''
    Generate a (content.shape[0]+n_dim-1, n_dim, content.shape[1]) tensor
    that convolves a ChiPhiFunc content along chi (axis 0) with another of
    width n_dim. Used for multiplication in FFT space during ODE solves.

    Convolution: x2_conv_y2 = jnp.einsum('ijk,jk->ik', conv_x2, y2.content)

    Input:  content — (len_chi, len_phi); n_dim — column count for the target.
    Output: (len_chi+n_dim-1, n_dim, len_phi) convolution matrix stack.
    '''
    len_phi = content.shape[1]
    content_padded = jnp.concatenate((content, jnp.zeros((n_dim - 1, len_phi))), axis=0)
    content_padded = jnp.tile(content_padded[:, None, :], (1, n_dim, 1))
    shift = jnp.arange(n_dim)
    return batch_roll_axis_0(content_padded, shift[None, :])


roll_fft_last_axis = lambda a, shift: jnp.roll(a, shift, axis=-1)
batch_roll_fft_last_axis = vmap(roll_fft_last_axis, in_axes=(-2, 0), out_axes=2)


def fft_conv_tensor_batch(source: jnp.ndarray):
    '''
    Generates a 4D convolution operator in the phi axis for a 3D "tensor coef"
    (see looped_solver.py for explanation).

    Input:  source — (a, b, len_phi) array.
    Output: (a, b, len_phi, len_phi) operator that acts on a content by
            transposing axes 1 and 2 then tensordotting.
    '''
    len_phi = source.shape[2]
    arange = jnp.arange(len_phi)
    out = jnp.repeat(source[:, :, None, :], len_phi, axis=2)
    split = (len_phi + 1) // 2
    split_b = jnp.roll(jnp.arange(len_phi % 2, len_phi + len_phi % 2), split)
    out = batch_roll_fft_last_axis(out, arange)
    split_start = jnp.where(split_b < split, split_b, split)
    split_end = jnp.where(split_b < split, split, split_b)
    mask = jnp.where(
        jnp.logical_and(arange[None, :] >= split_start[:, None], arange[None, :] < split_end[:, None]),
        0, 1
    )
    out = out * mask[None, None, :, :]
    return jnp.transpose(out, (0, 1, 3, 2)) / len_phi


# ---------------------------------------------------------------------------
# II. Chi-mode operator (deconvolution for Yn)
# ---------------------------------------------------------------------------

def get_O_O_einv_from_A_B(chiphifunc_A: ChiPhiFunc, chiphifunc_B: ChiPhiFunc, rank_rhs: int, Y1c_mode: bool):
    '''
    Build O, O_einv and vector_free_coef that solve the equation system
    O Yn = (A + B dchi) Yn = RHS  <=>  Yn = O_einv@RHS - (Yn0 or Yn1p)*vec_free_coef.

    This is an under-determined problem. Yn is an (n+1)-dimensional vector.
    A and B are 2-d vectors. O is a known (n+2, n+1) convolution/differential
    matrix. The RHS (not provided here) is an (n+2) vector with (n)
    linearly-independent, phi-dependent components.

    In code:
        Yn_content = jnp.einsum('ijk,jk->ik', O_einv, chiphifunc_rhs_content)
                     + vec_free * vector_free_coef

    Inputs:  A, B — ChiPhiFunc coefficients; rank_rhs — number of RHS rows;
             Y1c_mode — whether to use the Y1c free variable convention.
    Outputs: O_matrices, O_einv, vector_free_coef (or (nan,)*4 on shape error).
    '''
    i_free = (rank_rhs + 1) // 2

    if chiphifunc_A.content.shape[1] != chiphifunc_B.content.shape[1] \
            and (chiphifunc_A.content.shape[1] != 1 and chiphifunc_B.content.shape[1] != 1):
        return jnp.nan, jnp.nan, jnp.nan, jnp.nan

    stretch_phi = jnp.zeros((1, max(chiphifunc_A.content.shape[1], chiphifunc_B.content.shape[1])))
    chiphifunc_A_content = chiphifunc_A.content + stretch_phi
    chiphifunc_B_content = chiphifunc_B.content + stretch_phi

    O_matrices = 0
    A_conv_matrices = conv_tensor(chiphifunc_A_content, rank_rhs + 1)
    O_matrices = O_matrices + A_conv_matrices

    dchi_matrix = dchi_op(rank_rhs + 1)
    B_conv_matrices = conv_tensor(chiphifunc_B_content, rank_rhs + 1)
    O_matrices = O_matrices + jnp.einsum('ijk,jl->ilk', B_conv_matrices, dchi_matrix)

    if Y1c_mode and rank_rhs % 2 == 1:
        O_matrices = jnp.einsum('ijp,jm->imp', O_matrices, trig_to_exp_op(O_matrices.shape[1]))
        O_matrices = jnp.einsum('ij,jmp->imp', exp_to_trig_op(O_matrices.shape[0]), O_matrices)

    O_einv = batch_matrix_inv_excluding_col(O_matrices)
    O_einv = jnp.concatenate((O_einv[:i_free], jnp.zeros((1, O_einv.shape[1], O_einv.shape[2])), O_einv[i_free:]))
    O_free_col = O_matrices[:, i_free, :]
    vector_free_coef = jnp.einsum('ijk,jk->ik', O_einv, O_free_col)
    vector_free_coef = vector_free_coef.at[i_free].set(-jnp.ones((vector_free_coef.shape[1])))

    if Y1c_mode and rank_rhs % 2 == 1:
        O_einv = jnp.einsum('ijp,jm->imp', O_einv, exp_to_trig_op(O_einv.shape[1]))
        O_einv = jnp.einsum('ij,jmp->imp', trig_to_exp_op(O_einv.shape[0]), O_einv)
        vector_free_coef = trig_to_exp_op(O_einv.shape[0]) @ vector_free_coef

    return O_matrices, O_einv, -vector_free_coef


# ---------------------------------------------------------------------------
# III. Phi ODE solvers
# ---------------------------------------------------------------------------

def fft_dphi_op(len_phi: int):
    '''
    dphi operator acting on the FFT of a content array along axis 1.
    Only used by solve_1d_fft.

    Input:  len_phi — number of phi grid points.
    Output: (len_phi, len_phi) diagonal matrix representing d/dphi in FFT space.
    '''
    fftfreq = jit_fftfreq_int(len_phi)
    return jnp.identity(len_phi) * 1j * fftfreq


def fft_conv_op(source):
    '''
    Convolution operator in FFT space that convolves a length-len_phi FFT with
    source. Only used by solve_1d_fft.

    Input:  source — 1d FFT array.
    Output: (len_phi, len_phi) convolution matrix.
    '''
    source_eff = source[None, None, :]
    tensor_eff = fft_conv_tensor_batch(source_eff)
    return tensor_eff[0][0]


def solve_1d_asym(p_eff, f_eff):
    '''
    Solves one linear ODE y' + p_eff*y = f_eff using an asymptotic series
    (deprecated — only kept for reference; the FFT method is preferred).

    When p_eff is 0, y is the anti-derivative of f with zero average.
    '''
    if jnp.array(p_eff).ndim == 0:
        p_eff = p_eff * jnp.ones_like(f_eff)
    ai = f_eff / p_eff
    asym_series = jnp.zeros((asymptotic_order + 1, len(ai))) + ai[None, :]
    for i in range(asymptotic_order):
        ai_new = -(ChiPhiFunc(ai[None, :], 1).dphi().content[0]) / p_eff
        ai = ai_new
        asym_series = asym_series.at[i + 1].set(ai_new)
    max_amp_for_each_term = jnp.max(jnp.abs(asym_series), axis=1)
    loc_smallest_max_amp = jnp.argmin(max_amp_for_each_term)
    asym_series = jnp.where(
        jnp.arange(len(max_amp_for_each_term))[:, None] > loc_smallest_max_amp,
        0, asym_series
    )
    return jnp.sum(asym_series, axis=0)


def solve_1d_fft(p_eff, f_eff, static_max_freq: int = None):
    '''
    Solves one linear ODE y' + p_eff*y = f_eff using the spectral (FFT) method.
    Assumes non-zero p. The p_eff==0 case is handled upstream in solve_ODE
    via jnp.where.

    Inputs: p_eff, f_eff — 1d arrays; static_max_freq — optional frequency cap.
    Output: solution array.
    '''
    len_phi = len(f_eff)
    if jnp.array(p_eff).ndim == 0:
        p_eff = p_eff * jnp.ones_like(f_eff)
    if static_max_freq is None or static_max_freq <= 0:
        target_length = len_phi
    else:
        target_length = min(len_phi, static_max_freq * 2)
    p_fft = fft_filter(jnp.fft.fft(p_eff), target_length, axis=0)
    f_fft = fft_filter(jnp.fft.fft(f_eff), target_length, axis=0)
    diff_matrix = fft_dphi_op(target_length)
    conv_matrix = fft_conv_op(p_fft)
    sln_fft = jnp.linalg.solve(diff_matrix + conv_matrix, f_fft)
    return jnp.fft.ifft(fft_pad(sln_fft, len_phi, axis=0), axis=0)


solve_1d_fft_batch = vmap(solve_1d_fft, in_axes=(0, 0, None), out_axes=0)


def solve_ODE(coeff_arr, coeff_dp_arr, f_arr: jnp.ndarray, static_max_freq: int = None):
    '''
    Solve simple linear first-order ODE systems in batch:
    (coeff_phi * d/dphi + coeff) y = f  =>  y' + p_eff*y = f_eff.

    Does not work well for p > 10 with zeros or resonant p.

    Inputs: coeff_arr, coeff_dp_arr, f_arr — 2d (n_eq x len_phi) matrices
            (all quantities assumed periodic); static_max_freq — optional cap.
    Output: solution array, same shape as f_arr.
    '''
    len_phi = f_arr.shape[1]
    if static_max_freq is None:
        static_max_freq = len_phi // 2
    f_eff = f_arr / coeff_dp_arr
    p_eff = coeff_arr / coeff_dp_arr
    if jnp.array(p_eff).ndim == 0:
        p_eff = p_eff + jnp.zeros_like(f_arr)
    if p_eff.shape[1] != f_eff.shape[1]:
        if p_eff.shape[1] == 1:
            p_eff = p_eff * jnp.ones_like(f_arr)
        else:
            return jnp.nan
    if p_eff.shape[0] != f_eff.shape[0]:
        return jnp.nan
    out_arr = jnp.where(
        (jnp.all(p_eff == 0, axis=1))[:, None],
        ChiPhiFunc(f_eff, nfp=1).integrate_phi_fft(zero_avg=True).content,
        solve_1d_fft_batch(p_eff, f_eff, static_max_freq)
    )
    return out_arr


def solve_ODE_chi(coeff, coeff_dp, coeff_dc, f, static_max_freq: int):
    '''
    Solve the periodic linear 1st-order ODE
    (coeff + coeff_dp*dphi + coeff_dc*dchi) y = f(phi, chi) via Fourier method.

    Inputs: coeffs are constants or content arrays; f is a content array.
    Output: solution content array.
    '''
    len_chi = f.shape[0]
    ind_chi = len_chi - 1
    mode_chi = 1j * jnp.linspace(-ind_chi, ind_chi, len_chi, axis=0)[:, None]
    coeff_eff = coeff_dc * mode_chi + coeff
    return solve_ODE(coeff_eff, coeff_dp, f, static_max_freq=static_max_freq)


def solve_dphi_iota_dchi(iota, f, static_max_freq: int):
    '''
    Solve the periodic linear 1st-order ODE
    (dphi + iota*dchi) y = f(phi, chi) via Fourier method.

    Inputs: iota — constant; f — ChiPhiFunc content array.
    Output: solution content array.
    '''
    return solve_ODE_chi(
        coeff=0,
        coeff_dp=1,
        coeff_dc=iota,
        f=f,
        static_max_freq=static_max_freq
    )


# ---------------------------------------------------------------------------
# IV. Tensor constructors for looped_solver.py
# ---------------------------------------------------------------------------

def to_tensor_fft_op(ChiPhiFunc_in: ChiPhiFunc, len_tensor: int):
    '''
    Build a 4D convolution tensor in FFT space from a ChiPhiFunc.
    Used in looped_solver.py and lambda_coefs_B_psi.py.

    Output: (1, 1, len_tensor, len_tensor) tensor operator.
    '''
    tensor_coef = ChiPhiFunc_in.content[:, None, :]
    tensor_fft_coef = fft_filter(jnp.fft.fft(tensor_coef, axis=2), len_tensor, axis=2)
    return fft_conv_tensor_batch(tensor_fft_coef)


def to_tensor_fft_op_multi_dim(
    ChiPhiFunc_in: ChiPhiFunc,
    dphi: int,
    dchi: int,
    num_mode: int,
    cap_axis0: int,
    len_tensor: int,
    nfp: int,
):
    '''
    Build a (cap_axis0, num_mode, len_tensor, len_tensor) convolution tensor
    in FFT space, optionally including chi/phi derivative factors.

    Used for n > 2 (n_eval > 3) recursion relations involving B_theta.

    Inputs:
        ChiPhiFunc_in — source coefficient.
        dphi, dchi    — derivative orders (positive = derivative, negative = integral).
        num_mode      — number of columns (target chi width).
        cap_axis0     — desired axis-0 length (must share parity with the raw tensor).
        len_tensor    — phi tensor dimension.
        nfp           — number of field periods.
    Output: (cap_axis0, num_mode, len_tensor, len_tensor) operator, or a NaN
            array on parity/range error.
    '''
    if ChiPhiFunc_in.nfp == 0:
        return 0
    len_chi = ChiPhiFunc_in.content.shape[0]
    tensor_coef_nD = conv_tensor(ChiPhiFunc_in.content, num_mode)
    if cap_axis0 % 2 != tensor_coef_nD.shape[0] % 2:
        return jnp.full((len_chi + num_mode - 1, num_mode, len_tensor, len_tensor), jnp.nan)
    if cap_axis0 > tensor_coef_nD.shape[0]:
        return jnp.full((len_chi + num_mode - 1, num_mode, len_tensor, len_tensor), jnp.nan)
    if tensor_coef_nD.shape[0] > cap_axis0:
        tensor_coef_nD = tensor_coef_nD[
            (tensor_coef_nD.shape[0] - cap_axis0) // 2:
            (tensor_coef_nD.shape[0] + cap_axis0) // 2
        ]
    if dchi != 0:
        dchi_array_temp = (1j * jnp.arange(-num_mode + 1, num_mode + 1, 2)[None, :, None])
        if dchi > 0:
            tensor_coef_nD = tensor_coef_nD * dchi_array_temp ** dchi
        elif dchi < 0:
            if num_mode % 2 == 0:
                tensor_coef_nD = tensor_coef_nD / dchi_array_temp ** (-dchi)
            else:
                return jnp.full((len_chi + num_mode - 1, num_mode, len_tensor, len_tensor), jnp.nan)
    tensor_fft_coef = fft_filter(jnp.fft.fft(tensor_coef_nD, axis=2), len_tensor, axis=2)
    tensor_fft_op = fft_conv_tensor_batch(tensor_fft_coef)
    if dphi != 0:
        if dphi < 0:
            return jnp.full((len_chi + num_mode - 1, num_mode, len_tensor, len_tensor), jnp.nan)
        fft_freq = jit_fftfreq_int(len_tensor)
        dphi_array = jnp.ones((len_tensor, len_tensor)) * 1j * fft_freq * nfp
        tensor_fft_op = tensor_fft_op * (dphi_array ** dphi)
    return tensor_fft_op


# ---------------------------------------------------------------------------
# V. Miscellaneous
# ---------------------------------------------------------------------------

def linear_least_sq_2d_svd(A, b):
    '''
    Solve the linear least-squares problem minimising ||Ax - b|| via SVD.

    Inputs: A — (n, m) matrix; b — (n,) vector.
    Output: x — (m,) solution vector.
    '''
    u, s, vh = jnp.linalg.svd(A, full_matrices=False)
    Eps_inv = jnp.diag(1 / s)
    return vh.conjugate().T @ Eps_inv @ u.conjugate().T @ b
