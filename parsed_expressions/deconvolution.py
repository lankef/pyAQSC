import numpy as np
import timeit
import scipy.signal
from matplotlib import pyplot as plt
from numba import jit, njit, prange
from numba import complex128, int64    # import jit value types
from functools import lru_cache # import functools for caching
import warnings

""" Solver for A*va = B*vb """
# This part solves the pointwise product function problem A*va = B*vb woth unknown va.
# by chi mode matching. It treats both pointwise products as matrix
# products of vectors (components are chi mode coeffs, which are rows in ChiPhiFunc.content)
# with convolution matrices A@va = B@vb,
# Where A, B are (m, n+1) (m, n) or (m, n) (m, n) matrices
#    of rank     (n+1)    (n)    or (n)    (n)
#    va, vb are  (n+1)    (n)    or (n)    (n)   -d vectors
#
# These type of equations underlies power-matched recursion relations:
# a(chi, phi) * va(chi, phi) = b(chi, phi) * vb(chi, phi)
# in grid realization.
# where a, b has the same number of chi modes (let's say o), as pointwise product
# with a, b are convolutions, which are (o+(n+1)-1, n+1) or (o+n-1, n) degenerate
# matrices.

""" I - Solver for 1D deconvolution A@va = B@vb """
# This part solves the 1D deconvolution problem where A, B are 2D matrices.
# Used in ChiPhiFuncGrid.

""" I.1 - va, vb with the same number of dimensions """
# Invert an (n,n) submatrix of a (m>n,n) rectangular matrix by taking the first
# n rows. "Taking the first n rows" is motivated by the RHS being rank n.
#
# -- Input --
# (m,n) matrix A
#
# -- Return --
# (m,m) matrix A_inv
@njit(complex128[:,:](complex128[:,:]))
def inv_square_jit(in_matrix):
    if in_matrix.ndim != 2:
        raise ValueError("Input should be 2d array")

    n_row = in_matrix.shape[0]
    n_col = in_matrix.shape[1]
    if n_row<=n_col:
        raise ValueError("Input should have more rows than cols")

    # Remove specfied column (slightly faster than delete)
    # and remove extra rows
    sqinv = np.linalg.inv(in_matrix[:n_col, :])

    padded = np.zeros((n_row, n_row), dtype = np.complex128)
    padded[:len(sqinv), :len(sqinv)] = sqinv
    return(padded)

# Solve degenerate underdetermined equation system A@va = B@vb, where A, B
# are (m,n+1) (m,n) matrices (m>n+1) of rank n, vb is a n-dim vector, and
# va is an n+1-dim vector with a free element vai at i.
#
# -- Input --
# (m,n+1), (m,n), rank-n 2d np arrays A, B
# n-dim np array-like vb
# or
# (m,n+1), rank n+1 A,
# m-dim np array-like v_rhs
#
# vb can be any array-like item, and is not necessarily 1d.
# Implemented with ChiPhiFunc in mind.
#
# -- Return --
# n+1 np array-like va
#
# -- Note --
# For recursion relations with ChiPhiFunc's, A and B should come from
# convolution matrices. That still needs implementation.
@njit(complex128[:](complex128[:,:], complex128[:]))
def solve_degenerate_jit(A, v_rhs):
    n_dim = A.shape[1]
    if A.shape[0] != v_rhs.shape[0]:
        raise ValueError("solve_underdetermined: A, v_rhs must have the same number of rows")
    A_inv = np.ascontiguousarray(inv_square_jit(A))
    # This vector is actually m-dim, with m-n blank elems at the end.
    va = (A_inv@np.ascontiguousarray(v_rhs))[:n_dim]
    return(va)

# @njit(complex128[:](complex128[:,:], complex128[:,:], complex128[:]))
# def solve_degenerate_jit(A, B, vb):
#     B_cont = np.ascontiguousarray(B)
#     vb_cont = np.ascontiguousarray(vb)
#     return(solve_degenerate_jit(A, B_cont@vb_cont))

""" I.2 - va has 1 more component than vb """
# Invert an (n,n) submatrix of a (m>n+1,n+1) rectangular matrix by taking the first
# n-1 rows and excluding the ind_col'th column. "Taking the first n rows" is motivated
# by the RHS being rank n-1
#
# -- Input --
# (m,n+1) matrix A
# ind_col < n+1
#
# -- Return --
# (m,m) matrix A_inv
@njit(complex128[:,:](complex128[:,:], int64))
def inv_square_excluding_col_jit(in_matrix, ind_col):
    if in_matrix.ndim != 2:
        raise ValueError("Input should be 2d array")

    n_row = in_matrix.shape[0]
    n_col = in_matrix.shape[1]
    if n_row<=n_col:
        raise ValueError("Input should have more rows than cols")

    if ind_col>=n_col:
        raise ValueError('ind_col should be smaller than column number')

    # Remove specfied column (slightly faster than delete)
    # and remove extra rows
    sub = in_matrix[:,np.arange(in_matrix.shape[1])!=ind_col][:n_col-1, :]
    sqinv = np.linalg.inv(sub)

    padded = np.zeros((n_row, n_row), dtype = np.complex128)
    padded[:len(sqinv), :len(sqinv)] = sqinv
    return(padded)

# Solve degenerate underdetermined equation system A@va = B@vb, where A, B
# are (m,n+1) (m,n) matrices (m>n+1) of rank n, vb is a n-dim vector, and
# va is an n+1-dim vector with a free element vai at i.
#
# -- Input --
# (m,n+1), (m,n), rank-n 2d np arrays A, B
# n-dim np array-like vb
# or
# (m,n+1), rank n+1 A,
# m-dim np array-like v_rhs
#
# vb can be any array-like item, and is not necessarily 1d.
# Implemented with ChiPhiFunc in mind.
#
# -- Return --
# n+1 np array-like va
#
# -- Note --
# For recursion relations with ChiPhiFunc's, A and B should come from
# convolution matrices. That still needs implementation.
@njit(complex128[:](complex128[:,:], complex128[:], int64, complex128))
def solve_degenerate_underdetermined_jit(A, v_rhs, i_free, vai):
    n_dim = A.shape[1]-1
    if A.shape[0] != v_rhs.shape[0]:
        raise ValueError("solve_underdetermined: A, v_rhs must have the same number of rows")
    A_einv = np.ascontiguousarray(inv_square_excluding_col_jit(A, i_free))
    A_free_col = np.ascontiguousarray(A.T[i_free])
    va_free_coef = (A_einv@A_free_col)
    # This vector is actually m-dim, with m-n blank elems at the end.
    va_fixed = (A_einv@np.ascontiguousarray(v_rhs) - vai * va_free_coef)[:n_dim]
    return(np.concatenate((va_fixed[:i_free], np.array([vai]) , va_fixed[i_free:])))

# @njit(complex128[:](complex128[:,:], complex128[:,:], complex128[:], int64, complex128))
# def solve_degenerate_underdetermined_jit(A, B, vb, i_free, vai):
#     B_cont = np.ascontiguousarray(B)
#     vb_cont = np.ascontiguousarray(vb)
#     return(solve_degenerate_underdetermined_jit(A, B_cont@vb_cont, i_free, vai))

""" I.3 - Convolution operator generator and ChiPhiFuncGrid.content numba wrapper """
# Generate convolution operator from a for an n_dim vector.
# Can't be compiled for parallel beacuase vec and out_transposed's sizes dont match?
@njit(complex128[:,:](complex128[:], int64))
def conv_matrix(vec, n_dim):
    out_transposed = np.zeros((n_dim,len(vec)+n_dim-1), dtype = np.complex128)
    for i in prange(n_dim):
        out_transposed[i, i:i+len(vec)] = vec
    return(out_transposed.T)

# For solving a*va = v_rhs, where va, vb have the same number of dimensions.
# In the context below, "#dim" represents number of chi mode components.
#
# -- Input --
# v_source_A: 2d matrix, content of ChiPhiFuncGrid, #dim = a
# v_rhs: 2d matrix, content of ChiPhiFuncGrid, #dim = m
# rank_rhs: int, rank of v_rhs.
#     Think of the problem A@va = B@vb, where
#     A and B are convolution matrices with the same row number.
#     n_dim_rhs is the dimensionality of vb. In a recursion relation,
#     this represents the highest mode number appearing in RHS.
#     The following relation must be satisfied:
#     a + #dim_va - 1 = m
#     a + (rank_rhs+1) - 1 = m
# i_free: int, the index of va's free element. Note that #dim_va = rank_rhs + 1.
# vai:  2d matrix with a single row, content of ChiPhiFuncGrid
#    represents a function of only phi given on grid.
#
# -- Output --
# va: 2d matrix, content of ChiPhiFuncGrid. Has #dim = rank_rhs+1.
@njit(complex128[:,:](complex128[:,:], complex128[:,:], int64, int64, complex128[:,:]), parallel=True)
def batch_underdetermined_degen_jit(v_source_A, v_rhs, rank_rhs, i_free, vai):
    A_slices = np.ascontiguousarray(v_source_A.T) # now the axis 0 is phi grid
    v_rhs_slices = np.ascontiguousarray(v_rhs.T) # now the axis 0 is phi grid
    # axis 0 is phi grid, axis 1 is chi mode
    va_transposed = np.zeros((len(A_slices), rank_rhs+1), dtype = np.complex128)
    if len(A_slices) != len(v_rhs_slices):
        raise ValueError('batch_underdetermined_deconv: A, v_rhs must have the same number of phi grids.')
    if len(v_source_A) + rank_rhs != len(v_rhs):
        raise ValueError('batch_underdetermined_deconv: #dim_A + rank_rhs = #dim_v_rhs does not hold.')
    for i in prange(A_slices.shape[0]):
        A_conv_matrix_i = conv_matrix(A_slices[i], rank_rhs+1)
        va_transposed[i, :] = solve_degenerate_underdetermined_jit(A_conv_matrix_i,\
                                         v_rhs_slices[i], i_free, np.ravel(vai)[i])
    return va_transposed.T

# For solving a*va = v_rhs, where va, vb have the same number of dimensions.
# In the context below, "#dim" represents number of chi mode components.
#
# -- Input --
# v_source_A: 2d matrix, content of ChiPhiFuncGrid, #dim = a
# v_rhs: 2d matrix, content of ChiPhiFuncGrid, #dim = m
# rank_rhs: int, rank of v_rhs.
#     Think of the problem A@va = B@vb, where
#     A and B are convolution matrices with the same row number.
#     n_dim_rhs is the dimensionality of vb. In a recursion relation,
#     this represents the highest mode number appearing in RHS.
#     The following relation must be satisfied:
#     a + rank_rhs - 1 = m
# -- Output --
# va: 2d matrix, content of ChiPhiFuncGrid. Has #dim = rank_rhs
@njit(complex128[:,:](complex128[:,:], complex128[:,:], int64), parallel=True)
def batch_degen_jit(v_source_A, v_rhs, rank_rhs):
#     if type(v_source_A) is not ChiPhiFuncGrid or type(v_source_B) is not ChiPhiFuncGrid:
#         raise TypeError('batch_underdetermined_deconv: input should be ChiPhiFuncGrid.')
    A_slices = np.ascontiguousarray(v_source_A.T) # now the axis 0 is phi grid
    v_rhs_slices = np.ascontiguousarray(v_rhs.T) # now the axis 0 is phi grid
    # axis 0 is phi grid, axis 1 is chi mode
    va_transposed = np.zeros((len(A_slices), rank_rhs), dtype = np.complex128)
    if len(A_slices) != len(v_rhs_slices):
        raise ValueError('batch_underdetermined_deconv: A, v_rhs must have the same number of phi grids.')
    if len(v_source_A) + rank_rhs - 1 != len(v_rhs):
        raise ValueError('batch_underdetermined_deconv: #dim_A + rank_rhs - 1 = #dim_v_rhs must hold.')
    for i in prange(A_slices.shape[0]):
        A_conv_matrix_i = conv_matrix(A_slices[i], rank_rhs)
        va_transposed[i, :] = solve_degenerate_jit(A_conv_matrix_i,v_rhs_slices[i])
    return va_transposed.T
