import numpy as np

import jax.numpy as jnp
from jax import jit, vmap, tree_util
from functools import partial # for JAX jit with static params

from math import floor, ceil
from joblib import Parallel, delayed
import scipy.fftpack
import chiphifunc
import warnings
from functools import lru_cache

n_jobs = 8

# Sum: implemented as a function taking in a single-argument func and the lower/upper bounds
# Can run in parallel, but runs serial by default.
# expr should be non-dynamic.
def py_sum(expr, lower, upper, n_jobs=1, backend='threading'):
    out = 0
    upper_floor = floor(upper)
    lower_ceil = ceil(lower)
    # If lower==upper then return expr(lower)
    if upper_floor==lower_ceil:
        return(expr(lower_ceil))
    # Warning for lower>upper
    if lower_ceil>upper_floor:
        # warnings.warn('Warning: lower bound higher than upper bound in '+str(expr) \
        # +'. Bound values: lower='+str(lower)+', upper='+str(upper), RuntimeWarning)
        return(chiphifunc.ChiPhiFuncNull())
    if n_jobs<1:
        raise ValueError('n_jobs must not be smaller than 1')
    if n_jobs==1:
        for i in range(lower_ceil,upper_floor+1):
            out = out + expr(i)
    else: # Running parallel evaluation for sum arguments
        out_list = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(expr)(i) for i in range(lower, upper+1)
        )
        for a in out_list:
            out = out+a
    return(out)

def py_sum_parallel(expr, lower, upper):
    return(py_sum(expr, lower, upper, n_jobs=n_jobs))

## Condition operators

# Used to make sure new indices of terms and new upper bounds are within the
# bound of the original summations
# is_seq(a,b): 1 if a<=b
@partial(jit, static_argnums=(0, 1,))
def is_seq(a, b):
    if a<=b:
        return(1)
    else:
        return(ChiPhiFuncSpecial(0))
# Used to ensure new index values (after removing the innermost sum) are integers.
# is_integer(a): 1 if a is integer
@partial(jit, static_argnums=(0,))
def is_integer(a):
    if a%1==0:
        return(1)
    else:
        return(ChiPhiFuncSpecial(0))

# Takes phi or chi derivative.
# y: ChiPhiFunc or const
# x_name: 'chi' or 'phi'
# order: number of times to take derivative
def diff_backend(y, x_name, order):
    if np.isscalar(y):
        return(0)
    out = y

    if not isinstance(y, chiphifunc.ChiPhiFunc):
        raise AttributeError('Warning: diff is being evaluated on: '+str(type(y))+\
        '. This should not happen unless you are testing.')
        #
        # if x_name=='phi':
        #     dphi = lambda i_chi : scipy.fftpack.diff(y[i_chi], order=order)
        #     out = np.array(Parallel(n_jobs=8, backend='threading')(
        #         delayed(dphi)(i_chi) for i_chi in range(len(y))
        #     ))
        #
        # if x_name=='chi':
        #     dchi = lambda i_phi : scipy.fftpack.diff(y.T[i_phi], order=order)
        #     out = np.array(Parallel(n_jobs=8, backend='threading')(
        #         delayed(dchi)(i_phi) for i_phi in range(len(y.T))
        #     )).T
    else:
        if x_name=='phi':
            out = out.dphi(order=order)

        if x_name=='chi':
            for i in range(order):
                out = out.dchi()
    return(out)

# Maxima sometimes merges a few diff's together.
@lru_cache(maxsize=1000)
def diff(y, x_name1, order1, x_name2=None, order2=None):
    out = diff_backend(y, x_name1, order1)
    if x_name2 is not None:
        out = diff_backend(out, x_name2, order2)
    if type(out) is chiphifunc.ChiPhiFunc:
        out=out#.filter() # TODO: REPLACE WITH REGULARITY
    return(out)

# # integrate over chi.
# def int_chi(y):
#     if isinstance(y, chiphifunc.ChiPhiFunc):
#         return(y.antid_chi())
#     elif y == 0:
#         return(0)
#     else:
#         raise TypeError('Illegal int_chi argument: ' + str(y))

# Faster. In this case, tensordot is faster than einsum.
def einsum_ijkl_jmln_to_imkn(array_A, array_B):
    if len(array_A.shape)!=4 or len(array_B.shape)!=4:
        raise ValueError('Both input need to be 4d arrays')
    # ikjl
    A_transposed = np.transpose(array_A, (0,2,1,3))
    # jlmn
    B_transposed = np.transpose(array_B, (0,2,1,3))
    # ikmn
    array_out = np.tensordot(A_transposed, B_transposed)
    return(np.transpose(array_out, (0,2,1,3)))

# Slower than Einsum ijkl, jl -> ik. For reference only.
def einsum_ijkl_jl_to_ik(array_A, array_B):
    if len(array_A.shape)!=4 or len(array_B.shape)!=2:
        raise ValueError('Both input need to be 4d arrays')
    # ikjl
    A_transposed = np.transpose(array_A, (0,2,1,3))
    # ikmn
    array_out = np.tensordot(A_transposed, array_B)
    return(array_out)
