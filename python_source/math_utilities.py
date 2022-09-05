import numpy as np
from math import floor, ceil
from joblib import Parallel, delayed
from numba import jit, njit, prange
from numba import int32, bool_, float32
import scipy.fftpack
import chiphifunc
import warnings

# int_chi() should not be run on a ChiPhiFuncGrid with a chi-independent component,
# because this produces a non-periodic function. However, zero-checking the
# component is not feasible, because cancellation is often not exact in numerical
# evaluations. Instead, we check if the maximum amplitude of the chi-independent
# component is greater than this noise_level_int
noise_level_int = 1e-5

# Sum: implemented as a function taking in a single-argument func and the lower/upper bounds
# Can run in parallel.
def py_sum(expr, lower, upper, n_jobs=2, backend='threading'):
    out = 0
    upper_floor = floor(upper)
    lower_ceil = ceil(lower)
    # If lower==upper then return expr(lower)
    if upper_floor==lower_ceil:
        return(expr(lower_ceil))
    # Warning for lower>upper
    if lower_ceil>upper_floor:
        warnings.warn('Warning: lower bound higher than upper bound in '+str(expr) \
        +'. Bound values: lower='+str(lower)+', upper='+str(upper), RuntimeWarning)
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

## Condition operators

# Used to make sure new indices of terms and new upper bounds are within the
# bound of the original summations
# is_seq(a,b): 1 if a<=b
def is_seq(a, b):
    if a<=b:
        return(1)
    else:
        return(0)
# Used to ensure new index values (after removing the innermost sum) are integers.
# is_integer(a): 1 if a is integer
def is_integer(a):
    if a%1==0:
        return(1)
    else:
        return(0)

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
def diff(y, x_name1, order1, x_name2=None, order2=None):
    out = diff_backend(y, x_name1, order1)
    if x_name2 is not None:
        out = diff_backend(out, x_name2, order2)
    return(out)

#
def int_chi(y):
    if isinstance(y, chiphifunc.ChiPhiFuncGrid):
        len_chi = y.get_shape()[0]
        if len_chi%2==1\
        and np.max(np.abs(y.content[len_chi//2]))>noise_level_int:
            raise ValueError('Integrand has a significant chi-independent '\
            'component!')
        return(chiphifunc.ChiPhiFuncGrid(chiphifunc.dchi_op(len_chi, True) @ y.content))
    elif y == 0:
        return(0)
    else:
        raise TypeError('Illegal int_chi argument: ' + str(y))
