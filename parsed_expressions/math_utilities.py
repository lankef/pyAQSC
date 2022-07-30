import numpy as np
from math import floor, ceil
from joblib import Parallel, delayed
from numba import jit, njit, prange
from numba import int32, bool_, float32
import chiphifunc
import warnings
# import sys
# import traceback
#
# # Implementation of warning with traceback to diagnose where index out of bound occurs.
# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#
#     log = file if hasattr(file,'write') else sys.stderr
#     traceback.print_stack(file=log)
#     log.write(warnings.formatwarning(message, category, filename, lineno, line))
#
# warnings.showwarning = warn_with_traceback

# Sum: implemented as a function taking in a single-argument func and the lower/upper bounds
# Can run in parallel.
def py_sum(expr, lower, upper, n_jobs=2, backend='threading'):
    out = 0
    upper_floor = floor(upper)
    lower_ceil = ceil(lower)
    if upper_floor==lower_ceil:
        return(expr(lower_ceil))
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

# dummy for testing parser
def diff(y, x_name , order):
    return(y)

def diff_ChiPhiFunc(y, x_name, order):
    if order == 'chi':
        diff_matrix = ChiPhiFunc.diff_chi_op(y.get_shape()[0])
    elif order == 'phi':
        diff_matrix = ChiPhiFunc.diff_phi_op(y.get_shape()[1])
    else:
        raise ValueError('x_name must be \'chi\' or \'phi\'')
    operator = np.linalg.matrix_power(diff_matrix, order)
    return(operator@y)
