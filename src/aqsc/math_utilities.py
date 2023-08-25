import jax.numpy as jnp
# from jax import jit, vmap, tree_util
# from functools import partial # for JAX jit with static params

from math import floor, ceil
from .chiphifunc import *

# Sum: implemented as a function taking in a single-argument func and the lower/upper bounds
# expr should be non-dynamic.
# Non-jitted because an argument is a callable. A wrapper for a py_sum with no
# callable argument can be jitted.
def py_sum(expr, lower:int, upper:int):
    # The integer 0 cannot be added to even ChiPhiFuncs,
    # because JAX does not support conditionals on traced arguments.
    out = ChiPhiFuncSpecial(0)
    upper_floor = floor(upper)
    lower_ceil = ceil(lower)
    # If lower==upper then return expr(lower)
    if upper_floor==lower_ceil:
        return(expr(lower_ceil))
    # Warning for lower>upper
    if lower_ceil>upper_floor:
        # This is classified as "out of bound".
        # Originally the code is -1. Since the formula
        # are checked correct, these are made 0.
        return(ChiPhiFuncSpecial(0))
    indices = list(range(lower_ceil,upper_floor+1))
    out_list = jax.tree_util.tree_map(expr, indices)
    for item in out_list:
        out = out+item
    return(out)

# In the JAX implementation, there is no distinction between how the outmost and
# inner sums are evaluated.
py_sum_parallel = py_sum

## Condition operators

# Used to make sure new indices of terms and new upper bounds are within the
# bound of the original summations
# is_seq(a,b): 1 if a<=b
# @partial(jit, static_argnums=(0, 1,))
def is_seq(a, b):
    if a<=b:
        return(1)
    else:
        return(ChiPhiFuncSpecial(0))
# Used to ensure new index values (after removing the innermost sum) are integers.
# is_integer(a): 1 if a is integer
# @partial(jit, static_argnums=(0,))
def is_integer(a):
    if a%1==0:
        return(1)
    else:
        return(ChiPhiFuncSpecial(0))

# @partial(jit, static_argnums=(1, 2, ))
def diff_backend(y, is_chi:bool, order):
    '''
    Takes phi or chi derivative.
    Input:
    y: ChiPhiFunc or const
    is_chi: True for 'chi' or False for 'phi'
    order: order of derivative
    '''
    if jnp.isscalar(y):
        return(0)
    out = y
    if isinstance(y, ChiPhiFunc):
        if is_chi:
            out = out.dchi(order)
        else:
            out = out.dphi(order)
    else:
        return(ChiPhiFuncSpecial(-13))
    return(out)

# Maxima sometimes merges a few diff's together.
# @partial(jit, static_argnums=(1, 2, 3, 4,))
def diff(y, is_chi1:bool, order1:int, is_chi2=None, order2=None):
    out = diff_backend(y, is_chi1, order1)
    if is_chi2 is not None:
        out = diff_backend(out, is_chi2, order2)
    return(out)

# Faster. In this case, tensordot is faster than einsum.
def einsum_ijkl_jmln_to_imkn(array_A, array_B):
    if len(array_A.shape)!=4 or len(array_B.shape)!=4:
        return(jnp.nan)# Both input need to be 4d arrays
    # ikjl
    A_transposed = jnp.transpose(array_A, (0,2,1,3))
    # jlmn
    B_transposed = jnp.transpose(array_B, (0,2,1,3))
    # ikmn
    array_out = jnp.tensordot(A_transposed, B_transposed)
    return(jnp.transpose(array_out, (0,2,1,3)))
