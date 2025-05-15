import jax.numpy as jnp
from jax import jit, jacfwd
from jax.tree_util import tree_map
from jax.lax import while_loop
from jax import jit # vmap, tree_util
from functools import partial, reduce # for JAX jit with static params
from operator import add
from math import floor, ceil
from .chiphifunc import *

# # Sum: implemented as a function taking in a single-argument func and the lower/upper bounds
# # expr should be non-dynamic.
# # Non-jitted because an argument is a callable. A wrapper for a py_sum with no
# # callable argument can be jitted.
# def py_sum(expr, lower:int, upper:int):
#     # The integer 0 cannot be added to even ChiPhiFuncs,
#     # because JAX does not support conditionals on traced arguments.
#     out = ChiPhiFuncSpecial(0)
#     upper_floor = floor(upper)
#     lower_ceil = ceil(lower)
#     # If lower==upper then return expr(lower)
#     if upper_floor==lower_ceil:
#         return(expr(lower_ceil))
#     # Warning for lower>upper
#     if lower_ceil>upper_floor:
#         # This is classified as "out of bound".
#         # Originally the code is -1. Since the formula
#         # are checked correct, these are made 0.
#         return(ChiPhiFuncSpecial(0))
#     # This scan implementation may be faster, but the index is a 
#     # traced var in fori_loop. Because of that, conditionals like is_seq,
#     # as well as indexing in ChiPhiEpsFunc will not work.
#     # body_fun = lambda i, val: val + expr(i)
#     # out = fori_loop(lower_ceil, upper_floor+1, body_fun, 0)
#     # return(out)
#     indices = list(range(lower_ceil,upper_floor+1))
#     out_list = tree_map(expr, indices)
#     for item in out_list:
#         out = out+item
#     return(out)
def py_sum(expr, lower: int, upper: int):
    upper_floor = floor(upper)
    lower_ceil = ceil(lower)
    
    if upper_floor == lower_ceil:
        return expr(lower_ceil)
    
    if lower_ceil > upper_floor:
        return ChiPhiFuncSpecial(0)

    indices = range(lower_ceil, upper_floor + 1)

    # Unroll the loop statically using reduce + operator.add
    # This builds a single expression tree
    return reduce(add, (expr(i) for i in indices), ChiPhiFuncSpecial(0))


# In the JAX implementation, there is no distinction between how the outmost and
# inner sums are evaluated.
py_sum_parallel = py_sum

## Condition operators
# Used to make sure new indices of terms and new upper bounds are within the
# bound of the original summations
# is_seq(a,b): 1 if a<=b
# Using where, not if saves 10% compile time
def is_seq(a, b):
    return(jnp.where(a<=b, 1, 0))
    # if a<=b:
    #     return(1)
    # else:
    #     return(0)
        # return(ChiPhiFuncSpecial(0))

# Used to ensure new index values (after removing the innermost sum) are integers.
# is_integer(a): 1 if a is integer
# Redundant for the series we are considering, by observation.
# Removing the if statement saves 10% compile time.
def is_integer(a):
    return(1)
    # if a%1==0:
    #     return(1)
    # else:
    #     return(ChiPhiFuncSpecial(0))

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

def newton_solver_scalar(f, f_prime, x0, tol=1e-6, max_iter=100):
    """
    Newton iteration function using jax.while_loop to solve f(x) = 0.

    Parameters:
    f: Callable - Function whose root is to be found.
    f_prime: Callable - Derivative of the function.
    x0: float - Initial guess for the root.
    tol: float - Tolerance for convergence.
    max_iter: int - Maximum number of iterations.

    Returns:
    x: float - Approximation to the root.
    """
    def condition(state):
        _, fx, fpx, iter_count = state
        return (fx**2 > tol**2) & (iter_count < max_iter)

    def body(state):
        x, fx, fpx, iter_count = state
        # Update x using Newton's method: x_{n+1} = x_n - f(x) / f'(x)
        x_new = x - fx / fpx
        fpx_new = f_prime(x_new)
        fx_new = f(x_new)
        return x_new, fx_new, fpx_new, iter_count + 1

    # Initialize state: (x, f(x), f'(x), iteration counter)
    initial_state = (x0, f(x0), f_prime(x0), 0)

    x_final, _, _, _ = while_loop(condition, body, initial_state)
    return x_final


def newton_solver(f, x0, tol=1e-6, max_iter=100):
    """
    Newton's method for solving f(x) = 0.
    
    Args:
        f: Callable function that takes in a 1D array x and returns a 1D array of the same length.
        x0: Initial guess as a 1D JAX array.
        tol: Convergence tolerance.
        max_iter: Maximum number of iterations.
    
    Returns:
        x: The solution as a 1D JAX array.
        converged: Boolean indicating whether convergence was achieved.
        num_iter: Number of iterations performed.
    """
    def body(state):
        x, i = state
        J = jacfwd(f)(x)          # Jacobian matrix
        fx = f(x)
        delta_x = jnp.linalg.lstsq(J, -fx)[0]
        x_next = x + delta_x
        return x_next, i + 1
    
    def cond(state):
        x, i = state
        return (jnp.linalg.norm(f(x)) > tol) & (i < max_iter)
    
    x_final, num_iter = while_loop(cond, body, (x0, 0))
    converged = jnp.linalg.norm(f(x_final)) <= tol
    
    return x_final, converged, num_iter



@partial(jit, static_argnums=(2))
def fourier_interpolation(y_data, x_interp, nfp):
    """
    Perform Fourier interpolation of a periodic vector function.
    Very memory intensive compared to FFT interpolation in interpax! Use carefully.

    Parameters:
    y_data (jnp.ndarray): 2D array where axis 0 represents vector components, and axis 1 represents sampled points.
    x_interp (jnp.ndarray): Points where interpolation is desired.
    nfp (int): Number of field periods in the function.

    Returns:
    jnp.ndarray: Interpolated values with the same number of components as y_data along axis 0.
    """
    n_components, n_samples = y_data.shape
    period = 2 * jnp.pi / nfp

    # Compute Fourier coefficients for each component via FFT
    fft_coeffs = jnp.fft.fft(y_data, axis=1) / n_samples

    # Frequency indices
    k_values = jnp.fft.fftfreq(n_samples) * n_samples

    # Wrap interpolation points to [0, period)
    x_interp_mod = x_interp % period

    # Compute Fourier series interpolation
    print('k_values', k_values.shape)
    print('x_interp_mod', x_interp_mod.shape)
    print('x_interp_mod', x_interp.shape)
    exponentials = jnp.exp(1j * jnp.outer(k_values, x_interp_mod) * (2 * jnp.pi / period))
    
    print('exponentials', exponentials.shape)
    result = jnp.sum(fft_coeffs[:, :, None] * exponentials, axis=1).real

    return result

