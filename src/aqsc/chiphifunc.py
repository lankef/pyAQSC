import jax.numpy as jnp
from jax import vmap, tree_util
# from jax import jit, vmap, tree_util
# from functools import partial # for JAX jit with static params

import math # factorial in dphi_direct
from matplotlib import pyplot as plt

# Configurations
from .config import *

''' Debug options and loading configs '''

# Loading configurations
if double_precision:
    import jax.config
    jax.config.update("jax_enable_x64", True)

# Maximum allowed asymptotic series order for y'+py=f
# This feature is depreciated and no longer included in py.
asymptotic_order = 6

''' I. Representing functions of chi and phi (ChiPhiFunc subclasses) '''
# Represents a function of chi and phi.
# Manages an complex128[m, n] 2d array called content.
# Axis 0 represents "m". Its length is n+1 for a nth-order term:
# each n-th order known term has n+1 non-zero coeffs due to regularity cond.
# F(chi, phi) = sum of Chi_coeff_-m*exp(-im phi) + Chi_coeff_-m*exp(-im phi)
# m=0, 2, 4... n or 1, 3, ... n)
# [                           # [
#     [Chi_coeff_-n(phi)],    #     [Chi_coeff_-n(phi)],
#     ...                     #     ...
#     [Chi_coeff_-2(phi)],    #     [Chi_coeff_-1(phi)],
#     [const(phi)],           #     [Chi_coeff_1(phi)],
#     [Chi_coeff_2(phi)],     #     ...
#     ...                     #     [Chi_coeff_n(phi)]
#     [Chi_coeff_n(phi)]      # ] for odd n
# ] for even n

# Axis 1 stores representation of a phi function as values on uniformly spaced
# grid points from phi = 0 to phi = 2pi(n_grid-1)/n_grid.
# Grid, rather than fourier representation is used because
# 1. The form of phi regularity is complicated
# 2. Grid rep favors multiplications, while Fourier rep favors diffs and integrals.
# Multiplications are simply more common.
# The number of phi modes will be tracked and cleaned up with a low-pass filter.

''' I.0 JIT methods used in the grid implementation '''
def dchi_op(n_dim:int):
    '''
    Generate chi differential operator diff_matrix. diff_matrix@f.content = dchi(f). 

    Input: -----

    n_dim: length of Chi series.

    Output: -----

    2d diagonal matrix with elements [-m, -m+2, ... m].
    '''
    ind_chi = n_dim-1
    mode_chi = jnp.linspace(-ind_chi, ind_chi, n_dim)
    return jnp.diag(1j*mode_chi)

def trig_to_exp_op(n_dim:int):
    ''' Converts a ChiPhiFunc from trig to exp fourier series. '''
    ones = jnp.ones(n_dim//2)
    if n_dim%2==0:
        arr_diag = jnp.concatenate([0.5j*ones, 0.5*ones])
        arr_anti_diag = jnp.concatenate([-0.5j*ones, 0.5*ones])
    if n_dim%2==1:
        arr_diag = jnp.concatenate([0.5j*ones, jnp.array([0.5]), 0.5*ones])
        arr_anti_diag = jnp.concatenate([-0.5j*ones, jnp.array([0.5]), 0.5*ones])
    return(jnp.diag(arr_diag)+jnp.flipud(jnp.diag(arr_anti_diag)))

def exp_to_trig_op(n_dim:int):
    ''' Converts a ChiPhiFunc from exp to trig fourier series. '''
    ones = jnp.ones(n_dim//2)
    # sin:  +0.5je^-ix-0.5je^ix
    # cos:  +0.5e^-ix+0.5e^ix
    if n_dim%2==0:
        arr_diag = jnp.concatenate([-1j*ones, ones])
        arr_anti_diag = jnp.concatenate([ones, 1j*ones])
    if n_dim%2==1:
        arr_diag = jnp.concatenate([-1j*ones, jnp.array([0.5]), ones])
        arr_anti_diag = jnp.concatenate([ones, jnp.array([0.5]), 1j*ones])
    return(jnp.diag(arr_diag)+jnp.flipud(jnp.diag(arr_anti_diag)))

def wrap_grid_content_jit(content:jnp.ndarray):
    '''
    Used for wrapping grid content. Defined outside the ChiPhiFunc class so that
    it can be used in @njit compiled methods.
    Input: -----

    (m, k) 2d array.

    Output: -----

    (m, k+1) 2d array. output[:, -1] = output[:, 0]
    '''
    first_col = content[:,0]
    return(jnp.concatenate((content, first_col[:,None]), axis=1))

def max_log10(input):
    '''
    Calculates the max amplitude's order of magnitude
    Input: -----

    An ndarray.

    Output: -----

    The log10 of the maximum element in input.
    '''
    return(jnp.log10(jnp.max(jnp.abs(input)))) # not nfp-sensitive

# Contains static argument.
# @partial(jit, static_argnums=(0,))
def jit_fftfreq_int(int_in:int):
    '''
    Shorthand for jnp.fft.fftfreq(n)*n rounded to the nearest int.
    The input should be static.

    Input: -----

    An integer equal to the length of an array (and its FFT)

    Output: -----

    np.fft.fft_freq(input)*input
    '''
    out = jnp.arange(int_in)
    return(jnp.where(out>(int_in-1)//2,out-int_in,out))

# Contains static argument.
# @partial(jit, static_argnums=(0,))
def ChiPhiFuncSpecial(error_code:int):
    '''
    Creates a special ChiPhiFunc that represents 0 or an error.
    This is necessary because JAX does not know the value of traced
    values at compile time. Having a special zero type allows simplifications
    like 0*n=0, and prevents even-oddness mismatch when zero odd order
    ChiPhiFunc's (which have even number of chi components) are treated as
    scalar 0. For example:

    odd + ChiPhiFuncSpecial(0)
    = ChiPhiFunc(nfp=odd) + ChiPhiFunc(nfp=0)
    = ChiPhiFunc(nfp=0)
    odd + 0
    = ChiPhiFunc(nfp=odd) + (traced int with unknown value)
    = illegal operation.

    Input: -----

    An integer. Must be smaller than or equal to 0. This represents the type
    of the special ChiPhiFunc and will be stored as its nfp.

    Output: -----

    A ChiPhiFunc with the given nfp (if error_code>0 then it's set to -2) and
    content=np.nan
    '''
    if error_code>0:
        return(ChiPhiFunc(jnp.nan, -2))
    return(ChiPhiFunc(jnp.nan, error_code)) # , is_special=True))

'''
Convolves 2 2d arrays along axis=0.
The inputs must have with equal lengths along axis=1
'''
batch_convolve = vmap(jnp.convolve, in_axes=1, out_axes=1)

def phi_avg(in_quant):
    '''
    A type-insensitive phi-averaging function that:
    - Averages along phi and output a ChiPhiFunc if the input is a ChiPhiFunc.
    - Does nothing if the input is a scalar.
    '''
    if isinstance(in_quant, ChiPhiFunc):
        # special ChiPhiFunc's
        if in_quant.is_special():
            return(in_quant)
        new_content = jnp.array([
            jnp.mean(in_quant.content,axis=1)
        ]).T
        return(ChiPhiFunc(new_content,in_quant.nfp))
    if not jnp.isscalar(in_quant):
        if in_quant.ndim!=0: # 0-d np array will check false for isscalar.
            return(ChiPhiFuncSpecial(-5))
    return(in_quant)

''' I.1 Grid implementation '''
class ChiPhiFunc:
    '''
    ChiPhiFunc represents a function of chi and phi in even(m = 0, 2, 4, ...)
    or odd (m = 1, 3, 5, ...) fourier series in chi. The coefficients are
    phi-dependent, and has nfp field periods. The phi dependence is stored
    on n grid points located at phi = (0, ... 2pi(n-1)/n/nfp).

    Members: -----

    1. (Traced) content: 2d ndarray, complex64 or complex 128 arrays storing a
    function of chi and phi. The chi dependence can be specified as either an
    exponential or trigonometric Fourier series. This is specified with
    the argument trig_mode in the constructor.
    The format of content is as follows:
    Exponential:
    [
        [ A_{-m}(phi = 0), ... A_{-m}(phi = 2pi(n-1)/n/nfp) ],
        [ A_{-(m-2)}(phi = 0), ... A_{-(m-1)}(phi = 2pi(n-1)/n/nfp) ],
        ...,
        [ A_{+m}(phi = 0), ... A_{+m}(phi = 2pi(n-1)/n/nfp) ],
    ]
    Trigonometric:
    [
        [ A^s_{m}(phi = 0), ... A^s_{-m}(phi = 2pi(n-1)/n/nfp) ],
        [ A^s_{m-1}(phi = 0), ... A^s_{m-1}(phi = 2pi(n-1)/n/nfp) ],
        ...,
        [ A^c_{+m}(phi = 0), ... A^c_{+m}(phi = 2pi(n-1)/n/nfp) ],
    ]

    2. (Static) nfp: int, number of field periods.

    It is necessary to represent values exactly known to be 0 as
    a special ChiPhiFunc, because JAX does not know the value of traced
    values at compile time. Having a special zero type allows simplifications
    like 0*n=0, and prevents even-oddness mismatch when zero odd order
    ChiPhiFunc's (which have even number of chi components) are treated as
    scalar 0. For example:

    odd + ChiPhiFuncSpecial(0)
    = ChiPhiFunc(nfp=odd) + ChiPhiFunc(nfp=0)
    = ChiPhiFunc(nfp=0)
    odd + 0
    = ChiPhiFunc(nfp=odd) + (traced int with unknown value)
    = illegal operation.

    Since JAX does not support exception raising, errors are also
    stored as special ChiPhiFunc. Special ChiPhiFunc's always have
    content=np.nan and nfp<=0:

    nfp = 0 -----
    Zero produced by non-traced conditionals in math_utilities, or quantities
    known to be zero. The former is needed because such conditionals enforces
    summation bounds, and need to cancel with out-of-bound power series
    coefficients. The latter is needed because *0 changes the even/oddness
    of a ChiPhiFunc, and need to be handled very carefully.

    Negative-nfp -----
    -1: Inconsistent even/oddness in ChiPhiEpsFUnc
    -2: Invalid/mismatched nfp
    -3: Invalid mode number
    -4: Mismatched even/oddness/direct division by chi-dependent ChiPhiFunc
    -5: Operation between ChiPhiFunc and another item that's not a
    constant or ChiPhiFunc
    -6: Operation between 2 ChiPhiFuncs with mismatching lengths.
    -7: Invalid content shape
    -8: /zero (special ChiPhiFunc)
    -9: Incorrect argument for pow
    -10: Incorrect argument for dchi
    -11: Incorrect argument for dphi
    -12: DEPRECIATED
    -13: Invalid object to diff()
    -14: Inconsistent type in ChiPhiEpsFunc
    -15: Insufficient argument in Equilibrium
    -16: Filter mode not recognized
    -17: Incorrect looped equation even/odd-ness

    When multiple error types are present, the error will be recorded as:
    100*error_a+10*error_b+error_c
    (example of error -4, -3 followed by -11: -40311.)
    '''
    def __init__(self, content:jnp.ndarray, nfp:int, trig_mode: bool=False):
        '''
        Constructor.

        Inputs: -----

        content: a 2d ndarray.

        nfp: number of field periods.

        trig_mode: Whether the ChiPhiFunc is given as trigonometric or exponential
        fourier series.
        When trig_mode=True, every row of a content array is treated as
        a sin/cos chi mode, rather than a e^(im phi) mode. The format for
        '''
        if nfp<=0:
            self.content = jnp.nan
            self.nfp = nfp
        else:
            if content.ndim!=2: # Checks content shape
                # self.is_special = True
                self.content = jnp.nan
                self.nfp = -7
            # Checks nfp type and sign
            elif nfp<=0 or not isinstance(nfp, int):
                # self.is_special = True
                self.content = jnp.nan
                self.nfp = -2
            # A ChiPhiFuncs satisfying all above conditions is legal
            else:
                # self.is_special = False
                self.nfp = nfp
                # Forcing complex128 type
                if double_precision:
                    content = content.astype(jnp.complex128)
                else:
                    content = content.astype(jnp.complex64)
                self.content = jnp.asarray(content)
                if trig_mode:
                    self.content = self.trig_to_exp().content

    def is_special(self):
        ''' Checks if a ChiPhiFunc is special. '''
        return(self.nfp<=0)

    def __str__(self):
        ''' For printing a ChiPhiFunc '''
        if self.is_special():
            if self.nfp==0:
                msg = 'conditional 0'
            else:
                msg = 'error '+str(self.nfp)
            return('ChiPhiFunc('+msg+')')
        return(
            'ChiPhiFunc(content.shape='+str(self.content.shape)+', nfp='+str(self.nfp)+')'
        )

    ''' Registers ChiPhiFunc as a pytree (nested list/dict) with JAX. '''
    def _tree_flatten(self):
        children = (self.content,)  # arrays / dynamic values
        aux_data = {'nfp': self.nfp}  # static values
        return (children, aux_data)
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    '''
    I.1.1 Operator overloads
    The following section contains overloads for algebraic operators. The
    outputs will always be ChiPhiFunc. Note that any term known to be 0
    will either be a ChiPhiFunc(nfp=0) or a ChiPhiFunc with content close to 0.
    The reasoning is explained in the constructor section.
    '''
    # Static argument
    def __getitem__(self, index: int):
        '''
        Obtains the m=index component of this ChiPhiFunc.
        DOES NOT WORK like list indexing.

        Input: -----

        index: m (mode number)

        Outputs: -----

        A ChiPhiFunc of shape (1, self.content.shape[1]).
        '''
        if self.is_special():
            return(self)
        len_chi = self.content.shape[0]
        # Checking even/oddness and total length
        if len_chi%2==index%2 or abs(index)>abs(len_chi-1):
            return(ChiPhiFuncSpecial(-3)) # invalid mode number
        new_content = jnp.array([self.content[len_chi//2+index//2]])
        if new_content.shape == (1,1):
            return new_content[0,0]
        return(ChiPhiFunc(new_content, self.nfp))

    def __neg__(self):
        '''
        Overloads the - operator. Returns a ChiPhiFunc with
        content=-self.content.
        '''
        # If self.is_special() is true, this still preserves the error message
        return ChiPhiFunc(-self.content, self.nfp, self.is_special())

    def __add__(self, other):
        '''
        Overloads the self + other operator. Internally, this is a sum of the
        two arguments' content with the center element of axis=0 aligned.
        Both ChiPhiFunc must have equal phi grid numbers and nfp. The legal combination of
        argument types are:
        even self + even ChiPhiFunc other
        even self + scalar or DeviceArrays(shape==[]) other
        odd self + odd ChiPhiFunc other
        even/odd self + ChiPhiFunc(nfp==0) other => self
        ChiPhiFunc(nfp==0) self + even/odd other => other
        even/odd self + ChiPhiFunc(nfp<0) other => ChiPhiFunc(nfp<0) other
        ChiPhiFunc(nfp<0) self + even/odd other => ChiPhiFunc(nfp<0) self
        ChiPhiFunc(nfp<=0) self + ChiPhiFunc(nfp<=0) other
            => ChiPhiFunc(self.nfp*100+other.nfp)
        '''
        if isinstance(other, ChiPhiFunc):
            if self.nfp==0:
                return(other)
            if other.nfp==0:
                return(self)
            if self.is_special():
                if other.is_special(): # Adding two nulls compound the error message
                    # if self.nfp==-1 or other.nfp==-1:
                    #     return(ChiPhiFuncSpecial(-1))
                    # else:
                    return(ChiPhiFuncSpecial(self.nfp*100+other.nfp))
                return(self)
            if other.is_special():
                return(other)
            if self.nfp!=other.nfp:
                return(ChiPhiFuncSpecial(-2)) # Mismatched shape
            if self.content.shape[0]%2 != other.content.shape[0]%2:
                return(ChiPhiFuncSpecial(-4)) # Mismatched even/oddness
            if self.content.shape[1] != other.content.shape[1]\
            and (self.content.shape[1]!=1 and other.content.shape[1]!=1):
                return(ChiPhiFuncSpecial(-6)) # Mismatched length
            a = self.content
            b = other.content
            # Now calculate the sum of two ChiPhiFuncs
            len_chi = max(a.shape[0], b.shape[0])
            a_pad_row = jnp.zeros(((len_chi - a.shape[0])//2, a.shape[1]))
            b_pad_row = jnp.zeros(((len_chi - b.shape[0])//2, b.shape[1]))
            a_padded = jnp.vstack([
                a_pad_row,
                a,
                a_pad_row
            ])
            b_padded = jnp.vstack([
                b_pad_row,
                b,
                b_pad_row
            ])
            return(ChiPhiFunc(a_padded+b_padded, self.nfp))
        else:
            if not jnp.isscalar(other):
                if other.ndim!=0: # 0-d np array will check false for isscalar.
                    return(ChiPhiFuncSpecial(-5))
            if self.is_special():
                if self.nfp==0:
                    return(other)
                return(self)
            if self.content.shape[0]%2==0:
                return(ChiPhiFuncSpecial(-4))

            # add to center
            center_loc = self.content.shape[0]//2
            updated_center = self.content[center_loc]+other
            updated_content = self.content.at[center_loc, :].set(updated_center)
            return(ChiPhiFunc(updated_content, self.nfp))

    def __radd__(self, other):
        ''' Overloads the other + self operator. See __add__() for details.'''
        return(self+other)

    def __sub__(self, other):
        ''' Overloads the self - other operator. See __add__() for details. '''
        return(self+(-other))

    def __rsub__(self, other):
        ''' Overloads the other - self operator. See __add__() for details. '''
        return(-(self-other))

    def __mul__(self, other):
        '''
        Overloads the self * other operator. Internally, this is a convolution
        of the two arguments' content along axis=0.
        Both ChiPhiFunc must have equal phi grid numbers and nfp. The legal
        combination of argument types are:
        even/odd self * even/odd ChiPhiFunc other
        even/odd self * scalar or DeviceArrays(shape==[]) other
        even/odd self * ChiPhiFunc(nfp==0) other => ChiPhiFunc(nfp==0) other
        ChiPhiFunc(nfp==0) self * even/odd other => ChiPhiFunc(nfp==0) self
        ChiPhiFunc(nfp==0) self * ChiPhiFunc(nfp==0) other
            => ChiPhiFunc(nfp==0) self
        ChiPhiFunc(nfp<0) self * even/odd other => ChiPhiFunc(nfp<0) self
        even/odd self * ChiPhiFunc(nfp<0) other => ChiPhiFunc(nfp<0) other
        ChiPhiFunc(nfp<0) * ChiPhiFunc(nfp<0) other
            => ChiPhiFunc(self.nfp*100+other.nfp)
        '''
        if isinstance(other, ChiPhiFunc):
            if self.nfp==0:
                if other.nfp<0: # nfp < 0 always means error message
                    return(other)
                return(self) # 0*out of bound and 0*non-trivial are both 0.
            if other.nfp==0:
                if self.nfp<0: # nfp < 0 always means error message
                    return(self)
                return(other) # 0*out of bound and 0*non-trivial are both 0.
            if self.is_special():
                if other.is_special(): # Adding two nulls compound the error message
                    # if self.nfp==-1 or other.nfp==-1:
                    #     return(ChiPhiFuncSpecial(-1))
                    # else:
                    return(ChiPhiFuncSpecial(self.nfp*100+other.nfp))
                return(self)
            if other.is_special():
                return(other)
            if self.nfp!=other.nfp:
                return(ChiPhiFuncSpecial(-2)) # Mismatched nfp
            if self.content.shape[1] != other.content.shape[1]\
            and (self.content.shape[1]!=1 and other.content.shape[1]!=1):
                return(ChiPhiFuncSpecial(-6)) # Mismatched length
            # One of self and other's content can have only 1 phi component.
            # Stretch both variables
            stretch_phi = jnp.zeros((1, max(self.content.shape[1], other.content.shape[1])))
            a = self.content+stretch_phi
            b = other.content+stretch_phi
            return(ChiPhiFunc(batch_convolve(a,b), self.nfp))
        else:
            if not jnp.isscalar(other):
                if other.ndim!=0: # 0-d np array will check false for isscalar.
                    return(ChiPhiFuncSpecial(-5))
            if self.is_special():
                return(self)
            return(ChiPhiFunc(other * self.content, self.nfp))


    def __rmul__(self, other):
        ''' Overloads the other * self operator. See __mul__() for details. '''
        return(self*other)

    # @jit
    def __truediv__(self, other):
        '''
        Overloads the self / other operator. "Division" in the context of NAE
        recursion relations is a deconvolution problem. For an other with no
        chi dependence, this operation is trivial, and for a chi-dependent
        other, this operation may be a exact or under-determined linear problem.
        Luckily, (see paper appendix for linear operations) there is only one
        case where "division" with chi-dependent other is necessary, in
        Yn.
        Both ChiPhiFunc must have equal phi grid numbers and nfp. The legal
        combination of argument types are:
        even/odd self / ChiPhiFunc other with one Chi component
        even/odd self / scalar or DeviceArrays(shape==[]) other
        even/odd self / ChiPhiFunc(nfp==0) other => /0 error
        ChiPhiFunc(nfp==0) self / even/odd other => ChiPhiFunc(nfp==0) self
        ChiPhiFunc(nfp==0) self * ChiPhiFunc(nfp==0) other
            => ChiPhiFunc(nfp==0) self
        ChiPhiFunc(nfp<0) self / even/odd other => ChiPhiFunc(nfp<0) self
        even/odd self / ChiPhiFunc(nfp<0) other => ChiPhiFunc(nfp<0) other
        ChiPhiFunc(nfp<0) self / ChiPhiFunc(nfp<0) other
            => ChiPhiFunc(self.nfp*100+other.nfp)
        '''
        if isinstance(other, ChiPhiFunc):
            # Handles zero/error, any/error and any/zero
            if other.is_special():
                if other.nfp==0:
                    return(ChiPhiFuncSpecial(-8)) # /zero error
                if self.is_special(): # dividing two nulls compound the error message
                    # if self.nfp==-1 or other.nfp==-1:
                    #     return(ChiPhiFuncSpecial(-1))
                    # else:
                    return(ChiPhiFuncSpecial(self.nfp*100+other.nfp))
                return(other)
            # Handles zero/(not error)
            if self.nfp==0:
                return(self) 
            # Mismatched nfp
            if self.nfp!=other.nfp:
                return(ChiPhiFuncSpecial(-2)) 
            # Mismatched shape
            if other.content.shape[0]!=1:
                return(ChiPhiFuncSpecial(-4))
            # Mismatched length
            if self.content.shape[1] != other.content.shape[1]\
            and (self.content.shape[1]!=1 and other.content.shape[1]!=1):
                return(ChiPhiFuncSpecial(-6)) 
            # Legal case and non-zero
            return(ChiPhiFunc(self.content/other.content, self.nfp))
        else:
            if self.nfp==0:
                return(self)
            if not jnp.isscalar(other):
                if other.ndim!=0: # 0-d np array will check false for isscalar.
                    return(ChiPhiFuncSpecial(-5))
            return(ChiPhiFunc(self.content/other, self.nfp))


    def __rtruediv__(self, other):
        '''
        Overloads the self / other operator. See __truediv__ for details.
        '''
        if self.is_special():
            # if self.nfp==-1:
            #     return(self)
            if self.nfp==0:
                return(ChiPhiFuncSpecial(-8))
            return(self)
        # handles wrong shape of self
        if self.content.shape[0]!=1:
            return(ChiPhiFuncSpecial(-4))
        else:
            if isinstance(other, ChiPhiFunc):
                # Handles zero/non-zero and error/non-zero
                if other.is_special():
                    return(other)
                # Handles non-zero/non-zero and error/non-zero
                return(ChiPhiFunc(other.content/self.content, self.nfp))
            else:
                if not jnp.isscalar(other):
                    if other.ndim!=0: # 0-d np array will check false for isscalar.
                        return(ChiPhiFuncSpecial(-5))
                return(ChiPhiFunc(other/self.content, self.nfp))


    def __rmatmul__(self, mat):
        '''
        Overloads the other @ self operator, for treating this object as a
        vector of Chi modes, and multiplying with a matrix. Directly multiplies
        other to self.content.
        '''
        return ChiPhiFunc(mat @ self.content, self.nfp)

    # Static argument
    # @partial(jit, static_argnums=(1,))
    def __pow__(self, other):
        '''
        Overloads the self ** other operator. Only supports integer other.
        A wrapper for multiple * operations.
        '''
        if self.is_special():
            return(self)
        new_content = self.content.copy()
        if other%1!=0:
            return(ChiPhiFuncSpecial(-9))
        if other == 0:
            return(1)
        for i in range(other-1):
            new_content = batch_convolve(new_content,self.content)
        return(ChiPhiFunc(new_content, self.nfp))

    ''' I.1.2 Derivatives, integrals and related methods '''

    def fft(self):
        ''' FFT the axis=1 of content and returns as a ChiPhiFunc '''
        return(ChiPhiFunc(jnp.fft.fft(self.content, axis=1), self.nfp))


    def ifft(self):
        ''' IFFT the axis=1 of content and returns as a ChiPhiFunc '''
        return(ChiPhiFunc(jnp.fft.ifft(self.content, axis=1), self.nfp))

    # Static argument
    # @partial(jit, static_argnums=(1,))
    def dchi(self, order=1):
        '''
        Derivative in chi.

        Input: -----

        order: order of derivative.

        Output: -----

        ChiPhiFunc.
        '''
        if self.is_special():
            return(self)
        len_chi = self.content.shape[0]
        if order<0:
            return(ChiPhiFuncSpecial(-10))
        mode_i = (1j*jnp.arange(-len_chi+1,len_chi+1,2)[:,None])**order
        return(ChiPhiFunc(mode_i * self.content, self.nfp))

    def antid_chi(self):
        '''
        Anti-derivative in chi. Ignores the m=0 component, if any.

        Output: -----

        ChiPhiFunc.
        '''
        if self.is_special():
            return(self)
        len_chi = self.content.shape[0]
        temp = jnp.arange(-len_chi+1,len_chi+1,2,dtype=jnp.float32)[:,None]
        if len_chi%2==1:
            temp = temp.at[len(temp)//2].set(jnp.inf)
        return(ChiPhiFunc(-1j * self.content/temp, self.nfp))

    # Static argument
    # @partial(jit, static_argnums=(1, 2,))
    def dphi(self, order:int=1, mode=0):
        if self.is_special():
            return(self)
        if order<0:
            return(ChiPhiFuncSpecial(-11))
        if mode==0:
            mode = diff_mode
        if mode==1:
            len_phi = self.content.shape[1]
            content_fft = jnp.fft.fft(self.content, axis=1)
            fftfreq_temp = jit_fftfreq_int(len_phi)*1j
            out_content_fft = content_fft*fftfreq_temp[None, :]**order
            out = jnp.fft.ifft(out_content_fft,axis=1)
        elif mode==2:
            out = self.content
            for i in range(order):
                out = (dphi_op_pseudospectral(self.content.shape[1]) @ out.T).T
        else:
            return(ChiPhiFuncSpecial(-11))
        return(ChiPhiFunc(out*self.nfp**order, self.nfp))

    def exp(self):
        '''
        Used to calculate e**(ChiPhiFunc). Only support ChiPhiFunc with no
        chi dependence.

        Outputs: -----

        ChiPhiFunc e**(self). Is chi-independent.
        '''
        if self.is_special():
            return(self)
        if self.content.shape[0]!=1:
            return(ChiPhiFuncSpecial(-7))
        return(ChiPhiFunc(jnp.exp(self.content), self.nfp))

    # Input is not order-dependent, and permitted to be static.
    # @partial(jit, static_argnums=(1,))
    def integrate_phi_fft(self, zero_avg):
        '''
        Phi-integrate a ChiPhiFunc over 0 to 2pi or 0 to a given phi.
        Has two choices of initial conditions, output(phi=0) = 0 or
        avg(output) = 0.

        Input: -----

        zero_avg: When True, use avg(output)=0. When False, use output(0)=0.

        Output: -----

        ChiPhiFunc
        '''
        if self.is_special():
            return(self)
        # number of phi grids
        len_chi = self.content.shape[0]
        len_phi = self.content.shape[1]
        if double_precision:
            phis = jnp.linspace(0, 2*jnp.pi*(1-1/len_phi), len_phi, dtype=jnp.complex128)
        else:
            phis = jnp.linspace(0, 2*jnp.pi*(1-1/len_phi), len_phi, dtype=jnp.complex64)
        # fft integral
        content_fft = jnp.fft.fft(self.content, axis=1)
        fftfreq_temp = jit_fftfreq_int(len_phi)*1j
        fftfreq_temp = fftfreq_temp.at[0].set(jnp.inf)
        out_content_fft = content_fft/fftfreq_temp[None, :]/self.nfp
        out_content = jnp.fft.ifft(out_content_fft,axis=1)
        # The fft.diff integral assumes zero average.
        if not zero_avg:
            out_content -= out_content[:,0][:,None]
            out_content += phis[None, :]*content_fft[:,0][:, None]/self.nfp/len_phi
        return(ChiPhiFunc(out_content, self.nfp))

    ''' I.1.3 phi Filters '''
    # Static arguments
    # @partial(jit, static_argnums=(2,))
    def filter(self, arg:float, mode:int=0):
        '''
        An expandable filter. Now only low-pass is available.

        Inputs: -----

        mode: filtering mode. Available modes are:
            0: low_pass.

        arg: filter for argument:
            Low-pass: cutoff frequency.

        Output: -----

        The filtered ChiPhiFunc.
        '''
        if self.is_special():
            return(self)
        if mode == 0:
            # Skip filtering if arg is negative.
            arg = jnp.where(arg<0, jnp.inf, arg)
            len_phi = self.content.shape[1]
            W = jnp.abs(jit_fftfreq_int(len_phi))
            f_signal = jnp.fft.fft(self.content, axis = 1)
            # If our original signal time was in seconds, this is now in Hz
            cut_f_signal = f_signal.copy()
            cut_f_signal = jnp.where(W[None, :]>arg, 0, cut_f_signal)
            return(ChiPhiFunc(jnp.fft.ifft(cut_f_signal, axis=1), self.nfp))
        else:
            return(ChiPhiFuncSpecial())

    # @partial(jit, static_argnums=(1,))
    def filter_reduced_length(self, arg:int):
        '''
        Low pass filter that reduces the length of a ChiPhiFunc.
        Inputs: -----

        arg: cut-off frequency.

        Output: -----

        The filtered ChiPhiFunc.
        '''
        if self.is_special():
            return(self)
        fft_content = jnp.fft.fft(self.content, axis = 1)
        short_fft_content = fft_filter(fft_content, arg*2, axis=1)
        short_content = jnp.fft.ifft(short_fft_content, axis=1)
        return(ChiPhiFunc(short_content, self.nfp))

    ''' I.1.4 Properties '''
    def get_amplitude(self):
        '''
        Getting the max amplitude of the content

        Outputs:

        A real scalar.
        '''
        if self.nfp==0:
            return(0)
        elif self.nfp<0:
            return(jnp.inf)
        return(jnp.max(jnp.abs(self.content)))

    # def real(self):
    #     '''
    #     Functions like jnp.real()

    #     Outputs:

    #     A real scalar.
    #     '''
    #     if self.is_special():
    #         return(self)
    #     return(ChiPhiFunc(j(self.content), self.nfp))

    # def imag(self):
    #     '''
    #     Functions like jnp.imag()

    #     Outputs:

    #     A real scalar.
    #     '''
    #     if self.is_special():
    #         return(self)
    #     return(ChiPhiFunc(jnp.imag(self.content), self.nfp))
    # @partial(jit, static_argnums=(1,))
    def cap_m(self, m:int):
        '''
        Takes the center m+1 rows of content. If the ChiPhiFunc
        contain less chi modes than m+1, It will be zero-padded.

        Inputs: -----

        m: number of m in the output.

        Outputs: -----

        A ChiPhiFunc with maximum chi mode number m (and m+1 chi components)
        '''
        if self.is_special():
            return(self)
        target_chi = m+1
        len_chi = self.content.shape[0]
        num_clip = len_chi - target_chi
        if target_chi == len_chi:
            return(self)
        if target_chi > len_chi:
            self.pad_m(m)
        if num_clip%2 != 0:
            return(ChiPhiFuncSpecial(-4))
        return(ChiPhiFunc(self.content[num_clip//2:-num_clip//2], self.nfp))

    def pad_m(self,m:int):
        '''
        Alias for pad_chi to have a known maximum m instead. Recall that the
        number of chi components in a ChiPhiFunc equals to m+1.
        '''
        return self.pad_chi(m+1)

    def pad_chi(self, target_chi:int):
        '''
        Pads a ChiPhiFunc to be have a given total number of mode components.
        Both self.content.shape[0] and must be even/odd. target_chi MUST BE
        larger than the current number of chi components in self.
        Note: This takes the total number of mode components, rather
        than m, because it's primarily used when solving the looped
        equations. It's sometimes less confusing to use the total number
        of mode components (equations) rather than the maximum mode number m
        during equation-counting.

        Input: -----

        target_chi: number of chi components in output. Must be greater than the
        number of chi components in self. Must have the same even/oddness as
        self.

        Outputs: -----

        ChiPhiFunc with target_chi chi components.
        '''
        if self.is_special():
            return(self)
        len_chi = self.content.shape[0]
        if len_chi%2!=target_chi%2:
            return(ChiPhiFuncSpecial(-4))
        if target_chi < len_chi:
            return(ChiPhiFuncSpecial(-12))
        padding = ChiPhiFunc(
            jnp.zeros((target_chi, self.content.shape[1])),
            self.nfp
        )
        return(self+padding)


    ''' I.1.5 Output and plotting '''
    def eval(self, chi, phi):
        '''
        Getting a 2d vectorized function, f(chi, phi) for plotting a ChiPhiFunc.

        Output: -----

        A vectorized callables f(chi, phi).
        '''
        len_chi = self.content.shape[0]
        len_phi = self.content.shape[1]

        # Create 'x' for interpolation. 'x' is wrapped for periodicity.
        phi_grid = jnp.linspace(0,2*jnp.pi/self.nfp*(1-1/len_phi), len_phi)

        # The outer dot product is summing along axis 0.
        out = 0
        for i in range(len_chi):
            out+=jnp.e**(1j*(chi)*(i*2-len_chi+1))\
                *jnp.interp(
                    phi, 
                    phi_grid, 
                    self.content[i], 
                    period=2*jnp.pi/self.nfp
                )
        return(out)

    def display_content(self, trig_mode=False, colormap_mode=False):
        '''
        Plot the content of a ChiPhiFunc.

        Input: -----

        trig_mode: bool. When True, plot the trig Chi fourier coefficients,
        rather than exponential

        colormap_mode: bool. When True, make colormaps. Otherwise makes line plots.
        '''

        if self.is_special():
            print('display_content(): input is ChiPhiFuncSpecial.')
            print(self)
            return()
        plt.rcParams['figure.figsize'] = [8,3]
        content = self.content
        if content.shape[1]==1:
            content = content*jnp.ones((1,100))
        len_phi = content.shape[1]
        phis = jnp.linspace(0,2*jnp.pi*(1-1/len_phi)/self.nfp, len_phi)
        if trig_mode:
            fourier = ChiPhiFunc(content, self.nfp).exp_to_trig()
            if len(fourier.content)%2==0:
                ax1 = plt.subplot(121)
                ax1.set_title('cos, nfp='+str(self.nfp))
                ax2 = plt.subplot(122)
                ax2.set_title('sin, nfp='+str(self.nfp))
                if colormap_mode:
                    modecos = jnp.linspace(1, len(fourier.content)-1, len(fourier.content)//2)
                    modesin = jnp.linspace(len(fourier.content)-1, 1, len(fourier.content)//2)
                    phi = jnp.linspace(0, 2*jnp.pi*(1-1/len_phi), len_phi)
                    ax1.pcolormesh(phi, modecos, jnp.real(fourier.content)[len(fourier.content)//2:])
                    ax2.pcolormesh(phi, modesin, jnp.real(fourier.content)[:len(fourier.content)//2])
                else:
                    ax1.plot(phis, jnp.real(fourier.content)[len(fourier.content)//2:].T)
                    ax2.plot(phis, jnp.real(fourier.content)[:len(fourier.content)//2].T)
            else:
                plt.rcParams['figure.figsize'] = [12,3]
                ax1 = plt.subplot(131)
                ax1.set_title('cos, nfp='+str(self.nfp))
                ax2 = plt.subplot(132)
                ax2.set_title('constant, nfp='+str(self.nfp))
                ax3 = plt.subplot(133)
                ax3.set_title('sin, nfp='+str(self.nfp))
                if colormap_mode and len(fourier.content) != 1:
                    modesin = jnp.linspace(2, len(fourier.content)-1, len(fourier.content)//2)
                    modecos = jnp.linspace(len(fourier.content)-1, 2, len(fourier.content)//2)
                    phi = jnp.linspace(0, 2*jnp.pi*(1-1/len_phi)/self.nfp, len_phi)
                    print('phi',phi.shape)
                    print('modesin',modesin.shape)
                    print('modecos',modecos.shape)
                    print('np.real(fourier.content)[len(fourier.content)//2+1:]', jnp.real(fourier.content)[len(fourier.content)//2+1:].shape)
                    ax1.pcolormesh(phi, modesin, jnp.real(fourier.content)[len(fourier.content)//2+1:])
                    ax3.pcolormesh(phi, modecos, jnp.real(fourier.content)[:len(fourier.content)//2])
                else:
                    ax1.plot(phis, jnp.real(fourier.content)[len(fourier.content)//2+1:].T)
                    ax3.plot(phis, jnp.real(fourier.content)[:len(fourier.content)//2].T)

                ax2.plot(phis, jnp.real(fourier.content)[len(fourier.content)//2])
        else:
            ax1 = plt.subplot(121)
            ax1.set_title('Real, nfp='+str(self.nfp))
            ax2 = plt.subplot(122)
            ax2.set_title('Imaginary, nfp='+str(self.nfp))

            if colormap_mode:
                mode = jnp.linspace(-len(content)+1, len(content)-1, len(content))
                phi = jnp.linspace(0, 2*jnp.pi*(1-1/len_phi)/self.nfp, len_phi)
                ax1.pcolormesh(phi, mode, jnp.real(content))
                ax2.pcolormesh(phi, mode, jnp.imag(content))
            else:
                ax1.plot(phis, jnp.real(content).T)
                ax2.plot(phis, jnp.imag(content).T)
        # if fname is None:
        plt.show()
        # else:
        #     plt.savefig(fname)

    def display(self, complex:bool=False, size=(100,100), avg_clim:bool=False):
        '''
        Plot a period in both chi and phi in colormap.

        Input: -----

        complex: bool, plot complex components

        size: (int, int), size of the plot

        avg_clim: limit color to only +- avg(self.content).
        '''
        plt.rcParams['figure.figsize'] = [4,3]
        # This would trigger an error for most complex,
        # static methods used for evaluation.
        chi = jnp.linspace(0, 2*jnp.pi*0.99, size[0])
        phi = jnp.linspace(0, 2*jnp.pi*0.99/self.nfp, size[1])
        eval = self.eval(chi, phi.reshape(-1,1))
        plt.pcolormesh(chi, phi, jnp.real(eval))
        plt.title('ChiPhiFunc, real component')
        plt.xlabel('chi')
        plt.ylabel('phi')
        if avg_clim:
            clim = jnp.average(jnp.abs(jnp.real(eval)))
            plt.clim(-clim, clim)
            plt.colorbar(extend='both')
        else:
            plt.colorbar()
        plt.show()
        if complex:
            plt.pcolormesh(chi, phi, jnp.imag(eval))
            plt.title('ChiPhiFunc, imaginary component')
            plt.xlabel('chi')
            plt.ylabel('phi')
            if avg_clim:
                clim = jnp.average(jnp.abs(jnp.imag(eval)))
                plt.clim(-clim, clim)
                plt.colorbar(extend='both')
            else:
                plt.colorbar()
            plt.show()

    def export_single_nfp(self):
        '''
        Outputs a ChiPhiFunc with just 1 nfp.
        '''
        new_content = jnp.tile(self.content, (1,self.nfp))
        return(ChiPhiFunc(new_content, 1))

    def trig_to_exp(self):
        ''' Converts a ChiPhiFunc from trig to exp fourier series. '''
        content=self.content
        n_dim = content.shape[0]
        return(ChiPhiFunc(trig_to_exp_op(n_dim)@content, self.nfp))

    def exp_to_trig(self):
        ''' Converts a ChiPhiFunc from exp to trig fourier series. '''
        content=self.content
        n_dim = content.shape[0]
        return(ChiPhiFunc(exp_to_trig_op(n_dim)@content, self.nfp))

# For JAX use. Registers ChiPhiFunc as a pytree.
tree_util.register_pytree_node(ChiPhiFunc,
                               ChiPhiFunc._tree_flatten,
                               ChiPhiFunc._tree_unflatten)

''' I.2 Utilities '''
# Used only in pseudosspectral method.
roll_axis_01 = lambda a, shift: jnp.roll(jnp.roll(a, shift, axis=0), shift, axis=1)
batch_roll_axis_01 = vmap(roll_axis_01, in_axes=0, out_axes=0)
# @lru_cache(maxsize=10)
# @partial(jit, static_argnums=(0,))
def dphi_op_pseudospectral(n:int):
    """
    Return the spectral differentiation matrix for n grid points
    on the periodic domain [xmax, xmax). This routine is a JAX port of
    qsc.spectral_diff_matrix in
    https://github.com/landreman/pyQSC/,
    and scipy.linalg.toeplitz in
    https://github.com/scipy/scipy.
    """
    h = 2 * jnp.pi / n
    kk = jnp.arange(1, n)
    n1 = n//2-(n+1)%2
    n2 = n//2
    if n%2 == 0:
        topc = 1 / jnp.tan(jnp.arange(1, n2 + 1) * h / 2)
        temp = jnp.concatenate((topc, -jnp.flip(topc[0:n1])))
    else:
        topc = 1 / jnp.sin(jnp.arange(1, n2 + 1) * h / 2)
        temp = jnp.concatenate((topc, jnp.flip(topc[0:n1])))

    # Calculating the Toeplitz matrix
    col1 = jnp.array([jnp.concatenate((jnp.array([0]), 0.5 * ((-1) ** kk) * temp))]).T
    col1 = jnp.concatenate([col1, jnp.zeros((n, n-1))], axis=1)
    row1 = -col1.T
    # Creating the first row and column
    raw = col1 + row1
    # Creating a new axis and shifting the rows and columns
    toeplitz = jnp.repeat(raw[None,:,:], n, axis=0)
    shifts = jnp.arange(n)
    # Rolling the first row and first col
    toeplitz = batch_roll_axis_01(toeplitz, shifts)
    # Masking the elements that are looped back
    masks = jnp.repeat(
        (shifts[:, None] + shifts[None, :])[None, :, :],
        n, axis=0
    )
    masks = jnp.sign(jnp.where((masks+1)//2<=shifts[:,None,None], 0, masks))
    # Applying the mask
    toeplitz = toeplitz*masks
    # Summing along and collapse the new axis.
    toeplitz = jnp.sum(toeplitz, axis=0)
    return(toeplitz)

''' II. Deconvolution ("dividing" chi-dependent terms) '''
# @partial(jit, static_argnums=(2,))
def get_O_O_einv_from_A_B(chiphifunc_A:ChiPhiFunc, chiphifunc_B:ChiPhiFunc, rank_rhs:int, Y1c_mode:bool):
    '''
    Get O, O_einv and vector_free_coef that solves the eqaution system
    O Yn = (A + B dchi) Yn = RHS <=>
      Yn = O_einv@RHS - (Yn0 or Yn1p) * vec_free_coef

    This is an under-determined problem. Here, Yn is an (n+1)-dim vector.
    A and B are 2-d vectors. O is a known (n+2, n+1) convolution/differential
    matrix with A and B as kernels. The RHS (not provided here) is a (n+2) vector
    with (n) linearly-independent, phi-dependent components.

    Or in code format,
    Yn_content = jnp.einsum('ijk,jk->ik',O_einv,chiphifunc_rhs_content)\
        + vec_free * vector_free_coef


    Inputs: -----

    A, B: ChiPhiFunc coefficients

    rank_rhs: The number of rows in the RHS.

    Outputs: -----

    O_matrices,

    O_einv,

    vector_free_coef: A (n+1*len_phi)-component array

    Y_nfp: the nfp of Y.
    '''
    i_free = (rank_rhs+1)//2 # We'll always use Yn0 or Yn1p as the free var.

    # Mismatched length
    if chiphifunc_A.content.shape[1] != chiphifunc_B.content.shape[1]\
    and (chiphifunc_A.content.shape[1]!=1 and chiphifunc_B.content.shape[1]!=1):
        return(jnp.nan, jnp.nan, jnp.nan, jnp.nan)

    stretch_phi = jnp.zeros((1, max(chiphifunc_A.content.shape[1], chiphifunc_B.content.shape[1])))
    chiphifunc_A_content = chiphifunc_A.content+stretch_phi
    chiphifunc_B_content = chiphifunc_B.content+stretch_phi

    # generate the LHS operator O_matrices = (va conv + vb conv dchi)
    O_matrices = 0
    A_conv_matrices = conv_tensor(chiphifunc_A_content, rank_rhs+1)
    O_matrices = O_matrices + A_conv_matrices

    dchi_matrix = dchi_op(rank_rhs+1)
    B_conv_matrices = conv_tensor(chiphifunc_B_content, rank_rhs+1)
    O_matrices = O_matrices + jnp.einsum('ijk,jl->ilk',B_conv_matrices,dchi_matrix)

    if Y1c_mode and rank_rhs%2==1:
        # This O, when inverted, yields an O_einv acting on the trig
        # representation of RHS and vec_free_coef:
        # Yn_trig = O_einv@RHS_trig - (Yn0 or Yn1c) * vec_free_coef
        # O_matrices = O_matrices@trig_to_exp_op(O_matrices.shape[1])
        O_matrices = jnp.einsum('ijp,jm->imp', O_matrices, trig_to_exp_op(O_matrices.shape[1]))
        O_matrices = jnp.einsum('ij,jmp->imp', exp_to_trig_op(O_matrices.shape[0]), O_matrices)

    O_einv = batch_matrix_inv_excluding_col(O_matrices)
    O_einv = jnp.concatenate((O_einv[:i_free], jnp.zeros((1,O_einv.shape[1],O_einv.shape[2])), O_einv[i_free:]))
    O_free_col = O_matrices[:,i_free,:]

    vector_free_coef = jnp.einsum('ijk,jk->ik',O_einv, O_free_col)#A_einv@A_free_col
    vector_free_coef = vector_free_coef.at[i_free].set(-jnp.ones((vector_free_coef.shape[1])))

    if Y1c_mode and rank_rhs%2==1:
        #             Yn_trig =             O_einv@            RHS_trig - (Yn0 or Yn1c) *             vec_free_coef
        # trig_to_exp@Yn_trig = trig_to_exp@O_einv@            RHS_trig - (Yn0 or Yn1c) * trig_to_exp@vec_free_coef
        #             Yn      = trig_to_exp@O_einv@            RHS_trig - (Yn0 or Yn1c) * trig_to_exp@vec_free_coef
        #             Yn      = trig_to_exp@O_einv@exp_to_trig@RHS      - (Yn0 or Yn1c) * trig_to_exp@vec_free_coef
        # O_einv = trig_to_exp_op(O_einv.shape[0])@O_einv@exp_to_trig_op(O_einv.shape[1])
        O_einv = jnp.einsum('ijp,jm->imp', O_einv, exp_to_trig_op(O_einv.shape[1]))
        O_einv = jnp.einsum('ij,jmp->imp', trig_to_exp_op(O_einv.shape[0]), O_einv)
        vector_free_coef = trig_to_exp_op(O_einv.shape[0])@vector_free_coef

    return(O_matrices, O_einv, -vector_free_coef)

'''
III. Grid 1D deconvolution (used for "dividing" a chi-dependent quantity)

This part solves the pointwise product function problem A*va = B*vb woth unknown va.
by chi mode matching. It treats both pointwise products as matrix
products of vectors (components are chi mode coeffs, which are rows in ChiPhiFunc.content)
with convolution matrices A@va = B@vb,
Where A, B are (m, n+1) (m, n) or (m, n) (m, n) matrices
   of rank     (n+1)    (n)    or (n)    (n)
   va, vb are  (n+1)    (n)    or (n)    (n)   -d vectors

These type of equations underlies power-matched recursion relations:
a(chi, phi) * va(chi, phi) = b(chi, phi) * vb(chi, phi)
in grid realization.
where a, b has the same number of chi modes (let's say o), as pointwise product
with a, b are convolutions, which are (o+(n+1)-1, n+1) or (o+n-1, n) degenerate
matrices.
Codes written in this part are specifically for 1D deconvolution used for ChiPhiFunc.
'''

''' III.2 va has 1 more component than vb '''
def batch_matrix_inv_excluding_col(in_matrices:jnp.ndarray):
    '''
    Invert an (n,n) submatrix of a (n+2,n+1) rectangular matrix by taking the
    center n rows and excluding the rank_rhs column. "Taking the center n rows"
    is motivated by the RHS being rank n-1. For solving for Yn. (a (n+1)-dim
    vector)

    Input: -----

    in_matrices: (n+2, n+1, len_phi) rank_rhs n matrix to invert

    Return: -----

    (m,m,len_phi) matrix A_inv
    '''
    # Checking dimensions
    if in_matrices.ndim != 3:
        return(jnp.nan)

    n_row = in_matrices.shape[0]
    n_col = in_matrices.shape[1]
    n_phi = in_matrices.shape[2]
    rank_rhs = n_col-1
    ind_col = (rank_rhs+1)//2

    # Checking shape
    if n_row-1!=n_col:
        return(jnp.nan)

    n_clip = (n_row-n_col+1)//2 # How much is the transposed array larger than Yn
    if n_row<=n_col:
        return(jnp.nan)
        # raise ValueError("Input should have more rows than cols")

    # Remove specfied column (slightly faster than delete)
    # and remove extra rows (take n_col-1 rows from the center)
    rows_to_remove = (n_row-(n_col-1))//2
    # sub = in_matrices[:,jnp.arange(in_matrices.shape[1])!=i_free,:][rows_to_remove:-rows_to_remove, :, :]
    sub = jnp.delete(in_matrices, ind_col, axis=1)[rows_to_remove:-rows_to_remove, :, :]
    sub = jnp.moveaxis(sub,2,0)
    sqinv = jnp.linalg.inv(sub)
    sqinv = jnp.moveaxis(sqinv,0,2)
    padded = jnp.pad(sqinv, ((0,0), (rows_to_remove, rows_to_remove), (0,0)))
    return(padded)

''' III.3 Convolution operator generator and ChiPhiFunc.content wrapper '''
# A jitted vectorized version of jnp.roll
roll_axis_0 = lambda a, shift: jnp.roll(a, shift, axis=0)
batch_roll_axis_0 = vmap(roll_axis_0, in_axes=1, out_axes=1)
# @partial(jit, static_argnums=(1,))
def conv_tensor(content:jnp.ndarray, n_dim:int):
    '''
    Generate a tensor_coef (see looped_solver.py) convolving a ChiPhiFunc
    to another along axis 0.
    For multiplication in FFT space during ODE solves.
    The convolution is done by:
    x2_conv_y2 = jnp.einsum('ijk,jk->ik',conv_x2, y2.content)

    Input: -----

    content: A content matrix

    n_dim: The convolution matrices act on a content with n_dim
    chi components.

    Output: -----

    A (content.shape[0]+n_dim-1, n_dim, content.shape[1]), representing
    a stack of content.shape[1] convolution matrices. The first two
    axes represent the row and column of a single convolution matrix.
    '''
    len_phi = content.shape[1]
    content_padded = jnp.concatenate((content, jnp.zeros((n_dim-1, len_phi))), axis=0)
    content_padded = jnp.tile(content_padded[:, None, :], (1, n_dim, 1))
    shift = jnp.arange(n_dim)
    return(batch_roll_axis_0(content_padded, shift[None, :]))

roll_fft_last_axis = lambda a, shift: jnp.roll(a, shift, axis=-1)
batch_roll_fft_last_axis = vmap(roll_fft_last_axis, in_axes=(-2,0), out_axes=2)

def fft_conv_tensor_batch(source:jnp.array):
    '''
    Generates a 4D convolution operator in the phi axis for a 3d "tensor coef"
    (see looped_solver for explanation).

    Input: -----

    An (a, b, len_phi) array, where a and b represents a matrix acting on the chi
    dependences of a content

    Output: -----

    An (a, b, len_phi, len_phi) array that acts on a content by transposing
    axis 1 and 2 and then tensordotting. (see explanation for a "tensor operator"
    in looped_solver)
    '''
    len_phi = source.shape[2]
    arange = jnp.arange(len_phi)
    # out = jnp.zeros((len_a, len_b, len_phi, len_phi))#, dtype = jnp.complex128)
    out = jnp.repeat(source[:,:,None,:], len_phi, axis=2)
    # Where does 0 elem start/end
    split = (len_phi+1)//2
    split_b = jnp.roll(jnp.arange(len_phi%2, len_phi+len_phi%2),split)
    # Turns out rolling by fftfreq and arange(len_phi) is the same.
    out = batch_roll_fft_last_axis(out, arange)
    # Masking
    split_start = jnp.where(split_b<split, split_b, split) # min
    split_end = jnp.where(split_b<split, split, split_b) # max
    mask = jnp.where(
        jnp.logical_and(arange[None,:]>=split_start[:,None], arange[None,:]<split_end[:,None]),
        0,1
    )
    out = out*mask[None,None,:,:]
    return(jnp.transpose(out, (0,1,3,2))/len_phi)  # not nfp-dependent

''' IV. Solving linear PDE in phi grids '''

''' IV.1 Solving the periodic linear PDE (a + b * dphi + c * dchi) y = f(phi, chi) '''
def solve_1d_asym(p_eff, f_eff): # not nfp-dependent
    '''
    Solves one linear ODE of form y' + p_eff*y = f_eff.

    Inputs: -----

    p_eff: coefficient of y. 1d array.

    f_eff: RHS. 1d array.

    Outputs: -----

    Solution to the equation
    when p_eff is 0, y is the anti-derivative of f with zero average.
    '''
    if jnp.isscalar(p_eff):
        p_eff = p_eff*jnp.ones_like(f_eff)

    ai = f_eff/p_eff # f/p
    # Make an 2d array containing asymptotic series terms.
    # axis=1 is phi dependence and axis=0 is the term number.
    # Then, add in the 0-th order term.
    asym_series = jnp.zeros((asymptotic_order+1, len(ai)))+ai[None, :]
    for i in range(asymptotic_order):
        # ai is periodic. We use the non-looped value to ensure
        # that dphi by fft functions correctly
        ai_new = -(ChiPhiFunc(ai[None, :],1).dphi().content[0])/p_eff
        ai = ai_new
        asym_series = asym_series.at[i+1].set(ai_new)

    # Optimal truncation: Find which term has the smallest maximum amplitude.
    # And set all following terms to 0.
    max_amp_for_each_term = jnp.max(jnp.abs(asym_series), axis=1)
    loc_smallest_max_amp = jnp.argmin(max_amp_for_each_term)
    asym_series = jnp.where(jnp.arange(len(max_amp_for_each_term))[:, None]>loc_smallest_max_amp, 0, asym_series)

    return(jnp.sum(asym_series, axis=0))

# @partial(jit, static_argnums=(2,))
def solve_1d_fft(p_eff, f_eff, static_max_freq:int=None): # not nfp-dependent
    '''
    Solves one linear ODE of form y' + p_eff*y = f_eff.
    Assumes non-zero p.

    Inputs: -----

    p_eff: coefficient of y. 1d array.
    f_eff: RHS. 1d array.

    Outputs: -----

    Solution to the equation.
    The p_eff == 0 case is handled in solve_ODE by a jnp.where
    statement.
    '''
    len_phi = len(f_eff)
    if jnp.isscalar(p_eff):
        p_eff = p_eff*jnp.ones_like(f_eff)

    if static_max_freq is None:
        target_length = len(f_eff)
    else:
        target_length = static_max_freq*2
    p_fft = fft_filter(jnp.fft.fft(p_eff), target_length, axis=0)
    f_fft = fft_filter(jnp.fft.fft(f_eff), target_length, axis=0)

    # Both are (len_phi, len_phi) matrices
    diff_matrix = fft_dphi_op(target_length)
    conv_matrix = fft_conv_op(p_fft)
    inv_dxpp = jnp.linalg.inv(diff_matrix + conv_matrix)
    sln_fft = inv_dxpp@f_fft
    sln = jnp.fft.ifft(fft_pad(sln_fft, len_phi, axis=0), axis=0)

    return(sln)

solve_1d_fft_batch = vmap(solve_1d_fft, in_axes=(0, 0, None), out_axes=0)
# @partial(jit, static_argnums=(3,))
def solve_ODE(coeff_arr, coeff_dp_arr, f_arr:jnp.ndarray, static_max_freq:int=None): # not nfp-dependent
    '''
    Solves simple linear first order ODE systems in batch:
    (coeff_phi d/dphi + coeff) y = f. ( y' + p_eff*y = f_eff )
    (Dchi = +- m * 1j)

    NOTE:
    Does not work well for p>10 with zeros or resonant p.

    Inputs: -----

    coeff_arr, coeff_dp_arr, f_arr: Components of the equations as 2d matrices.
    Axis=0 is equation indices and axis=1 is phi dependences. All quantities are
    assumed periodic

    static_max_freq: Maximum number of Fourier harmonics used.

    Output: -----

    The solution to the equation system as 2d arrays.
    '''
    len_phi = f_arr.shape[1]
    len_chi = f_arr.shape[0]

    if static_max_freq is None:
        static_max_freq = len_phi//2

    # Rescale the eq into y'+py=f
    f_eff = f_arr/coeff_dp_arr
    p_eff = coeff_arr/coeff_dp_arr
    # Necessary for integration factor, not necessary for spectral method.
    # f_eff_scaling = jnp.average(jnp.abs(f_eff))
    # f_eff = f_eff/f_eff_scaling

    if jnp.isscalar(p_eff):
        p_eff = p_eff+jnp.zeros_like(f_arr)

    # We always assume f is phi-dependent.
    if p_eff.shape[1] != f_eff.shape[1]:
        if p_eff.shape[1]==1:
            p_eff = p_eff*jnp.ones_like(f_arr)
        else:
            return(jnp.nan) # Mismatch

    # Uneven component number
    if p_eff.shape[0]!=f_eff.shape[0]:
        return(jnp.nan)

    # Asymptotic series is depreciated. It only works well when
    # minimum amplitude is large. In this case, the FFT method also works well.
    out_arr = jnp.where(
        (jnp.all(p_eff == 0, axis=1))[:, None],
        ChiPhiFunc(f_eff, nfp=1).\
            integrate_phi_fft(zero_avg=True).content,
        solve_1d_fft_batch(p_eff, f_eff, static_max_freq)
    )
    # return(out_arr*f_eff_scaling)
    return(out_arr)


''' ONLY USED IN solve_1d(). '''
# @partial(jit, static_argnums=(0,))
def fft_dphi_op(len_phi:int):
    '''
    ONLY USED IN solve_1d().

    Input: -----

    len_phi:
    Only used in solve_ODE.
    remove_zero: A list of booleans. When true, replaces the corresponding element
    in the second output with a matrix of jnp.nan's.

    Output: -----

    1. A dphi operator acting on the fft of
    a content along axis=1
    '''
    fftfreq = jit_fftfreq_int(len_phi)
    matrix = jnp.identity(len_phi) * 1j * fftfreq
    return(matrix)

''' ONLY USED IN solve_1d(). '''

def fft_conv_op(source):
    '''
    ONLY USED IN solve_1d().
    A convolution operator acting convolving
    a fft of len_phi with source.

    Input: -----

    A 1d array

    Output: -----

    A convoution operators acting on the fft of
    a content along axis=1. Shape is [len_phi, len_phi].
    '''
    # We first turn the axis=0 of source into two axes
    source_eff = source[None,None,:]
    tensor_eff = fft_conv_tensor_batch(source_eff)
    # Collapse the first two axes from a diagonal matrix to a 1d array by summing
    return(tensor_eff[0][0])

# @partial(jit, static_argnums=(4,))
def solve_ODE_chi(coeff, coeff_dp, coeff_dc, f, static_max_freq: int):
    '''
    For solving the periodic linear 1st order ODE
    (coeff + coeff_dp*dphi + coeff_dc*dchi) y = f(phi, chi)
    using Fourier method.

    Input: -----

    coeffs are constants or contents. f is a content (see ChiPhiFunc's description).
    Contant mode and offset are for evaluating the constant component equation
    dphi y0 = f0.

    Output: -----
    y is a ChiPhiFunc's content
    '''
    len_chi = f.shape[0]
    # Chi harmonics
    ind_chi = len_chi-1
    # Multiplies each row with its corresponding mode number.
    mode_chi = 1j*jnp.linspace(-ind_chi, ind_chi, len_chi, axis=0)[:,None]
    coeff_eff = (coeff_dc*mode_chi + coeff)

    return(
        solve_ODE(coeff_eff, coeff_dp, f, \
            static_max_freq=static_max_freq)
    )

# @partial(jit, static_argnums=(2,))
def solve_dphi_iota_dchi(iota, f, static_max_freq: int):
    '''
    For solving the periodic linear 1st order ODE
    (dphi+iota*dchi) y = f(phi, chi)
    using Fourier method.

    Input: -----

    iota is a constant and f is a ChiPhiFunc.

    Contant mode and offset are for evaluating the constant component equation
    dphi y0 = f0.

    Output: -----

    y is a ChiPhiFunc's content
    '''
    return(
        solve_ODE_chi(
            coeff=0,
            coeff_dp=1,
            coeff_dc = iota,
            f=f,
            static_max_freq=static_max_freq
        )
    )

''' V. utilities '''

''' V.1. Low-pass filter for simplifying tensor to invert '''
# @partial(jit, static_argnums=(1,2,))
def fft_filter(fft_in:jnp.ndarray, target_length:int, axis:int): # not nfp-dependent
    '''
    Shorten an array in FFT representation to leave only target_length elements.
    (which correspond to a low-pass filter with k<target_length/2*nfp)
    by removing the highest frequency modes. The resulting array can be IFFT'ed.
    Target length is the number of harmonics to keep in one field period.

    Inputs: -----

    fft_in: ndarray to filter. Must already be in FFT representation.

    target_length: length of the output array

    axis: axis to filter along

    Output: -----

    Filtered array.
    '''
    if target_length>=fft_in.shape[axis] or target_length<0:
        return(fft_in)
    # FFT of an array contains mode amplitude in the order given by
    # fftfreq(length)*length. For example, for length=7,
    # [ 0.,  1.,  2.,  3., -3., -2., -1.]
    left = fft_in.take(indices=jnp.arange(0, (target_length+1)//2), axis=axis)
    right = fft_in.take(indices=jnp.arange(-(target_length//2), 0), axis=axis)
    return(jnp.concatenate((left, right), axis=axis)*target_length/fft_in.shape[axis])

# @partial(jit, static_argnums=(1,2,))
def fft_pad(fft_in:jnp.array, target_length:int, axis:int): # not nfp-dependent
    '''
    Pad an array in FFT representation to target_length elements.
    by adding zeroes as highest frequency modes.
    The resulting array can be IFFT'ed.

    Inputs: -----

    fft_in: ndarray to pad. Must already be in FFT representation.

    target_length: length of the output array

    axis: axis to filter along

    Output: -----

    Padded array.
    '''
    if target_length<fft_in.shape[axis]:
        return(jnp.nan) # Target length smaller than current length
    elif target_length==fft_in.shape[axis]:
        return(fft_in)
    new_shape = list(fft_in.shape)
    original_length = new_shape[axis]
    new_shape[axis] = target_length - original_length
    center_array = jnp.zeros(new_shape)
    # FFT of an array contains mode amplitude in the order given by
    # fftfreq(length)*length. For example, for length=7,
    # [ 0.,  1.,  2.,  3., -3., -2., -1.]
    left = fft_in.take(indices=jnp.arange(0, (original_length+1)//2), axis=axis)
    right = fft_in.take(indices=jnp.arange(-(original_length//2), 0), axis=axis)
    return(jnp.concatenate((left, center_array, right), axis=axis)*target_length/fft_in.shape[axis])

'''
V.2 Tensor construction for looped_solver.py

Theses methods are for constructing differential/convolution tensors
'''

# @partial(jit, static_argnums=(1,))
def to_tensor_fft_op(ChiPhiFunc_in:ChiPhiFunc, len_tensor:int):
    '''
    For solving the looped equations. They are only used in looped_solver.py and
    lambda_coefs_B_psi.py.
    '''
    tensor_coef = ChiPhiFunc_in.content[:, None, :]
    tensor_fft_coef = fft_filter(jnp.fft.fft(tensor_coef, axis = 2), len_tensor, axis=2)
    tensor_fft_op = fft_conv_tensor_batch(tensor_fft_coef)
    return(tensor_fft_op)

# @partial(jit, static_argnums=(1,2,3,4,5,6))
def to_tensor_fft_op_multi_dim(
    ChiPhiFunc_in:ChiPhiFunc, dphi:int, dchi:int,
    num_mode:int, cap_axis0:int,
    len_tensor: int,
    nfp: int):
    '''
    (n, n-2, len_tensor, len_tensor), acting on the FFT of
    [
        [B_theta +n-1],
        ...
        [B_theta -n+1],
    ]
    Generating convolution tensors from B_theta coefficients.
    These are only needed for n>2 (n_eval>3)

    Inputs: -----

    num_mode: the number of columns of the resulting tensor
    (corresponds to input row number)

    cap_axis0: the length of axis=0 for the resulting tensor,
    used to remove outer components that are known to cancel.
    Must have the same even/oddness and smaller than the row
    number of the convolution tensor generated from ChiPhiFunc_in
    and num_mode.

    Output: -----

    A tensor of shape (len_chi+num_mode-1, num_mode, len_tensor, len_tensor)
    acting on a content by np.tensordot(operator, content, 2).
    It first takes chi and phi derivatives of specified orders
    and then multiplies
    '''
    if ChiPhiFunc_in.nfp == 0:
        return(0)
    # A stack of convolution matrices
    # shape is (len_chi+num_mode-1, num_mode, len_phi)
    len_chi = ChiPhiFunc_in.content.shape[0]
    tensor_coef_nD = conv_tensor(ChiPhiFunc_in.content, num_mode)
    # Putting in dchi
    # The outmost component of B_theta is 0.
    # B_theta coeffs carried by B_psi has 3 components,
    # and the convolution matrix is n_unknown+2 * n_unknown-1
    if cap_axis0%2!=tensor_coef_nD.shape[0]%2:
        return(jnp.full((len_chi+num_mode-1, num_mode, len_tensor, len_tensor),jnp.nan))
    if cap_axis0>tensor_coef_nD.shape[0]:
        return(jnp.full((len_chi+num_mode-1, num_mode, len_tensor, len_tensor),jnp.nan))
    if tensor_coef_nD.shape[0]>cap_axis0:
        tensor_coef_nD = tensor_coef_nD[
            (tensor_coef_nD.shape[0]-cap_axis0)//2:
            (tensor_coef_nD.shape[0]+cap_axis0)//2
        ]
    if dchi!=0:
        dchi_array_temp = (1j*jnp.arange(-num_mode+1,num_mode+1,2)[None, :, None])
        if dchi>0:
            tensor_coef_nD = tensor_coef_nD*dchi_array_temp**dchi
        elif dchi<0:
            if num_mode%2==0: # chi integrals are only supported when there is no constant componemnt
                tensor_coef_nD = tensor_coef_nD/dchi_array_temp**(-dchi)
            else:
                # This helper method does not support calculating
                # chi integrals (dchi<0) when the content being acted on
                # has chi-indep component (num_mode is odd)
                return(jnp.full((len_chi+num_mode-1, num_mode, len_tensor, len_tensor),jnp.nan))
    # Applying FFT
    # A stack of convolution matrices, but now axis=2 is
    # in frequency space and capped to len_tensor elements.
    # shape is (len_chi+num_mode-1, num_mode, len_tensor)
    tensor_fft_coef_B_theta = fft_filter(jnp.fft.fft(tensor_coef_nD, axis = 2), len_tensor, axis=2)
    # 'Tensor coefficients', dimension is (n_eval-1, n_eval-3, len_phi)
    # Last 2 dimensions are for convolving phi cells.
    # shape is (len_chi+num_mode-1, num_mode, len_tensor, len_tensor)
    tensor_fft_op_B_theta = fft_conv_tensor_batch(tensor_fft_coef_B_theta)
    # Applying dphi
    if dphi!=0:
        if dphi<0:
            # dphi must be positive
            return(jnp.full((len_chi+num_mode-1, num_mode, len_tensor, len_tensor),jnp.nan))
        # dphi matrix
        fft_freq = jit_fftfreq_int(len_tensor)
        dphi_array = jnp.ones((len_tensor,len_tensor)) * 1j * fft_freq * nfp
        tensor_fft_op_B_theta = tensor_fft_op_B_theta*(dphi_array**dphi)
    return(tensor_fft_op_B_theta)

''' V.3. Others '''
def linear_least_sq_2d_svd(A, b):
    '''
    Solves the linear least square problem minimizing ||Ax-b||

    Inputs: -----

    A, b: (n,m) and (n)

    Output: -----

    x
    '''
    u, s, vh = jnp.linalg.svd(A, full_matrices=False)
    Eps_inv = jnp.diag(1/s)
    return(vh.conjugate().T@Eps_inv@u.conjugate().T@b)
