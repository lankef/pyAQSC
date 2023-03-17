import numpy as np

import jax.numpy as jnp
from jax import jit, vmap, tree_util
from functools import partial # for JAX jit with static params

import math # factorial in dphi_direct
from math_utilities import *
from matplotlib import pyplot as plt
# Numba doesn't support scipy methods. Therefore, scipy integrals are sped up
# with joblib.
import scipy.integrate
import scipy.interpolate
from joblib import Parallel, delayed
from functools import lru_cache # import functools for caching

# Configurations
import config






''' Debug options and loading configs '''
# tolerance on how periodic I(phi) is during integration factor.
# When I(phi) is periodic, any C1 can satisfy the periodic BC,
# The assiciated error is thrown by comparing the difference
# between the integration factor's first and last element to
# this variable.
noise_level_periodic = 1e-10

# Loading configurations
if config.double_precision:
    import jax.config
    jax.config.update("jax_enable_x64", True)


# Loading default diff modes
# Converting differential mode string to an int for jitting.
if config.diff_mode=='fft':
    diff_mode=1
if config.diff_mode=='pseudo_spectral':
    diff_mode=2
if config.diff_mode=='finite_difference':
    diff_mode=3
# integral_mode = config.integral_mode
# two_pi_integral_mode = config.two_pi_integral_mode
# Maximum allowed asymptotic series order for y'+py=f
asymptotic_order = config.asymptotic_order

# Joblib settings
n_jobs = config.n_jobs_chiphifunc
# scipy.integrate is based on compiled codes. 'threading' is the best backend.
backend = config.backend_chiphifunc
# If set to ‘sharedmem’, the selected backend will be single-host and
# thread-based even if the user asked for a non-thread based backend
# with parallel_backend.
require = None

''' I. Representing functions of chi and phi (ChiPhiFunc subclasses) '''
# Represents a function of chi and phi.
# Manages an complex128[m, n] 2d array called content.
# Axis 0 represents "m". Its length is n+1 for a nth-order term:
# each n-th order known term has n+1 non-zero coeffs due to regularity cond.
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
@jit
def dchi_op(content:jnp.ndarray):
    '''
    Generate chi differential operator diff_matrix. diff_matrix@f.content = dchi(f).content
    invert = True generates anti-derivative operator. Cached for each new Chi length.
    Input: -----
    len_chi: length of Chi series.
    Output: -----
    2d diagonal matrix with elements [-m, -m+2, ... m].
    '''
    len_chi = content.shape[0]
    ind_chi = len_chi-1
    mode_chi = jnp.linspace(-ind_chi, ind_chi, len_chi)
    return jnp.diag(1j*mode_chi) # not nfp-sensitive

@jit
def wrap_grid_content_jit(content:jnp.ndarray):
    '''
    Used for wrapping grid content. Defined outside the ChiPhiFunc class so that
    it can be used in @njit compiled methods.
    '''
    first_col = content[:,0]
    return(jnp.concatenate((content, first_col[:,None]), axis=1)) # not nfp-sensitive

@jit
def max_log10(input):
    ''' Calculates the max amplitude's order of magnitude '''
    return(jnp.log10(jnp.max(jnp.abs(input)))) # not nfp-sensitive

# Input is not order-dependent, and permitted to be static.
@partial(jit, static_argnums=(0,))
def jit_fftfreq_int(int_in:int):
    ''' Shorthand for np.fft.fftfreq(n)*n rounded to the nearest int. '''
    return(jnp.rint(np.fft.fftfreq(int_in)*int_in).astype(jnp.int32))

# Input is order-dependent but the function is rarely used.
# Permitted to be static.
@jit
def fourier_to_exp(content):
    n_dim = content.shape[0]
    ones = jnp.ones(n_dim//2)
    if n_dim%2==0:
        arr_diag = jnp.concatenate([0.5j*ones, 0.5*ones])[:, None]
        arr_anti_diag = jnp.concatenate([-0.5j*ones, 0.5*ones])[:, None]
    if n_dim%2==1:
        arr_diag = jnp.concatenate([0.5j*ones, jnp.array([0.5]), 0.5*ones])[:, None]
        arr_anti_diag = jnp.concatenate([-0.5j*ones, jnp.array([0.5]), 0.5*ones])[:, None]
    return(arr_diag*content + jnp.flip(arr_anti_diag*content, axis=0))


def fourier_to_exp_op(content_ref): # not nfp-dependent
    '''
    Generates a matrix for converting self's content (can be full or skip) from
    trig fourier representation into exponential fourier representation.
    '''
    n_dim = content_ref.shape[0]
    if n_dim%2==0:
        n_mode = n_dim//2
        I_n = jnp.identity(n_mode)
        I_anti_n = jnp.fliplr(I_n)
        util_matrix = jnp.block([
            [ 0.5j*I_n            , 0.5*I_anti_n         ],
            [-0.5j*I_anti_n       , 0.5*I_n              ]
        ])
    else:
        n_mode = (n_dim-1)//2
        I_n = jnp.identity(n_mode)
        I_anti_n = jnp.fliplr(I_n)
        util_matrix = jnp.block([
            [ 0.5j*I_n            , np.zeros((n_mode, 1)), 0.5*I_anti_n         ],
            [np.zeros((1, n_mode)), 1                    , np.zeros((1, n_mode))],
            [-0.5j*I_anti_n       , np.zeros((n_mode, 1)), 0.5*I_n              ]
        ])
    return(util_matrix)

# Input is not order-dependent, and permitted to be static.
@partial(jit, static_argnums=(0,))
def ChiPhiFuncSpecial(error_code:int):
    return(ChiPhiFunc(jnp.nan, error_code, is_special=True))

# A jitted vectorized convolution function
batch_concolve = jit(vmap(jnp.convolve, in_axes=1, out_axes=1))

@jit
def phi_avg(in_quant):
    '''
    A type-insensitive phi-averaging function that:
    - Averages along phi and output a ChiPhiFunc if the input is a ChiPhiFunc.
    - Does nothing if the input is a scalar.
    '''
    if isinstance(in_quant, ChiPhiFunc):
        # special ChiPhiFunc's
        if in_quant.is_special:
            return(in_quant)
        new_content = jnp.array([
            jnp.mean(in_quant.content,axis=1)
        ]).T
        return(ChiPhiFunc(new_content,in_quant.nfp))
    if not jnp.isscalar(other):
        if test.ndim!=0: # 0-d np array will check false for isscalar.
            return(ChiPhiFuncSpecial(-5))
    return(in_quant)

''' I.1 Grid implementation '''
class ChiPhiFunc:
    '''
    ChiPhiFunc represents a function of chi and phi in even/odd fourier series in
    chi. The coefficients are assumed phi-dependent with nfp field periods.
    This dependence is stored on n grid points located at (0, ... 2pi(n-1)/n/nfp) in phi.
    Members -----
    1. content: 2d, complex64 or complex 128 arrays storing a function of chi and phi.

    2. nfp: int, number of field periods. Automatically set to 0 when content has
    only one column.

    3. is_special: bool. A ChiPhiFunc with is_special represents an out-of-bound item when
    loading from a ChiPhiEpsFunc. Such a ChiPhiFunc behaves identically to np.nan except
    when multiplied with 0, which yields 0. This behavior is needed to work with the
    way the maxima order-matcher enforces summation bounds.

    When is_special is True, nfp represents the type of special ChiPhiFunc:
    nfp = 0 -----
    Zero produced by non-traced conditionals in math_utilities.
    These conditionals enforces summation bounds.

    nfp = -1 -----
    Produced when accessing an out-of-bound element of a ChiPhiEpsFunc.
    Can be cancelled by a ChiPhiFuncSpecial with nfp=0.

    Negative-nfp -----
    -1: Out of bound
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
    -12: match_m() failed

    When multiple error types are present, the error will be recorded as:
    100*error_a+10*error_b+error_c
    (example of error -4, -3 followed by -11: -40311.)

    Rules:
    + ----
    1. special(nfp=0)+anything = anything
    2. special(nfp!=0)+anything not special = special(nfp!=0)
    3. special(nfp!=0)+special(nfp!=0) = special(merged error message)
    4. special(nfp=0)*special(nfp=1) = special(nfp=0)

    '''
    def __init__(self, content:jnp.ndarray, nfp:int, is_special: bool=False, fourier_mode: bool=False):
        '''
        Constructor. It has 1 extra parameter.
        fourier_mode=true, every row of a content array is treated as a sin/cos
        chi modes, rather than e^(im phi).
        '''
        if is_special:
            if nfp>0:
                self.nfp = -2
            self.is_special = True
            self.content = jnp.nan
            self.nfp = nfp
        else:
            if content.ndim!=2: # Checks content shape
                self.is_special = True
                self.content = jnp.nan
                self.nfp = -7
            # Checks nfp type and sign
            elif nfp<=0 or not isinstance(nfp, int):
                self.is_special = True
                self.content = jnp.nan
                self.nfp = -2
            # A ChiPhiFuncs satisfying all above conditions is legal
            else:
                self.is_special = False
                self.nfp = nfp
                # Forcing complex128 type
                if config.double_precision:
                    content = content.astype(jnp.complex128)
                else:
                    content = content.astype(jnp.complex64)
                if fourier_mode:
                    self.content = fourier_to_exp(content)
                else:
                    self.content = content

    def __str__(self):
        if self.is_special:
            if self.nfp==1:
                msg = 'ChiPhiEpsFunc index out of bound'
            elif self.nfp==0:
                msg = 'zero from summation index conditionals'
            else:
                msg = 'error '+str(self.nfp)
            return('Special ChiPhiFunc, '+msg+'.')
        return(
            'ChiPhiFunc, content.shape='+str(self.content.shape)+', nfp='+str(self.nfp)+'.'
        )

    ''' For JAX use. '''
    def _tree_flatten(self):
        children = (self.content,)  # arrays / dynamic values
        aux_data = {'nfp': self.nfp, 'is_special': self.is_special}  # static values
        return (children, aux_data)
    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    ''' I.1.1 Operator overloads '''
    # Input is not order-dependent, and permitted to be static.
    @partial(jit, static_argnums=(1,))
    def __getitem__(self, index: int):
        '''
        Obtains the m=index component of this ChiPhiFunc.
        DOES NOT WORK like list indexing.
        Input: -----
        index: m (mode number)
        '''
        len_chi = len(self.content)
        # Checking even/oddness and total length
        if len_chi%2==index%2 or np.abs(index)>np.abs(len_chi-1):
            return(ChiPhiFuncSpecial(-3)) # invalid mode number
        new_content = jnp.array([self.content[len_chi//2+index//2]])
        if new_content.shape == (1,1):
            return new_content[0,0]
        return(ChiPhiFunc(new_content, self.nfp))

    @jit
    def __neg__(self):
        # If self.is_special is true, this still preserves the error message
        return ChiPhiFunc(-self.content, self.nfp, self.is_special)

    @jit
    def __add__(self, other):
        '''
        Adds self with another ChiPhiFunc or scalar.
        '''
        if isinstance(other, ChiPhiFunc):
            if self.nfp==0:
                return(other)
            if other.nfp==0:
                return(self)
            if self.is_special:
                if other.is_special: # Adding two nulls compound the error message
                    return(ChiPhiFuncSpecial(self.nfp*100+other.nfp))
                return(self)
            if other.is_special:
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
            return(ChiPhiFunc(a_padded+b_padded, max(self.nfp, other.nfp)))
        else:
            if not jnp.isscalar(other):
                if test.ndim!=0: # 0-d np array will check false for isscalar.
                    return(ChiPhiFuncSpecial(-5))
            if self.content.shape[0]%2==0:
                return(ChiPhiFuncSpecial(-4))
            # add to center
            center_loc = self.content.shape[0]//2
            updated_center = self.content[center_loc]+other
            updated_content = self.content.at[center_loc, :].set(updated_center)
            return(ChiPhiFunc(updated_content, self.nfp))

    @jit
    def __radd__(self, other):
        ''' other + self '''
        return(self+other)

    @jit
    def __sub__(self, other):
        ''' self - other '''
        return(self+(-other))

    @jit
    def __rsub__(self, other):
        ''' other - self '''
        return(-(self-other))

    @jit
    def __mul__(self, other):
        '''
        Multiplies self with another ChiPhiFunc or scalar.
        '''
        if isinstance(other, ChiPhiFunc):
            if self.nfp==0:
                if other.nfp<-1: # nfp < 0 always means error message
                    return(other)
                return(self) # 0*out of bound and 0*non-trivial are both 0.
            if other.nfp==0:
                if self.nfp<-1: # nfp < 0 always means error message
                    return(self)
                return(other) # 0*out of bound and 0*non-trivial are both 0.
            if self.is_special:
                if other.is_special: # Adding two nulls compound the error message
                    return(ChiPhiFuncSpecial(self.nfp*100+other.nfp))
                return(self)
            if other.is_special:
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
            return(ChiPhiFunc(batch_concolve(a,b), max(self.nfp, other.nfp)))
        else:
            if not jnp.isscalar(other):
                if test.ndim!=0: # 0-d np array will check false for isscalar.
                    return(ChiPhiFuncSpecial(-5))
            return(ChiPhiFunc(other * self.content, self.nfp))

    @jit
    def __rmul__(self, other):
        ''' other * self '''
        return(self*other) # not nfp-sensitive

    @jit
    def __truediv__(self, other):
        ''' self / other, only supports division by scalar or functions of phi. '''
        if isinstance(other, ChiPhiFunc):
            # Handles zero/error, any/error and any/zero
            if other.is_special:
                if other.nfp==0:
                    return(ChiPhiFuncSpecial(-8)) # /zero error
                if self.is_special: # Adding two nulls compound the error message
                    return(ChiPhiFuncSpecial(self.nfp*100+other.nfp))
                return(other)
            # Handles zero/(not error)
            if self.nfp==0:
                return(self)
            if other.content.shape[0]!=1:
                return(ChiPhiFuncSpecial(-4))
            # non-zero/chi-indep
            return(ChiPhiFunc(self.content/other.content, self.nfp))
        else:
            if self.nfp==0:
                return(self)
            if not jnp.isscalar(other):
                if test.ndim!=0: # 0-d np array will check false for isscalar.
                    return(ChiPhiFuncSpecial(-5))
            return(ChiPhiFunc(self.content/other, self.nfp))

    @jit
    def __rtruediv__(self, other):
        ''' other/self, only supports division by scalar or functions of phi. '''
        if self.is_special:
            if self.nfp==0:
                return(ChiPhiFuncSpecial(-8))
            return(self)
        # handles wrong shape of self
        if self.content.shape[0]!=1:
            return(ChiPhiFuncSpecial(-4))
        else:
            if isinstance(other, ChiPhiFunc):
                # Handles zero/non-zero and error/non-zero
                if other.is_special:
                    return(other)
                # Handles non-zero/non-zero and error/non-zero
                return(ChiPhiFunc(other.content/self.content, self.nfp))
            else:
                if not jnp.isscalar(other):
                    if test.ndim!=0: # 0-d np array will check false for isscalar.
                        return(ChiPhiFuncSpecial(-5))
                return(ChiPhiFunc(other/self.content, self.nfp))

    @jit
    def __rmatmul__(self, mat):
        '''
        other@self, for treating this object as a vector of Chi modes,
        and multiplying with a matrix
        '''
        return ChiPhiFunc(mat @ self.content, self.nfp)

    # Input is not order-dependent, and permitted to be static.
    @partial(jit, static_argnums=(1,))
    def __pow__(self, other):
        ''' Integer power of ChiPhiFunc '''
        new_content = self.content.copy()
        if other%1!=0:
            return(ChiPhiFuncSpecial(-9))
        if other == 0:
            return(1)
        for i in range(other-1):
            new_content = batch_concolve(new_content,self.content)
        return(ChiPhiFunc(new_content, self.nfp))

    ''' I.1.2 Derivatives, integrals and related methods '''

    # Input is not order-dependent, and permitted to be static.
    @partial(jit, static_argnums=(1,))
    def dchi(self, order=1):
        ''' Derivative in chi. order gives order. '''
        out = self.content
        len_chi = len(out)
        if order<0:
            return(ChiPhiFuncSpecial(-10))
        mode_i = (1j*jnp.arange(-len_chi+1,len_chi+1,2)[:,None])**order
        out = mode_i * out
        return(ChiPhiFunc(out, self.nfp))

    @jit
    def antid_chi(self):
        ''' Anti-derivative in chi. order gives order. '''
        len_chi = self.content.shape[0]
        temp = jnp.arange(-len_chi+1,len_chi+1,2,dtype=jnp.float32)[:,None]
        if len_chi%2==1:
            temp = temp.at[len(temp)//2].set(jnp.inf)
        return(type(self)(-1j * self.content/temp, self.nfp))

    # Input is not order-dependent, and permitted to be static.
    @partial(jit, static_argnums=(1, 2,))
    def dphi(self, order:int=1, mode=0):  # nfp-sensitive!!
        if order<0:
            return(ChiPhiFuncSpecial(-11))
        if mode==0:
            mode = diff_mode
        if mode==1:
            len_phi = self.content.shape[1]
            content_fft = jnp.fft.fft(self.content, axis=1)
            fft_freq_temp = jit_fftfreq_int(len_phi)*1j
            out_content_fft = content_fft*fft_freq_temp[None, :]**order
            out = jnp.fft.ifft(out_content_fft,axis=1)
        elif mode==2:
            out = self.content
            for i in range(order):
                out = (dphi_op_pseudospectral_known @ out.T).T
        # elif mode==3:
        #     out = jnp.gradient(self.content, axis=1)/(jnp.pi*2/self.content.shape[1])
        # if mode[-6:]=='spline':
        #     if order>0:
        #         out = integrate_phi_spline(content, mode, periodic=False,
        #             diff=True, diff_order=order)
        #     if order<0:
        #         out = integrate_phi_spline(content, mode, periodic=False,
        #             diff=False, diff_order=-order)
        #     return(out)
        else:
            return(ChiPhiFuncSpecial(-11))
        return(ChiPhiFunc(out*self.nfp, self.nfp))

    @jit
    def dphi_iota_dchi(self, iota):  # not nfp-sensitive
        return(self.dphi()+iota*self.dchi())

    @jit
    def exp(self):
        '''
        Used to calculate e**(ChiPhiFunc). Only support ChiPhiFunc with no
        chi dependence
        '''
        if self.content.shape[0]!=1:
            return(ChiPhiFuncSpecial(-7))
        return(ChiPhiFunc(np.exp(self.content), self.nfp))

    # Input is not order-dependent, and permitted to be static.
    @partial(jit, static_argnums=(1,))
    def integrate_phi_fft(self, zero_avg):
        '''
        Used for solvability condition. phi-integrate a ChiPhiFunc over 0 to
        2pi or 0 to a given phi. The boundary condition output(phi=0) = 0 is enforced.
        Input: -----
        zero_avg: whether the integration constant is set to have F(0)=0 or avg(F)=0
        Output: -----
        a new ChiPhiFunc
        '''
        # number of phi grids
        len_chi = self.content.shape[0]
        len_phi = self.content.shape[1]
        phis = np.linspace(0, 2*np.pi*(1-1/len_phi), len_phi, dtype=np.complex128)
        # fft integral
        content_fft = np.fft.fft(self.content, axis=1)
        fft_freq_temp = jit_fftfreq_int(len_phi)*1j
        fft_freq_temp[0] = np.inf
        out_content_fft = content_fft/fft_freq_temp[None, :]/self.nfp
        out_content = np.fft.ifft(out_content_fft,axis=1)
        # The fft.diff integral assumes zero average.
        if not zero_avg:
            out_content -= out_content[:,0][:,None]
            out_content += phis[None, :]*content_fft[:,0][:, None]/self.nfp/len_phi
        return(ChiPhiFunc(out_content, self.nfp))

    ''' I.1.3 phi Filters '''
    # We will jit all iterations, but will not jit the equilibrium class.
    # This will not be jitted at the moment.
    def filter(self, mode, arg):
        ''' An expandable filter. Now only low-pass is available. '''
        if mode == 'low_pass':
            len_phi = self.content.shape[1]
            if freq*2>=len_phi:
                return(np.copy(self.content))
            W = np.abs(jit_fftfreq_int(len_phi))
            f_signal = np.fft.fft(self.content, axis = 1)
            # If our original signal time was in seconds, this is now in Hz
            cut_f_signal = f_signal.copy()
            cut_f_signal[:,(W>freq)] = 0
            return(ChiPhiFunc(np.fft.ifft(cut_f_signal, axis=1), self.nfp))
        else:
            return(ChiPhiFuncSpecial())

    # We will jit all iterations, but will not jit the equilibrium class.
    # This will not be jitted at the moment.
    def noise_filter(self, mode, arg):
        '''
        Measure the "noise" in a ChiPhiFunc by comparing it to the result of
        a filter
        '''
        return(self-self.filter(mode=mode, arg=arg))

    ''' I.1.4 Properties '''
    @jit
    def get_amplitude(self):
        ''' Getting the average amplitude of the content '''
        return(np.average(np.abs(self.content)))

    @jit
    def real(self):
        ''' Functions like np.imag() '''
        return(ChiPhiFunc(np.real(self.content), self.nfp))

    @jit
    def imag(self):
        ''' Functions like np.real() '''
        return(ChiPhiFunc(np.imag(self.content), self.nfp))

    @jit
    def get_constant(self):
        return(self[0])

    @jit
    def get_phi_zero(self):
        ''' Returns the value when phi=0. Copies. '''
        new_content = jnp.array([self.content[:,0]]).T
        if len(new_content) == 1:
            return(new_content[0][0])
        return(ChiPhiFunc(jnp.array([self.content[:,0]]).T, 0))

    @jit
    def match_m(self, other): # nfp-dependent!!
        '''
        Returns a padded/clipped copy of self that has the same number
        of chi components as a provided ChiPhiFunc.
        '''
        if not isinstance(other, ChiPhiFunc):
            return(ChiPhiFuncSpecial(-12))
        if other.is_special:
            return(ChiPhiFuncSpecial(-12))
        target_chi = other.content.shape[0]
        len_chi = self.content.shape[0]
        # Check even-oddness
        if len_chi%2!=target_chi%2:
            return(ChiPhiFuncSpecial(-4))
        # target chi comp # = current chi comp #
        if target_chi == len_chi:
            return(self)
        # target chi comp # > current chi comp #
        if target_chi > len_chi:
            padding = ChiPhiFunc(jnp.zeros((target_chi, self.content.shape[1])), self.nfp)
            return(self+padding)
        # target chi comp # < current chi comp #
        num_clip = len_chi - target_chi
        return(ChiPhiFunc(self.content[num_clip//2:-num_clip//2], self.nfp))

    # # Takes the center m+1 rows of content. If the ChiPhiFunc
    # # contain less chi modes than m+1, It will be zero-padded.
    # def cap_m(self, m): # nfp-dependent!!
    #     target_chi = m+1
    #     len_chi = self.content.shape[0]
    #     num_clip = len_chi - target_chi
    #     if target_chi == len_chi:
    #         return(self)
    #     if target_chi > len_chi:
    #         self.pad_m(m)
    #     if num_clip%2 != 0:
    #         return(ChiPhiFuncSpecial(-4))
    #     return(ChiPhiFunc(self.content[num_clip//2:-num_clip//2], self.nfp))

    # def pad_m(self,m):
    #     return self.pad_chi(m+1)
    #

    # def pad_chi(self, target_chi):
        '''
        Pads a ChiPhiFunc to be have  total mode components.
        Both self.get_shape[0] and  must be even/odd.
        Note: This takes the total number of mode components, rather
        than m, because it's primarily used when solving the looped
        equations. It's sometimes less confusing to use the total number
        of mode components (equations) than the m of the corresponding equations.
        '''
        len_chi = self.content.shape[0]
        if len_chi%2!=target_chi%2:
            return(ChiPhiFuncSpecial(-4))

        if target_chi < len_chi:
            return(ChiPhiFuncSpecial(-12))
        padding = ChiPhiFunc(jnp.zeros((target_chi, self.content.shape[1])), self.nfp)
        return(self+padding)


    ''' I.1.5 Output and plotting '''
    def get_lambda(self):
        ''' Get a 2d vectorized function, f(chi, phi) for plotting a ChiPhiFunc '''
        len_chi = self.content.shape[0]
        len_phi = self.content.shape[1]

        # Create 'x' for interpolation. 'x' is 1 longer than lengths specified due to wrapping
        phi_looped = np.linspace(0,2*np.pi/self.nfp, len_phi+1)
        # wrapping
        content_looped = wrap_grid_content_jit(self.content)

        ind_chi = len_chi-1
        mode_chi = np.linspace(-ind_chi, ind_chi, len_chi)

        # The outer dot product is summing along axis 0.
        return(jnp.vectorize(
            lambda chi, phi : sum(
                [jnp.e**(1j*(chi)*mode_chi[i]) * jnp.interp((phi)%(2*jnp.pi/self.nfp),
                 phi_looped, content_looped[i]) for i in range(len_chi)]
            )
        )) # nfp-dependent!!

    def display_content(self, fourier_mode = False, colormap_mode = False):
        '''
        Plot the content of a ChiPhiFunc.
        Input: -----
        fourier_mode: bool. When True, plot the trig Chi fourier coefficients,
        rather than exponential

        colormap_mode: bool. When True, make colormaps. Otherwise makes line plots.
        '''
        plt.rcParams['figure.figsize'] = [8,3]
        content = self.content
        if content.shape[1]==1:
            content = content*np.ones((1,100))
        if type(content) is ChiPhiFuncSpecial:
            print('display_content(): input is ChiPhiFuncSpecial.')
            return()
        len_phi = content.shape[1]
        phis = np.linspace(0,2*np.pi*(1-1/len_phi)/self.nfp, len_phi)
        if fourier_mode:
            fourier = ChiPhiFunc(content, self.nfp).export_trig()
            if len(fourier.content)%2==0:
                ax1 = plt.subplot(121)
                ax1.set_title('cos, nfp='+str(self.nfp))
                ax2 = plt.subplot(122)
                ax2.set_title('sin, nfp='+str(self.nfp))
                if colormap_mode:
                    modecos = np.linspace(1, len(fourier.content)-1, len(fourier.content)//2)
                    modesin = np.linspace(len(fourier.content)-1, 1, len(fourier.content)//2)
                    phi = np.linspace(0, 2*np.pi*(1-1/len_phi), len_phi)
                    ax1.pcolormesh(phi, modecos, np.real(fourier.content)[len(fourier.content)//2:])
                    ax2.pcolormesh(phi, modesin, np.real(fourier.content)[:len(fourier.content)//2])
                else:
                    ax1.plot(phis,np.real(fourier.content)[len(fourier.content)//2:].T)
                    ax2.plot(phis,np.real(fourier.content)[:len(fourier.content)//2].T)
            else:
                plt.rcParams['figure.figsize'] = [12,3]
                ax1 = plt.subplot(131)
                ax1.set_title('cos, nfp='+str(self.nfp))
                ax2 = plt.subplot(132)
                ax2.set_title('constant, nfp='+str(self.nfp))
                ax3 = plt.subplot(133)
                ax3.set_title('sin, nfp='+str(self.nfp))
                if colormap_mode and len(fourier.content) != 1:
                    modesin = np.linspace(2, len(fourier.content)-1, len(fourier.content)//2)
                    modecos = np.linspace(len(fourier.content)-1, 2, len(fourier.content)//2)
                    phi = np.linspace(0, 2*np.pi*(1-1/len_phi)/self.nfp, len_phi)
                    print('phi',phi.shape)
                    print('modesin',modesin.shape)
                    print('modecos',modecos.shape)
                    print('np.real(fourier.content)[len(fourier.content)//2+1:]', np.real(fourier.content)[len(fourier.content)//2+1:].shape)
                    ax1.pcolormesh(phi, modesin, np.real(fourier.content)[len(fourier.content)//2+1:])
                    ax3.pcolormesh(phi, modecos, np.real(fourier.content)[:len(fourier.content)//2])
                else:
                    ax1.plot(phis,np.real(fourier.content)[len(fourier.content)//2+1:].T)
                    ax3.plot(phis,np.real(fourier.content)[:len(fourier.content)//2].T)

                ax2.plot(phis,np.real(fourier.content)[len(fourier.content)//2])
        else:
            ax1 = plt.subplot(121)
            ax1.set_title('Real, nfp='+str(self.nfp))
            ax2 = plt.subplot(122)
            ax2.set_title('Imaginary, nfp='+str(self.nfp))

            if colormap_mode:
                mode = np.linspace(-len(content)+1, len(content)-1, len(content))
                phi = np.linspace(0, 2*np.pi*(1-1/len_phi)/self.nfp, len_phi)
                ax1.pcolormesh(phi, mode, np.real(content))
                ax2.pcolormesh(phi, mode, np.imag(content))
            else:
                ax1.plot(phis,np.real(content).T)
                ax2.plot(phis,np.imag(content).T)
        plt.show()

    def display(self, complex = False, size=(100,100), avg_clim = False):
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
        chi = np.linspace(0, 2*np.pi*0.99, size[0])
        phi = np.linspace(0, 2*np.pi*0.99/self.nfp, size[1])
        f = self.get_lambda()
        eval = f(chi, phi.reshape(-1,1))
        plt.pcolormesh(chi, phi, np.real(eval))
        plt.title('ChiPhiFunc, real component')
        plt.xlabel('chi')
        plt.ylabel('phi')
        if avg_clim:
            clim = np.average(np.abs(np.real(eval)))
            plt.clim(-clim, clim)
            plt.colorbar(extend='both')
        else:
            plt.colorbar()
        plt.show()
        if complex:
            plt.pcolormesh(chi, phi, np.imag(eval))
            plt.title('ChiPhiFunc, imaginary component')
            plt.xlabel('chi')
            plt.ylabel('phi')
            if avg_clim:
                clim = np.average(np.abs(np.imag(eval)))
                plt.clim(-clim, clim)
                plt.colorbar(extend='both')
            else:
                plt.colorbar()
            plt.show()

    @jit
    def fft(self): # nfp-dependent!!
        ''' FFT the content and returns as a ChiPhiFunc '''
        return(ChiPhiFunc(jnp.fft.fft(self.content, axis=1), self.nfp))

    @jit
    def ifft(self): # nfp-dependent!!
        ''' IFFT the content and returns a ChiPhiFunc '''
        return(ChiPhiFunc(jnp.fft.ifft(self.content, axis=1), self.nfp))

    ''' I.1.6 Utilities '''
    # no need to jit
    def export_trig(self): # nfp-dependent!!
        '''
        Converting fourier coefficients into exponential coeffs used in
        ChiPhiFunc's internal representation. Only used during super().__init__
        Does not copy.
        '''
        util_matrix_chi = np.linalg.inv(self.fourier_to_exp_op())
        # Apply the conversion matrix on chi axis
        return(ChiPhiFunc(util_matrix_chi @ self.content, self.nfp))

    # no need to jit
    def export_single_nfp(self):
        ''' Outputs a ChiPhiFunc with just 1 nfp. '''
        new_content = jnp.tile(self.content, (1,self.nfp))
        return(ChiPhiFunc(new_content, 1))

# For JAX use
tree_util.register_pytree_node(ChiPhiFunc,
                               ChiPhiFunc._tree_flatten,
                               ChiPhiFunc._tree_unflatten)

''' I.2 Utilities '''

# Not supported by JAX
def integrate_phi_simpson(content, dx = 'default', periodic = False):
    '''
    Integrates a function on a grid using Simpson's method.
    Produces a content where values along axis 1 is the input content's
    integral.
    periodic is a special mode that assumes integrates a content over a period.
    It assumes that the grid function is periodic, and does not repeat the first
    grid's value.
    A usual content has the first cell's LEFT edge at 0
    and the last cell's RIGHT edge at 2pi.
    A specal dx is provided for a grid that has the first cell's LEFT edge at 0
    and the last cell's LEFT edge at 2pi.
    NOTE
    Cell values are ALWAYS taken at the left edge.
    nfp dependence is NOT HANDLED HERE.
    '''
    raise NotImplementedError('Simpson integrals not implemented in JAX')
    # len_chi = content.shape[0]
    # len_phi = content.shape[1]
    # if dx == 'default':
    #     dx = 2*np.pi/len_phi
    # if periodic:
    #     # The result of the integral is an 1d array of chi coeffs.
    #     # This integrates the full period, and needs to be wrapped.
    #     # the periodic=False option does not integrate the full period and
    #     # does not wrap.
    #     new_content = scipy.integrate.simpson(\
    #         wrap_grid_content_jit(content),\
    #         dx=dx,\
    #         axis=1\
    #         )
    #     return(np.array([new_content]).T)
    # else:
    #     # Integrate up to each element's grid
    #     integrate = lambda i_phi : scipy.integrate.simpson(content[:,:i_phi+1], dx=dx)
    #     out_list = Parallel(n_jobs=n_jobs, backend=backend, require=require)(
    #         delayed(integrate)(i_phi) for i_phi in range(len_phi)
    #     )
    #     return(np.array(out_list).T) # not nfp-dependent

# Not supported by JAX
def integrate_phi_spline(content, mode, dx = 'default', periodic=False,
    diff=False, diff_order=None):
    '''
    Implementation of spline-based integrate_phi using Parallel.
    nfp dependence is NOT HANDLED HERE.
    mode can be 'b_spline' or 'cubic_spline'
    '''
    raise NotImplementedError('Spline integral not implemented in JAX')
    # len_chi = content.shape[0]
    # len_phi = content.shape[1]
    # # if dx == 'default':
    # #     dx = 2*np.pi/len_phi
    # #     # purely real.
    #
    # content_looped = wrap_grid_content_jit(content)
    # # Separating real and imag components
    # content_re = np.real(content_looped)
    # content_im = np.imag(content_looped)
    # content_looped = np.concatenate((content_re, content_im), axis=0)
    # phis = np.linspace(0, np.pi*2, len_phi+1)
    #
    # def generate_and_integrate_spline(i_chi):
    #     if mode == 'b_spline':
    #         new_spline = scipy.interpolate.make_interp_spline(phis, content_looped[i_chi], bc_type = 'periodic')
    #         if diff:
    #             return(scipy.interpolate.splder(new_spline, n=diff_order))
    #         print('Waring! B-spline antiderivative is known to produce a small constant offset to the result.')
    #         return(scipy.interpolate.splantider(new_spline))
    #     elif mode == 'cubic_spline':
    #         new_spline = scipy.interpolate.CubicSpline(phis, content_looped[i_chi], bc_type = 'periodic')
    #         if diff:
    #             return(new_spline.derivative(diff_order))
    #         return(new_spline.antiderivative())
    #     else:
    #         raise AttributeError('Spline mode \''+str(mode)+'\' is not recognized.')
    # A list of integrated splines
    # integrate_spline_list = Parallel(n_jobs=n_jobs, backend=backend, require=require)(
    #     delayed(generate_and_integrate_spline)(i_chi) for i_chi in range(len_chi*2)
    # )
    #
    # if periodic:
    #     # The result of the integral is an 1d array of chi coeffs.
    #     # This integrates the full period, and needs to be wrapped.
    #     # the periodic=False option does not integrate the full period and
    #     # does not wrap.
    #     evaluate_spline_2pi = lambda spline: spline(2*np.pi)
    #     out_list = Parallel(n_jobs=n_jobs, backend=backend, require=require)(
    #         delayed(evaluate_spline_2pi)(spline) for spline in integrate_spline_list
    #     )
    #     out_list = np.array(out_list)
    #     out_list = out_list[:len_chi]+1j*out_list[len_chi:]
    #     return(out_list[:, None])
    # else:
    #     evaluate_spline = lambda spline, phis : spline(phis)
    #     out_list = Parallel(n_jobs=n_jobs, backend=backend, require=require)(
    #         delayed(evaluate_spline)(spline, phis[:-1]) for spline in integrate_spline_list
    #     )
    #     out_list = np.array(out_list)
    #     out_list = out_list[:len_chi]+1j*out_list[len_chi:]
    #     return(out_list) # not nfp-dependent

# # pseudo-spectral method is not available for now. The JAX implementation is slow.
# # Because it need to be re-compiled for every different (n). The non-JAX
# # implementation is fast but cannot be jitted. n_phi is fixed for every
# # equilibrium. It is ideal to treat this as a global constant but at compile time
# # n_phi is not known. Fixing n_phi for each execution of the code also
# # makes plotting and debugging using existng configurations a great pain.
# @partial(jit, static_argnums=(0,))
@lru_cache(maxsize=10)
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
        topc = 1 / np.tan(np.arange(1, n2 + 1) * h / 2)
        temp = jnp.concatenate((topc, -np.flip(topc[0:n1])))
    else:
        topc = 1 / np.sin(np.arange(1, n2 + 1) * h / 2)
        temp = jnp.concatenate((topc, np.flip(topc[0:n1])))


    # Calculating the Toeplitz matrix
    col1 = jnp.array([jnp.concatenate((jnp.array([0]), 0.5 * ((-1) ** kk) * temp))]).T
    col1 = jnp.concatenate([col1, jnp.zeros((n, n-1))], axis=1)
    row1 = -col1.T

    raw = col1 + row1
    toeplitz = jnp.zeros_like(raw)
    for i in range(n):
        toeplitz = toeplitz+raw
        raw = jnp.pad(raw,(1,0))[:n, :n]
    D = 2 * jnp.pi / (2*jnp.pi) * toeplitz
    return D
#
# def dphi_op_pseudospectral_2(n_phi):
#     out = spectral_diff_matrix(n_phi, xmin=0, xmax=2*np.pi)
#     return(out) # not nfp-dependent



''' II. Deconvolution ("dividing" chi-dependent terms) '''
@jit
def get_O_O_einv_from_A_B(chiphifunc_A:ChiPhiFunc, chiphifunc_B:ChiPhiFunc, rhs_ref:ChiPhiFunc):
    '''
    Get O, O_einv and vector_free_coef that solves the eqaution system
    O y = (A + B dchi) y = RHS <=>
      y = O_einv - vec_free * vec_free_coef
    Or in code format,
    y = (np.einsum('ijk,jk->ik',O_einv,chiphifunc_rhs_content) + vec_free * vector_free_coef)
    Inputs: -----
    A, B: ChiPhiFunc coefficients

    rhs_ref: a reference RHS to know the number of component
    the method need to solve for. Can be incomplete as long as
    the number of chi component is right.

    Outputs: -----
    O_matrices,
    O_einv,
    vector_free_coef: A (n+1*len_phi)-component array
    Y_nfp: the nfp of Y.
    '''
    rank_rhs = rhs_ref.content.shape[0]
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
    A_conv_matrices = conv_tensor(chiphifunc_A_content, rhs_ref, 1)
    O_matrices = O_matrices + A_conv_matrices

    dchi_matrix = dchi_op(rank_rhs+1, False)
    B_conv_matrices = conv_tensor(chiphifunc_B_content, rhs_ref, 1)
    O_matrices = O_matrices + jnp.einsum('ijk,jl->ilk',B_conv_matrices,dchi_matrix)

    O_einv = batch_matrix_inv_excluding_col(O_matrices, rhs_ref)
    O_einv = np.concatenate((O_einv[:i_free], np.zeros((1,O_einv.shape[1],O_einv.shape[2])), O_einv[i_free:]))
    O_free_col = O_matrices[:,i_free,:]

    vector_free_coef = np.einsum('ijk,jk->ik',O_einv, O_free_col)#A_einv@A_free_col
    vector_free_coef[i_free] = -np.ones((vector_free_coef.shape[1]))

    return(O_matrices, O_einv, -vector_free_coef, max(chiphifunc_A.nfp, chiphifunc_B.nfp))



''' III. Grid 1D deconvolution (used for "dividing" a chi-dependent quantity)'''
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
# Codes written in this part are specifically for 1D deconvolution used for ChiPhiFunc.

''' III.2 va has 1 more component than vb '''
# Input is order-independent.
@jit
def batch_matrix_inv_excluding_col(in_matrices, rhs_ref):
    '''
    Invert an (n,n) submatrix of a (m>n+1,n+1) rectangular matrix by taking the center
    n-1 rows and excluding the ind_col'th column. "Taking the center n rows" is motivated
    by the RHS being rank n-1

    Input: -----
    (m,n+1,len_phi) matrix A
    ind_col < n+1

    Return: -----
    (m,m,len_phi) matrix A_inv
    '''
    rank_rhs = rhs_ref.content.shape[0]
    i_free = (rank_rhs+1)//2 # We'll always use Yn0 or Yn1p as the free var.
    if (in_matrices.shape[0] - in_matrices.shape[1])%2==0:
        return(jnp.nan)
        # raise AttributeError('This method takes rows from the middle. The array'\
        # 'shape must have an odd difference between row and col numbers.')
    if in_matrices.ndim != 3:
        return(jnp.nan)
        # raise ValueError("Input should be 3d array")
    n_row = in_matrices.shape[0]
    n_col = in_matrices.shape[1]
    n_phi = in_matrices.shape[2]
    n_clip = (n_row-n_col+1)//2 # How much is the transposed array larger than Yn
    if n_row<=n_col:
        return(jnp.nan)
        # raise ValueError("Input should have more rows than cols")

    if i_free>=n_col:
        return(jnp.nan)
        # raise ValueError('i_free should be smaller than column number')

    # Remove specfied column (slightly faster than delete)
    # and remove extra rows (take n_col-1 rows from the center)
    rows_to_remove = (n_row-(n_col-1))//2
    sub = in_matrices[:,jnp.arange(in_matrices.shape[1])!=i_free,:][rows_to_remove:-rows_to_remove, :, :]
    sub = jnp.moveaxis(sub,2,0)
    sqinv = jnp.linalg.inv(sub)
    sqinv = jnp.moveaxis(sqinv,0,2)
    padded = jnp.pad(sqinv, (rows_to_remove, rows_to_remove, 0))
    # padded[rows_to_remove:-rows_to_remove, rows_to_remove:-rows_to_remove,:] = sqinv
    return(padded[n_clip:-n_clip]) # not nfp-dependent

def batch_matrix_inv_excluding_col_orig(in_matrices, ind_col):
    if (in_matrices.shape[0] - in_matrices.shape[1])%2==0:
        raise AttributeError('This method takes rows from the middle. The array'\
        'shape must have an odd difference between row and col numbers.')
    if in_matrices.ndim != 3:
        raise ValueError("Input should be 3d array")
    n_row = in_matrices.shape[0]
    n_col = in_matrices.shape[1]
    n_phi = in_matrices.shape[2]
    n_clip = (n_row-n_col+1)//2 # How much is the transposed array larger than Yn
    if n_row<=n_col:
        raise ValueError("Input should have more rows than cols")

    if ind_col>=n_col:
        raise ValueError('ind_col should be smaller than column number')

    # Remove specfied column (slightly faster than delete)
    # and remove extra rows (take n_col-1 rows from the center)
    rows_to_remove = (n_row-(n_col-1))//2
    sub = in_matrices[:,np.arange(in_matrices.shape[1])!=ind_col,:][rows_to_remove:-rows_to_remove, :, :]
    sub = np.moveaxis(sub,2,0)
    sqinv = np.linalg.inv(sub)
    sqinv = np.moveaxis(sqinv,0,2)
    padded = np.zeros((n_row, n_row, n_phi), dtype = np. complex128)
    padded[rows_to_remove:-rows_to_remove, rows_to_remove:-rows_to_remove,:] = sqinv
    return(padded[n_clip:-n_clip]) # not nfp-dependent

''' III.3 Convolution operator generator and ChiPhiFunc.content wrapper '''
# A jitted vectorized version of np.roll
roll_axis_0 = lambda a, shift: jnp.roll(a, shift, axis=0)
batch_roll_axis_0 = jit(vmap(roll_axis_0, in_axes=1, out_axes=1))
@partial(jit, static_argnums=(2,))
def conv_tensor(content: jnp.ndarray, content2: jnp.ndarray, offset:int):
    '''
    Generate a tensor_coef (see looped_solver.py) convolving a ChiPhiFunc
    to another along axis 0.
    For multiplication in FFT space during ODE solves.
    The convolution is done by:
    x2_conv_y2 = np.einsum('ijk,jk->ik',conv_x2, y2.content)

    Input: -----

    content: A content matrix

    content2 and offset: content2 is a reference matrix, and offset is an int.
    The convolution matrices act on a content with content2.shape[0]+offset
    chi components.

    Output: -----

    A (content.shape[0]+n_dim-1, n_dim, content.shape[1]), representing
    a stack of content.shape[1] convolution matrices. The first two
    axes represent the row and column of a single convolution matrix.
    '''
    n_dim = content2.shape[0]+offset
    len_phi = content.shape[1]
    content_padded = jnp.concatenate((content, jnp.zeros((n_dim-1, len_phi))), axis=0)
    content_padded = jnp.tile(content_padded[:, None, :], (1, n_dim, 1))
    shift = jnp.array([range(n_dim)])
    return(batch_roll_axis_0(content_padded, shift))

# Generates a 4D convolution operator in the phi axis for a 3d "tensor coef"
# (see looped_solver for explanation)
# @njit(complex128[:,:,:,:](complex128[:,:,:]))
def fft_conv_tensor_batch(source):
    len_a = source.shape[0]
    len_b = source.shape[1]
    len_phi = source.shape[2]
    roll = jit_fftfreq_int(len_phi)
    out = np.zeros((len_a, len_b, len_phi, len_phi), dtype = np.complex128)
    # Where does 0 elem start/end
    split = (len_phi+1)//2
    split_b = np.roll(np.arange(len_phi%2, len_phi+len_phi%2, dtype = np.int64),split)
    for i in range(len_phi):
        index = roll[i]
        # Equivalent to
        # out[:, :, i, :] = np.roll(source, index, axis = 2)
        if index >= 0:
            out[:, :, i, index:] = source[:, :, :len_phi - index]
            out[:, :, i, :index] = source[:, :, len_phi - index:]
        else:
            out[:, :, i, index:] = source[:, :, :-index]
            out[:, :, i, :index] = source[:, :, -index:]
        split_start = min(split, split_b[i])
        split_end = max(split, split_b[i])
        out[:, :, i, split_start:split_end] = 0
    return(np.transpose(out, (0,1,3,2))/len_phi)  # not nfp-dependent

''' IV. Solving linear PDE in phi grids '''

''' IV.1 Solving the periodic linear PDE (a + b * dphi + c * dchi) y = f(phi, chi) '''
# Solves simple linear first order ODE systems in batch:
# (coeff_phi d/dphi + coeff) y = f. ( y' + p_eff*y = f_eff )
# (Dchi = +- m * 1j)
# All inputs are content matrices.
# coeff and coeff_dp are assumed periodic.
# P is preferrably non-resonant for small p amplitude.
# All coeffs' CONTENTS are point-wise multiplied to f, dpf or dcf's content.
# -----
# To apply nfp, use coeff, coeff_dp, f
# -----
# The asymptotic mode ASSUMES coeff/coeff_dp is non-zero.
# It works better when the amplitude of coeff/coeff_dp > ~17.
# where the regular method diverges because of very large
# value of integral(coeff/coeff_dp), which is the exponent in
# the integration factor.
# asymptotic_order decides how many order to truncate at.
# If optimum truncation is reached before that, the optimum
# truncation will be used instead.# a dphi operator acting on the fft of
# a content along axis=1
# Initial condition for y
# nfp dependence NOT HANDLED HERE!
solve_1d = lambda p_eff_1d, f_eff_1d, asymptotic_mode : \
    solve_integration_factor_single(
        p_eff_1d,
        f_eff_1d,
        asymptotic_mode=asymptotic_mode,
        fft_max_freq=fft_max_freq)
solve_1d_batch = jit(vmap(solve_1d, in_axes=0, out_axes=0))
# @jit

# todo: separate auto and non-auto
def solve_integration_factor(coeff_arr, coeff_dp_arr, f_arr, fft_max_freq=None): # not nfp-dependent
    '''
    Input: -----
    coeff_arr, coeff_dp_arr, f_arr
    '''

    len_phi = f.shape[1]
    len_chi = f.shape[0]

    # Rescale the eq into y'+py=f
    f_eff = f_arr/coeff_dp_arr
    p_eff = coeff_arr/coeff_dp_arr
    f_eff_scaling = jnp.average(jnp.abs(f_eff))
    f_eff = f_eff/f_eff_scaling
    # print('solve_integration_factor: average p_eff:', np.average(np.abs(p_eff)))
    # print('solve_integration_factor: average f_eff:', f_eff_scaling)


    if p_eff.shape[0]!=f_eff.shape[0]:
        # raise AttributeError('p_eff and f_eff has different component numbers!')
        return jnp.nan
    # We always assume f is phi-dependent.
    if p_eff.shape[1] != f_eff.shape[1]:
        if p_eff.shape[1]==1:
            p_eff = p_eff+jnp.zeros_like(f)
        else:
            return(jnp.nan) # Mismatched length

    effective_dx = 2*np.pi/(len_phi)

    p_amp = jnp.mean(jnp.abs(jnp.real(p_eff[i])), axis=1)
    asymptotic_mode = jnp.where(p_amp < config.asymptotic_threshold, -1, 1)
    out_arr = solve_1d_batch(p_eff, 1, f_eff, asymptotic_mode)
    return(out_arr*f_eff_scaling)

def solve_integration_factor_single(p_eff, f_eff, \
    asymptotic_mode, fft_max_freq=None): # not nfp-dependent
    # TODO: WRITE ASYMPTOTIC OPTIMAL TRUNCATION IN VMAP
    # Asymptotic mode
    if asymptotic_mode == -1:
        ai = f_eff/p_eff # f/p
        integration_factor = ai.copy()
        for i in range(asymptotic_order):
            # ai is periodic. We use the non-looped value to ensure
            # that dphi by fft functions correctly
            ai_new = -(ChiPhiFunc(ai,1).dphi().content)/p_eff
            if np.max(np.abs(ai_new)) > np.max(np.abs(ai)):
                print('Optimum truncation at order', i+1)
                print('Amplitude of the truncation term:', np.amax(np.abs(ai), axis = 1))
                break
            ai = ai_new
            integration_factor += ai
        # No longer outputs the truncation term for error tracking
        return(integration_factor*f_eff_scaling) # , np.amax(np.abs(ai), axis = 1)*f_eff_scaling)


    elif asymptotic_mode == 1:
        if fft_max_freq is None:
            raise AttributeError('fft_max_freq must be provided!')
        target_length = fft_max_freq*2
        p_fft = fft_filter(np.fft.fft(p_eff, axis = 1), target_length, axis=1)
        f_fft = fft_filter(np.fft.fft(f_eff, axis = 1), target_length, axis=1)

        # Find comps with p=0
        remove_zero = np.all(p_fft==0, axis=1)
        diff_matrix_single, diff_matrix = fft_dphi_op(target_length, remove_zero)
        conv_matrix = fft_conv_op_batch(p_fft)
        inv_dxpp = np.linalg.inv(diff_matrix + conv_matrix)
        # for now ignore p=0 comps.
        inv_dxpp[remove_zero,:,:] = 0
        sln_fft = (inv_dxpp@f_fft[:,:,None])[:,:,0]
        # RHS_with_sln_fft = ((diff_matrix + conv_matrix)@sln_fft[:,:,None])[:,:,0]
        # print('Recovering RHS? - RHS is recovered. Maybe the operator is incorrect.')
        # ChiPhiFunc(RHS_with_sln_fft,1).ifft().display_content()
        sln = np.fft.ifft(fft_pad(sln_fft, len_phi, axis = 1), axis = 1)
        # Calculating components with p=0 (now the ODE is y'=f)
        # These components are just phi integrals of f.
        # NOTE: BC is set to ZERO AVERAGE!
        if np.any(remove_zero):
            # nfp is not treated here.
            zero_comps = ChiPhiFunc(f_eff[remove_zero], nfp=1).integrate_phi_fft(zero_avg=True).filter('low_pass', fft_max_freq)
            sln[remove_zero] = zero_comps.content
        return(sln*f_eff_scaling)

    else:
        raise AttributeError('ODE solver mode not recognized')
        ''' Integration factor is now depreciated. It underperforms FFT and is unstable. '''
        # # The integrand of the integration factor (I = exp(int_p))
        # if integral_mode == 'simpson':
        #     int_p = integrate_phi_simpson(p_eff, periodic=False, dx = effective_dx)
        #     int_p_2pi = integrate_phi_simpson(p_eff, periodic=True, dx = effective_dx)
        # elif integral_mode[-6:] == 'spline' or integral_mode == 'asymptotic' :
        #     int_p = integrate_phi_spline(p_eff, integral_mode, periodic=False, dx = effective_dx)
        #     int_p_2pi = integrate_phi_spline(p_eff, integral_mode, periodic=True, dx = effective_dx)
        # else:
        #     raise AttributeError('integral_mode not recognized.')
        # # Solving with intermediate p by integrating factor
        # exp_neg2pi = np.exp(-int_p_2pi)
        # exp_phi = np.exp(int_p)
        # exp_negphi = np.exp(-int_p)
        # integrand = f*exp_phi
        #
        # # Here integration_factor_2pi can no longer be evaluated by
        # # integrate_phi_simpson(periodic=True), because the integrand
        # # is not generally periodic.
        # if integral_mode == 'simpson':
        #     integration_factor = integrate_phi_simpson(integrand, periodic=False)
        #     integration_factor_2pi = integrate_phi_simpson(integrand, periodic=True)
        # elif integral_mode[-6:] == 'spline':
        #     integration_factor = integrate_phi_spline(integrand, integral_mode, periodic=False)
        #     integration_factor_2pi = integrate_phi_spline(integrand, integral_mode, periodic=True)
        # else:
        #     raise AttributeError('integral_mode not recognized.')
        #     # The integration constant. Derived from the periodic boundary condition.
        #
        # integration_factor = integration_factor*exp_negphi
        # integration_factor_2pi = integration_factor_2pi*exp_neg2pi
        #
        # # If the integral of p is periodic, I is periodic.
        # # The BVP cannot get solved.
        # if np.average(np.abs(int_p[:,0] - int_p[:,-1])) < np.max(f_eff)*noise_level_periodic:
        #     print('I(phi) is periodic. Cannot yield an unique solution using only periodic BC.')
        #     print('returning integration factor.')
        #     c1=0
        # else:
        #     # exp_neg2pi may contain 1's.
        #     exp_neg2pi[exp_neg2pi == 1] = np.inf
        #     c1=integration_factor_2pi/(1-exp_neg2pi)
        # out = c1*exp_negphi+integration_factor
        # out = out[:,:-1]
        # return(out*f_eff_scaling)

# ONLY USED IN solve_integration_factor().
# A stack of dphi operators acting on the fft of
# a content along axis=1. Shape is [len_phi, len_phi, len_chi].
# Only used in solve_integration_factor.
# remove_zero is a list of booleans that when true, replaces the [0,0] element
# in the corresponding row of fft_freq
# with np.inf to accomodate the special cases where
# the ODE looks like y'=f.
# nfp dependence NOT HANDLED HERE!
def fft_dphi_op(len_phi, remove_zero = np.array([False])):  # not nfp-dependent
    len_chi = len(remove_zero)
    fft_freq = jit_fftfreq_int(len_phi)
    matrix = np.identity(len_phi) * 1j * fft_freq
    tiled_matrix = np.tile(matrix[:,:], (len_chi, 1, 1))
    tiled_matrix[remove_zero,:,:]=np.nan
    return(matrix, tiled_matrix)

# ONLY USED IN solve_integration_factor().
# A convolution operator acting convolving
# a fft of len_phi with source.
# see paper note for correct format of this matrix
# Sadly jit doesn't support np.fft.
# source has shape [n_chi, n_phi]
# returns [n_chi, n_phi_row, n_phi_col]
# nfp dependence NOT HANDLED HERE!
def fft_conv_op_batch(source):
    len_chi = source.shape[0]
    len_phi = source.shape[1]
    out = np.zeros((len_chi, len_phi, len_phi), dtype = np.complex128)
    # How much to roll source
    roll = jit_fftfreq_int(len_phi)
    # Where does 0 elem start/end
    split = (len_phi+1)//2
    split_b = np.roll(np.arange(len_phi%2, len_phi+len_phi%2, dtype = np.int64),split)
    for i in range(len_phi):
        index = roll[i]
        out[:, i, :] = np.roll(source, index, axis = 1)
        split_start = min(split, split_b[i])
        split_end = max(split, split_b[i])
        out[:, i,split_start:split_end] = 0
    return(np.transpose(out, (0,2,1))/len_phi) # not nfp-dependent

# For solving the periodic linear 1st order ODE (coeff + coeff_dp*dphi + coeff_dc*dchi) y = f(phi, chi)
# using integral factor.
# -- Input --
# coeffs are constants or contents. f is a content (see ChiPhiFunc's description).
# Contant mode and offset are for evaluating the constant component equation
# dphi y0 = f0.
# -- Output --
# y is a ChiPhiFunc's content
def solve_integration_factor_chi(coeff, coeff_dp, coeff_dc, f, \
    integral_mode,
    asymptotic_order=asymptotic_order, fft_max_freq=None): # not nfp-dependent

    len_chi = f.shape[0]
    len_phi = f.shape[1]
    # Chi harmonics
    ind_chi = len_chi-1
    # Multiplies each row with its corresponding mode number.
    mode_chi = 1j*np.linspace([-ind_chi], [ind_chi], len_chi, axis=0)

    coeff_eff = (coeff_dc*mode_chi + coeff)/coeff_dp
    f_eff = f/coeff_dp

    return(
        solve_integration_factor(coeff_eff, 1, f_eff, \
            integral_mode=integral_mode,
            asymptotic_order=asymptotic_order,
            fft_max_freq=fft_max_freq)
    )

# For solving the periodic linear 1st order ODE (dphi+iota*dchi) y = f(phi, chi)
# using integral factor.
# -- Input --
# iota is a constant and f is a ChiPhiFunc.
# Contant mode and offset are for evaluating the constant component equation
# dphi y0 = f0.
# -- Output --
# y is a ChiPhiFunc's content
def solve_dphi_iota_dchi(iota, f, \
    integral_mode='auto',
    asymptotic_order=asymptotic_order, fft_max_freq=None): # not nfp-dependent
    return(
        solve_integration_factor_chi(0, 1, iota, f, \
            integral_mode=integral_mode,
            asymptotic_order=asymptotic_order,
            fft_max_freq=fft_max_freq)
        )

''' V. utilities '''

''' V.1. Low-pass filter for simplifying tensor to invert '''
# Shorten an array in FFT representation to leave only target_length elements.
# (which correspond to a low-pass filter with k<target_length/2*nfp)
# by removing the highest frequency modes. The resulting array can be IFFT'ed.
# Target length is the number of harmonics to keep in one field period.
@partial(jit, static_argnums=(2,))
def fft_filter(fft_in:jnp.ndarray, target_length, axis): # not nfp-dependent
    todo
    # write in term of reference chiphifunc.
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
def fft_pad(fft_in, target_length, axis): # not nfp-dependent
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
