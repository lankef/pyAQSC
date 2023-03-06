import numpy as np
import timeit
import warnings
import scipy.signal
import scipy.fftpack
import math # factorial in dphi_direct
from math_utilities import *
from matplotlib import pyplot as plt
# Numba doesn't support scipy methods. Therefore, scipy integrals are sped up
# with joblib.
import scipy.integrate
import scipy.interpolate
from joblib import Parallel, delayed
from functools import lru_cache # import functools for caching
from numba import jit, njit, prange
from numba.types import Tuple
# import jit value types. Bizzarely using int32 causes typing issues
# in solve_underdet_degen with njit(parallel=True) because
# it seem to insist prange(...shape[0]) is int64, even when int64 is not imported.
from numba import complex128, int64, float64, boolean
# Configurations
import config

# The 'pseudo_spectral' derivative operator uses pyQSC.
if config.use_pyQSC:
    from qsc import spectral_diff_matrix

''' Debug options and loading configs '''
# When True, throws error if a power-fourier series coefficient (ChiPhiFunc)
# contains np.nan.
check_nan_content = True

# integrate_chi() should not be run on a ChiPhiFunc with a chi-independent component,
# because this produces a non-periodic function. However, zero-checking the
# component is not feasible, because cancellation is often not exact in numerical
# evaluations. Instead, we check if the maximum amplitude of the chi-independent
# component is greater than this noise_level_int
# The typing is for numba.
noise_level_int = np.float64(1e-5)

# tolerance on how periodic I(phi) is during integration factor.
# When I(phi) is periodic, any C1 can satisfy the periodic BC,
# The assiciated error is thrown by comparing the difference
# between the integration factor's first and last element to
# this variable.
noise_level_periodic = 1e-10

# Loading configurations
diff_mode = config.diff_mode
integral_mode = config.integral_mode
two_pi_integral_mode = config.two_pi_integral_mode
# Maximum allowed asymptotic series order for y'+py=f
asymptotic_order = config.asymptotic_order
# Debug mode
debug_mode = config.debug_mode
# Tracks the max and avg values of intermediate results. Compare with output to
# identify rounding errors.
debug_max_value = []
debug_avg_value = []
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
# Generate chi differential operator diff_matrix. diff_matrix@f.content = dchi(f).content
# invert = True generates anti-derivative operator. Cached for each new Chi length.
# -- Input:
# len_chi: length of Chi series.
# invert_mode=True only used for int_chi(). Should be False by default,
# but numba doesn't support default parameters.
# -- Output: 2d matrix.
@njit(complex128[:,:](int64, boolean))
def dchi_op(len_chi, invert=False):
    ind_chi = len_chi-1
    mode_chi = np.linspace(-ind_chi, ind_chi, len_chi)
    if invert:
        if len_chi%2==1:
            # dchi operator for odd order n should not be invertible,
            # because the constant element integrates to a non-periodic component.
            # However, zero-checking the constant element is also impossible
            # because of numerical noise.
            # The only way to deal with this right now is setting the constant
            # component of the integral oeprator to 0, and add a magnitude check
            # in int_chi().
            mode_chi[len_chi//2] = np.inf
        return(np.diag(-1j/mode_chi))
    return np.diag(1j*mode_chi) # not nfp-sensitive

# Used for wrapping grid content. Defined outside the ChiPhiFunc class so that
# it can be used in @njit compiled methods. Defined in front of the class because
# numba seem to care about the order.
@njit(complex128[:,:](complex128[:,:]))
def wrap_grid_content_jit(content):
    len_chi = content.shape[0]
    len_phi = content.shape[1]
    content_looped = np.zeros((len_chi, len_phi+1), dtype=np.complex128)
    content_looped[:, :-1] = content
    content_looped[:, -1:] = content[:,:1]
    return(content_looped) # not nfp-sensitive

# Used in operator *. Transposes 2 contents with equal phi grid (col) number,
# then loop over rows (now chi coeffs) and np.convolve().
@njit(complex128[:,:](complex128[:,:], complex128[:,:]), parallel=True)
def mul_grid_jit(a, b):
    aT = a.T
    bT = b.T
    phi_dim = a.shape[1]
    # Each row is a grid, and each column represents a chi mode.
    # End product of N, M dim vectors have N+M-1 dim.
    out_transposed = np.zeros((phi_dim, a.shape[0]+b.shape[0]-1), dtype=np.complex128)
    for i in prange(phi_dim):
        out_transposed[i,:] = np.convolve(aT[i], bT[i])
    return out_transposed.T # not nfp-sensitive

# An accelerated sum that aligns the center-point of 2-d arrays and zero-broadcasts the edges.
# Input arrays must both have even/odd cols/rows
# (such as (3,2), (13,6))
# Copies arguments
# -- Input: 2 2d arrays.
# -- Output: 2d array
# Switches between a compiled and a non-compiled implementation
# depending on debug_mode (because print and appending global)
# doesn't work in compiled methods.
@njit(complex128[:,:](complex128[:,:], complex128[:,:], int64))
def add_jit(a, b, sign):
    shape = (max(a.shape[0], b.shape[0]),max(a.shape[1],b.shape[1]))
    out = np.zeros(shape, dtype=np.complex128)
    a_pad_row = (shape[0] - a.shape[0])//2
    a_pad_col = (shape[1] - a.shape[1])//2
    b_pad_row = (shape[0] - b.shape[0])//2
    b_pad_col = (shape[1] - b.shape[1])//2
    out[a_pad_row:shape[0]-a_pad_row,a_pad_col:shape[1]-a_pad_col] = a
    out[b_pad_row:shape[0]-b_pad_row,b_pad_col:shape[1]-b_pad_col] += b*sign
    return(out)

# Wrapper for mul_grid_jit. Handles int power.
# -- Input: self and an int
# -- Output: a new ChiPhiFunc
@njit(complex128[:,:](complex128[:,:], int64))
def pow_jit(content, int_pow):
    new_content = content.copy()
    for i in prange(int_pow-1):
        new_content = mul_grid_jit(new_content, content)
    return(new_content)

# Calculates the max amplitude's order of magnitude
@njit(float64(complex128[:,:]))
def max_log10(input):
    return(np.log10(np.max(np.abs(input)))) # not nfp-sensitive

# Numba ver of np.fft.fftfreq. Used in fft_conv_tensor_batch.
@njit(int64[:](int64))
def jit_fftfreq_int(int_in):
    fftfreq_out = np.zeros(int_in, dtype = np.int64)
    half = int_in//2
    if int_in%2==0:
        fftfreq_out[:half] = np.arange(0,half)
        fftfreq_out[half:] = np.arange(-half,0)
    else:
        fftfreq_out[:half+1] = np.arange(0,half+1)
        fftfreq_out[half+1:] = np.arange(-half,0)
    return(fftfreq_out) # nfp-sensitive!!

# If input is a ChiPhiFunc, average along phi.
# If input is a scalar, do nothing
def phi_avg(in_quant):
    if type(in_quant) is ChiPhiFunc and type(in_quant) is not ChiPhiFuncNull:
        new_content = np.array([
            np.mean(in_quant.content,axis=1)
        ]).T
        return(ChiPhiFunc(new_content,0))
    elif np.isscalar(in_quant):
        return(in_quant)
    else:
        raise TypeError('phi_avg input can only be a scalar or a ChiPhiFunc. '\
        'Actual type is' + str(type(in_quant))) # nfp-sensitive!!

''' I.1 Grid implementation '''
# ChiPhiFunc represents a function of chi and phi in even/odd fourier series in
# chi. The coefficients are assumed phi-dependent with nfp field periods.
# This dependence is stored on n grid points located at (0, ... 2pi(n-1)/n/nfp) in phi.
class ChiPhiFunc:
    # Initializer. Fourier_mode==True converts sin, cos coeffs to exponential
    def __init__(self, content, nfp, fourier_mode=False):
        if debug_mode:
            debug_max_value.append(np.max(np.abs(content)))
            debug_avg_value.append(np.max(np.average(content)))

        if len(content.shape)!=2:
            raise ValueError('ChiPhiFunc content must be 2d arrays.')
        if nfp<0 or not isinstance(nfp, int):
            raise ValueError('Number of field period must be a non-negative int. ')
        if nfp==0 and content.shape[1]!=1:
            raise ValueError('nfp=0 only applies to phi-independent ChiPhiFunc. '\
            'content.shape[1]!=1.')
        # for definind special instances that are similar to nan, except yields 0 when *0.
        # copies and force types for numba
        self.nfp=nfp
        if check_nan_content:
            if np.any(np.isnan(content)):
                raise ValueError('ChiPhiFunc content contains nan element!')
        self.content = np.complex128(content)
        if fourier_mode:
            self.trig_to_exp() # nfp-sensitive!!

    # Obtains the m=index component of this ChiPhiFunc.
    # DOES NOT WORK like list indexing.
    def __getitem__(self, index):
        len_chi = len(self.content)
        # Checking even/oddness and total length
        if len_chi%2==index%2 or np.abs(index)>np.abs(len_chi-1):
            raise ValueError('Cannot get the '+str(index)+
                             '-th element from an n='
                             +str(-1)+' ChiPhiFunc.')
        new_content = np.array([self.content[len_chi//2+index//2]])
        return(ChiPhiFunc(new_content, self.nfp)) # nfp-sensitive!!

    def __setitem__(self, key, newvalue):
        raise NotImplementedError('ChiPhiFunc should only be modified by'\
        ' mathematical operations, not element-wise editing.') # not nfp-sensitive

    ''' I.1.1 Operators '''
    # -self (negative) operator.
    def __neg__(self):
        return type(self)(-self.content, self.nfp) # nfp-sensitive!!

    # self+other, with:
    # a scalar or another ChiPhiFunc of the same implementation
    def __add__(self, other): # not nfp-sensitive
        # When summing two ChiPhiFunc's, only allows summation
        # of the same implementation (fourier or grid)
        if isinstance(other, ChiPhiFuncNull):
            return(other)
        if isinstance(other, ChiPhiFunc):
            if not self.both_even_odd(other):
                raise ValueError('+ can only be evaluated between 2 ChiPhiFuncs '\
                                'that are both even or odd')
            return self.add_ChiPhiFunc(other)
        else:
            if not np.isscalar(other):
                raise TypeError('+ cannnot be evaluated with a vector.')
            return(self.add_const(other))

    # other+self
    def __radd__(self, other): # not nfp-sensitive
        return(self+other)

    # self-other, implemented using addition.
    def __sub__(self, other): # not nfp-sensitive

        # When summing two ChiPhiFunc's, only allows summation
        # of the same implementation (fourier or grid)
        if isinstance(other, ChiPhiFuncNull):
            return(other)
        if issubclass(type(other), ChiPhiFunc):
            if not self.both_even_odd(other):
                raise ValueError('+ can only be evaluated between 2 ChiPhiFuncs '\
                                'that are both even or odd')
            return self.add_ChiPhiFunc(other, sign=-1)
        else:
            if not np.isscalar(other):
                raise TypeError('+ cannnot be evaluated with a vector.')
            return self.add_const(other, sign=-1)

    # other-self
    def __rsub__(self, other): # not nfp-sensitive
        return(-(self-other))

    # other*self, POINTWISE multiplication of ChiPhiFunc with:
    # a scalar
    # another ChiPhiFunc of the same implementation.
    # even*even -> even
    # odd*odd -> odd
    # even*odd -> even

    def __mul__(self, other): # nfp-sensitive!!
        # When summing two ChiPhiFunc's, only allows summation
        # of the same implementation (fourier or grid)
        if isinstance(other, ChiPhiFuncNull):
            if not np.any(self.content):
                return(0)
            return(other)
        if isinstance(other, ChiPhiFunc):
            return(self.multiply(other))
        else:
            if not np.isscalar(other):
                raise TypeError('* cannnot be evaluated with a vector.')
            if other == 0: # to accomodate ChiPhiFuncNull.
                return(0)
            return(type(self)(other * self.content, self.nfp))

    # self*other
    def __rmul__(self, other):
        return(self*other) # not nfp-sensitive

    def __truediv__(self, other): # nfp-sensitive!!
        # When summing two ChiPhiFunc's, only allows summation
        # of the same implementation (fourier or grid)
        if isinstance(other, ChiPhiFuncNull):
            return(other)
        if isinstance(other, ChiPhiFunc):
            if other.content.shape[0]!=1:
                raise TypeError('/ can only be evaluated with a'\
                                    'ChiPhiFuncs with no chi dependence (only 1 row)')
            return(self.multiply(other, div=True))
        else:
            if not np.isscalar(other):
                raise TypeError('/ cannnot be evaluated with a vector.')
            return(type(self)(self.content/other, self.nfp))

    # other/self
    def __rtruediv__(self, other): # nfp-sensitive!!
        if self.content.shape[0]!=1:
            raise TypeError('/ can only be evaluated with a'\
                            'ChiPhiFuncs with no chi dependence (only 1 row)')
        if isinstance(other, ChiPhiFunc):
            return(other.multiply(self, div=True))
        else:
            if not np.isscalar(other):
                raise TypeError('/ cannnot be evaluated with a vector.')
            return(type(self)(other/self.content, self.nfp))

    # other@self, for treating this object as a vector of Chi modes, and multiplying with a matrix
    def __rmatmul__(self, mat): # nfp-sensitive!!
        return type(self)(mat @ self.content, self.nfp)

    # self@other, not supported. Throws an error.
    def __matmul__(self, mat): # not nfp-sensitive
        raise TypeError('ChiPhiFunc is treated as a vector of chi coeffs in @. '\
                       'Therefore, ChiPhiFunc@B is not supported.')

    # self^n, based on self.pow().
    @lru_cache(maxsize=config.max_size)
    def __pow__(self, other):
        if not np.isscalar(other):
            raise TypeError('**\'s other argument must be a non-negative scalar integer.')
        if int(other)!=other:
            raise ValueError('Only integer ** of ChiPhiFunc is supported.')
        if other == 0:
            return(1)
        if other < 1:
            raise ValueError('**\'s other argument must be non-negative.')
        return ChiPhiFunc(pow_jit(self.content, other), self.nfp) # nfp-sensitive!!

    # Used in operator * and /. First stretch the phi axis to match grid locations,
    # Then do pointwise product.
    # -- Input: self and another ChiPhiFunc
    # -- Output: a new ChiPhiFunc
    @lru_cache(maxsize=config.max_size)
    def multiply(self, other, div = False): # nfp-sensitive!!
        if self.nfp!=other.nfp and self.nfp!=0 and other.nfp!=0:
            raise ValueError('nfp does not match!')
        # mul_grid_jit(a, b) never modifies the original. Skip copying.
        a, b = self.stretch_phi_to_match(other, always_copy=False)
        if div:
            b = 1.0/b
        # Now both are stretch to be dim (n_a, n_phi), (n_b, n_phi).
        # Transpose and 1d convolve all constituents.
        return(ChiPhiFunc(mul_grid_jit(a, b), max(self.nfp, other.nfp)))

    # Addition of 2 ChiPhiFunc's.
    # Wrapper for numba method.
    # -- Input: self and another ChiPhiFunc
    # -- Output: a new ChiPhiFunc
    def add_ChiPhiFunc(self, other, sign=1): # nfp-sensitive!!
        if self.nfp!=other.nfp and self.nfp!=0 and other.nfp!=0:
            raise ValueError('nfp does not match!')
        # add_jit(a,b,sign) never modifies the original. Skip copying.
        a,b = self.stretch_phi_to_match(other, always_copy = False)
        # Now that grid points are matched by stretch_phi, we can invoke add_jit()
        # To add matching rows(chi coeffs) and grid points.
        return ChiPhiFunc(add_jit(a,b,sign), max(self.nfp, other.nfp))

    # Addition of a constant with a ChiPhiFunc.
    # Wrapper for numba method.
    def add_const(self, const, sign=1):
        if self.no_const(): # odd series, even number of elems
            if const==0:
                return(self)
            else:
                print('self.content:')
                print(self.content)
                print('other:')
                print(const)
                raise ValueError('A constant + with an even series. ')
        stretched_constant = np.full((1, self.get_shape()[1]), const, dtype=np.complex128)
        return ChiPhiFunc(add_jit(self.content, stretched_constant,sign), self.nfp) # nfp-sensitive!!

    ''' I.1.2 Derivatives, integrals and related methods '''
    # derivatives. Implemented through dchi_op
    def dchi(self, order=1): # nfp-sensitive!!
        out = self.content
        len_chi = len(out)
        if order<0:
            raise AttributeError('dchi order must be positive')
        mode_i = (1j*np.arange(-len_chi+1,len_chi+1,2)[:,None])**order
        out = mode_i * out
        return(type(self)(out, self.nfp))

    def integrate_chi(self, ignore_mode_0=False): # nfp-sensitive!!
        len_chi = self.get_shape()[0]
        temp = np.arange(-len_chi+1,len_chi+1,2,dtype=np.float32)[:,None]
        if len_chi%2==1:
            temp[len(temp)//2]=np.inf
            if np.max(np.abs(self.content[len_chi//2]))>noise_level_int\
            and not ignore_mode_0:
                raise ValueError('Integrand has a significant chi-independent '\
                'component!')
        return(type(self)(-1j * self.content/temp, self.nfp))

    # NOTE: will not be passed items awaiting for conditions.
    def dphi(self, order=1, mode='default'):  # nfp-sensitive!!
        return(ChiPhiFunc(dphi_direct(self.content, order=order, mode=mode)*self.nfp, self.nfp))

    def dphi_iota_dchi(self, iota):  # not nfp-sensitive
        return(self.dphi()+iota*self.dchi())

    # Used to calculate e**(ChiPhiFunc). Only support ChiPhiFunc with no
    # chi dependence
    def exp(self): # nfp-sensitive!!
        if self.get_shape()[0]!=1:
            raise ValueError('exp only supports ChiPhiFunc with no chi dependence!')
        return(ChiPhiFunc(np.exp(self.content), self.nfp))

    # Used for solvability condition. phi-integrate a ChiPhiFunc over 0 to
    # 2pi or 0 to a given phi. The boundary condition output(phi=0) = 0 is enforced.
    # -- Input: self and integral settings:
    # periodic=False evaluates integral from 0 to phi FOR EACH GRID POINT and
    # creates a ChiPhiFunc with phi dependence.
    # periodic=True evaluates integral from 0 to 2pi and creates a ChiPhiFunc
    # with NO phi dependence.
    # zero_avg: whether the integration constant is set to have F(0)=0 or avg(F)=0
    # mode='simpson' is reasonably accurate and applicable to funcs with integral!=0
    # over a period
    # mode='fft' uses FFT.
    # -- Output: a new ChiPhiFunc
    @lru_cache(maxsize=1000)
    def integrate_phi(self, periodic, zero_avg, mode = 'default'): # nfp-dependent!!
        # number of phi grids
        len_chi = self.get_shape()[0]
        len_phi = self.get_shape()[1]
        if periodic:
            zero_avg = False
        phis = np.linspace(0, 2*np.pi*(1-1/len_phi), len_phi, dtype=np.complex128)
        if mode == 'default':
            if periodic:
                mode = two_pi_integral_mode
            else:
                mode = integral_mode

        if mode == 'fft':
            if periodic:
                raise AttributeError('It is not advised to integrate over 2pi with spectral method.')
            def integral(i_chi):
                out = scipy.fftpack.diff(self.content[i_chi], order=-1)
                return(out)
            out_list = Parallel(n_jobs=n_jobs, backend=backend, require=require)(
                delayed(integral)(i_chi) for i_chi in range(len(self.content))
            )
            out_content = np.array(out_list)/self.nfp
            # The fft.diff integral assumes zero average.
            if not zero_avg:
                out_content -= out_content[:,0][:,None]
            return(ChiPhiFunc(out_content, self.nfp))

        elif mode == 'simpson':
            new_content = integrate_phi_simpson(self.content, periodic = periodic)
        elif mode == 'spline':
            new_content = integrate_phi_spline(self.content, periodic = periodic)
        else:
            raise AttributeError('integrate_phi mode not recognized')
        if zero_avg:
            new_content -= np.mean(new_content,axis=1)
        new_content/=self.nfp
        if new_content.shape == (1,1):
            return(new_content[0,0])
        return(ChiPhiFunc(new_content, self.nfp))

    # Compares if self and other both have even or odd chi series.
    def both_even_odd(self,other): # not nfp-dependent
        if not isinstance(other, ChiPhiFunc):
            raise TypeError('other must be a ChiPhiFunc.')
        return (self.get_shape()[0]%2 == other.get_shape()[0]%2)

    ''' I.1.3 Filters and phi regularity '''
    # A multi-mode filter
    def filter(self, mode, arg): # nfp-dependent!!
        # A simple filter calculating a 3-element rolling average:
        # [..., a, b, c, ...] = [..., 0.25a+0.5b+0.25c, ...]
        if mode == 'roll_avg':
            content = self.content
            a = np.roll(content, -1, axis=1)
            b = np.roll(content, 1, axis=1)
            return(ChiPhiFunc(0.5*content+0.25*a+0.25*b, self.nfp))
        elif mode == 'low_pass':
            return(ChiPhiFunc(low_pass_direct(self.content, arg), self.nfp))
        else:
            raise AttributeError('ChiPhiFunc.filter: mode not recognized.')

    # Measure the "noise" in a ChiPhiFunc by comparing it to the result of
    # a low-pass filter
    def noise_filter(self, mode, arg):
        return(self-self.filter(mode=mode, arg=arg)) # not nfp-dependent

    ''' I.1.4 Properties '''

    # Getting the shape of the content
    def get_shape(self):
        return(self.content.shape) # not nfp-dependent

    # Getting the average amplitude of the content
    def get_amplitude(self):
        return(np.average(np.abs(self.content))) # not nfp-dependent

    def mask_constant(self):  # nfp-dependent!!
        if self.get_shape()[0]%2!=1:
            raise ValueError('Only even order coeffs have constant components')
        new_content = self.content.copy()
        new_content[len(new_content)//2] = np.zeros(new_content.shape[1])
        return(ChiPhiFunc(new_content, self.nfp))

    # Functions like np.imag
    def real(self): # nfp-dependent!!
        return(ChiPhiFunc(np.real(self.content), self.nfp))

    # Functions like np.real
    def imag(self): # nfp-dependent!!
        return(ChiPhiFunc(np.imag(self.content), self.nfp))

    # Returns the constant component.
    def get_constant(self): # not nfp-dependent
        return(self[0])

    # Returns the value when phi=0. Copies.
    def get_phi_zero(self):  # nfp-dependent!!
        new_content = np.array([self.content[:,0]]).T
        if len(new_content) == 1:
            return(new_content[0][0])
        return(ChiPhiFunc(np.array([self.content[:,0]]).T, 0))

    # Mask the constant component with zero. Copies.
    def zero_mask_constant(self):  # not nfp-dependent
        zero_array = np.zeros(self.get_shape()[1], dtype = np.complex128)
        return(replace_constant(self, zero_array))

    # Mask the constant component with a given array. Copies.
    def replace_constant(self, array):  # nfp-dependent!!
        if self.get_shape()[0]%2 == 0:
            raise AttributeError('zero_mask_constant: only applicable to even n.')
        new_content = self.content.copy()
        new_content[self.get_shape()[0]//2] = array
        return(ChiPhiFunc(new_content, self.nfp))

    # Used in addition with constant. Only even chi series has constant component.
    # Odd chi series + constant results in error.
    def no_const(self): # not nfp-dependent
        # This would trigger an error for most complex,
        # static methods used for evaluation.
        return (self.get_shape()[0]%2==0)

    # Returns the first sin and cos components of a ChiPhiFunc
    # as a ChiPhiFunc of the same type with no chi dependence
    # (only one row in its content.)
    def get_Yn1s_Yn1c(self): # nfp-dependent!!
        n_col = self.get_shape()[0]
        if n_col%2!=0:
            raise ValueError('Yn1 can only be obtained from an odd n(order).')
        Yn1_pos = self.content[n_col//2]
        Yn1_neg = self.content[n_col//2-1]

        Yn1_s = np.array([(Yn1_pos-Yn1_neg)*1j])
        Yn1_c = np.array([Yn1_pos+Yn1_neg])

        return(ChiPhiFunc(Yn1_s, self.nfp), type(self)(Yn1_c, self.nfp))

    # Takes the center m+1 rows of content. If the ChiPhiFunc
    # contain less chi modes than m+1, It will be zero-padded.
    def cap_m(self, m): # nfp-dependent!!
        target_chi = m+1
        len_chi = self.get_shape()[0]
        num_clip = len_chi - target_chi
        if target_chi == len_chi:
            return(self)
        if target_chi > len_chi:
            self.pad_m(m)
        if num_clip%2 != 0:
            raise AttributeError('cap_m only works when input and '\
            'self.content are both even or odd.')
        return(ChiPhiFunc(self.content[num_clip//2:-num_clip//2], self.nfp))

    def pad_m(self,m): # not nfp-dependent
        return self.pad_chi(m+1)

    # Pads a ChiPhiFunc to be have  total mode components.
    # Both self.get_shape[0] and  must be even/odd.
    # Note: This takes the total number of mode components, rather
    # than m, because it's primarily used when solving the looped
    # equations. It's sometimes less confusing to use the total number
    # of mode components (equations) than the m of the corresponding equations.
    def pad_chi(self, target_chi): # nfp-dependent!!
        len_chi = self.get_shape()[0]
        if len_chi%2!=target_chi%2:
            raise AttributeError('pad_chi only works when target_chi and '\
            'self.get_shape[0] are both even or odd.')

        if target_chi < len_chi:
            raise AttributeError('pad_chi only works when m is smaller '\
            'than the highest mode number in self.content. For expanding '\
            'a ChiPhiFunc, please use cap_m')
        padding = ChiPhiFunc(np.zeros((target_chi, self.get_shape()[1])), self.nfp)
        return(self+padding)


    ''' I.1.5 Output and plotting '''
    # Get a 2d vectorized function for plotting this term
    def get_lambda(self):
        len_chi = self.get_shape()[0]
        len_phi = self.get_shape()[1]

        # Create 'x' for interpolation. 'x' is 1 longer than lengths specified due to wrapping
        phi_looped = np.linspace(0,2*np.pi/self.nfp, len_phi+1)
        # wrapping
        content_looped = wrap_grid_content_jit(self.content)

        ind_chi = len_chi-1
        mode_chi = np.linspace(-ind_chi, ind_chi, len_chi)

        # The outer dot product is summing along axis 0.
        return(np.vectorize(
            lambda chi, phi : sum(
                [np.e**(1j*(chi)*mode_chi[i]) * np.interp((phi)%(2*np.pi/self.nfp),
                 phi_looped, content_looped[i]) for i in range(len_chi)]
            )
        )) # nfp-dependent!!

    # Plot phi-dependent power-Fourier coefficients on overlapping line plots.
    def display_content(self, fourier_mode = False, colormap_mode = False):
        plt.rcParams['figure.figsize'] = [8,3]
        content = self.content
        if content.shape[1]==1:
            content = content*np.ones((1,100))
        if type(content) is ChiPhiFuncNull:
            print('display_content(): input is ChiPhiFuncNull.')
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
        plt.show() # nfp-dependent!!

    # FFT the content and returns a ChiPhiFunc
    # Note that fft_freq, (operators acting on the fft),
    # rather than the fft is modified.
    def fft(self): # nfp-dependent!!
        return(ChiPhiFunc(np.fft.fft(self.content, axis=1), self.nfp))

    # IFFT the content and returns a ChiPhiFunc
    def ifft(self): # nfp-dependent!!
        return(ChiPhiFunc(np.fft.ifft(self.content, axis=1), self.nfp))

    # Plot a period in both chi and phi
    def display(self, complex = False, size=(100,100), avg_clim = False):
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

    ''' I.1.6 Utilities '''
    # Used in operators, wrapper for stretch_phi. Match self's shape to another ChiPhiFunc.
    # returns 2 CONTENTS.
    # always_copy: always make copy for contents.
    # If set to False, when the two ChiPhiFunc have an equal number of
    # phi grids, this method directly points to their content.
    def stretch_phi_to_match(self, other, always_copy=True): # not nfp-dependent
        if type(other) is not ChiPhiFunc:
            raise TypeError('stretch_phi_to_match only takes ChiPhiFunc.')
        if self.get_shape()[1] == other.get_shape()[1]:
            if always_copy:
                return(np.copy(self.content), np.copy(other.content))
            else:
                return(self.content, other.content)
        # warnings.warn('Warning: phi grid stretching has occured. Shapes are:'\
        #     'self:'+str(self.get_shape())+'; other:'+str(other.get_shape()))
        max_phi_len = max(self.get_shape()[1], other.get_shape()[1])
        return(
            ChiPhiFunc.stretch_phi(self.content, max_phi_len),
            ChiPhiFunc.stretch_phi(other.content, max_phi_len)
        )

    # Stretching each row individually by interpolation.
    # The phi grid is assumed periodic, and the first column is always
    # wrapped after the last column to correctly interpolate
    # periodic function. Handles same-length cases by skipping.
    # -- Input: self and an int length
    # -- Output: a new ChiPhiFuncFourier
    @njit(complex128[:,:](complex128[:,:], int64), parallel=True)
    def stretch_phi(content, len_target): # not nfp-dependent
        len_phi = content.shape[1]
        if len_phi == len_target:
            return(np.copy(content))
        # Create empty output for numba
        stretched = np.zeros((content.shape[0],len_target+1), dtype=np.complex128)
        # Create 'x' for interpolation. 'x' is 1 longer than lengths specified due to wrapping
        unstreched_x = np.linspace(0,np.pi*2,len_phi+1)
        stretched_x = np.linspace(0,np.pi*2,len_target+1)
        # wrapping
        content_looped = wrap_grid_content_jit(content)
        # parallelize
        for i in prange(content.shape[0]):
            stretched[i] = np.interp(stretched_x, unstreched_x, content_looped[i])
        return(stretched[:,:-1])

    # Converting fourier coefficients into exponential coeffs used in
    # ChiPhiFunc's internal representation. Only used during super().__init__
    # Does not copy.
    def trig_to_exp(self):
        util_matrix_chi = self.fourier_to_exp_op()
        # Apply the conversion matrix on chi axis
        self.content = util_matrix_chi @ self.content  # not nfp-dependent

    # Converting fourier coefficients into exponential coeffs used in
    # ChiPhiFunc's internal representation. Only used during super().__init__
    # Does not copy.
    def export_trig(self): # nfp-dependent!!
        util_matrix_chi = np.linalg.inv(self.fourier_to_exp_op())
        # Apply the conversion matrix on chi axis
        return(ChiPhiFunc(util_matrix_chi @ self.content, self.nfp))

    def export_single_nfp(self):
        new_content = np.tile(self.content, (1,self.nfp))
        return ChiPhiFunc(new_content, 1)

    # Generates a matrix for converting a n-dim trig-fourier-representation vector (can be full or skip)
    # into exponential-fourier-representation.
    def fourier_to_exp_op(self): # not nfp-dependent
        n_dim = self.get_shape()[0]
        if n_dim%2==0:
            n_mode = n_dim//2
            I_n = np.identity(n_mode)
            I_anti_n = np.fliplr(I_n)
            util_matrix = np.block([
                [ 0.5j*I_n            , 0.5*I_anti_n         ],
                [-0.5j*I_anti_n       , 0.5*I_n              ]
            ])
        else:
            n_mode = (n_dim-1)//2
            I_n = np.identity(n_mode)
            I_anti_n = np.fliplr(I_n)
            util_matrix = np.block([
                [ 0.5j*I_n            , np.zeros((n_mode, 1)), 0.5*I_anti_n         ],
                [np.zeros((1, n_mode)), 1                    , np.zeros((1, n_mode))],
                [-0.5j*I_anti_n       , np.zeros((n_mode, 1)), 0.5*I_n              ]
            ])
        return(util_matrix)

    ''' I.1.8 Deconvolution ("dividing" chi-dependent terms) '''

    # Get O, O_einv and vector_free_coef that solves the eqaution system
    # O y = (A + B dchi) y = RHS <=>
    #   y = O_einv - vec_free * vec_free_coef
    # Or in code format,
    # y = (np.einsum('ijk,jk->ik',O_einv,chiphifunc_rhs_content) + vec_free * vector_free_coef)
    # rankrhs is the number of comp in Y minus 1.
    def get_O_O_einv_from_A_B(chiphifunc_A, chiphifunc_B, i_free, rank_rhs): # not nfp-dependent except in output

        if not (type(chiphifunc_A) is ChiPhiFunc\
            and type(chiphifunc_B) is ChiPhiFunc):
            raise TypeError('Both of chiphifunc_A, chiphifunc_B '\
                            'should be ChiPhiFunc. The actual types are:'
                            +str(type(chiphifunc_A))+', '
                            +str(type(chiphifunc_B)))

        chiphifunc_A_content, chiphifunc_B_content = chiphifunc_A.stretch_phi_to_match(chiphifunc_B, always_copy=False)

        # generate the LHS operator O_matrices = (va conv + vb conv dchi)
        O_matrices = 0
        A_conv_matrices = conv_tensor(chiphifunc_A_content, rank_rhs+1)
        O_matrices += A_conv_matrices

        dchi_matrix = dchi_op(rank_rhs+1, False)
        B_conv_matrices = conv_tensor(chiphifunc_B_content, rank_rhs+1)
        O_matrices += np.einsum('ijk,jl->ilk',B_conv_matrices,dchi_matrix)

        O_einv = batch_matrix_inv_excluding_col(O_matrices, i_free)
        O_einv = np.concatenate((O_einv[:i_free], np.zeros((1,O_einv.shape[1],O_einv.shape[2])), O_einv[i_free:]))
        O_free_col = O_matrices[:,i_free,:]

        vector_free_coef = np.einsum('ijk,jk->ik',O_einv, O_free_col)#A_einv@A_free_col
        vector_free_coef[i_free] = -np.ones((vector_free_coef.shape[1]))

        return(O_matrices, O_einv, -vector_free_coef, max(chiphifunc_A.nfp, chiphifunc_B.nfp))

''' I.2 Utilities '''
# Integrates a function on a grid using Simpson's method.
# Produces a content where values along axis 1 is the input content's
# integral.
# periodic is a special mode that assumes integrates a content over a period.
# It assumes that the grid function is periodic, and does not repeat the first
# grid's value.
# A usual content has the first cell's LEFT edge at 0
# and the last cell's RIGHT edge at 2pi.
# A specal dx is provided for a grid that has the first cell's LEFT edge at 0
# and the last cell's LEFT edge at 2pi.
# NOTE
# Cell values are ALWAYS taken at the left edge.
# nfp dependence is NOT HANDLED HERE.
def integrate_phi_simpson(content, dx = 'default', periodic = False):
    len_chi = content.shape[0]
    len_phi = content.shape[1]
    if dx == 'include_2pi':
        dx = 2*np.pi/(len_phi-1)
    elif dx == 'default':
        dx = 2*np.pi/len_phi
    if periodic:
        # The result of the integral is an 1d array of chi coeffs.
        # This integrates the full period, and needs to be wrapped.
        # the periodic=False option does not integrate the full period and
        # does not wrap.

        if dx == 'default':
            dx = 2*np.pi/(len_phi-1)
        new_content = scipy.integrate.simpson(\
            wrap_grid_content_jit(content),\
            dx=dx,\
            axis=1\
            )
        return(np.array([new_content]).T)
    else:
        # Integrate up to each element's grid
        integrate = lambda i_phi : scipy.integrate.simpson(content[:,:i_phi+1], dx=dx)
        out_list = Parallel(n_jobs=n_jobs, backend=backend, require=require)(
            delayed(integrate)(i_phi) for i_phi in range(len_phi)
        )
        return(np.array(out_list).T) # not nfp-dependent

# Implementation of spline-based integrate_phi using Parallel.
# nfp dependence is NOT HANDLED HERE.
def integrate_phi_spline(content, dx = 'default', periodic=False,
    diff=False, diff_order=None):

    print('This is  the naive implementation. Please implement '
    'periodic spline with scipy.interpolate.splrep')

    len_chi = content.shape[0]
    len_phi = content.shape[1]
    if dx == 'include_2pi':
        dx = 2*np.pi/(len_phi-1)
    elif dx == 'default':
        dx = 2*np.pi/len_phi
        # purely real.

    phis = np.linspace(0, dx*(len_phi-1), len_phi)

    def generate_and_integrate_spline(i_chi):
        new_spline = scipy.interpolate.make_interp_spline(phis, content[i_chi])
        if diff:
            return(scipy.interpolate.splder(new_spline, n=diff_order))
        return(scipy.interpolate.splantider(new_spline))

    # A list of integrated splines
    integrate_spline_list = Parallel(n_jobs=n_jobs, backend=backend, require=require)(
        delayed(generate_and_integrate_spline)(i_chi) for i_chi in range(len_chi)
    )

    if periodic:
        # The result of the integral is an 1d array of chi coeffs.
        # This integrates the full period, and needs to be wrapped.
        # the periodic=False option does not integrate the full period and
        # does not wrap.
        if dx == 'default':
            dx = 2*np.pi/(len_phi-1)
        evaluate_spline_2pi = lambda spline: spline(2*np.pi)
        out_list = Parallel(n_jobs=n_jobs, backend=backend, require=require)(
            delayed(evaluate_spline_2pi)(spline) for spline in integrate_spline_list
        )
        return(np.array([out_list]).T)
    else:
        evaluate_spline = lambda spline, phis : spline(phis)
        out_list = Parallel(n_jobs=n_jobs, backend=backend, require=require)(
            delayed(evaluate_spline)(spline, phis) for spline in integrate_spline_list
        )

        return(np.array(out_list)) # not nfp-dependent

# Generate phi differential operator diff_matrix. diff_matrix@f.content = dchi(f).content
# invert = True generates anti-derivative operator. Cached for each new Chi length.
# -- Input: len_chi: length of Chi series.
# -- Output: 2d matrix.
# nfp dependence is NOT HANDLED HERE.
def dphi_op_pseudospectral(n_phi):
    if not config.use_pyQSC:
        raise AttributeError('pyQSC is needed to calculate pseudo spectral phi derivatives.')
    out = spectral_diff_matrix(n_phi, xmin=0, xmax=2*np.pi)
    return(out) # not nfp-dependent

# a low pass filter acting on a content matrix
# nfp dependence is NOT HANDLED HERE.
def low_pass_direct(content, freq):
    len_phi = content.shape[1]
    if freq*2>=len_phi:
        return(np.copy(content))
    W = np.abs(jit_fftfreq_int(len_phi))
    f_signal = np.fft.fft(content, axis = 1)

    # If our original signal time was in seconds, this is now in Hz
    cut_f_signal = f_signal.copy()
    cut_f_signal[:,(W>freq)] = 0

    return(np.fft.ifft(cut_f_signal, axis=1)) # not nfp-dependent

# dphi of a content matrix
# nfp dependence is NOT HANDLED HERE.
def dphi_direct(content, order=1, mode='default'): # not nfp-dependent

    if order<0:
        raise AttributeError('Order must be positive for dphi_direct().')

    if mode=='default':
        mode = diff_mode

    if mode=='fft':
        derivative = lambda i_chi : scipy.fftpack.diff(content[i_chi], order=order)
        out_list = Parallel(n_jobs=n_jobs, backend=backend, require=require)(
            delayed(derivative)(i_chi) for i_chi in range(len(content))
        )
        return(np.array(out_list))

    if mode=='pseudo_spectral':
        out = content
        for i in range(order):
            out = (dphi_op_pseudospectral(content.shape[1]) @ out.T).T
        return(out)

    if mode=='finite_difference':
        return(np.gradient(content, axis=1, edge_order=2)\
        /(np.pi*2/content.shape[1]))

    if mode=='spline':
        if order>0:
            out = integrate_phi_spline(content, periodic=False,
                diff=True, diff_order=order)
        if order<0:
            out = integrate_phi_spline(content, periodic=False,
                diff=False, diff_order=-order)
        return(out)

    else:
        raise AttributeError('dphi mode not recognized.')

''' II. Singleton for conditionals '''
# A singleton subclass. The instance behaves like
# nan, except during multiplication with zero, where it becomes 0.
# This object never copies during operations.
# This works in conjunction with quirks in power_matching.mac: instead of ensuring that
# new upper bounds and suffixes (made by replacing the innermost index with
# a value satisfying eps^{f(i)} = eps^n) are consistent with,
# all original upper and lower bounds and are integers, it adds heaviside-like functions,
# is_integer and is_seq to make terms out-of-bound and non-integer suffixes zero.
class ChiPhiFuncNull(ChiPhiFunc):
    def __new__(cls, content=np.nan, nfp=0):
        if not hasattr(cls, 'instance'):
            cls.instance = ChiPhiFunc.__new__(cls)
        return cls.instance

    def get_shape(self):
        raise TypeError('Cannot use get_shape() on ChiPhiFuncNull.')

    # The contents is dummy to enable calling of this singleton using
    # the default constructor
    def __init__(self, content=np.nan, nfp=0):
        self.content = np.nan
        self.nfp = 0

    def __getitem__(self, index):
        raise TypeError('__getitem__ cannot operate on ChiPhiFuncNull.')

    def __neg__(self):
        return(self)

    def __add__(self, other):
        return(self)

    def __radd__(self, other):
        return(self)

    def __sub__(self, other):
        return(self)

    def __rsub__(self, other):
        return(self)

    def __mul__(self, other):
        if other == 0:
            return(0)
        if isinstance(other, ChiPhiFunc):
            if not np.any(other.content):
                return(0)
        return(self)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return(self)

    def __rtruediv__(self, other):
        return(self)

    def __pow__(self, other):
        return(self)

    def __rmatmul__(self, mat):
        raise ValueError('@ (a matrix representation of an operator, not generated by Maxima) should not '\
                             'encounter a null term')

    def dchi(self, order=1):
        return(self)

    def dphi(self, order=1):
        return(self)

    def get_constant(self):
        raise TypeError('Cannot get constant of a ChiPhiFuncNull.')

    def cap_m(self, dummy):
        raise TypeError('Cannot cap_m() a ChiPhiFuncNull.')

    def real(self):
        raise TypeError('Cannot get real components from a ChiPhiFuncNull.')

    def imag(self):
        raise TypeError('Cannot get imag components from a ChiPhiFuncNull.')  # nfp-independent except in constructor

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
# Invert an (n,n) submatrix of a (m>n+1,n+1) rectangular matrix by taking the first
# n-1 rows and excluding the ind_col'th column. "Taking the first n rows" is motivated
# by the RHS being rank n-1
#
# -- Input --
# (m,n+1,len_phi) matrix A
# ind_col < n+1
#
# -- Return --
# (m,m,len_phi) matrix A_inv
def batch_matrix_inv_excluding_col(in_matrices, ind_col):
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

# @njit(complex128[:](complex128[:,:], complex128[:,:], complex128[:], int64, complex128))
# def solve_degenerate_underdetermined_jit(A, B, vb, i_free, Yn_free):
#     B_cont = np.ascontiguousarray(B)
#     vb_cont = np.ascontiguousarray(vb)
#     return(solve_degenerate_underdetermined_jit(A, B_cont@vb_cont, i_free, Yn_free))

''' III.3 Convolution operator generator and ChiPhiFunc.content numba wrapper '''
# Generate convolution operator from a for an n_dim vector.
# Can't be compiled for parallel beacuase vec and out_transposed's sizes dont match?
@njit(complex128[:,:](complex128[:], int64))
def conv_matrix(vec, n_dim):
    out_transposed = np.zeros((n_dim,len(vec)+n_dim-1), dtype = np.complex128)
    for i in prange(n_dim):
        out_transposed[i, i:i+len(vec)] = vec
    return(out_transposed.T) # not nfp-dependent

# Generate a tensor_coef (see looped_solver.py) convolving a ChiPhiFunc
# content with n_dim chi modes.
# For multiplication in FFT space during ODE solves.
# The convolution is done by:
# x2_conv_y2 = np.einsum('ijk,jk->ik',conv_x2, y2.content)
@njit(complex128[:,:,:](complex128[:,:], int64))
def conv_tensor(content, n_dim):
    len_chi = content.shape[0]
    len_phi = content.shape[1]
    out = np.zeros((len_chi+n_dim-1,n_dim,len_phi), dtype = np.complex128)
    for i in prange(n_dim):
        out[i:i+len_chi, i, :] = content
    return(out) # not nfp-dependent

# Generates a 4D convolution operator in the phi axis for a 3d "tensor coef"
# (see looped_solver for explanation)
@njit(complex128[:,:,:,:](complex128[:,:,:]))
def fft_conv_tensor_batch(source):
    len_a = source.shape[0]
    len_b = source.shape[1]
    len_phi = source.shape[2]
    roll = jit_fftfreq_int(len_phi)
    out = np.zeros((len_a, len_b, len_phi, len_phi), dtype = np.complex128)
    # Where does 0 elem start/end
    split = (len_phi+1)//2
    split_b = np.roll(np.arange(len_phi%2, len_phi+len_phi%2, dtype = np.int64),split)
    for i in prange(len_phi):
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

# # The first finite differene implementation
# @njit(complex128[:,:](complex128[:], int64))
# def finite_diff_matrix(stencil, n_dim):
#     if len(stencil)%2!=1 or len(stencil.shape)!=1:
#         raise AttributeError('The stencil must be 1d and have odd number of elements')
#     half_len = len(stencil)//2
#     out_transposed = np.zeros((n_dim, n_dim), dtype = np.complex128)
#     first_row = np.zeros((n_dim, n_dim), dtype = np.complex128)
#     first_row[-half_len:] = stencil[:half_len]
#     first_row[:half_len+1] = stencil[-half_len-1:]
#     for i in prange(n_dim):
#         out_transposed[i] = np.roll(first_row, i)
#     return(out_transposed.T)

''' IV. Solving linear PDE in phi grids '''

''' IV.1 Solving the periodic linear PDE (a + b * dphi + c * dchi) y = f(phi, chi) '''
# Solves simple linear first order ODE systems in batch:
# (coeff + coeff_phi d/dphi) y = f. ( y' + p_eff*y = f_eff )
# (Dchi = +- m * 1j)
# All inputs are content matrices.
# p and f are assumed periodic.
# P is preferrably non-resonant for small p amplitude.
# All coeffs' CONTENTS are point-wise multiplied to f, dpf or dcf's content.

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
def solve_integration_factor(coeff, coeff_dp, f, \
    integral_mode='auto', asymptotic_order=asymptotic_order,
    masking = True, fft_max_freq=None): # not nfp-dependent

    len_phi = f.shape[1]
    len_chi = f.shape[0]

    # Rescale the eq into y'+py=f
    f_eff = f/coeff_dp
    p_eff = coeff/coeff_dp
    f_eff_scaling = np.average(np.abs(f_eff))
    f_eff = f_eff/f_eff_scaling
    # print('solve_integration_factor: average p_eff:', np.average(np.abs(p_eff)))
    # print('solve_integration_factor: average f_eff:', f_eff_scaling)

    # Make f_eff and p_eff both shaped like f
    if np.isscalar(p_eff):
        p_eff = p_eff*np.ones_like(f, dtype = np.complex128)
    elif p_eff.shape[0]!=f_eff.shape[0]:
        raise AttributeError('p_eff and f_eff has different component numbers!')
    elif p_eff.shape[1]!=f_eff.shape[1]:
        p_eff = ChiPhiFunc.stretch_phi(p_eff, len_phi)


    effective_dx = 2*np.pi/(len_phi)

    # Depending on each component's amplitude and maximum real component, decide which method to use
    if integral_mode == 'auto':
        modes = []
        for i in range(f.shape[0]):
            if np.average(np.abs(np.real(p_eff[i])))<config.asymptotic_threshold:
                modes.append('fft')
            else:
                modes.append('asymptotic')

        print('Modes:',modes)
        # Each component needs different modes
        if not np.all(np.array(modes) == modes[0]):
            print('solve_integration_factor: not all components use the same '\
                 'integral_mode. Solving each component individially')
            solve_1d = lambda coeff_1d, coeff_dp_1d, f_1d, integral_mode : \
                solve_integration_factor(
                    np.array([coeff_1d]),
                    np.array([coeff_dp_1d]),
                    np.array([f_1d]),
                    integral_mode=integral_mode,
                    asymptotic_order = asymptotic_order,
                    fft_max_freq=fft_max_freq)
            out_list = Parallel(n_jobs=n_jobs, backend=backend, require=require)(
                delayed(solve_1d)(p_eff[i], 1, f_eff[i], modes[i])\
                for i in range(len_chi)
            )
            return(np.array(out_list)[:,0,:]*f_eff_scaling)
        # Batch solve the whole group of equations
        # with asymptotic series if all of them has large amplitudes
        integral_mode = modes[0]

    print('integral_mode is', integral_mode)
    print(
        'WARNING: spline and simpson produces slightly non-periodic Delta. '\
        'This inconsistency is shown by masking p in Delta and adding '\
        '-B_denom_coef_c * p_perp_coef_cp[n]. Why is this is still not understood yet.'
    )

    if integral_mode == 'asymptotic':
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


    elif integral_mode == 'fft':
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
        sln = np.fft.ifft(fft_pad(sln_fft, len_phi, axis = 1), axis = 1)
        # Calculating components with p=0 (now the ODE is y'=f)
        # These components are just phi integrals of f.
        # NOTE: BC is set to ZERO AVERAGE!
        if np.any(remove_zero):
            # nfp is not treated here.
            zero_comps = ChiPhiFunc(f_eff[remove_zero], nfp=1).integrate_phi(periodic=False, zero_avg=True).filter('low_pass', fft_max_freq)
            sln[remove_zero] = zero_comps.content
        return(sln*f_eff_scaling)

    else:
        f_looped = wrap_grid_content_jit(f_eff)
        p_looped = wrap_grid_content_jit(p_eff)

        # The integrand of the integration factor (I = exp(int_p))
        if integral_mode == 'simpson':
            int_p = integrate_phi_simpson(p_looped, periodic=False, dx = effective_dx)
        elif integral_mode == 'spline' or integral_mode == 'asymptotic' or integral_mode == 'piecewise' :
            int_p = integrate_phi_spline(p_looped, periodic=False, dx = effective_dx)
        else:
            raise AttributeError('integral_mode not recognized.')
        # Solving with intermediate p by integrating factor
        int_p_2pi = np.array([int_p[:,-1]]).T
        exp_neg2pi = np.exp(-int_p_2pi)
        exp_phi = np.exp(int_p)
        exp_negphi = np.exp(-int_p)
        integrand = f_looped*exp_phi

        # Here integration_factor_2pi can no longer be evaluated by
        # integrate_phi_simpson(periodic=True), because the integrand
        # is not generally periodic.
        if integral_mode == 'simpson':
            integration_factor = integrate_phi_simpson(integrand, periodic=False, dx = effective_dx)
            integration_factor_2pi = np.array([integration_factor[:,-1]]).T
        elif integral_mode == 'spline':
            integration_factor = integrate_phi_spline(integrand, periodic=False, dx = effective_dx)
            integration_factor_2pi = np.array([integration_factor[:,-1]]).T
        else:
            raise AttributeError('integral_mode not recognized.')
            # The integration constant. Derived from the periodic boundary condition.

        integration_factor = integration_factor*exp_negphi
        integration_factor_2pi = integration_factor_2pi*exp_neg2pi

        # If the integral of p is periodic, I is periodic.
        # The BVP cannot get solved.
        if np.average(np.abs(int_p[:,0] - int_p[:,-1])) < np.max(f_looped)*noise_level_periodic:
            print('I(phi) is periodic. Cannot yield an unique solution using only periodic BC.')
            print('returning integration factor.')
            c1=0
        else:
            # exp_neg2pi may contain 1's.
            exp_neg2pi[exp_neg2pi == 1] = np.inf
            c1=integration_factor_2pi/(1-exp_neg2pi)
        out = c1*exp_negphi+integration_factor
        out = out[:,:-1]
        return(out*f_eff_scaling)

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
# using integral factor. Not numba accelerated since scipy is not supported by numba.
# -- Input --
# coeffs are constants or contents. f is a content (see ChiPhiFunc's description).
# Contant mode and offset are for evaluating the constant component equation
# dphi y0 = f0.
# -- Output --
# y is a ChiPhiFunc's content
def solve_integration_factor_chi(coeff, coeff_dp, coeff_dc, f, \
    integral_mode=two_pi_integral_mode,
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
# using integral factor. Not numba accelerated since scipy is not supported by numba.
# -- Input --
# iota is a constant and f is a ChiPhiFunc.
# Contant mode and offset are for evaluating the constant component equation
# dphi y0 = f0.
# -- Output --
# y is a ChiPhiFunc's content
def solve_dphi_iota_dchi(iota, f, \
    integral_mode=two_pi_integral_mode,
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
def fft_filter(fft_in, target_length, axis): # not nfp-dependent
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
