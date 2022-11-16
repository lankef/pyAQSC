import numpy as np
import timeit
import warnings
import scipy.signal
import scipy.fftpack
from matplotlib import pyplot as plt

# Numba doesn't support scipy methods. Therefore, scipy integrals are sped up
# with joblib.
import scipy.integrate
import scipy.interpolate
from joblib import Parallel, delayed
from functools import lru_cache # import functools for caching

# TODO: Add filtering to dphi?

# integrate_chi() should not be run on a ChiPhiFuncGrid with a chi-independent component,
# because this produces a non-periodic function. However, zero-checking the
# component is not feasible, because cancellation is often not exact in numerical
# evaluations. Instead, we check if the maximum amplitude of the chi-independent
# component is greater than this noise_level_int
# The typing is for numba.
noise_level_int = np.float64(1e-5)
# Sometimes extra components have to be cut away in the RHS of ynp1 solve.
# since cancellation sometimes happen and won't be exact. This decides how much
# max extra component is okay to cut.
noise_order_solve = np.float64(7)

# tolerance on how periodic I(phi) is during integration factor.
# When I(phi) is periodic, any C1 can satisfy the periodic BC.
noise_level_periodic = 1e-10

# Default frequency for the low pass filtering
low_pass_freq=50

# Default diff and integration modes can be modified.
diff_mode = 'pseudo_spectral' # available: pseudo_spectral, finite_difference, fft, spline
integral_mode = 'fft' # available: spline, simpson, fft
non_periodic_integral_mode = 'spline' # available: spline, simpson, fft


# Threshold for p amplitude to use asymptotic expansion for y'+py=f
asymptotic_threshold = 30
# Maximum allowed asymptotic series order for y'+py=f
asymptotic_order = 6

# Debugging variables.
# When debug_mode is true, content of all new ChiPhiFunc's are tracked.
# Enable and access by debug_mode and debug_max_value.
debug_mode = False

# Tracks the max and avg values of intermediate results. Compare with output to
# identify rounding errors.
debug_max_value = []
debug_avg_value = []

# Tracks the difference between the power of 10 of the 2 arguments in a '+' operation
# to identify rounding errors.
debug_pow_diff_add = []

# Joblib settings
import multiprocessing

print('Detected', multiprocessing.cpu_count(), 'CPU\'s. Setting n_jobs to #CPU/2.')
n_jobs = multiprocessing.cpu_count()//2
backend = 'threading' # scipy.integrate is based on a compiled
require = None


from numba import jit, njit, prange
from numba.types import Tuple
# import jit value types. Bizzarely using int32 causes typing issues
# in solve_underdet_degen with njit(parallel=True) because
# it seem to insist prange(...shape[0]) is int64, even when int64 is not imported.
from numba import complex128, int64, float64, boolean

''' Representing functions of chi and phi (ChiPhiFunc subclasses) '''
# Represents a function of chi and phi.
# Manages an complex128[m, n] 2d array called content.
# Axis 0 represents "m". Its length is n+1 for a nth-order term:
# each n-th order known term has n+1 non-zero coeffs due to regularity cond.
# zero terms are marked by *. I'm not sure whether to implement them as zeroes
# and later check regularity, or implement them as skip series.
# [                           # [
#     [Chi_coeff_-n(phi)],    #     [Chi_coeff_-n(phi)],
#     ...                     #     ...
#     [Chi_coeff_-2(phi)],    #     [Chi_coeff_-1(phi)],
#     [const(phi)],           #     [Chi_coeff_1(phi)],
#     [Chi_coeff_2(phi)],     #     ...
#     ...                     #     [Chi_coeff_n(phi)]
#     [Chi_coeff_n(phi)]      # ] for odd n
# ] for even n

# Axis 1 stores representation of a phi function. It may be:
# 1. Grid values from phi = 0  to phi = 2pi
# 2. FULL Fourier exp coefficients
# which will be implemented as subclasses.

# At the moment the grid representation is preferred, since no regularity constraint
# may exist for phi fourier coefficients.


''' I. Abstract superclass and a null-like singleton '''
# ChiPhiFunc is an abstract superclass.
class ChiPhiFunc:
    # Initializer. Fourier_mode==True converts sin, cos coeffs to exponential
    def __init__(self, content=np.nan, fourier_mode=False):
        if debug_mode:
            debug_max_value.append(np.max(np.abs(content)))
            debug_avg_value.append(np.max(np.average(content)))

        if len(content.shape)!=2:
            raise ValueError('ChiPhiFunc content must be 2d arrays.')
        # for definind special instances that are similar to nan, except yields 0 when *0.
        if type(self) is ChiPhiFunc:
            raise TypeError('ChiPhiFunc is intended to be an abstract superclass.')
        # copies and force types for numba
        self.content = np.complex128(content)
        if fourier_mode:
            self.trig_to_exp()

    # Operators -----------------------------------------------------------------------------

    # -self (negative) operator.
    def __neg__(self):
        return type(self)(-self.content)

    # self+other, with:
    # a scalar or another ChiPhiFunc of the same implementation
    def __add__(self, other):

        # When summing two ChiPhiFunc's, only allows summation
        # of the same implementation (fourier or grid)
        if issubclass(type(other), ChiPhiFunc):
            if isinstance(other, ChiPhiFuncNull):
                return(other)
            if not isinstance(other, type(self)):
                raise TypeError('+ can only be evaluated with another '\
                                'ChiPhiFuncs of the same implementation.')
            if not self.both_even_odd(other):
                raise ValueError('+ can only be evaluated between 2 ChiPhiFuncs '\
                                'that are both even or odd')
            return self.add_ChiPhiFunc(other)
        else:
            if not np.isscalar(other):
                raise TypeError('+ cannnot be evaluated with a vector.')
            return(self.add_const(other))

    # other+self
    def __radd__(self, other):
        return(self+other)

    # self-other, implemented using addition.
    def __sub__(self, other):

        # When summing two ChiPhiFunc's, only allows summation
        # of the same implementation (fourier or grid)
        if issubclass(type(other), ChiPhiFunc):
            if isinstance(other, ChiPhiFuncNull):
                return(other)
            if not isinstance(other, type(self)):
                raise TypeError('+ can only be evaluated with another '\
                                'ChiPhiFuncs of the same implementation.')
            if not self.both_even_odd(other):
                raise ValueError('+ can only be evaluated between 2 ChiPhiFuncs '\
                                'that are both even or odd')
            return self.add_ChiPhiFunc(other, sign=-1)
        else:
            if not np.isscalar(other):
                raise TypeError('+ cannnot be evaluated with a vector.')
            return self.add_const(other, sign=-1)

    # other-self
    def __rsub__(self, other):
        return(-(self-other))

    # other*self, POINTWISE multiplication of ChiPhiFunc with:
    # a scalar
    # another ChiPhiFunc of the same implementation.
    # even*even -> even
    # odd*odd -> odd
    # even*odd -> even

    def __mul__(self, other):

        # When summing two ChiPhiFunc's, only allows summation
        # of the same implementation (fourier or grid)
        if isinstance(other, ChiPhiFunc):
            if isinstance(other, ChiPhiFuncNull):
                if not np.any(self.content):
                    return(0)
                return(other)
            if not isinstance(other, type(self)):
                raise TypeError('* can only be evaluated with another '\
                                'ChiPhiFunc\'s of the same implementation.')
            return(self.multiply(other))
        else:
            if not np.isscalar(other):
                raise TypeError('* cannnot be evaluated with a vector.')
            if other == 0: # to accomodate ChiPhiFuncNull.
                return(0)
            return(type(self)(other * self.content))

    # self*other
    def __rmul__(self, other):
        return(self*other)

    def __truediv__(self, other):
        # When summing two ChiPhiFunc's, only allows summation
        # of the same implementation (fourier or grid)
        if isinstance(other, ChiPhiFunc):
            if isinstance(other, ChiPhiFuncNull):
                return(other)
            if not isinstance(other, type(self)):
                raise TypeError('/ can only be evaluated with another '\
                                'ChiPhiFunc\'s of the same implementation.')
            if other.content.shape[0]!=1:
                raise TypeError('/ can only be evaluated with a'\
                                    'ChiPhiFuncs with no chi dependence (only 1 row)')
            return(self.multiply(other, div=True))
        else:
            if not np.isscalar(other):
                raise TypeError('/ cannnot be evaluated with a vector.')
            return(type(self)(self.content/other))

    # other/self
    def __rtruediv__(self, other):
        if self.content.shape[0]!=1:
            raise TypeError('/ can only be evaluated with a'\
                            'ChiPhiFuncs with no chi dependence (only 1 row)')
        if isinstance(other, ChiPhiFunc):
            return(other.multiply(self, div=True))
        else:
            if not np.isscalar(other):
                raise TypeError('/ cannnot be evaluated with a vector.')
            return(type(self)(other/self.content))

    # other@self, for treating this object as a vector of Chi modes, and multiplying with a matrix
    def __rmatmul__(self, mat):
        return type(self)(mat @ self.content)

    # self@other, not supported. Throws an error.
    def __matmul__(self, mat):
        raise TypeError('ChiPhiFunc is treated as a vector of chi coeffs in @. '\
                       'Therefore, ChiPhiFunc@B is not supported.')

    # self^n, based on self.pow().
    @lru_cache(maxsize=1000)
    def __pow__(self, other):
        if not np.isscalar(other):
            raise TypeError('**\'s other argument must be a non-negative scalar integer.')
        if int(other)!=other:
            raise ValueError('Only integer ** of ChiPhiFunc is supported.')
        if other == 0:
            return(1)
        if other < 1:
            raise ValueError('**\'s other argument must be non-negative.')

        return self.pow(other)

    # Multiplication with a ChiPhiFunc of the same type. Abstract method.\
    def multiply(a, b, div=False):
        raise NotImplementedError()

    # Math methods -----------------------------------------------------------------------------
    # derivatives. Implemented through dchi_op
    def dchi(self, order=1):
        out = self.content
        len_chi = len(out)
        if order<0:
            raise AttributeError('dchi order must be positive')
        mode_i = (1j*np.arange(-len_chi+1,len_chi+1,2)[:,None])**order
        out = mode_i * out
        return(type(self)(out))

    def dphi(self, order=1, mode='default'):
        raise NotImplementedError()

    def dphi_iota_dchi(self, iota):
        return(self.dphi()+iota*self.dchi())

    # Compares if self and other both have even or odd chi series.
    def both_even_odd(self,other):
        if not isinstance(other, ChiPhiFunc):
            raise TypeError('other must be a ChiPhiFunc.')
        return (self.get_shape()[0]%2 == other.get_shape()[0]%2)


    # Properties -----------------------------------------------------------------------------
    def get_shape(self):
        # This would trigger an error for most complex,
        # static methods used for evaluation.
        return(self.content.shape)

    # Returns the constant component.
    def get_constant(self):
        len_chi = self.get_shape()[0]
        if len_chi%2!=1:
            raise ValueError('No constant component found.')
        return(type(self)(np.array([self.content[len_chi//2]])))


    # Mask the constant component with zero. Copies.
    def zero_mask_constant(self):
        zero_array = np.zeros(self.get_shape()[1], dtype = np.complex128)
        return(replace_constant(self, zero_array))

    # Mask the constant component with a given array. Copies.
    def replace_constant(self, array):
        if self.get_shape()[0]%2 == 0:
            raise AttributeError('zero_mask_constant: only applicable to even n.')
        new_content = self.content.copy()
        new_content[self.get_shape()[0]//2] = array
        return(type(self)(new_content))

    # Used in addition with constant. Only even chi series has constant component.
    # Odd chi series + constant results in error.
    def no_const(self):
        # This would trigger an error for most complex,
        # static methods used for evaluation.
        return (self.get_shape()[0]%2==0)

    # Returns the first sin and cos components of a ChiPhiFunc
    # as a ChiPhiFunc of the same type with no chi dependence
    # (only one row in its content.)
    def get_Yn1s_Yn1c(self):
        n_col = self.get_shape()[0]
        if n_col%2!=0:
            raise ValueError('Yn1 can only be obtained from an odd n(order).')
        Yn1_pos = self.content[n_col//2]
        Yn1_neg = self.content[n_col//2-1]

        Yn1_s = np.array([(Yn1_pos-Yn1_neg)*1j])
        Yn1_c = np.array([Yn1_pos+Yn1_neg])

        return(type(self)(Yn1_s), type(self)(Yn1_c))

    # Takes the center m+1 rows of content.
    def cap_m(self, m):
        num_row = m+1
        tot_row = len(self.content)
        num_clip = tot_row - num_row
        if num_row == tot_row:
            return(self)
        if num_clip%2 != 0:
            raise AttributeError('cap_mode only works when input and '\
            'self.content are both even or odd.')
        return(type(self)(self.content[num_clip//2:-num_clip//2]))

    # Plotting -----------------------------------------------------------------------------
    # Get a 2-argument lambda function for plotting this term
    def get_lambda(self):
        raise NotImplementedError()

    # FFT the content and returns a ChiPhiFuncGrid
    def get_spectrum(self):
        return(ChiPhiFuncGrid(np.fft.fft(self.content, axis=1)))

    # Plot a period in both chi and phi
    def display(self, complex = False, size=(100,100), avg_clim = False):
        plt.rcParams['figure.figsize'] = [4,3]
        # This would trigger an error for most complex,
        # static methods used for evaluation.
        chi = np.linspace(0, 2*np.pi*0.99, size[0])
        phi = np.linspace(0, 2*np.pi*0.99, size[1])
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

    # JIT -----------------------------------------------------------------------------
    # An accelerated sum that aligns the center-point of 2-d arrays and zero-broadcasts the edges.
    # Input arrays must both have even/odd cols/rows
    # (such as (3,2), (13,6))
    # Copies arguments
    # -- Input: 2 2d arrays.
    # -- Output: 2d array
    # Switches between a compiled and a non-compiled implementation
    # depending on debug_mode (because print and appending global)
    # doesn't work in compiled methods.
    def add_jit(a, b, sign):
        if debug_mode:
            return(ChiPhiFunc.add_jit_debug(a, b, sign))
        else:
            return(ChiPhiFunc.add_jit_compiled(a, b, sign))

    def add_jit_debug(a, b, sign):
        shape = (max(a.shape[0], b.shape[0]),max(a.shape[1],b.shape[1]))
        out = np.zeros(shape, dtype=np.complex128)
        a_pad_row = (shape[0] - a.shape[0])//2
        a_pad_col = (shape[1] - a.shape[1])//2
        b_pad_row = (shape[0] - b.shape[0])//2
        b_pad_col = (shape[1] - b.shape[1])//2
        out[a_pad_row:shape[0]-a_pad_row,a_pad_col:shape[1]-a_pad_col] += a
        out[b_pad_row:shape[0]-b_pad_row,b_pad_col:shape[1]-b_pad_col] += b*sign

        # Debug. Compares the orders of magnitude of inputs.
        a_padded = np.empty(shape, dtype=np.complex128)
        a_padded[:] = np.nan
        b_padded = np.empty(shape, dtype=np.complex128)
        b_padded[:] = np.nan
        a_padded[a_pad_row:shape[0]-a_pad_row,a_pad_col:shape[1]-a_pad_col]\
            = np.log10(np.abs(a))
        b_padded[b_pad_row:shape[0]-b_pad_row,b_pad_col:shape[1]-b_pad_col]\
            = np.log10(np.abs(b))
        pow_diff = np.abs(a_padded - b_padded)

        # inf values shows up because often a and/or b is 0. Ignore them.
        pow_diff[pow_diff == np.inf] = np.nan
        debug_pow_diff_add.append(pow_diff.flatten())

        return(out)

    # The original add_jit
    @njit(complex128[:,:](complex128[:,:], complex128[:,:], int64))
    def add_jit_compiled(a, b, sign):
        shape = (max(a.shape[0], b.shape[0]),max(a.shape[1],b.shape[1]))
        out = np.zeros(shape, dtype=np.complex128)
        a_pad_row = (shape[0] - a.shape[0])//2
        a_pad_col = (shape[1] - a.shape[1])//2
        b_pad_row = (shape[0] - b.shape[0])//2
        b_pad_col = (shape[1] - b.shape[1])//2
        out[a_pad_row:shape[0]-a_pad_row,a_pad_col:shape[1]-a_pad_col] = a
        out[b_pad_row:shape[0]-b_pad_row,b_pad_col:shape[1]-b_pad_col] += b*sign
        return(out)

# A singleton subclass. The instance behaves like
# nan, except during multiplication with zero, where it becomes 0.
# This object never copies during operations.
# This works in conjunction with quirks in power_matching.mac: instead of ensuring that
# new upper bounds and suffixes (made by replacing the innermost index with
# a value satisfying eps^{f(i)} = eps^n) are consistent with,
# all original upper and lower bounds and are integers, it adds heaviside-like functions,
# is_integer and is_seq to make terms out-of-bound and non-integer suffixes zero.
class ChiPhiFuncNull(ChiPhiFunc):
    def __new__(cls, content=np.nan):
        if not hasattr(cls, 'instance'):
            cls.instance = ChiPhiFunc.__new__(cls)
        return cls.instance

    # The contents is dummy to enable calling of this singleton using
    # the default constructor
    def __init__(self, content=np.nan):
        self.content = np.nan

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

''' I.1 Utilities '''
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
    return np.diag(1j*mode_chi)

# Generates a matrix for converting a n-dim trig-fourier-representation vector (can be full or skip)
# into exponential-fourier-representation.
def fourier_to_exp_op(n_dim):
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
    return util_matrix

# # Remove the center row of a content array.
# def remove_center(content_in):
#     return(content_in[np.abs(np.arange(len(content_in))+0.5-len(content_in)/2)>=1])

# Add a given element at the center of a content array.
# def add_center(content_in, center_elem):
#     if len(content_in)%2!=0:
#         raise AttributeError('add_zero_center input must be of even length')
#     shape = list(content_in.shape)
#     shape[0] = shape[0]+1
#     out = np.zeros(shape)
#     out[:len(content_in)//2] = content_in[:len(content_in)//2]
#     out[-len(content_in)//2:] = content_in[-len(content_in)//2:]
#     out[-len(content_in)//2-1] = center_elem
#     return(out)

# two representations for phi dependence are implemented: Fourier and grid.

''' II. Grid representation for phi dependence '''
# Used for wrapping grid content. Defined outside the ChiPhiFuncGrid class so that
# it can be used in @njit compiled methods. Defined in front of the class because
# numba seem to care about the order
@njit(complex128[:,:](complex128[:,:]))
def wrap_grid_content_jit(content):
    len_chi = content.shape[0]
    len_phi = content.shape[1]
    content_looped = np.zeros((len_chi, len_phi+1), dtype=np.complex128)
    content_looped[:, :-1] = content
    content_looped[:, -1:] = content[:,:1]
    return(content_looped)

# A ChiPhiFunc where chi coeffs (free funcs in phi) are represented by function values in a grid.
# Initialization:
# ChiPhiFuncGrid(
#     np.array([
#         [Chi_coeff_sin n(phi)],    or    [Chi_coeff_-n(phi)],
#         ...                     or    ...
#         [Chi_coeff_sin 2(phi)],    or    [Chi_coeff_-1(phi)],
#         [const(phi)],           or    [Chi_coeff_1(phi)],
#         [Chi_coeff_cos 2(phi)],     or    ...
#         ...                     or    [Chi_coeff_n(phi)]
#         [Chi_coeff_cos n(phi)]
#     ])
# )
class ChiPhiFuncGrid(ChiPhiFunc):

    def __init__(self, content, fourier_mode = False):
        super().__init__(content, fourier_mode)
        # if content.shape[1]%2==0:
        #     raise ValueError('ChiPhiFuncGrid.__init__(): content has even# phi grids,'\
        #                      ' not supported. central difference is only invertible for odd-n matrices.')

    # Operator handlers -------------------------------------------------------
    # NOTE: will not be passed items awaiting for conditions.
    def dphi(self, order=1, mode='default'):
        return(ChiPhiFuncGrid(dphi_direct(self.content, order=order, mode=mode)))

    # Used in operator *. First stretch the phi axis to match grid locations,
    # Then do pointwise product.
    # -- Input: self and another ChiPhiFuncGrid
    # -- Output: a new ChiPhiFuncGrid
    @lru_cache(maxsize=1000)
    def multiply(self, other, div = False):
        a, b = self.stretch_phi_to_match(other)
        if div:
            b = 1.0/b
        # Now both are stretch to be dim (n_a, n_phi), (n_b, n_phi).
        # Transpose and 1d convolve all constituents.
        return(ChiPhiFuncGrid(ChiPhiFuncGrid.mul_grid_jit(a, b)))

    # Wrapper for mul_grid_jit. Handles int power.
    # -- Input: self and an int
    # -- Output: a new ChiPhiFuncGrid
    @lru_cache(maxsize=1000)
    def pow(self, int_pow):
        new_content = self.content.copy()
        for i in range(int_pow-1):
            new_content = ChiPhiFuncGrid.mul_grid_jit(new_content, self.content)
        return(ChiPhiFuncGrid(new_content))

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

        return out_transposed.T

    # Addition of 2 ChiPhiFuncGrid's.
    # Wrapper for numba method.
    # -- Input: self and another ChiPhiFuncGrid
    # -- Output: a new ChiPhiFuncGrid
    def add_ChiPhiFunc(self, other, sign=1):
        a,b = self.stretch_phi_to_match(other)
        # Now that grid points are matched by stretch_phi, we can invoke add_jit()
        # To add matching rows(chi coeffs) and grid points.

        return ChiPhiFuncGrid(ChiPhiFunc.add_jit(a,b,sign))

    # Used in operators, wrapper for stretch_phi. Match self's shape to another ChiPhiFuncGrid.
    # returns 2 CONTENTS.
    def stretch_phi_to_match(self, other):
        if self.get_shape()[1] == other.get_shape()[1]:
            return(self.content, other.content)
        # warnings.warn('Warning: phi grid stretching has occured. Shapes are:'\
        #     'self:'+str(self.get_shape())+'; other:'+str(other.get_shape()))
        max_phi_len = max(self.get_shape()[1], other.get_shape()[1])
        return(
            ChiPhiFuncGrid.stretch_phi(self.content, max_phi_len),
            ChiPhiFuncGrid.stretch_phi(other.content, max_phi_len)
        )

    # Addition of a constant with a ChiPhiFuncGrid.
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
        return ChiPhiFuncGrid(ChiPhiFunc.add_jit(self.content, stretched_constant,sign))


    # Properties ---------------------------------------------------------------
    # Returns the value when phi=0. Copies.
    def get_phi_zero(self):
        new_content = np.array([self.content[:,0]]).T
        if len(new_content) == 1:
            return(new_content[0][0])
        return(ChiPhiFuncGrid(np.array([self.content[:,0]]).T))

    # Math ---------------------------------------------------------------
    def integrate_chi(self, ignore_mode_0=False):
        len_chi = self.get_shape()[0]
        temp = np.arange(-len_chi+1,len_chi+1,2,dtype=np.float32)[:,None]
        if len_chi%2==1:
            temp[len(temp)//2]=np.inf
            if np.max(np.abs(self.content[len_chi//2]))>noise_level_int\
            and not ignore_mode_0:
                raise ValueError('Integrand has a significant chi-independent '\
                'component!')

        mode_i = -1j/temp
        return(type(self)(mode_i * self.content))


    # Used for solvability condition. phi-integrate a ChiPhiFuncGrid over 0 to
    # 2pi or 0 to a given phi. The boundary condition output(phi=0) = 0 is enforced.
    # -- Input: self and integral settings:
    # periodic=False evaluates integral from 0 to phi FOR EACH GRID POINT and
    # creates a ChiPhiFuncGrid with phi dependence.
    # periodic=True evaluates integral from 0 to 2pi and creates a ChiPhiFuncGrid
    # with NO phi dependence.
    # mode='simpson' is reasonably accurate and applicable to funcs with integral!=0
    # over a period
    # mode='fft' uses FFT.
    # -- Output: a new ChiPhiFuncGrid
    @lru_cache(maxsize=1000)
    def integrate_phi(self, periodic, mode = 'default'):
        # number of phi grids
        len_chi = self.get_shape()[0]
        len_phi = self.get_shape()[1]
        phis = np.linspace(0, 2*np.pi*(1-1/len_phi), len_phi, dtype=np.complex128)
        if mode == 'default':
            if periodic:
                mode = non_periodic_integral_mode
            else:
                mode = integral_mode

        if mode == 'fft':
            def integral(i_chi):
                out = scipy.fftpack.diff(self.content[i_chi], order=-1)
                out = out - out[0] # Enforces zero at phi=0 boundary condition.
                return(out)
            out_list = Parallel(n_jobs=n_jobs, backend=backend, require=require)(
                delayed(integral)(i_chi) for i_chi in range(len(self.content))
            )
            return(ChiPhiFuncGrid(np.array(out_list)))

        elif mode == 'simpson':
            new_content = integrate_phi_simpson(self.content, periodic = periodic)
        elif mode == 'spline':
            new_content = integrate_phi_spline(self.content, periodic = periodic)
        else:
            raise AttributeError('integrate_phi mode not recognized')

        if new_content.shape == (1,1):
            return(new_content[0,0])
        return(ChiPhiFuncGrid(new_content))
    # Used to calculate e**(ChiPhiFuncGrid). Only support ChiPhiFuncGrid with no
    # chi dependence
    def exp(self):
        if self.get_shape()[0]!=1:
            raise ValueError('exp only supports ChiPhiFuncGrid with no chi dependence!')
        return(ChiPhiFuncGrid(np.exp(self.content)))

    # SOLVING - CRITICAL! ------------------------------------------------

    # Solve under-determined degenerate system equivalent
    # v_source_A(chi, phi) * va_{n+1}(chi, phi) = v_rhs{n}(chi, phi).
    # When Y_mode=True, solves
    # wrapper for jit-enabled method batch_underdetermined_degen_jit.
    # -- Input --
    # v_source_A, v_rhs: ChiPhiFuncGrid,
    # rank_rhs: int, 'n' in the equation above
    # i_free: int, index of free var in va,
    # vai: ChiPhiFuncGrid with a single row, value of free component.
    def solve_underdet_degen(v_source_A, v_source_B, v_rhs, rank_rhs, i_free, vai,
        Y_mode = False, ignore_extra=False):
        # Checking input validity
        if type(v_source_A) is not ChiPhiFuncGrid\
            or type(v_source_B) is not ChiPhiFuncGrid\
            or type(v_rhs) is not ChiPhiFuncGrid:
            raise TypeError('ChiPhiFuncGrid.solve_underdetermined: '\
                            'v_source_A, v_rhs, should all be '\
                            'ChiPhiFuncGrid. The actual types are:'
                            +str(type(v_source_A))+', '
                            +str(type(v_rhs)))

        v_source_A_content, v_source_B_content = v_source_A.stretch_phi_to_match(v_source_B)
        v_source_A_content, v_rhs_content = v_source_A.stretch_phi_to_match(v_rhs)

        if type(vai) is not ChiPhiFuncGrid:
            if np.isscalar(vai):
                vai_content = np.full((1,v_source_A_content.shape[1]),vai, dtype=np.complex128)
            else:
                raise TypeError('ChiPhiFuncGrid.solve_underdetermined: '\
                                'vai is not scalar ChiPhiFuncGrid. '\
                                'The actual type is: '+str(type(vai)))
        else:
            vai_content = vai.content

        # Center-pad v_rhs if it's too short
        if v_source_A.get_shape()[0] + rank_rhs != v_rhs.get_shape()[0]:
            # warnings.warn('Warning: A, v_rhs and rank_rhs doesn\'t satisfy mode'
            #               ' number requirements. Zero-padding rhs chi components.'+
            #               ' v_source_A shape=' + str(v_source_A.get_shape()) +
            #               ', _rhs shape=' + str(v_rhs.get_shape()) +
            #               ', rank_rhs=' + str(rank_rhs))
             # This creates a padded content for v_rhs in case some components are zero. However, we still need to put in check for
             # LHS and RHS's even and oddness.
            v_rhs_content = ChiPhiFunc.add_jit(\
                v_rhs_content,\
                np.zeros((v_source_A.get_shape()[0]+rank_rhs, v_rhs.get_shape()[1]), dtype=np.complex128),\
                1 # Sign is 1
            )

        if Y_mode:
            va_content = batch_ynp1_jit(
                v_source_A_content,
                v_source_B_content,
                v_rhs_content,
                rank_rhs,
                i_free,
                vai_content, ignore_extra=ignore_extra
            )
        else:
            va_content = batch_underdetermined_degen_jit(
                v_source_A_content,
                v_rhs_content,
                rank_rhs,
                i_free,
                vai_content, ignore_extra=ignore_extra
            )

        return(ChiPhiFuncGrid(va_content))

    # Solve exactly determined degenerate system equivalent to
    # v_source_A(chi, phi) * va_{n}(chi, phi) = v_rhs{n}(chi, phi).
    # You can also think of this as "/" with chi-dependent "other" argument.
    # wrapper for jit-enabled method batch_degen_jit.
    def solve_degen(v_source_A, v_rhs, rank_rhs):

        # checking input types
        if type(v_source_A) is not ChiPhiFuncGrid\
            or type(v_rhs) is not ChiPhiFuncGrid:
            raise TypeError('ChiPhiFuncGrid.solve_underdetermined: '\
                            'v_source_A, v_rhs, vai should all be ChiPhiFuncGrid.')

        if v_source_A.get_shape()[0] + rank_rhs - 1 != v_rhs.get_shape()[0]:
            print("v_source_A shape:", v_source_A.get_shape())
            print("v_rhs shape:", v_rhs.get_shape())
            print("rank_rhs:", rank_rhs)
            warnings.warn('Warning: A, v_rhs and rank_rhs doesn\'t satisfy mode'
                          ' number requirements. Zero-padding rhs chi components.')
            # This creates a padded content for v_rhs in case some components are zero. However, we still need to put in check for
            # LHS and RHS's even and oddness.
            v_rhs_content = ChiPhiFunc.add_jit(
                v_rhs_content,
                np.zeros((v_source_A.get_shape()[0]+rank_rhs-1, v_rhs.get_shape()[1]), dtype=np.complex128)
            )
        v_source_A_content = v_source_A.content
        v_rhs_content = v_rhs.content

        va_content = batch_degen_jit(
            v_source_A.content,
            v_rhs.content,
            rank_rhs
        )
        return(ChiPhiFuncGrid(va_content))


    # Plotting -----------------------------------------------------------
    # Get a 2d vectorized function for plotting this term
    def get_lambda(self):
        len_chi = self.get_shape()[0]
        len_phi = self.get_shape()[1]

        # Create 'x' for interpolation. 'x' is 1 longer than lengths specified due to wrapping
        phi_looped = np.linspace(0,2*np.pi, len_phi+1)
        # wrapping
        content_looped = wrap_grid_content_jit(self.content)

        ind_chi = len_chi-1
        mode_chi = np.linspace(-ind_chi, ind_chi, len_chi)

        # The outer dot product is summing along axis 0.
        return(np.vectorize(
            lambda chi, phi : sum(
                [np.e**(1j*(chi)*mode_chi[i]) * np.interp((phi)%(2*np.pi), phi_looped, content_looped[i]) for i in range(len_chi)]
            )
        ))

    def display_content(self, fourier_mode = True, colormap_mode = False):
        plt.rcParams['figure.figsize'] = [8,3]
        len_phi = self.get_shape()[1]
        phis = np.linspace(0,2*np.pi*(1-1/len_phi), len_phi)
        if fourier_mode:
            fourier = self.export_trig()
            if len(fourier.content)%2==0:
                ax1 = plt.subplot(121)
                ax1.set_title('cos')
                ax2 = plt.subplot(122)
                ax2.set_title('sin')
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
                ax1.set_title('cos')
                ax2 = plt.subplot(132)
                ax2.set_title('constant')
                ax3 = plt.subplot(133)
                ax3.set_title('sin')
                if colormap_mode and len(fourier.content) != 1:
                    modesin = np.linspace(2, len(fourier.content)-1, len(fourier.content)//2)
                    modecos = np.linspace(len(fourier.content)-1, 2, len(fourier.content)//2)
                    phi = np.linspace(0, 2*np.pi*(1-1/len_phi), len_phi)
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
            ax1.set_title('Real')
            ax2 = plt.subplot(122)
            ax2.set_title('Imaginary')

            if colormap_mode:
                mode = np.linspace(-len(self.content)+1, len(self.content)-1, len(self.content))
                phi = np.linspace(0, 2*np.pi*(1-1/len_phi), len_phi)
                ax1.pcolormesh(phi, mode, np.real(self.content))
                ax2.pcolormesh(phi, mode, np.imag(self.content))
            else:
                ax1.plot(phis,np.real(self.content).T)
                ax2.plot(phis,np.imag(self.content).T)
        plt.show()

    # Utilities --------------------------------------------------------
    # A simple filter calculating a 3-element rolling average:
    # [..., a, b, c, ...] = [..., 0.25a+0.5b+0.25c, ...]
    def filter(self, mode='low_pass', arg=low_pass_freq):
        if mode == 'roll_avg':
            content = self.content
            a = np.roll(content, -1, axis=1)
            b = np.roll(content, 1, axis=1)
            return(ChiPhiFuncGrid(0.5*content+0.25*a+0.25*b))
        elif mode == 'low_pass':
            return(ChiPhiFuncGrid(low_pass_direct(self.content, arg)))
        else:
            raise AttributeError('ChiPhiFuncGrid.filter: mode not recognized.')

    def noise_filter(self, mode='low_pass', arg=low_pass_freq):
        return(self-self.filter(mode=mode, arg=arg))


    # Stretching each row individually by interpolation.
    # The phi grid is assumed periodic, and the first column is always
    # wrapped after the last column to correctly interpolate
    # periodic function. Handles same-length cases by skipping.
    # -- Input: self and an int length
    # -- Output: a new ChiPhiFuncFourier
    @njit(complex128[:,:](complex128[:,:], int64), parallel=True)
    def stretch_phi(content, len_target):
        len_phi = content.shape[1]
        if len_phi == len_target:
            return(content)
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


    # Converts a single-argument function to values on len_phi grid points located
    # at 0, 1*2pi/len_phi, 2*2pi/len_phi, ......, 2pi(1-1/len_phi)
    # -- Input: a function and a int specifying grid number
    # -- Output: an array
    def func_to_grid(f_phi, len_phi):
        x = np.linspace(0,2*np.pi*(1-1/len_phi) ,len_phi)
        return(f_phi(x))

    # Converting fourier coefficients into exponential coeffs used in
    # ChiPhiFunc's internal representation. Only used during super().__init__
    # Does not copy.
    def trig_to_exp(self):
        util_matrix_chi = fourier_to_exp_op(self.get_shape()[0])
        # Apply the conversion matrix on chi axis
        self.content = util_matrix_chi @ self.content


    # Converting fourier coefficients into exponential coeffs used in
    # ChiPhiFunc's internal representation. Only used during super().__init__
    # Does not copy.
    def export_trig(self):
        util_matrix_chi = np.linalg.inv(fourier_to_exp_op(self.get_shape()[0]))
        # Apply the conversion matrix on chi axis
        return(type(self)(util_matrix_chi @ self.content))

''' II.1 Grid utilities '''

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
def integrate_phi_simpson(content, dx = 'default', periodic = False):
    print('simpson')
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
        return(np.array(out_list).T)

# Implementation of spline-based integrate_phi using Parallel.
def integrate_phi_spline(content, dx = 'default', periodic=False,
    diff=False, diff_order=None):

    len_chi = content.shape[0]
    len_phi = content.shape[1]
    if dx == 'include_2pi':
        dx = 2*np.pi/(len_phi-1)
    elif dx == 'default':
        dx = 2*np.pi/len_phi
        # purely real.

    phi = np.linspace(0, dx*(len_phi-1), len_phi)

    def generate_and_integrate_spline(i_chi):
        new_spline = scipy.interpolate.make_interp_spline(phi, content[i_chi])
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
        evaluate_spline = lambda spline, phi : spline(phi)
        out_list = Parallel(n_jobs=n_jobs, backend=backend, require=require)(
            delayed(evaluate_spline)(spline, phi) for spline in integrate_spline_list
        )

        return(np.array(out_list))

# Generate phi differential operator diff_matrix. diff_matrix@f.content = dchi(f).content
# invert = True generates anti-derivative operator. Cached for each new Chi length.
# -- Input: len_chi: length of Chi series.
# -- Output: 2d matrix.
def dphi_op_pseudospectral(n_phi):
    from qsc import spectral_diff_matrix
    out = spectral_diff_matrix(n_phi, xmin=0, xmax=2*np.pi)
    return(out)

# a low pass filter acting on  a content matrix
def low_pass_direct(content, freq):
    len_phi = content.shape[1]
    W = np.abs(np.fft.fftfreq(len_phi))
    f_signal = np.fft.fft(content, axis = 1)

    # If our original signal time was in seconds, this is now in Hz
    cut_f_signal = f_signal.copy()
    cut_f_signal[:,(W>freq/len_phi)] = 0

    return(np.fft.ifft(cut_f_signal, axis=1))

# dphi of a content matrix
def dphi_direct(content, order=1, mode='default'):
    if order<0:
        raise AttributeError('dphi order must be positive.')

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
        out = integrate_phi_spline(content, periodic=False,
            diff=True, diff_order=order)
        return(out)

    else:
        raise AttributeError('dphi mode not recognized.')

# Calculates the max amplitude's order of magnitude
@njit(float64(complex128[:,:]))
def max_log10(input):
    return(np.log10(np.max(np.abs(input))))

''' III. Fourier representation for phi dependence - DEPRECIATED '''

# Implementation of ChiPhiFunc using FULL, exponential Fourier series to represent
# free functions of phi.
# When fourier_mode is enabled during initialization, content would be treated
# as fourier coefficients of format:
# [
#      s_chi_n = [s_k, ..., s_2, s_1, const, c_1, c_2, ..., c_k],
#      ...
#      s_chi_1 = [s_k, ..., s_2, s_1, const, c_1, c_2, ..., c_k],
#      c_chi_1 = [s_k, ..., s_2, s_1, const, c_1, c_2, ..., c_k],
#      ...
#      c_chi_n = [s_k, ..., s_2, s_1, const, c_1, c_2, ..., c_k],
# ]
class ChiPhiFuncFourier(ChiPhiFunc):
    def __init__(self, content, fourier_mode = False):
        super().__init__(content, fourier_mode)
        if content.shape[1]%2 == 0:
            raise ValueError('Phi coefficients should be a full fourier series. Even phi_dim detected.')
    # Operator handlers -------------------------------------------------
    # NOTE: will not be passed items awaiting for conditions.

    # Addition of 2 ChiPhiFuncFourier's.
    # Wrapper for numba method.
    # -- Input: self and another ChiPhiFuncFourier
    # -- Output: a new ChiPhiFuncFourier
    def add_ChiPhiFunc(self, other):
        return ChiPhiFuncFourier(ChiPhiFunc.add_jit(self.content, other.content))

    # Addition of a constant with a ChiPhiFuncFourier.
    # Wrapper for numba method.
    def add_const(self, other):
        return ChiPhiFuncFourier(ChiPhiFunc.add_jit(self.content, np.complex128([[other]])))

    # Handles pointwise multiplication (* operator)
    # Convolve2d is already compiled. No need to jit.
    # -- Input: self and other
    # -- Output: a new ChiPhiFuncFourier
    def multiply(self, other, div=False):
        if div:
            raise NotImplementedError()
        return(ChiPhiFuncFourier(scipy.signal.convolve2d(self.content, other.content)))

    # Calculates an integer power of a ChiPhiFuncFourier
    # Convolve2d is already compiled. No need to jit.
    # Also here we assume all powers are to fairly low orders (2)
    # -- Input: self and power (int)
    # -- Output: self and other
    def pow(self, int_pow):
        new_content = self.content.copy()
        for n in range(int_pow-1):
            new_content = scipy.signal.convolve2d(new_content, new_content)
        return(ChiPhiFuncFourier(new_content))

    # Get a 2-argument lamnda function for plotting this term
    def get_lambda(self):
        len_chi = self.get_shape()[0]
        len_phi = self.get_shape()[1]

        if len_phi%2!=1:
            raise ValueError('coeffs_chi must have an odd number of components on phi axis')

        ind_phi = int((len_phi-1)/2)
        mode_phi = np.linspace(-ind_phi, ind_phi, len_phi)

        ind_chi = len_chi-1
        mode_chi = np.linspace(-ind_chi, ind_chi, len_chi).reshape(-1,1)
        # The outer dot product is summing along axis 0.
        # The inner @ (equivalent to dot product) is summing along axis 1.
        return(np.vectorize(lambda chi, phi : np.dot(self.content@(np.e**(1j*phi*mode_phi)), (np.e**(1j*(-chi)*mode_chi)))))


    # Utilities ---------------------------------------------------

    # Converting fourier coefficients into exponential coeffs used in
    # ChiPhiFunc's internal representation. Only used during super().__init__
    # Does not copy.
    def trig_to_exp(self):
        util_matrix_chi = fourier_to_exp_op(self.get_shape()[0])
        util_matrix_phi = fourier_to_exp_op(self.get_shape()[1])
        # Apply the conversion matrix on phi axis
        # (two T's because np.matmul can't choose axis)
#         self.content = (util_matrix_phi @ self.content.T).T
        self.content = self.content @ util_matrix_phi.T
        # Apply the conversion matrix on chi axis
        self.content = util_matrix_chi @ self.content

    # DEPRECIATED
    # Generate phi central difference operator diff_matrixT. content @ diff_matrixT.T = dchi(f).
    # (.T actually not needed since this the operator is diagonal)
    # -- Input: len_phi: length of phi series.
    # -- Output: 2d matrix.
    @njit
    def dphi_op(len_phi, invert = False):
        ind_phi = int((len_phi-1)/2)
        mode_phi = np.linspace(-ind_phi, ind_phi, len_phi)
        if invert:
            return(np.diag(-1j/mode_phi))
        return(np.diag(1j*mode_phi))


''' IV. Grid 1D deconvolution '''
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
# Codes written in this part are specifically for 1D deconvolution used for ChiPhiFuncGrid.

''' IV.1 va, vb with the same number of dimensions '''
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
# For recursion relations with ChiPhiFunc's, A should come from
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

''' IV.2 va has 1 more component than vb '''
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
    if (in_matrix.shape[0] - in_matrix.shape[1])%2!=1:
        raise AttributeError('This method takes rows from the middle. The array'\
        'shape must have an even difference between row and col numbers.')
    if in_matrix.ndim != 2:
        raise ValueError("Input should be 2d array")

    n_row = in_matrix.shape[0]
    n_col = in_matrix.shape[1]
    if n_row<=n_col:
        raise ValueError("Input should have more rows than cols")

    if ind_col>=n_col:
        raise ValueError('ind_col should be smaller than column number')

    # Remove specfied column (slightly faster than delete)
    # and remove extra rows (take n_col-1 rows from the center)
    rows_to_remove = (n_row-(n_col-1))//2
    sub = in_matrix[:,np.arange(in_matrix.shape[1])!=ind_col][rows_to_remove:-rows_to_remove, :]
    sqinv = np.linalg.inv(sub)
    padded = np.zeros((n_row, n_row), dtype = np.complex128)
    padded[rows_to_remove:-rows_to_remove, rows_to_remove:-rows_to_remove] = sqinv
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
# When Y_mode=True, vai is the sum between the center two elements.
# (A must have an even number of rows)
# (Yn1p + Yn1n = Yn1c = Y11s * sigma_twiddle_n)
@njit(complex128[:](complex128[:,:], complex128[:], int64, complex128, boolean))
def solve_degenerate_underdetermined_jit(A, v_rhs, i_free, vai, Y_mode=False):
    n_dim = A.shape[1]-1 # #dim of vb, which is #dim_va-1. vb is v_rhs before the convolution
    if Y_mode:
        # if A.shape[1]%2!=0:
        #     warnings.warn('Warning: Y_mode=True on even order. vai should be Yn0')
        i_1p = A.shape[1]//2   # Index of Yn1p (or if n is even, Yn0)
        i_1n = A.shape[1]//2-1 # Index of Yn1n (or if n is even, Yn2_n)
        i_free = i_1p # set Yn1p as free var
    if A.shape[0] != v_rhs.shape[0]:
        raise ValueError("solve_underdetermined: A, v_rhs must have the same number of rows")
    A_einv = np.ascontiguousarray(inv_square_excluding_col_jit(A, i_free))
    A_free_col = np.ascontiguousarray(A.T[i_free])
    v_rhs = np.ascontiguousarray(v_rhs)
    va_free_coef = (A_einv@A_free_col)
    clip_n = (A.shape[0]-n_dim)//2 # How much is the transposed array larger than Yn
    # This vector is actually m-dim, with m-n blank elems at the end.
    if Y_mode and A.shape[1]%2==0: # Y_mode=True and on odd order (ODE exists)
        # The rest of the procedure is carried out normally with
        # i_free pointing at Yn1n. The resulting Yn should be
        # Yn = (A_einv@np.ascontiguousarray(v_rhs) - Yn1n * va_free_coef)[:n_dim]
        # where va_free_coef is a vector. This gives Yn1n = Yn1n and
        #
        # Yn1p = Yn[i_1p] = (A_einv@v_rhs - Yn1n * va_free_coef)[i_1p]
        # = A_einv[i_1p]@v_rhs - Yn1n * va_free_coef[i_1p]
        #
        # Therefore,
        # Yn1p+Yn1n=vai is equivalent to
        # A_einv[i_1n]@v_rhs - Yn1p * va_free_coef[i_1n] + Yn1n = vai
        # A_einv[i_1n]@v_rhs - Yn1p * (va_free_coef[i_1n]-1) = vai
        # A_einv[i_1n]@v_rhs - vai = Yn1p * (va_free_coef[i_1n]-1)
        # (A_einv[i_1n]@v_rhs - vai)/(va_free_coef[i_1n]-1) = Yn1p
        # i_1n is shifted by the padding
        vai = (A_einv[clip_n+i_1n]@v_rhs - vai)/(va_free_coef[clip_n+i_1n]-1)
    Yn = (A_einv@v_rhs - vai * va_free_coef)
    Yn = Yn[clip_n:-clip_n]
    return(np.concatenate((Yn[:i_free], np.array([vai]) , Yn[i_free:])))

# @njit(complex128[:](complex128[:,:], complex128[:,:], complex128[:], int64, complex128))
# def solve_degenerate_underdetermined_jit(A, B, vb, i_free, vai):
#     B_cont = np.ascontiguousarray(B)
#     vb_cont = np.ascontiguousarray(vb)
#     return(solve_degenerate_underdetermined_jit(A, B_cont@vb_cont, i_free, vai))

''' IV.3 Convolution operator generator and ChiPhiFuncGrid.content numba wrapper '''
# Generate convolution operator from a for an n_dim vector.
# Can't be compiled for parallel beacuase vec and out_transposed's sizes dont match?
@njit(complex128[:,:](complex128[:], int64))
def conv_matrix(vec, n_dim):
    out_transposed = np.zeros((n_dim,len(vec)+n_dim-1), dtype = np.complex128)
    for i in prange(n_dim):
        out_transposed[i, i:i+len(vec)] = vec
    return(out_transposed.T)

# The first
@njit(complex128[:,:](complex128[:], int64))
def finite_diff_matrix(stencil, n_dim):
    if len(stencil)%2!=1 or len(stencil.shape)!=1:
        raise AttributeError('The stencil must be 1d and have odd number of elements')
    half_len = len(stencil)//2
    out_transposed = np.zeros((n_dim, n_dim), dtype = np.complex128)
    first_row = np.zeros((n_dim, n_dim), dtype = np.complex128)
    first_row[-half_len:] = stencil[:half_len]
    first_row[:half_len+1] = stencil[-half_len-1:]
    for i in prange(n_dim):
        out_transposed[i] = np.roll(first_row, i)
    return(out_transposed.T)


@njit(float64[:,:](float64[:], int64))
def conv_matrix_re(vec, n_dim):
    out_transposed = np.zeros((n_dim,len(vec)+n_dim-1), dtype = np.float64)
    for i in prange(n_dim):
        out_transposed[i, i:i+len(vec)] = vec
    return(out_transposed.T)

# For solving a*va = v_rhs, where va, vb have the same number of dimensions.
# In the context below, "#dim" represents number of chi mode components.
# Note: "vector" means a series of chi coefficients in this context.
#
# -- Input --
# v_source_A: 2d matrix, content of ChiPhiFuncGrid, #dim = a
# v_rhs: 2d matrix, content of ChiPhiFuncGrid. Should be #dim = m vector
#     produced by convolution of a #dim = rank_rhs vector.
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
@njit(complex128[:,:](complex128[:,:], complex128[:,:], int64, int64, complex128[:,:], boolean), parallel=True)
def batch_underdetermined_degen_jit(v_source_A, v_rhs, rank_rhs, i_free, vai, ignore_extra):

    # Checking dimensionality
    if len(v_source_A) + rank_rhs != len(v_rhs):
        if len(v_source_A) + rank_rhs < len(v_rhs):
            num_extra = (len(v_rhs) - (len(v_source_A) + rank_rhs))//2
            # ignoring extra component is allowed and extra components are small
            # Can't have an even number of extra components
            # Checking extra components' size
            if ignore_extra\
                and (len(v_rhs) - (len(v_source_A) + rank_rhs))%2==0\
                and max_log10(v_rhs[:num_extra]) < max_log10(v_rhs[num_extra:-num_extra]) - noise_order_solve\
                and max_log10(v_rhs[-num_extra:]) < max_log10(v_rhs[num_extra:-num_extra]) - noise_order_solve:
                v_rhs = v_rhs[num_extra:-num_extra]
            else:
                raise ValueError('batch_underdetermined_deconv: #dim_A + rank_rhs < #dim'\
                    '_v_rhs.')
        else:
            raise ValueError('batch_underdetermined_deconv: #dim_A + rank_rhs > #dim'\
                '_v_rhs.')

    A_slices = np.ascontiguousarray(v_source_A.T) # now the axis 0 is phi grid
    v_rhs_slices = np.ascontiguousarray(v_rhs.T) # now the axis 0 is phi grid
    # axis 0 is phi grid, axis 1 is chi mode
    va_transposed = np.zeros((len(A_slices), rank_rhs+1), dtype = np.complex128)
    if len(A_slices) != len(v_rhs_slices):
        raise ValueError('batch_underdetermined_deconv: A, v_rhs must have the same number of phi grids.')
    for i in prange(A_slices.shape[0]):
        A_conv_matrix_i = conv_matrix(A_slices[i], rank_rhs+1)
        va_transposed[i, :] = solve_degenerate_underdetermined_jit(A_conv_matrix_i,\
                                         v_rhs_slices[i], i_free, np.ravel(vai)[i], False)
    return va_transposed.T

# Modification of batch_underdetermined_degen_jit for solving (conv(a) + conv(b)@dchi)@Yn+1 = RHS - LHS(Yn+1 = 0).
#@njit(complex128[:,:](complex128[:,:], complex128[:,:], complex128[:,:], int64, int64, complex128[:,:]), parallel=True)
@njit(complex128[:,:](complex128[:,:], complex128[:,:], complex128[:,:], int64, int64, complex128[:,:], boolean), parallel=True)
def batch_ynp1_jit(v_source_A, v_source_B, v_rhs, rank_rhs, i_free, vai, ignore_extra):

    # Checking dimensionality
    if len(v_source_A) + rank_rhs != len(v_rhs):
        if len(v_source_A) + rank_rhs < len(v_rhs):
            num_extra = (len(v_rhs) - (len(v_source_A) + rank_rhs))//2
            # ignoring extra component is allowed and extra components are small
            # Can't have an even number of extra components
            # Checking extra components' size
            if ignore_extra\
                and (len(v_rhs) - (len(v_source_A) + rank_rhs))%2==0\
                and max_log10(v_rhs[:num_extra]) < max_log10(v_rhs[num_extra:-num_extra]) - noise_order_solve\
                and max_log10(v_rhs[-num_extra:]) < max_log10(v_rhs[num_extra:-num_extra]) - noise_order_solve:
                v_rhs = v_rhs[num_extra:-num_extra]
            else:
                print('#dim_A, rank_rhs, #dim_v_rhs:', len(v_source_A), rank_rhs, len(v_rhs))
                raise ValueError('batch_underdetermined_deconv: #dim_A + rank_rhs < #dim'\
                    '_v_rhs.')
        else:
            print('#dim_A, rank_rhs, #dim_v_rhs:', len(v_source_A), rank_rhs, len(v_rhs))
            raise ValueError('batch_underdetermined_deconv: #dim_A + rank_rhs > #dim'\
                '_v_rhs.')

    A_slices = np.ascontiguousarray(v_source_A.T) # now the axis 0 is phi grid
    B_slices = np.ascontiguousarray(v_source_B.T) # now the axis 0 is phi grid
    v_rhs_slices = np.ascontiguousarray(v_rhs.T) # now the axis 0 is phi grid
    # axis 0 is phi grid, axis 1 is chi mode
    va_transposed = np.zeros((len(A_slices), rank_rhs+1), dtype = np.complex128)
    if len(A_slices) != len(v_rhs_slices):
        raise ValueError('batch_underdetermined_deconv: A, v_rhs must have the same number of phi grids.')

    # generate dchi operators
    dchi_matrix = np.ascontiguousarray(dchi_op(rank_rhs+1, False))
    for i in prange(A_slices.shape[0]): # Loop for each point in the phi grid
        # For Yn1, these are (rank_rhs+1+1,rank_rhs+1) because a and b are (2,n_grid) matrices.
        A_conv_matrix_i = conv_matrix(A_slices[i], rank_rhs+1)
        B_conv_matrix_i = np.ascontiguousarray(conv_matrix(B_slices[i], rank_rhs+1))
        total_matrix = A_conv_matrix_i + B_conv_matrix_i@dchi_matrix
        va_transposed[i, :] = solve_degenerate_underdetermined_jit(total_matrix,\
                                         v_rhs_slices[i], 0, np.ravel(vai)[i], True)
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

''' V. Solving linear PDE in phi grids '''

''' V.1 Solving the periodic linear PDE (a + b * dphi + c * dchi) y = f(phi, chi) '''
# Solves simple linear first order ODE systems in batch:
# (coeff + coeff_phi d/dphi) y = f. ( y' + p_eff*y = f_eff )
# (Dchi = +- m * 1j)
# All inputs are content matrices.
# p and f are assumed periodic.
# P is preferrably non-resonant for small p amplitude.
# All coeffs' CONTENTS are point-wise multiplied to f, dpf or dcf's content.

# The code solves the equation exactly with integration factor for small p amplitude.
# When p_eff's max amplitude>17, The exponential in the integrating factor
# usually blow up. An asymptotic series method based on Orszag
# P258 - IBP for Laplace integrals would be used. It can also be
# Manually engaged by integral_mode = 'asymptotic'

# When the avg amplitude of P falls between 11 and 30, both spline
# and asymptotic series works poorly. The code uses a spectral method
# that inverts a #grid x #grid convolution matrix. It is be highly accurate
# but slow, and is only used in this region. Notably, the code also attempts
# FFT if p is resonant.

# FFT can break down if the mode coeff of the first two modes exceeds 22000 and
# 33000. The code defaults to asymptotic series in that case.

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
def solve_integration_factor(coeff, coeff_dp, f, \
    integral_mode='auto', asymptotic_order=asymptotic_order, masking = True):

    len_phi = f.shape[1]
    len_chi = f.shape[0]

    # Rescale the eq into y'+py=f
    f_eff = f/coeff_dp
    p_eff = coeff/coeff_dp

    f_eff_scaling = np.average(np.abs(f_eff))
    f_eff = f_eff/f_eff_scaling

    print('solve_integration_factor: average p_eff:', np.average(np.abs(p_eff)))
    print('solve_integration_factor: average f_eff:', f_eff_scaling)

    # Make f_eff and p_eff both shaped like f
    if np.isscalar(p_eff):
        p_eff = p_eff*np.ones_like(f, dtype = np.complex128)
    elif p_eff.shape[0]!=f_eff.shape[0]:
        raise AttributeError('p_eff and f_eff has different component numbers!')
    elif p_eff.shape[1]!=f_eff.shape[1]:
        p_eff = ChiPhiFuncGrid.stretch_phi(p_eff, len_phi)


    effective_dx = 2*np.pi/(len_phi)

    # Depending on each component's amplitude and maximum real component, decide which method to use
    if integral_mode == 'auto':
        modes = []
        for i in range(f.shape[0]):
            if np.average(np.abs(np.real(p_eff[i])))<8:
                modes.append('spline')
            elif np.average(np.abs(np.real(p_eff[i])))>25:
                modes.append('asymptotic')
            else:
                modes.append('double_spline')
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
                    asymptotic_order = asymptotic_order)[0]
            out_list = Parallel(n_jobs=n_jobs, backend=backend, require=require)(
                delayed(solve_1d)(p_eff[i], 1, f_eff[i], modes[i])\
                for i in range(len_chi)
            )
            return(np.array(out_list))
        # Batch solve the whole group of equations
        # with asymptotic series if all of them has large amplitudes
        integral_mode = modes[0]

    print('integral_mode is', integral_mode)
    if integral_mode == 'asymptotic':
        ai = f/coeff # f/p
        integration_factor = ai.copy()
        for i in range(asymptotic_order):
            # ai is periodic. We use the non-looped value to ensure
            # that dphi by fft functions correctly
            ai_new = -(ChiPhiFuncGrid(ai).dphi().content)/p_eff
            if np.max(np.abs(ai_new)) > np.max(np.abs(ai)):
                print('Optimum truncation at order', i+1)
                print('Amplitude of the truncation term:', np.amax(np.abs(ai), axis = 1))
                break
            ai = ai_new
            integration_factor += ai

        return(integration_factor*f_eff_scaling, np.amax(np.abs(ai), axis = 1)*f_eff_scaling)

    # elif integral_mode == 'fft':
    #     print('first 5 modes in p:',
    #           np.max(np.fft.fft(p_eff, axis = 1)[:,1]),
    #           np.max(np.fft.fft(p_eff, axis = 1)[:,2]),
    #           np.max(np.fft.fft(p_eff, axis = 1)[:,3]),
    #           np.max(np.fft.fft(p_eff, axis = 1)[:,4]),
    #           np.max(np.fft.fft(p_eff, axis = 1)[:,5]),)
    #     print('max mode amplitude of p:',
    #           np.max(np.fft.fft(p_eff, axis = 1)))
    #
    #     f_fft = np.fft.fft(f_eff, axis = 1)
    #     diff_matrix = fft_dphi_op(len_phi)
    #     conv_matrix = fft_conv_op_batch(p_fft)
    #     inv_dxpp = np.linalg.inv(diff_matrix[None, :, :] + conv_matrix)
    #     sln_fft = (inv_dxpp@f_fft[:,:,None])[:,:,0]
    #     sln = np.fft.ifft(sln_fft, axis = 1)
    #     return(sln*f_eff_scaling, None)

    elif integral_mode == 'double_spline':
        asymptotic_y, error = solve_integration_factor(coeff, coeff_dp, f, \
        integral_mode='asymptotic', asymptotic_order=10)
        y_bound = np.amax(np.abs(asymptotic_y), axis = 1) + error
        C0_guess = asymptotic_y[:,0]
        # testing the idea of BC matching
        f_double = np.zeros((len_chi, len_phi*2), dtype = np.complex128)
        p_double = np.zeros((len_chi, len_phi*2), dtype = np.complex128)
        f_double[:, :len_phi] = f_eff
        f_double[:, len_phi:] = f_eff
        p_double[:, :len_phi] = p_eff
        p_double[:, len_phi:] = p_eff

        int_p = integrate_phi_spline(p_eff, periodic=False, dx = effective_dx)
        int_p_double = integrate_phi_spline(p_double, periodic=False, dx = effective_dx)

        exp_phi_double = np.exp(int_p_double)
        exp_negphi_double = np.exp(-int_p_double)

        integrand_double = f_double*exp_phi_double
        integration_factor_double = integrate_phi_spline(integrand_double, periodic=False, dx = effective_dx)

        # Finding location to pick BC
        # The periodic BC is applied where 1/I has the largest difference in period 1 and period 2
        # int_factor_over_I_double = exp_negphi_double*integration_factor_double
        int_factor_over_I_double = np.exp(np.log(integration_factor_double) - np.log(exp_phi_double))
        metric = int_factor_over_I_double[:, len_phi:]-int_factor_over_I_double[:, :len_phi]

        loc_BC1 = np.argmax(np.abs(metric), axis=1)
        loc_BC2 = loc_BC1+len_phi
        loc_BC2[loc_BC2>len_phi*2] -= len_phi*2

        I_neg_1 = exp_negphi_double[np.arange(len_chi), loc_BC1]
        I_neg_2 = exp_negphi_double[np.arange(len_chi), loc_BC2]

        int1 = int_factor_over_I_double[np.arange(len_chi), loc_BC1]
        int2 = int_factor_over_I_double[np.arange(len_chi), loc_BC2]

        Ci = (int2 - int1)/(I_neg_1-I_neg_2)
        out = (Ci[:, None]*exp_negphi_double+int_factor_over_I_double)[:, :len_phi]
        # Plotting
        # for i in range(len_chi):
        #     plt.title('Integrating factor '+str(i))
        #     plt.plot(exp_negphi_double[i])
        #     plt.show()
        #     plt.title('1/I '+str(i))
        #     plt.plot(int_factor_over_I_double[i])
        #     plt.show()

        # # Optimizing for variance... Produces non-smooth
        # for i in range(len_chi):
        #     def cost(A):
        #         u = A[0]
        #         v = A[1]
        #         ans = (u+1j*v)*exp_negphi_double[i]+int_factor_over_I_double[i]
        #         var = ans.var()
        #         BC = np.abs(ans[0] - ans[999])#ans[loc_BC1[i]] - ans[loc_BC2[i]]
        #         plt.plot(ans, alpha = 0.1)
        #         plt.title(var)
        #         return(var + BC)
        #         # return(np.sum(np.abs(int_factor * extra * (u+1j*v))))
        #     plt.ylim(-5,5)
        #     plt.show()
        #     optimized = scipy.optimize.minimize(cost, x0 = [np.real(C0_guess), np.imag(C0_guess)])
        #     A = optimized.x
        #     out[i] = ((A[0]+1j*A[1])*exp_negphi_double[i]+int_factor_over_I_double[i])[:len_phi]

        return(out*f_eff_scaling, 0)

    elif integral_mode == 'fft':
        p_fft = np.fft.fft(p_eff, axis = 1)
        f_fft = np.fft.fft(f_eff, axis = 1)
        diff_matrix = fft_dphi_op(len_phi)
        conv_matrix = fft_conv_op_batch(p_fft)
        inv_dxpp = np.linalg.inv(diff_matrix[None, :, :] + conv_matrix)
        sln_fft = (inv_dxpp@f_fft[:,:,None])[:,:,0]
        sln = np.fft.ifft(sln_fft, axis = 1)
        print('sln.shape',sln.shape)
        return(sln, 0)

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

        print('int_p')
        print(int_p)
        print('f_looped')
        print(f_looped)
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
        print('exp_neg2pi')
        print(exp_neg2pi)
        print('integration_factor')
        print(integration_factor)
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
        return(out*f_eff_scaling, 0)


# a dphi operator acting on the fft of
# a content along axis=1
def fft_dphi_op(len_phi):
    fft_freq = np.fft.fftfreq(len_phi)*len_phi
    return(np.identity(len_phi) * 1j * fft_freq)

# A convolution operator acting convolving
# a fft of len_phi with source.
# see paper not for correct format of this matrix
# Sadly jit doesn't support np.fft.
# source has shape [n_chi, n_phi]
# returns [n_chi, n_phi_row, n_phi_col]
def fft_conv_op_batch(source):
    len_chi = source.shape[0]
    len_phi = source.shape[1]
    out = np.zeros((len_chi, len_phi, len_phi), dtype = np.complex128)
    # How much to roll source
    roll = (np.fft.fftfreq(len_phi)*len_phi).astype(np.int64)
    # Where does 0 elem start/end
    split = (len_phi+1)//2
    split_b = np.roll(np.arange(len_phi%2, len_phi+len_phi%2, dtype = np.int64),split)
    for i in range(len_phi):
        index = roll[i]
        out[:, i, :] = np.roll(source, index, axis = 1)
        split_start = min(split, split_b[i])
        split_end = max(split, split_b[i])
        out[:, i,split_start:split_end] = 0
    return(np.transpose(out, (0,2,1))/len_phi)

# For solving the periodic linear 1st order ODE (coeff + coeff_dp*dphi + coeff_dc*dchi) y = f(phi, chi)
# using integral factor. Not numba accelerated since scipy is not supported by numba.
# -- Input --
# coeffs are constants or contents. f is a content (see ChiPhiFunc's description).
# Contant mode and offset are for evaluating the constant component equation
# dphi y0 = f0.
# -- Output --
# y is a ChiPhiFuncGrid's content
def solve_integration_factor_chi(coeff, coeff_dp, coeff_dc, f, \
    integral_mode=non_periodic_integral_mode, asymptotic_order=asymptotic_order):

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
            asymptotic_order=asymptotic_order)
    )

# For solving the periodic linear 1st order ODE (dphi+iota*dchi) y = f(phi, chi)
# using integral factor. Not numba accelerated since scipy is not supported by numba.
# -- Input --
# iota is a constant and f is a ChiPhiFuncGrid.
# Contant mode and offset are for evaluating the constant component equation
# dphi y0 = f0.
# -- Output --
# y is a ChiPhiFuncGrid's content
def solve_dphi_iota_dchi(iota, f, \
    integral_mode=non_periodic_integral_mode, asymptotic_order=asymptotic_order):
    return(
        solve_integration_factor_chi(0, 1, iota, f, \
            integral_mode=integral_mode,
            asymptotic_order=asymptotic_order)
        )

# Whenever the real value of a phi function crosses
# multiples of 5, move the whole sequence to have that
# element be 0. Used in piecewise ODE solve.
# Input should be a content matrix.
# Outputs:
# segment_start[n_seg, 2]
# segment_end[n_seg, 2]
# Segments always have 1 overlapping element to make sure BC works properly.
@njit(Tuple((int64[:,:], int64[:,:]))(complex128[:,:]))
def find_segments(content_in):
    list_segment_start = []
    list_segment_end = []
    len_chi = content_in.shape[0]
    len_phi = content_in.shape[1]
    sign = np.sign(content_in)
    sign = np.sign(sign+0.5)
    # abs: behavior of // is different for negative number
    # +1: otherwise sign transition does not count as change pof "//5"
    segment_height = 2
    real_amplitude = np.abs(np.real(content_in))
    cond = ((real_amplitude - real_amplitude%segment_height)/segment_height+1)*sign # was 5
    prev_i = -1

    for i in prange(len_chi):
        for j in prange(len_phi-1):
            if cond[i,j] != cond[i,j+1]:
                # adds in beginning and end of rows
                if prev_i != i:
                    list_segment_start.append([i,0])
                    if prev_i >= 0:
                        list_segment_end.append([prev_i,len_phi-1])
                    prev_i = i
                list_segment_start.append([i,j])
                list_segment_end.append([i,j+1])
        # close the last, unclosed segment
    list_segment_end.append([list_segment_start[-1][0], len_phi-1])
    return(np.array(list_segment_start), np.array(list_segment_end))
