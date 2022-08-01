import numpy as np
import timeit
import scipy.signal
from matplotlib import pyplot as plt

# Numba doesn't support scipy methods. Therefore, scipy integrals are sped up
# with joblib.
import scipy.integrate
from joblib import Parallel, delayed
simpson_mode = True
n_jobs = 4
backend = 'threading' # scipy.integrate is based on a compiled

from numba import jit, njit, prange
# import jit value types. Bizzarely using int32 causes typing issues
# in deconvolution.solve_underdet_degen with njit(parallel=True) because
# it seem to insist prange(...shape[0]) is int64, even when int64 is not imported.
from numba import complex128, int64, float64

from functools import lru_cache # import functools for caching
import warnings

import deconvolution

""" Representing functions of chi and phi (ChiPhiFunc subclasses) """
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

""" I. Abstract superclass and singleton """
# ChiPhiFunc is an abstract superclass.
class ChiPhiFunc:
    # Initializer. Fourier_mode==True converts sin, cos coeffs to exponential
    def __init__(self, content=np.nan, fourier_mode=False):
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

    # Multiplication with a ChiPhiFunc of the same type. Abstract method.
    def multiply(a, b, div=False):
        raise NotImplementedError()

    # Math methods -----------------------------------------------------------------------------
    # derivatives. Implemented through dchi_op and dphi_op
    def dchi(self):
        return type(self)(ChiPhiFunc.dchi_op(self.get_shape()[0]) @ self.content)

    def dphi(self):
        return(ChiPhiFuncGrid(np.gradient(self.content, axis=1, edge_order=2)/(np.pi*2/self.get_shape()[1])))
        #return type(self)(self.content @ type(self).dphi_op(self.get_shape()[1]).T)

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
        return type(self)(np.array([self.content[len_chi//2]]))

    # Used in addition with constant. Only even chi series has constant component.
    # Odd chi series + constant results in error.
    def no_const(self):
        # This would trigger an error for most complex,
        # static methods used for evaluation.
        return (self.get_shape()[0]%2==0)


    # Plotting -----------------------------------------------------------------------------
    # Get a 2-argument lambda function for plotting this term
    def get_lambda(self):
        raise NotImplementedError()

    # Plot a period in both chi and phi
    def display(self, complex = False):
        # This would trigger an error for most complex,
        # static methods used for evaluation.
        chi = np.linspace(-np.pi,np.pi,100)
        phi = np.linspace(-np.pi,np.pi,100)
        f = self.get_lambda()
        plt.pcolormesh(chi, phi, np.real(f(chi, phi.reshape(-1,1))))
        plt.title('ChiPhiFunc, real component')
        plt.xlabel('chi')
        plt.ylabel('phi')
        plt.colorbar()
        plt.show()
        if complex:
            plt.pcolormesh(chi, phi, np.imag(f(chi, phi.reshape(-1,1))))
            plt.title('ChiPhiFunc, imaginary component')
            plt.xlabel('chi')
            plt.ylabel('phi')
            plt.colorbar()
            plt.show()

    # Utility -----------------------------------------------------------------------------
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

    # Generate chi differential operator diff_matrix. diff_matrix@f.content = dchi(f).content
    # invert = True generates anti-derivative operator. Cached for each new Chi length.
    # -- Input: len_chi: length of Chi series.
    # -- Output: 2d matrix.
    @lru_cache(maxsize=100)
    @njit
    def dchi_op(len_chi, invert = False):
        ind_chi = len_chi-1
        mode_chi = np.linspace(-ind_chi, ind_chi, len_chi)
        if invert:
            return(np.diag(-1j/mode_chi))
        return np.diag(1j*mode_chi)

    # An accelerated sum that aligns the center-point of 2-d arrays and zero-broadcasts the edges.
    # Input arrays must both have even/odd cols/rows
    # (such as (3,2), (13,6))
    # Copies arguments
    # -- Input: 2 2d arrays.
    # -- Output: 2d array.
    @njit(complex128[:,:](complex128[:,:], complex128[:,:], int64))
    def add_jit(a, b, sign):
        shape = (max(a.shape[0], b.shape[0]),max(a.shape[1],b.shape[1]))
        out = np.zeros(shape, dtype=np.complex128)
        a_pad_row = (shape[0] - a.shape[0])//2
        a_pad_col = (shape[1] - a.shape[1])//2
        b_pad_row = (shape[0] - b.shape[0])//2
        b_pad_col = (shape[1] - b.shape[1])//2
        out[a_pad_row:shape[0]-a_pad_row,a_pad_col:shape[1]-a_pad_col] += a
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
    def __new__(cls, content = np.nan):
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

    def dchi(self):
        return(self)

    def dphi(self):
        return(self)

# two representations for phi dependence are implemented: Fourier and grid.
""" II. Grid representation for phi dependence """

# Used for wrapping grid content. Defined outside the ChiPhiFuncGrid class so that
# it can be used in @njit compiled methods.
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

    # Used in operator *. First stretch the phi axis to match grid locations,
    # Then do pointwise product.
    # -- Input: self and another ChiPhiFuncFourier
    # -- Output: a new ChiPhiFuncFourier
    def multiply(self, other, div = False):
        a, b = self.stretch_phi_to_match(other)
        if div:
            b = 1.0/b
        # Now both are stretch to be dim (n_a, n_phi), (n_b, n_phi).
        # Transpose and 1d convolve all constituents.
        return(ChiPhiFuncGrid(ChiPhiFuncGrid.mul_grid_jit(a, b)))

    # Wrapper for mul_grid_jit. Handles int power.
    # -- Input: self and an int
    # -- Output: a new ChiPhiFuncFourier
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
    # -- Input: self and another ChiPhiFuncFourier
    # -- Output: a new ChiPhiFuncFourier
    def add_ChiPhiFunc(self, other, sign=1):
        a,b = self.stretch_phi_to_match(other)
        # Now that grid points are matched by stretch_phi, we can invoke add_jit()
        # To add matching rows(chi coeffs) and grid points.
        return ChiPhiFuncGrid(ChiPhiFunc.add_jit(a,b,sign))

    # Used in operators, wrapper for stretch_phi. Match self's shape to another ChiPhiFuncGrid.
    # returns 2 contents.
    def stretch_phi_to_match(self, other):
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
                print(other)
                raise ValueError('A constant + with an even series. ')
        stretched_constant = np.full((1, self.get_shape()[1]), const, dtype=np.complex128)
        return ChiPhiFuncGrid(ChiPhiFunc.add_jit(self.content, stretched_constant,sign))


    # Properties ---------------------------------------------------------------
    # Returns the value when phi=0. Copies.
    def get_phi_zero(self):
        return ChiPhiFuncGrid(self.content[:,0])

    # Math ---------------------------------------------------------------
    # Used for solvability condition. phi-integrate a ChiPhiFuncGrid over 0 to
    # 2pi or 0 to a given phi.
    # periodic=False evaluates integral from 0 to phi FOR EACH GRID POINT and
    # creates a ChiPhiFuncGrid with phi dependence.
    # periodic=True evaluates integral from 0 to 2pi and creates a ChiPhiFuncGrid
    # with NO phi dependence.
    def integrate_phi(self, periodic):
        # number of phi grids
        len_chi = self.get_shape()[0]
        len_phi = self.get_shape()[1]
        phis = np.linspace(0,2*np.pi*(1-1/len_phi), len_phi, dtype=np.complex128)
        if periodic:
            # Integrate self.content along axis 1 (rows, phi axis)
            # with spacing (dx) of 2pi/#grid

            # The result of the integral is an 1d array of chi coeffs.
            # This integrates the full period, and needs to be wrapped.
            # the periodic=False option does not integrate the full period and
            # does not wrap.
            new_content = scipy.integrate.simpson(wrap_grid_content_jit(self.content), dx=2*np.pi/len_phi, axis=1)

            # if the result of the integral is a constant, return that constant.
            if len(new_content)==1:
                return(new_content[0])

            # Otherwise, add an axis, transpose and return a ChiPhiFuncGrid with
            # no chi dependence
            else:
                return(ChiPhiFuncGrid(np.array([new_content]).T))
        else:
            if simpson_mode:
                integrate = lambda i_phi : scipy.integrate.simpson(self.content[:,:i_phi+1], dx=2*np.pi/len_phi)
                out_list = Parallel(n_jobs=n_jobs, backend=backend)(
                    delayed(integrate)(i_phi) for i_phi in range(len_phi)
                )
                return(ChiPhiFuncGrid(np.array(out_list).T))

            else:
                @jit(complex128[:,:](complex128[:], complex128[:,:], int64, int64), parallel = True)
                def phi_integrate_jit(phis, content, len_chi, len_phi):
                    out = np.zeros_like(content, dtype=np.complex128)
                    for i_phi in prange(len_phi):
                        out[:, i_phi] = scipy.integrate.simpson(content[:,:i_phi+1], dx=2*np.pi/len_phi)
                    return(out)
                return(ChiPhiFuncGrid(phi_integrate_jit(phis, self.content, len_chi, len_phi)))

    # Used to calculate e**(ChiPhiFuncGrid). Only support ChiPhiFuncGrid with no
    # chi dependence
    def exp(self):
        if self.get_shape()[0]!=1:
            raise ValueError('exp only supports ChiPhiFuncGrid with no chi dependence!')
        return(ChiPhiFuncGrid(np.exp(self.content)))

    # SOLVING - CRITICAL! ------------------------------------------------

    # Solve under-determined degenerate system equivalent
    # v_source_A(chi, phi) * va_{n+1}(chi, phi) = v_rhs{n}(chi, phi).
    # wrapper for jit-enabled method batch_underdetermined_degen_jit.
    # -- Input --
    # v_source_A, v_rhs: ChiPhiFuncGrid,
    # rank_rhs: int, 'n' in the equation above
    # i_free: int, index of free var in va,
    # vai: ChiPhiFuncGrid with a single row, value of free component.
    def solve_underdet_degen(v_source_A, v_rhs, rank_rhs, i_free, vai):

        if isinstance(v_source_A, ChiPhiFuncNull) \
        or isinstance(v_rhs, ChiPhiFuncNull) \
        or isinstance(vai, ChiPhiFuncNull):
            raise ValueError('One or more input is awaiting cond: '+\
                            'v_source_A: ' + str(type(v_source_A)) +\
                            '; v_rhs: ' + str(type(v_rhs)) +\
                            '; vai: ' + str(type(vai)))

        v_source_A_content = v_source_A.content
        v_rhs_content = v_rhs.content
        vai_content = vai.content

        # checking input types
        if type(v_source_A) is not ChiPhiFuncGrid\
            or type(v_rhs) is not ChiPhiFuncGrid\
            or type(vai) is not ChiPhiFuncGrid:
            raise TypeError('ChiPhiFuncGrid.solve_underdetermined: '\
                            'v_source_A, v_rhs, vai should all be ChiPhiFuncGrid.')

        if v_source_A.get_shape()[0] + rank_rhs != v_rhs.get_shape()[0]:
            print("v_source_A shape:", v_source_A.get_shape())
            print("v_rhs shape:", v_rhs.get_shape())
            print("rank_rhs:", rank_rhs)
            warnings.warn('Warning: A, v_rhs and rank_rhs doesn\'t satisfy mode'
                          ' number requirements. Zero-padding rhs chi components.')
            # This creates a padded content for v_rhs in case some components are zero. However, we still need to put in check for
            # LHS and RHS's even and oddness.
            v_rhs_content = ChiPhiFunc.add_jit(
                v_rhs_content,
                np.zeros((v_source_A.get_shape()[0]+rank_rhs, v_rhs.get_shape()[1]), dtype=np.complex128)
            )

        va_content = deconvolution.batch_underdetermined_degen_jit(
            v_source_A_content,
            v_rhs_content,
            rank_rhs,
            i_free,
            vai_content
        )
        return(ChiPhiFuncGrid(va_content))

    # Solve exactly determined degenerate system equivalent
    # v_source_A(chi, phi) * va_{n}(chi, phi) = v_rhs{n}(chi, phi).
    # wrapper for jit-enabled method deconvolution.batch_degen_jit.
    def solve_degen(v_source_A, v_rhs, rank_rhs):

        if isinstance(v_source_A, ChiPhiFuncNull) or isinstance(v_rhs, ChiPhiFuncNull):
            raise ValueError('One or more input is awaiting cond: '+\
                            'v_source_A: ' + str(type(v_source_A)) +\
                            '; v_rhs: ' + str(type(v_rhs)))

        v_source_A_content = v_source_A.content
        v_rhs_content = v_rhs.content

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


        va_content = deconvolution.batch_degen_jit(
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
        unstreched_x = np.linspace(0,2*np.pi, len_phi+1)
        # wrapping
        content_looped = wrap_grid_content_jit(self.content)

        # Chi harmonics[lambda a=i: np.interp(b, unstreched_x, content_looped[a]) for i in range(len_chi)][lambda a=i: np.interp(b, unstreched_x, content_looped[a]) for i in range(len_chi)][lambda a=i: np.interp(b, unstreched_x, content_looped[a]) for i in range(len_chi)][lambda a=i: np.interp(b, unstreched_x, content_looped[a]) for i in range(len_chi)]
        ind_chi = len_chi-1
        mode_chi = np.linspace(-ind_chi, ind_chi, len_chi)

        # The outer dot product is summing along axis 0.
        return(np.vectorize(
            lambda chi, phi : sum(
                [np.e**(1j*(chi)*mode_chi[i]) * np.interp((phi)%(2*np.pi), unstreched_x, content_looped[i]) for i in range(len_chi)]
            )
        ))


    # Utilities --------------------------------------------------------
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

    # Generate phi differential operator diff_matrixT. content @ diff_matrixT.T = dchi(f).
    # (.T actually not needed since this the operator is diagonal)
    # the simplestcentral difference operator is not invertible.
    # we add an all-1 column to it to constrain average and make it invertible.
    # -- Input: a phi grid number int
    # -- Output: a 2d matrix
    @lru_cache(maxsize=None)
    def dphi_op(len_phi, invert = False):
        vec = np.zeros(len_phi, dtype=np.complex128)
        # circulant circulates column, not row.
        vec[len_phi-1] = 1
        vec[1] = -1
        # A matrix with all 1's on the first row and 0 on the rest of the rows
        all_ones = np.block([[np.ones((1,len_phi), dtype=np.complex128)], [np.zeros((len_phi-1, len_phi), dtype=np.complex128)]])
        diff_matrix = scipy.linalg.circulant(vec)*len_phi/4/np.pi
        if invert:
            return(np.linalg.inv(all_ones + diff_matrix)@(np.identity(len_phi, dtype=np.complex128) + all_ones))
        return(np.linalg.inv(all_ones + np.identity(len_phi, dtype=np.complex128))
            @(all_ones + diff_matrix))

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
        util_matrix_chi = ChiPhiFunc.fourier_to_exp_op(self.get_shape()[0])
        # Apply the conversion matrix on chi axis
        self.content = util_matrix_chi @ self.content




""" III. Fourier representation for phi dependence """
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
# class ChiPhiFuncFourier(ChiPhiFunc):
#     def __init__(self, content, fourier_mode = False):
#         super().__init__(content, fourier_mode)
#         if content.shape[1]%2 == 0:
#             raise ValueError('Phi coefficients should be a full fourier series. Even phi_dim detected.')
#     # Operator handlers -------------------------------------------------
#     # NOTE: will not be passed items awaiting for conditions.
#
#     # Addition of 2 ChiPhiFuncFourier's.
#     # Wrapper for numba method.
#     # -- Input: self and another ChiPhiFuncFourier
#     # -- Output: a new ChiPhiFuncFourier
#     def add_ChiPhiFunc(self, other):
#         return ChiPhiFuncFourier(ChiPhiFunc.add_jit(self.content, other.content))
#
#     # Addition of a constant with a ChiPhiFuncFourier.
#     # Wrapper for numba method.
#     def add_const(self, other):
#         return ChiPhiFuncFourier(ChiPhiFunc.add_jit(self.content, np.complex128([[other]])))
#
#     # Handles pointwise multiplication (* operator)
#     # Convolve2d is already compiled. No need to jit.
#     # -- Input: self and other
#     # -- Output: a new ChiPhiFuncFourier
#     def multiply(self, other, div=False):
#         if div:
#             raise NotImplementedError()
#         return(ChiPhiFuncFourier(scipy.signal.convolve2d(self.content, other.content)))
#
#     # Calculates an integer power of a ChiPhiFuncFourier
#     # Convolve2d is already compiled. No need to jit.
#     # Also here we assume all powers are to fairly low orders (2)
#     # -- Input: self and power (int)
#     # -- Output: self and other
#     def pow(self, int_pow):
#         new_content = self.content.copy()
#         for n in range(int_pow-1):
#             new_content = scipy.signal.convolve2d(new_content, new_content)
#         return(ChiPhiFuncFourier(new_content))
#
#     # Get a 2-argument lamnda function for plotting this term
#     def get_lambda(self):
#         len_chi = self.get_shape()[0]
#         len_phi = self.get_shape()[1]
#
#         if len_phi%2!=1:
#             raise ValueError('coeffs_chi must have an odd number of components on phi axis')
#
#         ind_phi = int((len_phi-1)/2)
#         mode_phi = np.linspace(-ind_phi, ind_phi, len_phi)
#
#         ind_chi = len_chi-1
#         mode_chi = np.linspace(-ind_chi, ind_chi, len_chi).reshape(-1,1)
#         # The outer dot product is summing along axis 0.
#         # The inner @ (equivalent to dot product) is summing along axis 1.
#         return(np.vectorize(lambda chi, phi : np.dot(self.content@(np.e**(1j*phi*mode_phi)), (np.e**(1j*(-chi)*mode_chi)))))
#
#
#     # Utilities ---------------------------------------------------
#
#     # Converting fourier coefficients into exponential coeffs used in
#     # ChiPhiFunc's internal representation. Only used during super().__init__
#     # Does not copy.
#     def trig_to_exp(self):
#         util_matrix_chi = ChiPhiFunc.fourier_to_exp_op(self.get_shape()[0])
#         util_matrix_phi = ChiPhiFunc.fourier_to_exp_op(self.get_shape()[1])
#         # Apply the conversion matrix on phi axis
#         # (two T's because np.matmul can't choose axis)
# #         self.content = (util_matrix_phi @ self.content.T).T
#         self.content = self.content @ util_matrix_phi.T
#         # Apply the conversion matrix on chi axis
#         self.content = util_matrix_chi @ self.content
#
#     # Generate phi differential operator diff_matrixT. content @ diff_matrixT.T = dchi(f).
#     # (.T actually not needed since this the operator is diagonal)
#     # -- Input: len_phi: length of phi series.
#     # -- Output: 2d matrix.
#     @lru_cache(maxsize=None)
#     @njit
#     def dphi_op(len_phi, invert = False):
#         ind_phi = int((len_phi-1)/2)
#         mode_phi = np.linspace(-ind_phi, ind_phi, len_phi)
#         if invert:
#             return(np.diag(-1j/mode_phi))
#         return(np.diag(1j*mode_phi))
