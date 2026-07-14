import jax.numpy as jnp
from jax import vmap, tree_util
# import lineax as lx
# from jax import jit, vmap, tree_util
# from functools import partial # for JAX jit with static params

from matplotlib import pyplot as plt

# Configurations
from .config import *
# from .math_utilities import fourier_interpolation
from interpax import interp1d
''' Debug options and loading configs '''

# Loading configurations
if double_precision:
    from jax import config as jax_config
    jax_config.update("jax_enable_x64", True)

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

def phi_avg(in_quant):
    '''
    A type-insensitive phi-averaging function that:
    - Averages along phi and output a ChiPhiFunc if the input is a ChiPhiFunc.
    - Does nothing if the input is a scalar.

    Uses duck typing (not isinstance(in_quant, ChiPhiFunc)) so this also
    works for ChiPhiFuncPadded, which chiphifunc.py cannot import directly
    (chiphifunc_padded.py imports from here, so the reverse would be
    circular) -- see chiphifunc_padded.py's module docstring.
    '''
    if hasattr(in_quant, 'content') and hasattr(in_quant, 'is_special'):
        # special ChiPhiFunc's
        if in_quant.is_special():
            return(in_quant)
        new_content = jnp.array([
            jnp.mean(in_quant.content,axis=1)
        ]).T
        return(type(in_quant)(new_content,in_quant.nfp))
    if not jnp.array(in_quant).ndim==0:
        return(ChiPhiFuncSpecial(-5))
    return(in_quant)

def display_content_shared(content, nfp, trig_mode, colormap_mode): 
    if content.shape[1]==1:
        content = content*jnp.ones((1,100))
    len_phi = content.shape[1]
    phis = jnp.linspace(0,2*jnp.pi*(1-1/len_phi)/nfp, len_phi)
    if trig_mode:
        fourier = ChiPhiFunc(content, nfp).exp_to_trig()
        if len(fourier.content)%2==0:
            ax1 = plt.subplot(121)
            ax1.set_title('cos, nfp='+str(nfp))
            ax2 = plt.subplot(122)
            ax2.set_title('sin, nfp='+str(nfp))
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
            ax1.set_title('cos, nfp='+str(nfp))
            ax2 = plt.subplot(132)
            ax2.set_title('constant, nfp='+str(nfp))
            ax3 = plt.subplot(133)
            ax3.set_title('sin, nfp='+str(nfp))
            if colormap_mode and len(fourier.content) != 1:
                modesin = jnp.linspace(2, len(fourier.content)-1, len(fourier.content)//2)
                modecos = jnp.linspace(len(fourier.content)-1, 2, len(fourier.content)//2)
                phi = jnp.linspace(0, 2*jnp.pi*(1-1/len_phi)/nfp, len_phi)
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
        ax1.set_title('Real, nfp='+str(nfp))
        ax2 = plt.subplot(122)
        ax2.set_title('Imaginary, nfp='+str(nfp))

        if colormap_mode:
            mode = jnp.linspace(-len(content)+1, len(content)-1, len(content))
            phi = jnp.linspace(0, 2*jnp.pi*(1-1/len_phi)/nfp, len_phi)
            ax1.pcolormesh(phi, mode, jnp.real(content))
            ax2.pcolormesh(phi, mode, jnp.imag(content))
        else:
            ax1.plot(phis, jnp.real(content).T)
            ax2.plot(phis, jnp.imag(content).T)
    # if fname is None:
    plt.show()
    # else:
    #     plt.savefig(fname)


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
            => ChiPhiFunc(self.nfp)
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
                    return(ChiPhiFuncSpecial(self.nfp))
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
        elif isinstance(other, ChiPhiEpsFunc):
            return(other+self)
        else:
            # If other isn't array-convertible at all (e.g. a
            # ChiPhiFuncPadded, which this class doesn't know about by
            # design -- see chiphifunc_padded.py), return NotImplemented
            # rather than crashing inside jnp.array(): this lets Python's
            # standard operator protocol retry via other.__radd__(self),
            # which does know how to handle a genuinely-special self (see
            # ChiPhiFuncPadded._coerce_operand). Array-convertible but
            # wrong-shaped values still get the existing error sentinel.
            try:
                other_is_scalar = jnp.array(other).ndim == 0
            except TypeError:
                return NotImplemented
            if not other_is_scalar:
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
            => ChiPhiFunc(self.nfp)
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
                    return(ChiPhiFuncSpecial(self.nfp))
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
            full = batch_convolve(a, b)
            return(ChiPhiFunc(full, self.nfp))
        elif isinstance(other, ChiPhiEpsFunc):
            return(other*self)
        else:
            try:
                other_is_scalar = jnp.array(other).ndim == 0
            except TypeError:
                return NotImplemented
            if not other_is_scalar:
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
                    return(ChiPhiFuncSpecial(self.nfp))
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
            try:
                other_is_scalar = jnp.array(other).ndim == 0
            except TypeError:
                return NotImplemented
            if not other_is_scalar:
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
                try:
                    other_is_scalar = jnp.array(other).ndim == 0
                except TypeError:
                    return NotImplemented
                if not other_is_scalar:
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
    # # @partial(jit, static_argnums=(1,))
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
        len_phi = self.content.shape[1]
        # if double_precision:
        #     phis = jnp.linspace(0, 2*jnp.pi*(1-1/len_phi), len_phi, dtype=jnp.complex128)
        # else:
        #     phis = jnp.linspace(0, 2*jnp.pi*(1-1/len_phi), len_phi, dtype=jnp.complex64)
        phis = jnp.linspace(0, 2*jnp.pi*(1-1/len_phi), len_phi)
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
            return(ChiPhiFuncSpecial(-16))

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
    # @partial(jit, static_argnums=(1,2))
    def get_max(self, len_chi:int=100, len_phi:int=100):
        '''
        Getting the max absolute value of the content

        Outputs:

        A real scalar.
        '''
        if self.nfp==0:
            return(0)
        elif self.nfp<0:
            return(jnp.inf)
        chis = jnp.arange(len_chi)/len_chi*jnp.pi*2
        phis = jnp.arange(len_phi)/len_phi*jnp.pi*2
        return(jnp.max(jnp.abs(self.eval(chis[:, None], phis[None, :]))))

    def get_l2(self):
        '''
        Getting the L2 norm of the underlying function over the full torus
        (chi in [0, 2*pi), phi in [0, 2*pi)), computed via Parseval's
        theorem directly from content rather than by evaluating on a grid.

        content's chi axis (axis=0) already holds exact chi-Fourier
        coefficients c_m(phi), so Parseval gives
        integral_0^2pi |f|^2 dchi = 2*pi * sum_m |c_m(phi)|^2 exactly.
        content's phi axis (axis=1) is a uniform grid over one field
        period [0, 2*pi/nfp); since it's assumed to exactly reproduce the
        underlying periodic function via trigonometric interpolation (the
        same assumption dphi's FFT-based derivative relies on), its plain
        grid average equals the continuous phi average, and nfp copies of
        one field period tile the full torus, giving
        integral_0^2pi (...) dphi = (2*pi)^2 * mean_phi(sum_m |c_m(phi)|^2)
        (the nfp cancels: nfp periods of length 2*pi/nfp).

        Output: -----

        A real scalar, sqrt(integral_0^2pi integral_0^2pi |f|^2 dchi dphi).
        '''
        if self.nfp==0:
            return(0.)
        elif self.nfp<0:
            return(jnp.inf)
        return get_l2_shared(self.content)

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
        # Broadcasting phi, chi to the same shape
        phi += jnp.zeros_like(chi)
        chi += jnp.zeros_like(phi)

        if self.nfp == 0:
            return(0)
        if self.nfp < 0:
            return(jnp.nan)

        len_chi = self.content.shape[0]
        len_phi = self.content.shape[1]

        # Create 'x' for interpolation. 'x' is wrapped for periodicity.
        phi_grid = jnp.linspace(0, 2 * jnp.pi / self.nfp, len_phi, endpoint=False)
        # Interpolating in the phi direction
        # Has shape: 
        # shape, n_harmonics
        # interp = fourier_interpolation(
        #     y_data=self.content,
        #     x_interp=phi.flatten(),
        #     nfp=self.nfp
        # ).T
        interp = interp1d(
            phi.flatten(),
            phi_grid, 
            self.content.T, 
            method=interp1d_method,
            period=2*jnp.pi/self.nfp
        ).reshape(list(phi.shape)+[-1])

        # Calculating the chi dependence
        chi_harm_m = (jnp.arange(len_chi) * 2 - len_chi + 1)
        # Has shape: shape, n_harmonics
        phase = jnp.exp(1j * (chi.flatten()[:, None] * chi_harm_m).reshape(list(phi.shape)+[-1]))
        # Has shape: n_harmonics, (chi and phi broadcast together).shape 
        result = jnp.sum(interp * phase, axis=-1)
        return(result)

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
        display_content_shared(content, self.nfp, trig_mode=trig_mode, colormap_mode=colormap_mode)

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



# Contains static argument.
# Does not create new instances of the class.
# This speeds up compole.
ChiPhiFuncSpecial_originals=[]
for i in range (18):
    error_code=-i
    ChiPhiFuncSpecial_originals.append(ChiPhiFunc(jnp.nan, error_code))
    
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
    if error_code>0 or error_code<=-len(ChiPhiFuncSpecial_originals):
        return(ChiPhiFuncSpecial_originals[2])
        # return(ChiPhiFunc(jnp.nan, -2))

    return(ChiPhiFuncSpecial_originals[-error_code])
    # return(ChiPhiFunc(jnp.nan, error_code)) # , is_special=True))


# ---------------------------------------------------------------------------
# Bottom-of-file imports — placed here rather than at the top to break the
# mutual dependency between chiphifunc and math_utilities:
#   math_utilities imports ChiPhiFunc (this module) at its top;
#   chiphifunc methods use the pure helpers below at call time (not
#   at class-definition time), so fetching them after the class is defined
#   is safe.  Python returns the partially-initialized math_utilities module
#   from sys.modules when the circular import resolves, and by that point
#   all helpers below are already defined there.
from .math_utilities import (
    dchi_op,
    trig_to_exp_op,
    exp_to_trig_op,
    dphi_op_pseudospectral,
    fft_filter,
    fft_pad,
    batch_convolve,
    centered_resize_content,
    jit_fftfreq_int,
    wrap_grid_content_jit,
    get_l2_shared,
    max_log10,
    roll_axis_01,
    batch_roll_axis_01,
)
# Existing cycle break: ChiPhiEpsFunc.__add__/__mul__ dispatch back to ChiPhiFunc.
from .chiphiepsfunc import ChiPhiEpsFunc