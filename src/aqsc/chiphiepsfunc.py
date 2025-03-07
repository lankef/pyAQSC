import jax.numpy as jnp
import numpy as np # Used in saving
# from functools import partial
# from jax import jit, tree_util
from jax import tree_util, jit
from functools import partial # for JAX jit with static params

from .chiphifunc import *
from .chiphifunc import ChiPhiFunc

'''ChiPhiEpsFunc'''
# A container for lists of ChiPhiFuncs. Primarily used to handle array index out of bound
# error in Maxima-translated codes. Produces a ChiPhiFuncNull when index is out of bound.
# Initialization:
# ChiPhiEpsFunc([X0, X1, X2, ... Xn], n, False) or
# ChiPhiEpsFunc([X0, X2, ... Xn], n, True)
class ChiPhiEpsFunc:
    def __init__(self, list:list, nfp:int, square_eps_series:bool, check_consistency:bool=False): # nfp-dependent!!
        self.chiphifunc_list = list
        self.nfp = nfp
        self.square_eps_series = square_eps_series
        if check_consistency:
            self.chiphifunc_list = self.check_nfp_consistency()

    ''' For JAX use '''
    def _tree_flatten(self):
        children = (self.chiphifunc_list,)  # arrays / dynamic values
        aux_data = {
            'nfp': self.nfp,
            'square_eps_series': self.square_eps_series
        }  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    # This consistency check cannot be jitted.
    # Because otherwise every element will have a consistency check if
    # on it.
    def check_nfp_consistency(self):
        '''
        Check the nfp of all constituents in self.chiphifunc_list
        '''
        new_chiphifunc_list = self.chiphifunc_list.copy()
        for i in range(len(new_chiphifunc_list)):
            item = new_chiphifunc_list[i]
            # Do not modify a CHiPhiFUnc if it has consistent nfp or is an error
            if isinstance(item, ChiPhiFunc) and (item.nfp==self.nfp or item.nfp<=0):
                # if jnp.all(item.content==0):
                #     self.chiphifunc_list[i] = ChiPhiFuncSpecial(0)
                #     continue
                continue
            # Do not modify scalars
            if jnp.array(item).ndim==0:
                # Force known zeros to be special zeros.
                # if item==0:
                #     self.chiphifunc_list[i] = ChiPhiFuncSpecial(0)
                #     continue
                continue
            new_chiphifunc_list[i] = ChiPhiFuncSpecial(-14)
        return(new_chiphifunc_list)

    ''' Operator overload '''

    def __neg__(self):
        '''
        Overloads the - operator. 
        '''
        list_new = []
        for item in self.chiphifunc_list:
            list_new.append(-item)
        return(ChiPhiEpsFunc(list_new, self.nfp, self.square_eps_series))

    def __add__(self, other):
        '''
        Overloads the self + other operator. 
        If other is a ChiPhiEpsFunc, add each elements individually. 
        Otherwise, add to the 0th order only. 
        ChiPhiEpsFunc produced by this may no longer satisfy regularity!
        '''
        if isinstance(other, ChiPhiEpsFunc):
            if self.square_eps_series == other.square_eps_series:
                # n is len-1
                n_max = max(self.get_order(), other.get_order())
                list_new = []
                for i in range(n_max+1):
                    list_new.append(self[i] + other[i])
                return(ChiPhiEpsFunc(list_new, self.nfp, self.square_eps_series))
            elif self.square_eps_series:
                return(self.remove_square() + other)
            elif other.square_eps_series:
                return(self + other.remove_square())
        else:
            leading = self[0]
            leading_new = leading + other
            list_n_geq_1 = self.chiphifunc_list[1:]
            return(ChiPhiEpsFunc([leading_new]+list_n_geq_1, self.nfp, self.square_eps_series))
        
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
        Overloads the self * other operator. Generates a new power series. May or may
        not satisfy regularity.
        '''
        if isinstance(other, ChiPhiEpsFunc):
            a = self.remove_square()
            b = other.remove_square()
            len_a = len(a.chiphifunc_list)
            len_b = len(b.chiphifunc_list)
            list_new = [ChiPhiFuncSpecial(0)] * (len_a + len_b - 1)
            for i in range(len_a):
                for j in range(len_b):
                    list_new[i + j] += a[i] * b[j]
            return(ChiPhiEpsFunc(list_new, self.nfp, False))
        else: 
            list_new = []
            for item in self.chiphifunc_list:
                list_new.append(item * other)
            return(ChiPhiEpsFunc(list_new, self.nfp, self.square_eps_series))
            
    def __rmul__(self, other):
        ''' Overloads the other * self operator. See __mul__() for details. '''
        return(self*other)

    def __getitem__(self, index):
        '''
        If array index out of bound or <0, then return a special element,
        ChiPhiFuncSpecial(-1). Unlike all other error codes, this special element
        has the special property that
        ChiPhiFuncSpecial(-1) * ChiPhiFuncSpecial(0) = 0.
        '''
        if (index>len(self.chiphifunc_list)-1 or index<0):
            # return(ChiPhiFuncSpecial(-1))
            return(ChiPhiFuncSpecial(0))
        return(self.chiphifunc_list[index])

    ''' List operations '''
    def remove_square(self):
        ''' 
        If self is a series containing only even orders, convert 
        self to a ChiPhiEpsFunc(square_eps_series=False)
        '''
        if self.square_eps_series:
            list_new = []
            for item in self.chiphifunc_list:
                list_new.append(ChiPhiFuncSpecial(0))
                list_new.append(item)
            return(ChiPhiEpsFunc(list_new[1:], self.nfp, False))
        else:
            return(self)

    def append(self, item):
        '''
        Returns a new ChiPhiEpsFunc with a new item appended to the end.
        '''
        if isinstance(item, ChiPhiFunc):
            # Mismatched nfp. If item is special then we still append it directly
            # to preserve the error message
            if item.nfp!=self.nfp and not item.is_special():
                return(ChiPhiEpsFunc(self.chiphifunc_list+[ChiPhiFuncSpecial(-14)], self.nfp, self.square_eps_series))
        elif jnp.array(item).ndim!=0:
            # Jax scalars are 0-d DeviceArrays.
            return(ChiPhiEpsFunc(self.chiphifunc_list+[ChiPhiFuncSpecial(-14)], self.nfp, self.square_eps_series))
        return(ChiPhiEpsFunc(self.chiphifunc_list+[item], self.nfp, self.square_eps_series))

    # @partial(jit, static_argnums=(1,))
    def zero_append(self, n=1):
        '''
        Append one or more zeros to the end of the list.
        For evaluating higher order terms with recursion relation. Sometimes
        expressions used to evaluate a higher order term includes the term itself,
        and requires ChiPhiEpsFunc to provide zero (rather than ChiPhiFuncNull
        from __getitem__ when array index is out of bound)

        Was required before when out of range returns a special ChiPhiFunc,
        rather than 0 to check for logical error. Is now redundant.
        '''
        return(self)
        # zeros = [ChiPhiFuncSpecial(0)]*n
        # # we know the new element has consistent nfp.
        # return(ChiPhiEpsFunc(self.chiphifunc_list+zeros, self.nfp, self.square_eps_series))

    # @partial(jit, static_argnums=(1,))
    def mask(self, n):
        '''
        Produces a new ChiPhiEpsFunc containing up to (including) order n
        When n exceeds the maximum order known, fill with ChiPhiFuncSpecial(0)
        for tracing incorrect logic or formulae.
        Originally the fill is with special ChiPhiFunc, but since JAX cannot
        read traced quantities' content, out-of-bound terms cannot cancel out.
        (n-n)*(out of bound) = out of bound. Since all parsed formulae are checked
        correct, in the JAX implementation we make out-of-bound terms 0 instead.
        '''
        if n == float('inf'):
            return(self)
        n_diff = n-(len(self.chiphifunc_list)-1)
        if n_diff>0:
             return(ChiPhiEpsFunc(
                 #self.chiphifunc_list[:n+1]+[ChiPhiFuncSpecial(-1)]*n_diff,
                 self.chiphifunc_list[:n+1]+[ChiPhiFuncSpecial(0)]*n_diff,
                 self.nfp,
                 self.square_eps_series
             ))
        return(ChiPhiEpsFunc(self.chiphifunc_list[:n+1], self.nfp, self.square_eps_series))

    # Cannot be jitted. The formula involvng len(traced) need to be
    # substituted into other jitted functions 'symbolically'.
    def get_order(self):
        ''' Gets the currently known order of a power series. '''
        return(len(self.chiphifunc_list)-1)

    # Convert to a list of amplitudes
    def get_max_order_by_order(self, len_chi:int=100, len_phi:int=100):
        amp_list = []
        for item in self.chiphifunc_list:
            if isinstance(item, ChiPhiFunc):
                amp_list.append(item.get_max(len_chi=len_chi, len_phi=len_phi))
            else:
                amp_list.append(jnp.abs(item))
        return(jnp.array(amp_list))

    # @partial(jit, static_argnums=(0,))
    def zeros_like(other):
        '''
        Make a ChiPhiFunc with zero elements with the same order
        and nfp as another.
        '''
        return(ChiPhiEpsFunc([ChiPhiFuncSpecial(0)]*(other.get_order()+1), other.nfp, other.square_eps_series))
    
    ''' Evaluation '''
    def deps(self):
        if self.square_eps_series:
            list_to_shift = []
            for i in range(len(self.chiphifunc_list)):
                list_to_shift.append(self.chiphifunc_list[i])
                list_to_shift.append(ChiPhiFuncSpecial(0))
        else:
            list_to_shift = self.chiphifunc_list
        new_chiphifunc_list = []
        for i in range(len(list_to_shift)-1):
            order_i = i+1
            new_chiphifunc_list.append(order_i*list_to_shift[order_i])
        return(ChiPhiEpsFunc(new_chiphifunc_list, self.nfp, False))
    
    def dchi_or_phi(self, chi_mode):
        new_chiphifunc_list = []
        for i in range(len(self.chiphifunc_list)):
            item = self.chiphifunc_list[i]
            if isinstance(item, ChiPhiFunc) and (item.nfp==self.nfp or item.nfp<=0):
                if chi_mode:
                    new_chiphifunc_list.append(item.dchi())
                else:
                    new_chiphifunc_list.append(item.dphi())
            elif jnp.array(item).ndim==0:
                new_chiphifunc_list.append(ChiPhiFuncSpecial(0))
            else:
                new_chiphifunc_list.append(ChiPhiFuncSpecial(-14))
        return(ChiPhiEpsFunc(new_chiphifunc_list, self.nfp, self.square_eps_series))

    def dchi(self):
        return(self.dchi_or_phi(True))

    def dphi(self):
        return(self.dchi_or_phi(False))

    def eval(self, psi, chi, phi, n_max=float('inf')):
        return(self.eval_eps(jnp.sqrt(jnp.abs(psi)), chi, phi, n_max=n_max))

    def eval_eps(self, eps, chi, phi, n_max=float('inf')):
        # Broadcasting
        if self.square_eps_series:
            power_arg = eps**2
        else:
            power_arg = eps
        out = 0
        for n in range(min(len(self.chiphifunc_list), n_max+1)):
            item = self.chiphifunc_list[n]
            if isinstance(item, ChiPhiFunc):
                if item.nfp==0:
                    pass
                elif item.nfp==self.nfp:
                    out += item.eval(chi, phi)*power_arg**n 
                else:
                    return(jnp.nan)
            # isscalar fails to detect jax float (single-element array)
            elif jnp.array(item).ndim==0:
                out += item*power_arg**n 
            else:
                return(jnp.nan)
        return(out)

    ''' Printing '''
    def __str__(self):
        string = '['
        first = True
        # Convert all items into strings to aid readability
        for item in self.chiphifunc_list:
            if first:
                first=False
            else:
                string = string + ', '
            string += str(item)
        string = string + '], nfp=' + str(self.nfp) + ', square_eps_series=' + str(self.square_eps_series)
        return(string)

    ''' Saving and loading '''
    # Converting to a list of ints and tuples. For saving and loading.
    # see recursion_relation.py.
    def to_content_list(self): # not nfp-dependent
        content_list = []
        for i in range(len(self.chiphifunc_list)):
            item = self.chiphifunc_list[i]
            if isinstance(item, ChiPhiFunc):
                if item.nfp==0:
                    content_list.append(0)
                elif item.nfp<0:
                    content_list.append('err'+str(item.nfp))
                else:
                    content_list.append(np.asarray(item.content))
            else:
                content_list.append(item)
        return(content_list)

    # Note that error types will not be saved.
    def from_content_list(content_list, nfp): # nfp-dependent!!
        chiphifunc_list = []
        for item in content_list:
            if isinstance(item, str):
                error_code = int(item[3:])
                chiphifunc_list.append(ChiPhiFuncSpecial(error_code))
            elif jnp.array(item).ndim==0: # JAX scalar
                if item==0:
                    chiphifunc_list.append(ChiPhiFuncSpecial(0))
                else:
                    chiphifunc_list.append(item)
            else:
                chiphifunc_list.append(ChiPhiFunc(item, nfp))
        out_chiphiepsfunc = ChiPhiEpsFunc(chiphifunc_list, nfp, False)
        return(out_chiphiepsfunc)

# For JAX use
tree_util.register_pytree_node(ChiPhiEpsFunc,
                               ChiPhiEpsFunc._tree_flatten,
                               ChiPhiEpsFunc._tree_unflatten)
