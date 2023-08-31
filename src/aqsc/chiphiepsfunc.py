import jax.numpy as jnp
import numpy as np # Used in saving
# from functools import partial
# from jax import jit, tree_util
from jax import tree_util

from .chiphifunc import *

'''ChiPhiEpsFunc'''
# A container for lists of ChiPhiFuncs. Primarily used to handle array index out of bound
# error in Maxima-translated codes. Produces a ChiPhiFuncNull when index is out of bound.
# Initialization:
# ChiPhiEpsFunc([X0, X1, X2, ... Xn])
class ChiPhiEpsFunc:
    def __init__(self, list:list, nfp:int, check_consistency:bool=False): # nfp-dependent!!
        self.chiphifunc_list = list
        self.nfp = nfp
        if check_consistency:
            self.chiphifunc_list = self.check_nfp_consistency()

    ''' For JAX use '''
    def _tree_flatten(self):
        children = (self.chiphifunc_list,)  # arrays / dynamic values
        aux_data = {
            'nfp': self.nfp
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
            if jnp.isscalar(item) or item.ndim==0:
                # Force known zeros to be special zeros.
                # if item==0:
                #     self.chiphifunc_list[i] = ChiPhiFuncSpecial(0)
                #     continue
                continue
            new_chiphifunc_list[i] = ChiPhiFuncSpecial(-14)
        return(new_chiphifunc_list)

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

    def append(self, item):
        '''
        Returns a new ChiPhiEpsFunc with a new item appended to the end.
        '''
        if isinstance(item, ChiPhiFunc):
            # Mismatched nfp. If item is special then we still append it directly
            # to preserve the error message
            if item.nfp!=self.nfp and not item.is_special():
                return(ChiPhiEpsFunc(self.chiphifunc_list+[ChiPhiFuncSpecial(-14)], self.nfp))
        elif not jnp.isscalar(item):
            # Jax scalars are 0-d DeviceArrays.
            if item.ndim!=0:
                return(ChiPhiEpsFunc(self.chiphifunc_list+[ChiPhiFuncSpecial(-14)], self.nfp))
        return(ChiPhiEpsFunc(self.chiphifunc_list+[item], self.nfp))

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
        # return(ChiPhiEpsFunc(self.chiphifunc_list+zeros, self.nfp))

    # @partial(jit, static_argnums=(1,))
    def mask(self, n):
        '''
        Produces a sub-list up to the nth element (order).
        When n exceeds the maximum order known, fill with ChiPhiFuncSpecial(0)
        for tracing incorrect logic or formulae.
        Originally the fill is with special ChiPhiFunc, but since JAX cannot
        read traced quantities' content, out-of-bound terms cannot cancel out.
        (n-n)*(out of bound) = out of bound. Since all parsed formulae are checked
        correct, in the JAX implementation we make out-of-bound terms 0 instead.
        '''
        n_diff = n-(len(self.chiphifunc_list)-1)
        if n_diff>0:
             return(ChiPhiEpsFunc(
                 #self.chiphifunc_list[:n+1]+[ChiPhiFuncSpecial(-1)]*n_diff,
                 self.chiphifunc_list[:n+1]+[ChiPhiFuncSpecial(0)]*n_diff,
                 self.nfp,
                 False
             ))
        return(ChiPhiEpsFunc(self.chiphifunc_list[:n+1], self.nfp))

    # Cannot be jitted. The formula involvng len(traced) need to be
    # substituted into other jitted functions 'symbolically'.
    def get_order(self):
        ''' Gets the currently known order of a power series. '''
        return(len(self.chiphifunc_list)-1)

    # @partial(jit, static_argnums=(0,))
    def zeros_like(other):
        '''
        Make a ChiPhiFunc with zero elements with the same order
        and nfp as another.
        '''
        return(ChiPhiEpsFunc([ChiPhiFuncSpecial(0)]*(other.get_order()+1), other.nfp))
    
    ''' Evaluation '''
    def eval(self, psi, chi=0, phi=0, sq_eps_series:bool=False, n_max=float('inf')):
        if sq_eps_series:
            power_arg = psi
        else:
            power_arg = jnp.sqrt(psi)
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
            elif jnp.isscalar(item):
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
        string = string + '], nfp=' +str(self.nfp)
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
        out_chiphiepsfunc = ChiPhiEpsFunc(chiphifunc_list, nfp)
        return(out_chiphiepsfunc)
    
# Replaces all zeros with ChiPhiFunc(nfp=0). Cannot be jitted.
def ChiPhiEpsFunc_remove_zero(list:list, nfp:int, check_consistency:bool=False):
    for i in range(len(list)):
        item = list[i]
        if isinstance(item, ChiPhiFunc):
            if jnp.all(item.content==0):
                list[i] = ChiPhiFuncSpecial(0)
        elif jnp.isscalar(item) or item.ndim==0:
            if item==0:
                list[i] = ChiPhiFuncSpecial(0)
    return(ChiPhiEpsFunc(list, nfp, check_consistency))

# For JAX use
tree_util.register_pytree_node(ChiPhiEpsFunc,
                               ChiPhiEpsFunc._tree_flatten,
                               ChiPhiEpsFunc._tree_unflatten)
