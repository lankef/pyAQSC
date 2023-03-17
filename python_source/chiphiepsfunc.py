import chiphifunc
import warnings
import numpy as np # Used in saving and loading
import jax.numpy as jnp

'''ChiPhiEpsFunc'''
# A container for lists of ChiPhiFuncs. Primarily used to handle array index out of bound
# error in Maxima-translated codes. Produces a ChiPhiFuncNull when index is out of bound.
# Initialization:
# ChiPhiEpsFunc([X0, X1, X2, ... Xn])
class ChiPhiEpsFunc:
    def __init__(self, list:list, nfp:int, check_consistency:bool=True): # nfp-dependent!!
        self.chiphifunc_list = list
        self.nfp = nfp
        if check_consistency:
            self.check_nfp_consistency()

    def _tree_flatten(self):
        children = (self.x,)  # arrays / dynamic values
        aux_data = {'chiphifunc_list': self.chiphifunc_list}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    # Check the nfp of all constituents in self.chiphifunc_list
    def check_nfp_consistency(self):
        for i in range(len(self.chiphifunc_list)):
            if not jnp.isscalar(self.chiphifunc_list[i]):
                if self.chiphifunc_list[i].nfp!=0 and self.chiphifunc_list[i].nfp!=self.nfp:
                    self.chiphifunc_list[i] = \
                        'inconsistent item, ' \
                        +'item.nfp'+str(item.nfp) \
                        +', self.nfp'+str(self.nfp)



    def __getitem__(self, index): # not nfp-dependent
        # If array index out of bound then put in an null item first
        if (index>len(self.chiphifunc_list)-1 or index<0):
            return(chiphifunc.ChiPhiFuncNull())
        return(self.chiphifunc_list[index])

    def __setitem__(self, key, newvalue): # not nfp-dependent
        raise NotImplementedError('ChiPhiEpsFunc should only be edited with'\
        ' ChiPhiEpsFunc.append() to prevent changes to known terms.')

    # Implementation of list append
    # def append(self, item):
    #     TODO add type check
    #     self.chiphifunc_list.append(item)  # nfp-dependent!!

    # Append one or more zeros to the end of the list.
    # For evaluating higher order terms with recursion relation. Sometimes
    # expressions used to evaluate a higher order term includes the term itself,
    # and requires ChiPhiEpsFunc to provide zero (rather than ChiPhiFuncNull
    # from __getitem__ when array index is out of bound)
    def zero_append(self, n=1): # nfp-dependent!!
        new_list = self.chiphifunc_list.copy()
        for i in range(n):
            new_list.append(0)
        # we know the new element has consistent nfp.
        return(ChiPhiEpsFunc(new_list, self.nfp, False))

    # For testing recursion. Produces a sub-list up to the nth element (order).
    def mask(self, n): # nfp-dependent!!
        if n>len(self.chiphifunc_list)-1:
            raise IndexError('Mask size is larger than the list\'s size')
        # we know the new element has consistent nfp.
        return(ChiPhiEpsFunc(self.chiphifunc_list[:n+1], self.nfp, False))

    # Copies a ChiPhiEpsFunc. Does not copy the constituents of self.chiphifunc_list.
    def copy(self): # nfp-dependent!!
        return(ChiPhiEpsFunc(self.chiphifunc_list.copy(), self.nfp, False))

    def __list__(self): # not nfp-dependent
        return(self.chiphifunc_list)


    def __len__(self): # not nfp-dependent
        return(len(self.chiphifunc_list))

    def get_order(self): # not nfp-dependent
        return(len(self.chiphifunc_list)-1)

    # Make a list with order+1 zero elements
    # not nfp-dependent
    def zeros_to_order(order, nfp=0):
        new_list = []
        for i in range(order+1):
            new_list.append(0)
        return(ChiPhiEpsFunc(new_list, nfp))

    # Make a list with order+1 zero elements
    # not nfp-dependent
    def zeros_like(chiphiepsfunc_in, nfp=0):
        return(ChiPhiEpsFunc.zeros_to_order(chiphiepsfunc_in.get_order(), nfp))

    # Converting to a list of arrays. For saving and loading.
    # see recursion_relation.py.
    def to_content_list(self): # not nfp-dependent
        content_list = []
        for i in range(len(self.chiphifunc_list)):
            item = self.chiphifunc_list[i]
            if np.isscalar(item):
                content_list.append(item)
            else:
                content_list.append(item.content)
        return(content_list)

    def from_content_list(content_list, nfp): # nfp-dependent!!
        chiphifunc_list = []
        for item in content_list:
            if np.isscalar(item):
                chiphifunc_list.append(item)
            else:
                chiphifunc_list.append(chiphifunc.ChiPhiFunc(item, nfp))
        out_chiphiepsfunc = ChiPhiEpsFunc(chiphifunc_list, nfp)
        return(out_chiphiepsfunc)
