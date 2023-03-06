import chiphifunc
import warnings
import numpy as np

'''ChiPhiEpsFunc'''
# A container for lists of ChiPhiFuncs. Primarily used to handle array index out of bound
# error in Maxima-translated codes. Produces a ChiPhiFuncNull when index is out of bound.
# Initialization:
# ChiPhiEpsFunc([X0, X1, X2, ... Xn])
class ChiPhiEpsFunc:
    def __init__(self, list, nfp, check_consistency = True): # nfp-dependent!!
        self.chiphifunc_list = list
        self.nfp = nfp
        if check_consistency:
            self.check_nfp_consistency()

    # Check the nfp of all constituents in self.chiphifunc_list
    def check_nfp_consistency(self):
        for item in self.chiphifunc_list:
            self.check_nfp(item) # nfp-dependent!!

    # Throws an error if item is a ChiPhiFunc and has non-zero nfp thats
    # not equal to self.nfp
    def check_nfp(self, item):
        if not np.isscalar(item):
            if item.nfp!=0 and item.nfp!=self.nfp:
                raise ValueError('A ChiPhiEpsFunc must contain ChiPhiFunc\'s with '\
                'nfp = ChiPhiEpsFunc.nfp or 0.') # nfp-dependent!!

    def __getitem__(self, index): # not nfp-dependent
        # If array index out of bound then put in an null item first
        if (index>len(self.chiphifunc_list)-1 or index<0):
            return(chiphifunc.ChiPhiFuncNull())
        return(self.chiphifunc_list[index])

    def __setitem__(self, key, newvalue): # not nfp-dependent
        raise NotImplementedError('ChiPhiEpsFunc should only be edited with'\
        ' ChiPhiEpsFunc.append() to prevent changes to known terms.')

    # Implementation of list append
    def append(self, item):
        self.check_nfp(item)
        self.chiphifunc_list.append(item)  # nfp-dependent!!

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
    def zeros_to_order(order):
        new_list = []
        for i in range(order+1):
            new_list.append(0)
        return(ChiPhiEpsFunc(new_list, 0))

    # Make a list with order+1 zero elements
    # not nfp-dependent
    def zeros_like(chiphiepsfunc_in):
        return(ChiPhiEpsFunc.zeros_to_order(chiphiepsfunc_in.get_order()))

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
