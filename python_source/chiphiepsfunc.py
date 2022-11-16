import chiphifunc
import warnings
import numpy as np

'''ChiPhiEpsFunc'''
# A container for lists of ChiPhiFuncs. Primarily used to handle array index out of bound
# error in Maxima-translated codes. Produces a ChiPhiFuncNull when index is out of bound.
# Initialization:
# ChiPhiEpsFunc([X0, X1, X2, ... Xn])
warn_index_out_of_bound = False # Index out of bound error can be disabled to increase speed
class ChiPhiEpsFunc:
    def __init__(self, list):
        self.chiphifunc_list = list

    def __getitem__(self, index):
        # If array index out of bound then put in an null item first
        if (index>len(self.chiphifunc_list)-1 or index<0):
            if warn_index_out_of_bound:
                warnings.warn('Warning: handling array index out-of-bound. Index = '+str(index))
            return(chiphifunc.ChiPhiFuncNull())
        return(self.chiphifunc_list[index])

    def __setitem__(self, key, newvalue):
        raise NotImplementedError('ChiPhiEpsFunc should only be edited with'\
        ' ChiPhiEpsFunc.append() to prevent changes to known terms.')

    # Implementation of list append
    def append(self, item):
        self.chiphifunc_list.append(item)

    # Append one or more zeros to the end of the list.
    # For evaluating higher order terms with recursion relation. Sometimes
    # expressions used to evaluate a higher order term includes the term itself,
    # and requires ChiPhiEpsFunc to provide zero (rather than ChiPhiFuncNull
    # from __getitem__ when array index is out of bound)
    def zero_append(self, n=1):
        new_list = self.chiphifunc_list.copy()
        for i in range(n):
            new_list.append(0)
        return(ChiPhiEpsFunc(new_list))

    # For testing recursion. Produces a sub-list up to the nth element (order).
    def mask(self, n):
        if n>len(self.chiphifunc_list)-1:
            warnings.warn('Mask size is larger than the list\'s size')
        return(ChiPhiEpsFunc(self.chiphifunc_list[:n+1]))


    # Copy and replacing an item for comparing
    def replace(self, n, new_elem):
        new_list = self.chiphifunc_list.copy()
        new_list[n] = new_elem
        return(ChiPhiEpsFunc(new_list))

    # Copies a ChiPhiEpsFunc. Does not copy the constituents of self.chiphifunc_list.
    def copy(self):
        return(ChiPhiEpsFunc(self.chiphifunc_list.copy()))

    def __list__(self):
        return(self.chiphifunc_list)


    def __len__(self):
        return(len(self.chiphifunc_list))

    def get_order(self):
        return(len(self.chiphifunc_list)-1)

    # Make a list with order+1 zero elements
    def zeros_to_order(order):
        new_list = []
        for i in range(order+1):
            new_list.append(0)
        return(ChiPhiEpsFunc(new_list))

    # Make a list with order+1 zero elements
    def zeros_like(chiphiepsfunc_in):
        return(ChiPhiEpsFunc.zeros_to_order(chiphiepsfunc_in.get_order()))

    # Converting to a list of arrays. For saving and loading.
    # see recursion_relation.py.
    def to_content_list(self):
        content_list = []
        for i in range(len(self.chiphifunc_list)):
            item = self.chiphifunc_list[i]
            if np.isscalar(item):
                content_list.append(item)
            else:
                content_list.append(item.content)
        return(content_list)

    def from_content_list(content_list):
        chiphifunc_list = []
        for item in content_list:
            if np.isscalar(item):
                chiphifunc_list.append(item)
            else:
                chiphifunc_list.append(chiphifunc.ChiPhiFunc(item))
        out_chiphiepsfunc = ChiPhiEpsFunc(chiphifunc_list)
        return(out_chiphiepsfunc)
