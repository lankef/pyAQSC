import chiphifunc
import warnings

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

    # Implementation of list append
    def append(self, item):
        self.chiphifunc_list.append(item)

    # Append one or more zeros to the end of the list. Copies.
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
        return(ChiPhiEpsFunc(self.chiphifunc_list[:n+1]))

    def __list__(self):
        return(self.chiphifunc_list)


    def __len__(self):
        return(len(self.chiphifunc_list))
