# Xi_n, but without doing the 0th order matching. 
# The constant component is the actual sigma_n(phi). 
# Must evaluate with Yn, Zn+1=0 
from math import floor, ceil
from math_utilities import *
import chiphifunc
def eval_full_Xi_n_p(n, X_coef_cp, Y_coef_cp, Z_coef_cp, \
    kap_p, dl_p, tau_p, iota_coef):    
    def sum_arg_13(i310):
        # Child args for sum_arg_13
        return(X_coef_cp[i310]*diff(Y_coef_cp[n-i310+1],'chi',1))
    
    def sum_arg_12(i306):
        # Child args for sum_arg_12
        return(Y_coef_cp[i306]*diff(X_coef_cp[n-i306+1],'chi',1))
    
    def sum_arg_11(i312):
        # Child args for sum_arg_11
        return(X_coef_cp[i312]*diff(Z_coef_cp[n-i312+1],'chi',1))
    
    def sum_arg_10(i308):
        # Child args for sum_arg_10
        return(Z_coef_cp[i308]*diff(X_coef_cp[n-i308+1],'chi',1))
    
    def sum_arg_9(i303):
        # Child args for sum_arg_9    
        def sum_arg_8(i304):
            # Child args for sum_arg_8
            return(diff(Z_coef_cp[i304],'chi',1)*diff(Z_coef_cp[(-n)-i304+2*i303-1],'chi',1)*is_seq(n-i303+1,i303-i304))
        
        return(is_seq(0,n-i303+1)*iota_coef[n-i303+1]*is_integer(n-i303+1)*py_sum(sum_arg_8,0,i303))
    
    def sum_arg_7(i302):
        # Child args for sum_arg_7
        return(diff(Z_coef_cp[i302],'chi',1)*diff(Z_coef_cp[n-i302+1],'phi',1))
    
    def sum_arg_6(i299):
        # Child args for sum_arg_6    
        def sum_arg_5(i300):
            # Child args for sum_arg_5
            return(diff(Y_coef_cp[i300],'chi',1)*diff(Y_coef_cp[(-n)-i300+2*i299-1],'chi',1)*is_seq(n-i299+1,i299-i300))
        
        return(is_seq(0,n-i299+1)*iota_coef[n-i299+1]*is_integer(n-i299+1)*py_sum(sum_arg_5,0,i299))
    
    def sum_arg_4(i298):
        # Child args for sum_arg_4
        return(diff(Y_coef_cp[i298],'chi',1)*diff(Y_coef_cp[n-i298+1],'phi',1))
    
    def sum_arg_3(i295):
        # Child args for sum_arg_3    
        def sum_arg_2(i296):
            # Child args for sum_arg_2
            return(diff(X_coef_cp[i296],'chi',1)*diff(X_coef_cp[(-n)-i296+2*i295-1],'chi',1)*is_seq(n-i295+1,i295-i296))
        
        return(is_seq(0,n-i295+1)*iota_coef[n-i295+1]*is_integer(n-i295+1)*py_sum(sum_arg_2,0,i295))
    
    def sum_arg_1(i294):
        # Child args for sum_arg_1
        return(diff(X_coef_cp[i294],'chi',1)*diff(X_coef_cp[n-i294+1],'phi',1))
    
    
    out = ((is_seq(0,n+1)*dl_p*is_integer(n+1)*py_sum_parallel(sum_arg_12,0,n+1)-is_seq(0,n+1)*dl_p*is_integer(n+1)*py_sum_parallel(sum_arg_13,0,n+1))*tau_p)\
        +(py_sum_parallel(sum_arg_9,ceil((n+1)/2),floor(n)+1))\
        +(is_seq(0,n+1)*is_integer(n+1)*py_sum_parallel(sum_arg_7,0,n+1))\
        +(py_sum_parallel(sum_arg_6,ceil((n+1)/2),floor(n)+1))\
        +(is_seq(0,n+1)*is_integer(n+1)*py_sum_parallel(sum_arg_4,0,n+1))\
        +(py_sum_parallel(sum_arg_3,ceil((n+1)/2),floor(n)+1))\
        +(-is_seq(0,n+1)*dl_p*kap_p*is_integer(n+1)*py_sum_parallel(sum_arg_11,0,n+1))\
        +(is_seq(0,n+1)*dl_p*kap_p*is_integer(n+1)*py_sum_parallel(sum_arg_10,0,n+1))\
        +(is_seq(0,n+1)*is_integer(n+1)*py_sum_parallel(sum_arg_1,0,n+1))\
        +(is_seq(0,n+1)*dl_p*is_integer(n+1)*diff(Z_coef_cp[n+1],'chi',1))
    return(out)
