# Xi_n, but without doing the 0th order matching. 
# The constant component is the actual sigma_n(phi). 
# Must evaluate with Yn, Zn+1=0 
from math import floor, ceil
from math_utilities import *
import chiphifunc
def eval_full_Xi_n_p(n, X_coef_cp, Y_coef_cp, Z_coef_cp, \
    kap_p, dl_p, tau_p, iota_coef):    
    def sum_arg_13(i332):
        # Child args for sum_arg_13
        return(X_coef_cp[i332]*diff(Y_coef_cp[n-i332+1],'chi',1))
    
    def sum_arg_12(i328):
        # Child args for sum_arg_12
        return(Y_coef_cp[i328]*diff(X_coef_cp[n-i328+1],'chi',1))
    
    def sum_arg_11(i334):
        # Child args for sum_arg_11
        return(X_coef_cp[i334]*diff(Z_coef_cp[n-i334+1],'chi',1))
    
    def sum_arg_10(i330):
        # Child args for sum_arg_10
        return(Z_coef_cp[i330]*diff(X_coef_cp[n-i330+1],'chi',1))
    
    def sum_arg_9(i325):
        # Child args for sum_arg_9    
        def sum_arg_8(i326):
            # Child args for sum_arg_8
            return(diff(Z_coef_cp[i326],'chi',1)*diff(Z_coef_cp[(-n)-i326+2*i325-1],'chi',1)*is_seq(n-i325+1,i325-i326))
        
        return(is_seq(0,n-i325+1)*iota_coef[n-i325+1]*is_integer(n-i325+1)*py_sum(sum_arg_8,0,i325))
    
    def sum_arg_7(i324):
        # Child args for sum_arg_7
        return(diff(Z_coef_cp[i324],'chi',1)*diff(Z_coef_cp[n-i324+1],'phi',1))
    
    def sum_arg_6(i321):
        # Child args for sum_arg_6    
        def sum_arg_5(i322):
            # Child args for sum_arg_5
            return(diff(Y_coef_cp[i322],'chi',1)*diff(Y_coef_cp[(-n)-i322+2*i321-1],'chi',1)*is_seq(n-i321+1,i321-i322))
        
        return(is_seq(0,n-i321+1)*iota_coef[n-i321+1]*is_integer(n-i321+1)*py_sum(sum_arg_5,0,i321))
    
    def sum_arg_4(i320):
        # Child args for sum_arg_4
        return(diff(Y_coef_cp[i320],'chi',1)*diff(Y_coef_cp[n-i320+1],'phi',1))
    
    def sum_arg_3(i317):
        # Child args for sum_arg_3    
        def sum_arg_2(i318):
            # Child args for sum_arg_2
            return(diff(X_coef_cp[i318],'chi',1)*diff(X_coef_cp[(-n)-i318+2*i317-1],'chi',1)*is_seq(n-i317+1,i317-i318))
        
        return(is_seq(0,n-i317+1)*iota_coef[n-i317+1]*is_integer(n-i317+1)*py_sum(sum_arg_2,0,i317))
    
    def sum_arg_1(i316):
        # Child args for sum_arg_1
        return(diff(X_coef_cp[i316],'chi',1)*diff(X_coef_cp[n-i316+1],'phi',1))
    
    
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
