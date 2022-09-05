# Xi_n, but without doing the 0th order matching. 
# The constant component is the actual sigma_n(phi). 
# Must evaluate with Yn, Zn+1=0 
from math import floor, ceil
from math_utilities import *
import chiphifunc
def eval_full_Xi_n_p(n, X_coef_cp, Y_coef_cp, Z_coef_cp, \
    kap_p, dl_p, tau_p, iota_coef):    
    def sum_arg_13(i344):
        # Child args for sum_arg_13
        return(X_coef_cp[i344]*diff(Y_coef_cp[n-i344+1],'chi',1))
    
    def sum_arg_12(i340):
        # Child args for sum_arg_12
        return(Y_coef_cp[i340]*diff(X_coef_cp[n-i340+1],'chi',1))
    
    def sum_arg_11(i346):
        # Child args for sum_arg_11
        return(X_coef_cp[i346]*diff(Z_coef_cp[n-i346+1],'chi',1))
    
    def sum_arg_10(i342):
        # Child args for sum_arg_10
        return(Z_coef_cp[i342]*diff(X_coef_cp[n-i342+1],'chi',1))
    
    def sum_arg_9(i337):
        # Child args for sum_arg_9    
        def sum_arg_8(i338):
            # Child args for sum_arg_8
            return(diff(Z_coef_cp[i338],'chi',1)*diff(Z_coef_cp[(-n)-i338+2*i337-1],'chi',1)*is_seq(n-i337+1,i337-i338))
        
        return(is_seq(0,n-i337+1)*iota_coef[n-i337+1]*is_integer(n-i337+1)*py_sum(sum_arg_8,0,i337))
    
    def sum_arg_7(i336):
        # Child args for sum_arg_7
        return(diff(Z_coef_cp[i336],'chi',1)*diff(Z_coef_cp[n-i336+1],'phi',1))
    
    def sum_arg_6(i333):
        # Child args for sum_arg_6    
        def sum_arg_5(i334):
            # Child args for sum_arg_5
            return(diff(Y_coef_cp[i334],'chi',1)*diff(Y_coef_cp[(-n)-i334+2*i333-1],'chi',1)*is_seq(n-i333+1,i333-i334))
        
        return(is_seq(0,n-i333+1)*iota_coef[n-i333+1]*is_integer(n-i333+1)*py_sum(sum_arg_5,0,i333))
    
    def sum_arg_4(i332):
        # Child args for sum_arg_4
        return(diff(Y_coef_cp[i332],'chi',1)*diff(Y_coef_cp[n-i332+1],'phi',1))
    
    def sum_arg_3(i329):
        # Child args for sum_arg_3    
        def sum_arg_2(i330):
            # Child args for sum_arg_2
            return(diff(X_coef_cp[i330],'chi',1)*diff(X_coef_cp[(-n)-i330+2*i329-1],'chi',1)*is_seq(n-i329+1,i329-i330))
        
        return(is_seq(0,n-i329+1)*iota_coef[n-i329+1]*is_integer(n-i329+1)*py_sum(sum_arg_2,0,i329))
    
    def sum_arg_1(i328):
        # Child args for sum_arg_1
        return(diff(X_coef_cp[i328],'chi',1)*diff(X_coef_cp[n-i328+1],'phi',1))
    
    
    out = (-is_seq(0,n+1)*dl_p*is_integer(n+1)*py_sum(sum_arg_13,0,n+1)*tau_p)\
        +(is_seq(0,n+1)*dl_p*is_integer(n+1)*py_sum(sum_arg_12,0,n+1)*tau_p)\
        +(py_sum(sum_arg_9,ceil(0.5*n+0.5),floor(n)+1))\
        +(is_seq(0,n+1)*is_integer(n+1)*py_sum(sum_arg_7,0,n+1))\
        +(py_sum(sum_arg_6,ceil(0.5*n+0.5),floor(n)+1))\
        +(is_seq(0,n+1)*is_integer(n+1)*py_sum(sum_arg_4,0,n+1))\
        +(py_sum(sum_arg_3,ceil(0.5*n+0.5),floor(n)+1))\
        +(-is_seq(0,n+1)*dl_p*kap_p*is_integer(n+1)*py_sum(sum_arg_11,0,n+1))\
        +(is_seq(0,n+1)*dl_p*kap_p*is_integer(n+1)*py_sum(sum_arg_10,0,n+1))\
        +(is_seq(0,n+1)*is_integer(n+1)*py_sum(sum_arg_1,0,n+1))\
        +(is_seq(0,n+1)*dl_p*is_integer(n+1)*diff(Z_coef_cp[n+1],'chi',1))
    return(out)
