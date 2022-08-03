# Sigma_n, but without doing the 0th order matching. 
# The constant component is the actual sigma_n(phi). 
# Must evaluate with Yn, Zn+1=0 
from math import floor, ceil
from math_utilities import is_seq, py_sum, is_integer, diff
import chiphifunc
def eval_full_Xi_n(n, X_coef_cp, Y_coef_cp, Z_coef_cp, \
    kap_p, dl_p, tau_p, iota_coef):    
    def sum_arg_13(i76):
        # Child args for sum_arg_13
        return(X_coef_cp[i76]*diff(Y_coef_cp[n-i76+1],'chi',1))
    
    def sum_arg_12(i72):
        # Child args for sum_arg_12
        return(Y_coef_cp[i72]*diff(X_coef_cp[n-i72+1],'chi',1))
    
    def sum_arg_11(i78):
        # Child args for sum_arg_11
        return(X_coef_cp[i78]*diff(Z_coef_cp[n-i78+1],'chi',1))
    
    def sum_arg_10(i74):
        # Child args for sum_arg_10
        return(Z_coef_cp[i74]*diff(X_coef_cp[n-i74+1],'chi',1))
    
    def sum_arg_9(i69):
        # Child args for sum_arg_9    
        def sum_arg_8(i70):
            # Child args for sum_arg_8
            return(diff(Z_coef_cp[i70],'chi',1)*diff(Z_coef_cp[(-n)-i70+2*i69-1],'chi',1)*is_seq(n-i69+1,i69-i70))
        
        return(is_seq(0,n-i69+1)*iota_coef[n-i69+1]*is_integer(n-i69+1)*py_sum(sum_arg_8,0,i69))
    
    def sum_arg_7(i68):
        # Child args for sum_arg_7
        return(diff(Z_coef_cp[i68],'chi',1)*diff(Z_coef_cp[n-i68+1],'phi',1))
    
    def sum_arg_6(i65):
        # Child args for sum_arg_6    
        def sum_arg_5(i66):
            # Child args for sum_arg_5
            return(diff(Y_coef_cp[i66],'chi',1)*diff(Y_coef_cp[(-n)-i66+2*i65-1],'chi',1)*is_seq(n-i65+1,i65-i66))
        
        return(is_seq(0,n-i65+1)*iota_coef[n-i65+1]*is_integer(n-i65+1)*py_sum(sum_arg_5,0,i65))
    
    def sum_arg_4(i64):
        # Child args for sum_arg_4
        return(diff(Y_coef_cp[i64],'chi',1)*diff(Y_coef_cp[n-i64+1],'phi',1))
    
    def sum_arg_3(i61):
        # Child args for sum_arg_3    
        def sum_arg_2(i62):
            # Child args for sum_arg_2
            return(diff(X_coef_cp[i62],'chi',1)*diff(X_coef_cp[(-n)-i62+2*i61-1],'chi',1)*is_seq(n-i61+1,i61-i62))
        
        return(is_seq(0,n-i61+1)*iota_coef[n-i61+1]*is_integer(n-i61+1)*py_sum(sum_arg_2,0,i61))
    
    def sum_arg_1(i60):
        # Child args for sum_arg_1
        return(diff(X_coef_cp[i60],'chi',1)*diff(X_coef_cp[n-i60+1],'phi',1))
    
    
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
