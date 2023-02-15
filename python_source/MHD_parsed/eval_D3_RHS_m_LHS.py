# D3_RHS[n] - D3_LHS[n] for calculating Y[n,+1]. 
# The constant component is what we need. 
# Must evaluate with Yn, Zn+1=0 
from math import floor, ceil
from math_utilities import *
import chiphifunc
def eval_D3_RHS_m_LHS(n, X_coef_cp, Y_coef_cp, Z_coef_cp, B_theta_coef_cp, 
    B_denom_coef_c, B_alpha_coef, iota_coef, dl_p, tau_p, kap_p):    
    def sum_arg_15(i270):
        # Child args for sum_arg_15
        return(X_coef_cp[i270]*diff(Y_coef_cp[n-i270],'chi',1))
    
    def sum_arg_14(i266):
        # Child args for sum_arg_14
        return(Y_coef_cp[i266]*diff(X_coef_cp[n-i266],'chi',1))
    
    def sum_arg_13(i272):
        # Child args for sum_arg_13
        return(X_coef_cp[i272]*diff(Z_coef_cp[n-i272],'chi',1))
    
    def sum_arg_12(i268):
        # Child args for sum_arg_12
        return(Z_coef_cp[i268]*diff(X_coef_cp[n-i268],'chi',1))
    
    def sum_arg_11(i263):
        # Child args for sum_arg_11    
        def sum_arg_10(i264):
            # Child args for sum_arg_10
            return(diff(Z_coef_cp[i264],'chi',1)*diff(Z_coef_cp[(-n)-i264+2*i263],'chi',1)*is_seq(n-i263,i263-i264))
        
        return(is_seq(0,n-i263)*iota_coef[n-i263]*is_integer(n-i263)*py_sum(sum_arg_10,0,i263))
    
    def sum_arg_9(i262):
        # Child args for sum_arg_9
        return(diff(Z_coef_cp[i262],'chi',1)*diff(Z_coef_cp[n-i262],'phi',1))
    
    def sum_arg_8(i259):
        # Child args for sum_arg_8    
        def sum_arg_7(i260):
            # Child args for sum_arg_7
            return(diff(Y_coef_cp[i260],'chi',1)*diff(Y_coef_cp[(-n)-i260+2*i259],'chi',1)*is_seq(n-i259,i259-i260))
        
        return(is_seq(0,n-i259)*iota_coef[n-i259]*is_integer(n-i259)*py_sum(sum_arg_7,0,i259))
    
    def sum_arg_6(i258):
        # Child args for sum_arg_6
        return(diff(Y_coef_cp[i258],'chi',1)*diff(Y_coef_cp[n-i258],'phi',1))
    
    def sum_arg_5(i255):
        # Child args for sum_arg_5    
        def sum_arg_4(i256):
            # Child args for sum_arg_4
            return(diff(X_coef_cp[i256],'chi',1)*diff(X_coef_cp[(-n)-i256+2*i255],'chi',1)*is_seq(n-i255,i255-i256))
        
        return(is_seq(0,n-i255)*iota_coef[n-i255]*is_integer(n-i255)*py_sum(sum_arg_4,0,i255))
    
    def sum_arg_3(i254):
        # Child args for sum_arg_3
        return(diff(X_coef_cp[i254],'chi',1)*diff(X_coef_cp[n-i254],'phi',1))
    
    def sum_arg_2(i209):
        # Child args for sum_arg_2    
        def sum_arg_1(i210):
            # Child args for sum_arg_1
            return(B_theta_coef_cp[i210]*B_denom_coef_c[(-n)-i210+2*i209]*is_seq(n-i209,i209-i210))
        
        return(is_seq(0,n-i209)*B_alpha_coef[n-i209]*is_integer(n-i209)*py_sum(sum_arg_1,0,i209))
    
    
    out = ((is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_14,0,n)-is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_15,0,n))*tau_p)\
        +(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_9,0,n))\
        +(py_sum_parallel(sum_arg_8,ceil(n/2),floor(n)))\
        +(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_6,0,n))\
        +(py_sum_parallel(sum_arg_5,ceil(n/2),floor(n)))\
        +(is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_3,0,n))\
        +(-py_sum_parallel(sum_arg_2,ceil(n/2),floor(n)))\
        +(-is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_13,0,n))\
        +(is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_12,0,n))\
        +(py_sum_parallel(sum_arg_11,ceil(n/2),floor(n)))\
        +(is_seq(0,n)*dl_p*is_integer(n)*diff(Z_coef_cp[n],'chi',1))
    return(out)
