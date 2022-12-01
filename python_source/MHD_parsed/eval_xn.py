# This script calculates Xn. Values of:
# n,
# X_coef_cp[0, ... , n-1],
# Y_coef_cp[0, ... , n-1],
# Z_coef_cp[0, ... , n],
# B_denom_coef_c[0, ... , n],
# B_alpha_coef[0, ... , (n-1)/2 or n/2],
# kap_p, dl_p, tau_p, iota[0, ... ,  (n-3)/2 or (n-2)/2 ]
# must be provided.
#
# Must provide X_coef_cp[n]=0 using ChPhiEpsFunc.zero_append().
# Contains dchi Zn and dphi Zn with coeff: 
# C_dchi = 2*iota_coef[0]*dl_p
# C_dphi = 2*dl_p
from math import floor, ceil
from math_utilities import *
import chiphifunc
def eval_Xn_cp(n,
    X_coef_cp,
    Y_coef_cp,
    Z_coef_cp,
    B_denom_coef_c,
    B_alpha_coef,
    kap_p, dl_p, tau_p,
    iota_coef):    
    def sum_arg_10(i20):
        # Child args for sum_arg_10
        return(dl_p**2*Y_coef_cp[i20]*is_integer(n)*Y_coef_cp[n-i20]+dl_p**2*X_coef_cp[i20]*is_integer(n)*X_coef_cp[n-i20])
    
    def sum_arg_9(i30):
        # Child args for sum_arg_9
        return((-2*dl_p*X_coef_cp[i30]*is_integer(n)*diff(Y_coef_cp[n-i30],'phi',1))+2*dl_p*Y_coef_cp[i30]*is_integer(n)*diff(X_coef_cp[n-i30],'phi',1)+2*dl_p**2*Y_coef_cp[i30]*kap_p*is_integer(n)*Z_coef_cp[n-i30])
    
    def sum_arg_8(i23):
        # Child args for sum_arg_8    
        def sum_arg_7(i26):
            # Child args for sum_arg_7
            return((2*is_seq(0,n-i23)*dl_p*Y_coef_cp[i26]*diff(X_coef_cp[(-n)-i26+2*i23],'chi',1)-2*is_seq(0,n-i23)*dl_p*X_coef_cp[i26]*diff(Y_coef_cp[(-n)-i26+2*i23],'chi',1))*is_seq(n-i23,i23-i26))
        
        return(iota_coef[n-i23]*is_integer(n-i23)*py_sum(sum_arg_7,0,i23))
    
    def sum_arg_6(i53):
        # Child args for sum_arg_6    
        def sum_arg_5(i54):
            # Child args for sum_arg_5    
            def sum_arg_4(i55):
                # Child args for sum_arg_4
                return((is_seq(0,n-i55-i53)*diff(Z_coef_cp[(-i55)-i54+i53],'chi',1)*iota_coef[i55]*diff(Z_coef_cp[(-n)+i55+i54+i53],'chi',1)+is_seq(0,n-i55-i53)*diff(Y_coef_cp[(-i55)-i54+i53],'chi',1)*iota_coef[i55]*diff(Y_coef_cp[(-n)+i55+i54+i53],'chi',1)+is_seq(0,n-i55-i53)*diff(X_coef_cp[(-i55)-i54+i53],'chi',1)*iota_coef[i55]*diff(X_coef_cp[(-n)+i55+i54+i53],'chi',1))*iota_coef[n-i55-i53]*is_integer(n-i55-i53)*is_seq(n-i55-i53,i54))
            
            return(py_sum(sum_arg_4,0,i53-i54)+((2*is_seq(0,n-i53)*diff(Z_coef_cp[i54],'phi',1)-2*is_seq(0,n-i53)*dl_p*X_coef_cp[i54]*kap_p)*diff(Z_coef_cp[(-n)-i54+2*i53],'chi',1)+2*is_seq(0,n-i53)*diff(Y_coef_cp[i54],'phi',1)*diff(Y_coef_cp[(-n)-i54+2*i53],'chi',1)+(2*is_seq(0,n-i53)*dl_p*Z_coef_cp[i54]*kap_p+2*is_seq(0,n-i53)*diff(X_coef_cp[i54],'phi',1))*diff(X_coef_cp[(-n)-i54+2*i53],'chi',1))*iota_coef[n-i53]*is_integer(n-i53)*is_seq(n-i53,i53-i54))
        
        return(py_sum(sum_arg_5,0,i53)+2*is_seq(0,n-i53)*dl_p*diff(Z_coef_cp[2*i53-n],'chi',1)*iota_coef[n-i53]*is_integer(n-i53)*is_seq(n-i53,i53))
    
    def sum_arg_3(i9):
        # Child args for sum_arg_3    
        def sum_arg_2(i8):
            # Child args for sum_arg_2
            return(B_alpha_coef[i8]*B_alpha_coef[n-i9-i8])
        
        return(is_seq(0,2*i9-n)*B_denom_coef_c[2*i9-n]*is_integer(2*i9-n)*is_seq(2*i9-n,i9)*py_sum(sum_arg_2,0,n-i9))
    
    def sum_arg_1(i32):
        # Child args for sum_arg_1
        return((diff(Z_coef_cp[i32],'phi',1)-2*dl_p*X_coef_cp[i32]*kap_p)*is_integer(n)*diff(Z_coef_cp[n-i32],'phi',1)+diff(Y_coef_cp[i32],'phi',1)*is_integer(n)*diff(Y_coef_cp[n-i32],'phi',1)+(2*dl_p*Z_coef_cp[i32]*kap_p+diff(X_coef_cp[i32],'phi',1))*is_integer(n)*diff(X_coef_cp[n-i32],'phi',1)+dl_p**2*Z_coef_cp[i32]*kap_p**2*is_integer(n)*Z_coef_cp[n-i32]+dl_p**2*X_coef_cp[i32]*kap_p**2*is_integer(n)*X_coef_cp[n-i32])
    
    
    out = reg_div(py_sum_parallel(sum_arg_10,0,n)*tau_p**2+(py_sum_parallel(sum_arg_9,0,n)+py_sum_parallel(sum_arg_8,ceil(n/2),floor(n)))*tau_p+py_sum_parallel(sum_arg_6,ceil(n/2),floor(n))-py_sum_parallel(sum_arg_3,ceil(n/2),floor(n))+py_sum_parallel(sum_arg_1,0,n)+2*dl_p*is_integer(n)*diff(Z_coef_cp[n],'phi',1)-2*dl_p**2*kap_p*X_coef_cp[n]*is_integer(n),2*dl_p**2*kap_p)
    return(out)
