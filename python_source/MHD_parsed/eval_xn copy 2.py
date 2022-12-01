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
    def sum_arg_5(i15):
        # Child args for sum_arg_5    
        def sum_arg_2(i8):
            # Child args for sum_arg_2
            return(B_alpha_coef[i8]*B_alpha_coef[n-i8-i15])
            
        def sum_arg_4(i26):
            # Child args for sum_arg_4    
            def sum_arg_3(i55):
                # Child args for sum_arg_3
                return((is_seq(0,n-i55-i15)*diff(Z_coef_cp[(-i55)-i26+i15],'chi',1)*iota_coef[i55]*diff(Z_coef_cp[(-n)+i55+i26+i15],'chi',1)+is_seq(0,n-i55-i15)*diff(Y_coef_cp[(-i55)-i26+i15],'chi',1)*iota_coef[i55]*diff(Y_coef_cp[(-n)+i55+i26+i15],'chi',1)+is_seq(0,n-i55-i15)*diff(X_coef_cp[(-i55)-i26+i15],'chi',1)*iota_coef[i55]*diff(X_coef_cp[(-n)+i55+i26+i15],'chi',1))*iota_coef[n-i55-i15]*is_integer(n-i55-i15)*is_seq(n-i55-i15,i26))
            
            return((2*is_seq(0,n-i15)*dl_p*Y_coef_cp[i26]*diff(X_coef_cp[(-n)-i26+2*i15],'chi',1)-2*is_seq(0,n-i15)*dl_p*X_coef_cp[i26]*diff(Y_coef_cp[(-n)-i26+2*i15],'chi',1))*iota_coef[n-i15]*is_integer(n-i15)*is_seq(n-i15,i15-i26)*tau_p+py_sum(sum_arg_3,0,i15-i26)+((2*is_seq(0,n-i15)*diff(Z_coef_cp[i26],'phi',1)-2*is_seq(0,n-i15)*dl_p*X_coef_cp[i26]*kap_p)*diff(Z_coef_cp[(-n)-i26+2*i15],'chi',1)+2*is_seq(0,n-i15)*diff(Y_coef_cp[i26],'phi',1)*diff(Y_coef_cp[(-n)-i26+2*i15],'chi',1)+(2*is_seq(0,n-i15)*dl_p*Z_coef_cp[i26]*kap_p+2*is_seq(0,n-i15)*diff(X_coef_cp[i26],'phi',1))*diff(X_coef_cp[(-n)-i26+2*i15],'chi',1))*iota_coef[n-i15]*is_integer(n-i15)*is_seq(n-i15,i15-i26))
        
        return(py_sum(sum_arg_4,0,i15)-is_seq(0,2*i15-n)*B_denom_coef_c[2*i15-n]*is_integer(2*i15-n)*is_seq(2*i15-n,i15)*py_sum(sum_arg_2,0,n-i15)+2*is_seq(0,n-i15)*dl_p*diff(Z_coef_cp[2*i15-n],'chi',1)*iota_coef[n-i15]*is_integer(n-i15)*is_seq(n-i15,i15))
    
    def sum_arg_1(i20):
        # Child args for sum_arg_1
        return(((is_seq(0,n)*dl_p**2*Y_coef_cp[i20]*is_integer(n)*Y_coef_cp[n-i20]+is_seq(0,n)*dl_p**2*X_coef_cp[i20]*is_integer(n)*X_coef_cp[n-i20])*tau_p**2)\
            +(((-2*is_seq(0,n)*dl_p*X_coef_cp[i20]*is_integer(n)*diff(Y_coef_cp[n-i20],'phi',1))+2*is_seq(0,n)*dl_p*Y_coef_cp[i20]*is_integer(n)*diff(X_coef_cp[n-i20],'phi',1)+2*is_seq(0,n)*dl_p**2*Y_coef_cp[i20]*kap_p*is_integer(n)*Z_coef_cp[n-i20])*tau_p)\
            +((is_seq(0,n)*diff(Z_coef_cp[i20],'phi',1)-2*is_seq(0,n)*dl_p*X_coef_cp[i20]*kap_p)*is_integer(n)*diff(Z_coef_cp[n-i20],'phi',1))\
            +(is_seq(0,n)*diff(Y_coef_cp[i20],'phi',1)*is_integer(n)*diff(Y_coef_cp[n-i20],'phi',1))\
            +((2*is_seq(0,n)*dl_p*Z_coef_cp[i20]*kap_p+is_seq(0,n)*diff(X_coef_cp[i20],'phi',1))*is_integer(n)*diff(X_coef_cp[n-i20],'phi',1))\
            +(is_seq(0,n)*dl_p**2*Z_coef_cp[i20]*kap_p**2*is_integer(n)*Z_coef_cp[n-i20])\
            +(is_seq(0,n)*dl_p**2*X_coef_cp[i20]*kap_p**2*is_integer(n)*X_coef_cp[n-i20]))
    
    
    out = (py_sum_parallel(sum_arg_5,ceil(n/2),floor(n))+py_sum_parallel(sum_arg_1,0,n)+2*is_seq(0,n)*dl_p*is_integer(n)*diff(Z_coef_cp[n],'phi',1)-2*is_seq(0,n)*dl_p**2*kap_p*X_coef_cp[n]*is_integer(n))/(2*dl_p**2*kap_p)
    return(out)
