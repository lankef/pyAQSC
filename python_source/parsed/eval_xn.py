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
    def sum_arg_38(i16):
        # Child args for sum_arg_38
        return(X_coef_cp[i16]*X_coef_cp[n-i16])
    
    def sum_arg_37(i14):
        # Child args for sum_arg_37
        return(Y_coef_cp[i14]*Y_coef_cp[n-i14])
    
    def sum_arg_36(i26):
        # Child args for sum_arg_36
        return(Y_coef_cp[i26]*diff(X_coef_cp[n-i26],'phi',1))
    
    def sum_arg_35(i24):
        # Child args for sum_arg_35
        return(X_coef_cp[i24]*diff(Y_coef_cp[n-i24],'phi',1))
    
    def sum_arg_34(i21):
        # Child args for sum_arg_34    
        def sum_arg_33(i22):
            # Child args for sum_arg_33
            return(Y_coef_cp[i22]*diff(X_coef_cp[(-n)-i22+2*i21],'chi',1)*is_seq(n-i21,i21-i22))
        
        return(is_seq(0,n-i21)*iota_coef[n-i21]*is_integer(n-i21)*py_sum(sum_arg_33,0,i21))
    
    def sum_arg_32(i19):
        # Child args for sum_arg_32    
        def sum_arg_31(i20):
            # Child args for sum_arg_31
            return(X_coef_cp[i20]*diff(Y_coef_cp[(-n)-i20+2*i19],'chi',1)*is_seq(n-i19,i19-i20))
        
        return(is_seq(0,n-i19)*iota_coef[n-i19]*is_integer(n-i19)*py_sum(sum_arg_31,0,i19))
    
    def sum_arg_30(i18):
        # Child args for sum_arg_30
        return(Y_coef_cp[i18]*Z_coef_cp[n-i18])
    
    def sum_arg_29(i51):
        # Child args for sum_arg_29    
        def sum_arg_28(i52):
            # Child args for sum_arg_28    
            def sum_arg_27(i53):
                # Child args for sum_arg_27
                return((is_seq(0,n-i53-i51))\
                    *(diff(Z_coef_cp[(-i53)-i52+i51],'chi',1))\
                    *(iota_coef[i53])\
                    *(diff(Z_coef_cp[(-n)+i53+i52+i51],'chi',1))\
                    *(iota_coef[n-i53-i51])\
                    *(is_integer(n-i53-i51))\
                    *(is_seq(n-i53-i51,i52)))
            
            return(py_sum(sum_arg_27,0,i51-i52))
        
        return(py_sum(sum_arg_28,0,i51))
    
    def sum_arg_26(i45):
        # Child args for sum_arg_26    
        def sum_arg_25(i46):
            # Child args for sum_arg_25    
            def sum_arg_24(i47):
                # Child args for sum_arg_24
                return((is_seq(0,n-i47-i45))\
                    *(diff(X_coef_cp[(-i47)-i46+i45],'chi',1))\
                    *(iota_coef[i47])\
                    *(diff(X_coef_cp[(-n)+i47+i46+i45],'chi',1))\
                    *(iota_coef[n-i47-i45])\
                    *(is_integer(n-i47-i45))\
                    *(is_seq(n-i47-i45,i46)))
            
            return(py_sum(sum_arg_24,0,i45-i46))
        
        return(py_sum(sum_arg_25,0,i45))
    
    def sum_arg_23(i39):
        # Child args for sum_arg_23    
        def sum_arg_22(i40):
            # Child args for sum_arg_22    
            def sum_arg_21(i41):
                # Child args for sum_arg_21
                return((is_seq(0,n-i41-i39))\
                    *(diff(Y_coef_cp[(-i41)-i40+i39],'chi',1))\
                    *(iota_coef[i41])\
                    *(diff(Y_coef_cp[(-n)+i41+i40+i39],'chi',1))\
                    *(iota_coef[n-i41-i39])\
                    *(is_integer(n-i41-i39))\
                    *(is_seq(n-i41-i39,i40)))
            
            return(py_sum(sum_arg_21,0,i39-i40))
        
        return(py_sum(sum_arg_22,0,i39))
    
    def sum_arg_20(i60):
        # Child args for sum_arg_20
        return(diff(X_coef_cp[i60],'phi',1)*diff(X_coef_cp[n-i60],'phi',1))
    
    def sum_arg_19(i58):
        # Child args for sum_arg_19
        return(diff(Y_coef_cp[i58],'phi',1)*diff(Y_coef_cp[n-i58],'phi',1))
    
    def sum_arg_18(i56):
        # Child args for sum_arg_18
        return(diff(Z_coef_cp[i56],'phi',1)*diff(Z_coef_cp[n-i56],'phi',1))
    
    def sum_arg_17(i5):
        # Child args for sum_arg_17    
        def sum_arg_16(i4):
            # Child args for sum_arg_16
            return(B_alpha_coef[i4]*B_alpha_coef[n-i5-i4])
        
        return(is_seq(0,2*i5-n)*B_denom_coef_c[2*i5-n]*is_integer(2*i5-n)*is_seq(2*i5-n,i5)*py_sum(sum_arg_16,0,n-i5))
    
    def sum_arg_15(i49):
        # Child args for sum_arg_15    
        def sum_arg_14(i50):
            # Child args for sum_arg_14
            return(diff(X_coef_cp[i50],'phi',1)*diff(X_coef_cp[(-n)-i50+2*i49],'chi',1)*is_seq(n-i49,i49-i50))
        
        return(is_seq(0,n-i49)*iota_coef[n-i49]*is_integer(n-i49)*py_sum(sum_arg_14,0,i49))
    
    def sum_arg_13(i43):
        # Child args for sum_arg_13    
        def sum_arg_12(i44):
            # Child args for sum_arg_12
            return(diff(Y_coef_cp[i44],'phi',1)*diff(Y_coef_cp[(-n)-i44+2*i43],'chi',1)*is_seq(n-i43,i43-i44))
        
        return(is_seq(0,n-i43)*iota_coef[n-i43]*is_integer(n-i43)*py_sum(sum_arg_12,0,i43))
    
    def sum_arg_11(i38):
        # Child args for sum_arg_11
        return(Z_coef_cp[i38]*diff(X_coef_cp[n-i38],'phi',1))
    
    def sum_arg_10(i36):
        # Child args for sum_arg_10
        return(X_coef_cp[i36]*diff(Z_coef_cp[n-i36],'phi',1))
    
    def sum_arg_9(i33):
        # Child args for sum_arg_9    
        def sum_arg_8(i34):
            # Child args for sum_arg_8
            return(X_coef_cp[i34]*diff(Z_coef_cp[(-n)-i34+2*i33],'chi',1)*is_seq(n-i33,i33-i34))
        
        return(is_seq(0,n-i33)*iota_coef[n-i33]*is_integer(n-i33)*py_sum(sum_arg_8,0,i33))
    
    def sum_arg_7(i31):
        # Child args for sum_arg_7    
        def sum_arg_6(i32):
            # Child args for sum_arg_6
            return(Z_coef_cp[i32]*diff(X_coef_cp[(-n)-i32+2*i31],'chi',1)*is_seq(n-i31,i31-i32))
        
        return(is_seq(0,n-i31)*iota_coef[n-i31]*is_integer(n-i31)*py_sum(sum_arg_6,0,i31))
    
    def sum_arg_5(i251):
        # Child args for sum_arg_5    
        def sum_arg_4(i252):
            # Child args for sum_arg_4
            return(diff(Z_coef_cp[i252],'phi',1)*diff(Z_coef_cp[(-n)-i252+2*i251],'chi',1)*is_seq(n-i251,i251-i252))
        
        return(is_seq(0,n-i251)*iota_coef[n-i251]*is_integer(n-i251)*py_sum(sum_arg_4,0,i251))
    
    def sum_arg_3(i30):
        # Child args for sum_arg_3
        return(X_coef_cp[i30]*X_coef_cp[n-i30])
    
    def sum_arg_2(i28):
        # Child args for sum_arg_2
        return(Z_coef_cp[i28]*Z_coef_cp[n-i28])
    
    def sum_arg_1(i11):
        # Child args for sum_arg_1
        return(is_seq(0,n-i11)*diff(Z_coef_cp[2*i11-n],'chi',1)*iota_coef[n-i11]*is_integer(n-i11)*is_seq(n-i11,i11))
    
    
    out = (is_seq(0,n)*dl_p**2*is_integer(n)*py_sum(sum_arg_38,0,n)*tau_p**2+is_seq(0,n)*dl_p**2*is_integer(n)*py_sum(sum_arg_37,0,n)*tau_p**2+2*is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_36,0,n)*tau_p-2*is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_35,0,n)*tau_p+2*dl_p*py_sum(sum_arg_34,ceil(0.5*n),floor(n))*tau_p-2*dl_p*py_sum(sum_arg_32,ceil(0.5*n),floor(n))*tau_p+2*is_seq(0,n)*dl_p**2*kap_p*is_integer(n)*py_sum(sum_arg_30,0,n)*tau_p-2*dl_p*kap_p*py_sum(sum_arg_9,ceil(0.5*n),floor(n))+2*dl_p*kap_p*py_sum(sum_arg_7,ceil(0.5*n),floor(n))+2*py_sum(sum_arg_5,ceil(0.5*n),floor(n))+is_seq(0,n)*dl_p**2*kap_p**2*is_integer(n)*py_sum(sum_arg_3,0,n)+py_sum(sum_arg_29,ceil(0.5*n),floor(n))+py_sum(sum_arg_26,ceil(0.5*n),floor(n))+py_sum(sum_arg_23,ceil(0.5*n),floor(n))+is_seq(0,n)*is_integer(n)*py_sum(sum_arg_20,0,n)+is_seq(0,n)*dl_p**2*kap_p**2*is_integer(n)*py_sum(sum_arg_2,0,n)+is_seq(0,n)*is_integer(n)*py_sum(sum_arg_19,0,n)+is_seq(0,n)*is_integer(n)*py_sum(sum_arg_18,0,n)-py_sum(sum_arg_17,ceil(0.5*n),floor(n))+2*py_sum(sum_arg_15,ceil(0.5*n),floor(n))+2*py_sum(sum_arg_13,ceil(0.5*n),floor(n))+2*is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum(sum_arg_11,0,n)-2*is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum(sum_arg_10,0,n)+2*dl_p*py_sum(sum_arg_1,ceil(0.5*n),floor(n))+2*is_seq(0,n)*dl_p*is_integer(n)*diff(Z_coef_cp[n],'phi',1)-2*is_seq(0,n)*dl_p**2*kap_p*X_coef_cp[n]*is_integer(n))/(2*dl_p**2*kap_p)
    return(out)
