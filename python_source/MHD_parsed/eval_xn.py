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
def eval_Xn_cp(arguments):    
    def sum_arg_38(i20):
        # Child args for sum_arg_38
        return(X_coef_cp[i20]*X_coef_cp[n-i20])
    
    def sum_arg_37(i18):
        # Child args for sum_arg_37
        return(Y_coef_cp[i18]*Y_coef_cp[n-i18])
    
    def sum_arg_36(i30):
        # Child args for sum_arg_36
        return(Y_coef_cp[i30]*diff(X_coef_cp[n-i30],'phi',1))
    
    def sum_arg_35(i28):
        # Child args for sum_arg_35
        return(X_coef_cp[i28]*diff(Y_coef_cp[n-i28],'phi',1))
    
    def sum_arg_34(i25):
        # Child args for sum_arg_34    
        def sum_arg_33(i26):
            # Child args for sum_arg_33
            return(Y_coef_cp[i26]*diff(X_coef_cp[(-n)-i26+2*i25],'chi',1)*is_seq(n-i25,i25-i26))
        
        return(is_seq(0,n-i25)*iota_coef[n-i25]*is_integer(n-i25)*py_sum(sum_arg_33,0,i25))
    
    def sum_arg_32(i23):
        # Child args for sum_arg_32    
        def sum_arg_31(i24):
            # Child args for sum_arg_31
            return(X_coef_cp[i24]*diff(Y_coef_cp[(-n)-i24+2*i23],'chi',1)*is_seq(n-i23,i23-i24))
        
        return(is_seq(0,n-i23)*iota_coef[n-i23]*is_integer(n-i23)*py_sum(sum_arg_31,0,i23))
    
    def sum_arg_30(i22):
        # Child args for sum_arg_30
        return(Y_coef_cp[i22]*Z_coef_cp[n-i22])
    
    def sum_arg_29(i53):
        # Child args for sum_arg_29    
        def sum_arg_28(i54):
            # Child args for sum_arg_28    
            def sum_arg_27(i55):
                # Child args for sum_arg_27
                return((is_seq(0,n-i55-i53))\
                    *(diff(X_coef_cp[(-i55)-i54+i53],'chi',1))\
                    *(iota_coef[i55])\
                    *(diff(X_coef_cp[(-n)+i55+i54+i53],'chi',1))\
                    *(iota_coef[n-i55-i53])\
                    *(is_integer(n-i55-i53))\
                    *(is_seq(n-i55-i53,i54)))
            
            return(py_sum(sum_arg_27,0,i53-i54))
        
        return(py_sum(sum_arg_28,0,i53))
    
    def sum_arg_26(i47):
        # Child args for sum_arg_26    
        def sum_arg_25(i48):
            # Child args for sum_arg_25    
            def sum_arg_24(i49):
                # Child args for sum_arg_24
                return((is_seq(0,n-i49-i47))\
                    *(diff(Y_coef_cp[(-i49)-i48+i47],'chi',1))\
                    *(iota_coef[i49])\
                    *(diff(Y_coef_cp[(-n)+i49+i48+i47],'chi',1))\
                    *(iota_coef[n-i49-i47])\
                    *(is_integer(n-i49-i47))\
                    *(is_seq(n-i49-i47,i48)))
            
            return(py_sum(sum_arg_24,0,i47-i48))
        
        return(py_sum(sum_arg_25,0,i47))
    
    def sum_arg_23(i43):
        # Child args for sum_arg_23    
        def sum_arg_22(i44):
            # Child args for sum_arg_22    
            def sum_arg_21(i45):
                # Child args for sum_arg_21
                return((is_seq(0,n-i45-i43))\
                    *(diff(Z_coef_cp[(-i45)-i44+i43],'chi',1))\
                    *(iota_coef[i45])\
                    *(diff(Z_coef_cp[(-n)+i45+i44+i43],'chi',1))\
                    *(iota_coef[n-i45-i43])\
                    *(is_integer(n-i45-i43))\
                    *(is_seq(n-i45-i43,i44)))
            
            return(py_sum(sum_arg_21,0,i43-i44))
        
        return(py_sum(sum_arg_22,0,i43))
    
    def sum_arg_20(i9):
        # Child args for sum_arg_20    
        def sum_arg_19(i8):
            # Child args for sum_arg_19
            return(B_alpha_coef[i8]*B_alpha_coef[n-i9-i8])
        
        return(is_seq(0,2*i9-n)*B_denom_coef_c[2*i9-n]*is_integer(2*i9-n)*is_seq(2*i9-n,i9)*py_sum(sum_arg_19,0,n-i9))
    
    def sum_arg_18(i64):
        # Child args for sum_arg_18
        return(diff(X_coef_cp[i64],'phi',1)*diff(X_coef_cp[n-i64],'phi',1))
    
    def sum_arg_17(i62):
        # Child args for sum_arg_17
        return(diff(Y_coef_cp[i62],'phi',1)*diff(Y_coef_cp[n-i62],'phi',1))
    
    def sum_arg_16(i60):
        # Child args for sum_arg_16
        return(diff(Z_coef_cp[i60],'phi',1)*diff(Z_coef_cp[n-i60],'phi',1))
    
    def sum_arg_15(i57):
        # Child args for sum_arg_15    
        def sum_arg_14(i58):
            # Child args for sum_arg_14
            return(diff(X_coef_cp[i58],'phi',1)*diff(X_coef_cp[(-n)-i58+2*i57],'chi',1)*is_seq(n-i57,i57-i58))
        
        return(is_seq(0,n-i57)*iota_coef[n-i57]*is_integer(n-i57)*py_sum(sum_arg_14,0,i57))
    
    def sum_arg_13(i51):
        # Child args for sum_arg_13    
        def sum_arg_12(i52):
            # Child args for sum_arg_12
            return(diff(Y_coef_cp[i52],'phi',1)*diff(Y_coef_cp[(-n)-i52+2*i51],'chi',1)*is_seq(n-i51,i51-i52))
        
        return(is_seq(0,n-i51)*iota_coef[n-i51]*is_integer(n-i51)*py_sum(sum_arg_12,0,i51))
    
    def sum_arg_11(i42):
        # Child args for sum_arg_11
        return(Z_coef_cp[i42]*diff(X_coef_cp[n-i42],'phi',1))
    
    def sum_arg_10(i40):
        # Child args for sum_arg_10
        return(X_coef_cp[i40]*diff(Z_coef_cp[n-i40],'phi',1))
    
    def sum_arg_9(i37):
        # Child args for sum_arg_9    
        def sum_arg_8(i38):
            # Child args for sum_arg_8
            return(Z_coef_cp[i38]*diff(X_coef_cp[(-n)-i38+2*i37],'chi',1)*is_seq(n-i37,i37-i38))
        
        return(is_seq(0,n-i37)*iota_coef[n-i37]*is_integer(n-i37)*py_sum(sum_arg_8,0,i37))
    
    def sum_arg_7(i35):
        # Child args for sum_arg_7    
        def sum_arg_6(i36):
            # Child args for sum_arg_6
            return(X_coef_cp[i36]*diff(Z_coef_cp[(-n)-i36+2*i35],'chi',1)*is_seq(n-i35,i35-i36))
        
        return(is_seq(0,n-i35)*iota_coef[n-i35]*is_integer(n-i35)*py_sum(sum_arg_6,0,i35))
    
    def sum_arg_5(i255):
        # Child args for sum_arg_5    
        def sum_arg_4(i256):
            # Child args for sum_arg_4
            return(diff(Z_coef_cp[i256],'phi',1)*diff(Z_coef_cp[(-n)-i256+2*i255],'chi',1)*is_seq(n-i255,i255-i256))
        
        return(is_seq(0,n-i255)*iota_coef[n-i255]*is_integer(n-i255)*py_sum(sum_arg_4,0,i255))
    
    def sum_arg_3(i34):
        # Child args for sum_arg_3
        return(X_coef_cp[i34]*X_coef_cp[n-i34])
    
    def sum_arg_2(i32):
        # Child args for sum_arg_2
        return(Z_coef_cp[i32]*Z_coef_cp[n-i32])
    
    def sum_arg_1(i15):
        # Child args for sum_arg_1
        return(is_seq(0,n-i15)*diff(Z_coef_cp[2*i15-n],'chi',1)*iota_coef[n-i15]*is_integer(n-i15)*is_seq(n-i15,i15))
    
    
    out = ((is_seq(0,n)*dl_p**2*is_integer(n)*py_sum_parallel(sum_arg_38,0,n)+is_seq(0,n)*dl_p**2*is_integer(n)*py_sum_parallel(sum_arg_37,0,n))*tau_p**2+(2*is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_36,0,n)-2*is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_35,0,n)+2*dl_p*py_sum_parallel(sum_arg_34,ceil(n/2),floor(n))-2*dl_p*py_sum_parallel(sum_arg_32,ceil(n/2),floor(n))+2*is_seq(0,n)*dl_p**2*kap_p*is_integer(n)*py_sum_parallel(sum_arg_30,0,n))*tau_p+2*dl_p*kap_p*py_sum_parallel(sum_arg_9,ceil(n/2),floor(n))-2*dl_p*kap_p*py_sum_parallel(sum_arg_7,ceil(n/2),floor(n))+2*py_sum_parallel(sum_arg_5,ceil(n/2),floor(n))+is_seq(0,n)*dl_p**2*kap_p**2*is_integer(n)*py_sum_parallel(sum_arg_3,0,n)+py_sum_parallel(sum_arg_29,ceil(n/2),floor(n))+py_sum_parallel(sum_arg_26,ceil(n/2),floor(n))+py_sum_parallel(sum_arg_23,ceil(n/2),floor(n))-py_sum_parallel(sum_arg_20,ceil(n/2),floor(n))+is_seq(0,n)*dl_p**2*kap_p**2*is_integer(n)*py_sum_parallel(sum_arg_2,0,n)+is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_18,0,n)+is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_17,0,n)+is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_16,0,n)+2*py_sum_parallel(sum_arg_15,ceil(n/2),floor(n))+2*py_sum_parallel(sum_arg_13,ceil(n/2),floor(n))+2*is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_11,0,n)-2*is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_10,0,n)+2*dl_p*py_sum_parallel(sum_arg_1,ceil(n/2),floor(n))+2*is_seq(0,n)*dl_p*is_integer(n)*diff(Z_coef_cp[n],'phi',1)-2*is_seq(0,n)*dl_p**2*kap_p*X_coef_cp[n]*is_integer(n))/(2*dl_p**2*kap_p)
    return(out)
