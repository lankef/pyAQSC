# Evaluating dchi B_psi_n-2. No masking needed. 
# Uses Xn, Yn, Zn,  B_theta_n, B_psi_n-3, iota_coef (n-1)/2 or (n-2)/2
# Must be evaluated with Z_coef_cp[n] = 0 and B_psi_coef_cp[n-2] = 0 
from math import floor, ceil
from math_utilities import *
import chiphifunc
def eval_dchi_B_psi_cp_nm2(n, X_coef_cp, Y_coef_cp, Z_coef_cp, \
    B_theta_coef_cp, B_psi_coef_cp, B_alpha_coef, B_denom_coef_c, \
    kap_p, dl_p, tau_p, iota_coef):    
    def sum_arg_30(i288):
        # Child args for sum_arg_30
        return(X_coef_cp[i288]*diff(Y_coef_cp[n-i288],'chi',1))
    
    def sum_arg_29(i284):
        # Child args for sum_arg_29
        return(Y_coef_cp[i284]*diff(X_coef_cp[n-i284],'chi',1))
    
    def sum_arg_28(i290):
        # Child args for sum_arg_28
        return(X_coef_cp[i290]*diff(Z_coef_cp[n-i290],'chi',1))
    
    def sum_arg_27(i286):
        # Child args for sum_arg_27
        return(Z_coef_cp[i286]*diff(X_coef_cp[n-i286],'chi',1))
    
    def sum_arg_26(i281):
        # Child args for sum_arg_26    
        def sum_arg_25(i282):
            # Child args for sum_arg_25
            return(diff(Z_coef_cp[i282],'chi',1)*diff(Z_coef_cp[(-n)-i282+2*i281],'chi',1)*is_seq(n-i281,i281-i282))
        
        return(is_seq(0,n-i281)*iota_coef[n-i281]*is_integer(n-i281)*py_sum(sum_arg_25,0,i281))
    
    def sum_arg_24(i280):
        # Child args for sum_arg_24
        return(diff(Z_coef_cp[i280],'chi',1)*diff(Z_coef_cp[n-i280],'phi',1))
    
    def sum_arg_23(i277):
        # Child args for sum_arg_23    
        def sum_arg_22(i278):
            # Child args for sum_arg_22
            return(diff(Y_coef_cp[i278],'chi',1)*diff(Y_coef_cp[(-n)-i278+2*i277],'chi',1)*is_seq(n-i277,i277-i278))
        
        return(is_seq(0,n-i277)*iota_coef[n-i277]*is_integer(n-i277)*py_sum(sum_arg_22,0,i277))
    
    def sum_arg_21(i276):
        # Child args for sum_arg_21
        return(diff(Y_coef_cp[i276],'chi',1)*diff(Y_coef_cp[n-i276],'phi',1))
    
    def sum_arg_20(i273):
        # Child args for sum_arg_20    
        def sum_arg_19(i274):
            # Child args for sum_arg_19
            return(diff(X_coef_cp[i274],'chi',1)*diff(X_coef_cp[(-n)-i274+2*i273],'chi',1)*is_seq(n-i273,i273-i274))
        
        return(is_seq(0,n-i273)*iota_coef[n-i273]*is_integer(n-i273)*py_sum(sum_arg_19,0,i273))
    
    def sum_arg_18(i272):
        # Child args for sum_arg_18
        return(diff(X_coef_cp[i272],'chi',1)*diff(X_coef_cp[n-i272],'phi',1))
    
    def sum_arg_17(i205):
        # Child args for sum_arg_17    
        def sum_arg_16(i206):
            # Child args for sum_arg_16
            return(B_theta_coef_cp[i206]*B_denom_coef_c[(-n)-i206+2*i205]*is_seq(n-i205,i205-i206))
        
        return(is_seq(0,n-i205)*B_alpha_coef[n-i205]*is_integer(n-i205)*py_sum(sum_arg_16,0,i205))
    
    def sum_arg_15(i256):
        # Child args for sum_arg_15
        return(i256*X_coef_cp[i256]*diff(Y_coef_cp[n-i256],'chi',1)+i256*diff(X_coef_cp[i256],'chi',1)*Y_coef_cp[n-i256])
    
    def sum_arg_14(i254):
        # Child args for sum_arg_14
        return(X_coef_cp[i254]*(n-i254)*diff(Y_coef_cp[n-i254],'chi',1)+diff(X_coef_cp[i254],'chi',1)*(n-i254)*Y_coef_cp[n-i254])
    
    def sum_arg_13(i269):
        # Child args for sum_arg_13    
        def sum_arg_12(i270):
            # Child args for sum_arg_12
            return(i270*Z_coef_cp[i270]*diff(Z_coef_cp[(-n)-i270+2*i269],'chi',2)*is_seq(n-i269,i269-i270)+i270*diff(Z_coef_cp[i270],'chi',1)*diff(Z_coef_cp[(-n)-i270+2*i269],'chi',1)*is_seq(n-i269,i269-i270))
        
        return(is_seq(0,n-i269)*iota_coef[n-i269]*is_integer(n-i269)*py_sum(sum_arg_12,0,i269))
    
    def sum_arg_11(i267):
        # Child args for sum_arg_11    
        def sum_arg_10(i268):
            # Child args for sum_arg_10
            return(i268*Y_coef_cp[i268]*diff(Y_coef_cp[(-n)-i268+2*i267],'chi',2)*is_seq(n-i267,i267-i268)+i268*diff(Y_coef_cp[i268],'chi',1)*diff(Y_coef_cp[(-n)-i268+2*i267],'chi',1)*is_seq(n-i267,i267-i268))
        
        return(is_seq(0,n-i267)*iota_coef[n-i267]*is_integer(n-i267)*py_sum(sum_arg_10,0,i267))
    
    def sum_arg_9(i265):
        # Child args for sum_arg_9    
        def sum_arg_8(i266):
            # Child args for sum_arg_8
            return(i266*X_coef_cp[i266]*diff(X_coef_cp[(-n)-i266+2*i265],'chi',2)*is_seq(n-i265,i265-i266)+i266*diff(X_coef_cp[i266],'chi',1)*diff(X_coef_cp[(-n)-i266+2*i265],'chi',1)*is_seq(n-i265,i265-i266))
        
        return(is_seq(0,n-i265)*iota_coef[n-i265]*is_integer(n-i265)*py_sum(sum_arg_8,0,i265))
    
    def sum_arg_7(i264):
        # Child args for sum_arg_7
        return(i264*diff(Z_coef_cp[i264],'chi',1)*diff(Z_coef_cp[n-i264],'phi',1)+i264*Z_coef_cp[i264]*diff(Z_coef_cp[n-i264],'chi',1,'phi',1))
    
    def sum_arg_6(i262):
        # Child args for sum_arg_6
        return(i262*diff(Y_coef_cp[i262],'chi',1)*diff(Y_coef_cp[n-i262],'phi',1)+i262*Y_coef_cp[i262]*diff(Y_coef_cp[n-i262],'chi',1,'phi',1))
    
    def sum_arg_5(i260):
        # Child args for sum_arg_5
        return(i260*diff(X_coef_cp[i260],'chi',1)*diff(X_coef_cp[n-i260],'phi',1)+i260*X_coef_cp[i260]*diff(X_coef_cp[n-i260],'chi',1,'phi',1))
    
    def sum_arg_4(i258):
        # Child args for sum_arg_4
        return(i258*X_coef_cp[i258]*diff(Z_coef_cp[n-i258],'chi',1)+i258*diff(X_coef_cp[i258],'chi',1)*Z_coef_cp[n-i258])
    
    def sum_arg_3(i252):
        # Child args for sum_arg_3
        return(X_coef_cp[i252]*(n-i252)*diff(Z_coef_cp[n-i252],'chi',1)+diff(X_coef_cp[i252],'chi',1)*(n-i252)*Z_coef_cp[n-i252])
    
    def sum_arg_2(i197):
        # Child args for sum_arg_2    
        def sum_arg_1(i198):
            # Child args for sum_arg_1
            return(diff(B_psi_coef_cp[i198],'chi',1)*B_denom_coef_c[(-n)-i198+2*i197+2]*is_seq(n-i197-2,i197-i198)+B_psi_coef_cp[i198]*diff(B_denom_coef_c[(-n)-i198+2*i197+2],'chi',1)*is_seq(n-i197-2,i197-i198))
        
        return(is_seq(0,n-i197-2)*B_alpha_coef[n-i197-2]*is_integer(n-i197-2)*py_sum(sum_arg_1,0,i197))
    
    
    out = ((n*(is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_30,0,n)*tau_p-is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_29,0,n)*tau_p+is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum(sum_arg_28,0,n)-is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum(sum_arg_27,0,n)-py_sum(sum_arg_26,ceil(0.5*n),floor(n))-is_seq(0,n)*is_integer(n)*py_sum(sum_arg_24,0,n)-py_sum(sum_arg_23,ceil(0.5*n),floor(n))-is_seq(0,n)*is_integer(n)*py_sum(sum_arg_21,0,n)-py_sum(sum_arg_20,ceil(0.5*n),floor(n))-is_seq(0,n)*is_integer(n)*py_sum(sum_arg_18,0,n)+py_sum(sum_arg_17,ceil(0.5*n),floor(n))-is_seq(0,n)*dl_p*is_integer(n)*diff(Z_coef_cp[n],'chi',1)))/2+(is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_15,0,n)*tau_p)/2-(is_seq(0,n)*dl_p*is_integer(n)*py_sum(sum_arg_14,0,n)*tau_p)/2+py_sum(sum_arg_9,ceil(0.5*n),floor(n))/2+(is_seq(0,n)*is_integer(n)*py_sum(sum_arg_7,0,n))/2+(is_seq(0,n)*is_integer(n)*py_sum(sum_arg_6,0,n))/2+(is_seq(0,n)*is_integer(n)*py_sum(sum_arg_5,0,n))/2+(is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum(sum_arg_4,0,n))/2-(is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum(sum_arg_3,0,n))/2-py_sum(sum_arg_2,ceil(0.5*n)-1,floor(n)-2)+py_sum(sum_arg_13,ceil(0.5*n),floor(n))/2+py_sum(sum_arg_11,ceil(0.5*n),floor(n))/2+(is_seq(0,n)*dl_p*n*is_integer(n)*diff(Z_coef_cp[n],'chi',1))/2)/(B_alpha_coef[0]*B_denom_coef_c[0])
    return(out)
