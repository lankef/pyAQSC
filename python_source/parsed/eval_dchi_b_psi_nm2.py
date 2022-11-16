# Evaluating dchi B_psi_n-2. No masking needed. 
# Uses Xn-1, Yn-1, Zn-1,  B_theta_n, B_psi_n-3, iota_coef (n-1)/2 or (n-2)/2
# Must be evaluated with Z_coef_cp[n] = 0 and B_psi_coef_cp[n-2] = 0 
from math import floor, ceil
from math_utilities import *
import chiphifunc
def eval_dchi_B_psi_cp_nm2(n, X_coef_cp, Y_coef_cp, Z_coef_cp, \
    B_theta_coef_cp, B_psi_coef_cp, B_alpha_coef, B_denom_coef_c, \
    kap_p, dl_p, tau_p, iota_coef):    
    def sum_arg_30(i622):
        # Child args for sum_arg_30
        return(X_coef_cp[i622]*diff(Y_coef_cp[n-i622],'chi',1))
    
    def sum_arg_29(i618):
        # Child args for sum_arg_29
        return(Y_coef_cp[i618]*diff(X_coef_cp[n-i618],'chi',1))
    
    def sum_arg_28(i590):
        # Child args for sum_arg_28
        return(i590*X_coef_cp[i590]*diff(Y_coef_cp[n-i590],'chi',1)+i590*diff(X_coef_cp[i590],'chi',1)*Y_coef_cp[n-i590])
    
    def sum_arg_27(i588):
        # Child args for sum_arg_27
        return((X_coef_cp[i588]*n-i588*X_coef_cp[i588])*diff(Y_coef_cp[n-i588],'chi',1)+(diff(X_coef_cp[i588],'chi',1)*n-i588*diff(X_coef_cp[i588],'chi',1))*Y_coef_cp[n-i588])
    
    def sum_arg_26(i624):
        # Child args for sum_arg_26
        return(X_coef_cp[i624]*diff(Z_coef_cp[n-i624],'chi',1))
    
    def sum_arg_25(i620):
        # Child args for sum_arg_25
        return(Z_coef_cp[i620]*diff(X_coef_cp[n-i620],'chi',1))
    
    def sum_arg_24(i615):
        # Child args for sum_arg_24    
        def sum_arg_23(i616):
            # Child args for sum_arg_23
            return(diff(Z_coef_cp[i616],'chi',1)*diff(Z_coef_cp[(-n)-i616+2*i615],'chi',1)*is_seq(n-i615,i615-i616))
        
        return(is_seq(0,n-i615)*iota_coef[n-i615]*is_integer(n-i615)*py_sum(sum_arg_23,0,i615))
    
    def sum_arg_22(i614):
        # Child args for sum_arg_22
        return(diff(Z_coef_cp[i614],'chi',1)*diff(Z_coef_cp[n-i614],'phi',1))
    
    def sum_arg_21(i611):
        # Child args for sum_arg_21    
        def sum_arg_20(i612):
            # Child args for sum_arg_20
            return(diff(Y_coef_cp[i612],'chi',1)*diff(Y_coef_cp[(-n)-i612+2*i611],'chi',1)*is_seq(n-i611,i611-i612))
        
        return(is_seq(0,n-i611)*iota_coef[n-i611]*is_integer(n-i611)*py_sum(sum_arg_20,0,i611))
    
    def sum_arg_19(i610):
        # Child args for sum_arg_19
        return(diff(Y_coef_cp[i610],'chi',1)*diff(Y_coef_cp[n-i610],'phi',1))
    
    def sum_arg_18(i607):
        # Child args for sum_arg_18    
        def sum_arg_17(i608):
            # Child args for sum_arg_17
            return(diff(X_coef_cp[i608],'chi',1)*diff(X_coef_cp[(-n)-i608+2*i607],'chi',1)*is_seq(n-i607,i607-i608))
        
        return(is_seq(0,n-i607)*iota_coef[n-i607]*is_integer(n-i607)*py_sum(sum_arg_17,0,i607))
    
    def sum_arg_16(i606):
        # Child args for sum_arg_16
        return(diff(X_coef_cp[i606],'chi',1)*diff(X_coef_cp[n-i606],'phi',1))
    
    def sum_arg_15(i603):
        # Child args for sum_arg_15    
        def sum_arg_14(i604):
            # Child args for sum_arg_14
            return((i604*Z_coef_cp[i604]*diff(Z_coef_cp[(-n)-i604+2*i603],'chi',2)+i604*diff(Z_coef_cp[i604],'chi',1)*diff(Z_coef_cp[(-n)-i604+2*i603],'chi',1))*is_seq(n-i603,i603-i604))
        
        return(is_seq(0,n-i603)*iota_coef[n-i603]*is_integer(n-i603)*py_sum(sum_arg_14,0,i603))
    
    def sum_arg_13(i601):
        # Child args for sum_arg_13    
        def sum_arg_12(i602):
            # Child args for sum_arg_12
            return((i602*Y_coef_cp[i602]*diff(Y_coef_cp[(-n)-i602+2*i601],'chi',2)+i602*diff(Y_coef_cp[i602],'chi',1)*diff(Y_coef_cp[(-n)-i602+2*i601],'chi',1))*is_seq(n-i601,i601-i602))
        
        return(is_seq(0,n-i601)*iota_coef[n-i601]*is_integer(n-i601)*py_sum(sum_arg_12,0,i601))
    
    def sum_arg_11(i599):
        # Child args for sum_arg_11    
        def sum_arg_10(i600):
            # Child args for sum_arg_10
            return((i600*X_coef_cp[i600]*diff(X_coef_cp[(-n)-i600+2*i599],'chi',2)+i600*diff(X_coef_cp[i600],'chi',1)*diff(X_coef_cp[(-n)-i600+2*i599],'chi',1))*is_seq(n-i599,i599-i600))
        
        return(is_seq(0,n-i599)*iota_coef[n-i599]*is_integer(n-i599)*py_sum(sum_arg_10,0,i599))
    
    def sum_arg_9(i598):
        # Child args for sum_arg_9
        return(i598*diff(Z_coef_cp[i598],'chi',1)*diff(Z_coef_cp[n-i598],'phi',1)+i598*Z_coef_cp[i598]*diff(Z_coef_cp[n-i598],'chi',1,'phi',1))
    
    def sum_arg_8(i596):
        # Child args for sum_arg_8
        return(i596*diff(Y_coef_cp[i596],'chi',1)*diff(Y_coef_cp[n-i596],'phi',1)+i596*Y_coef_cp[i596]*diff(Y_coef_cp[n-i596],'chi',1,'phi',1))
    
    def sum_arg_7(i594):
        # Child args for sum_arg_7
        return(i594*diff(X_coef_cp[i594],'chi',1)*diff(X_coef_cp[n-i594],'phi',1)+i594*X_coef_cp[i594]*diff(X_coef_cp[n-i594],'chi',1,'phi',1))
    
    def sum_arg_6(i592):
        # Child args for sum_arg_6
        return(i592*X_coef_cp[i592]*diff(Z_coef_cp[n-i592],'chi',1)+i592*diff(X_coef_cp[i592],'chi',1)*Z_coef_cp[n-i592])
    
    def sum_arg_5(i586):
        # Child args for sum_arg_5
        return((X_coef_cp[i586]*n-i586*X_coef_cp[i586])*diff(Z_coef_cp[n-i586],'chi',1)+(diff(X_coef_cp[i586],'chi',1)*n-i586*diff(X_coef_cp[i586],'chi',1))*Z_coef_cp[n-i586])
    
    def sum_arg_4(i501):
        # Child args for sum_arg_4    
        def sum_arg_3(i502):
            # Child args for sum_arg_3
            return(B_theta_coef_cp[i502]*B_denom_coef_c[(-n)-i502+2*i501]*is_seq(n-i501,i501-i502))
        
        return(is_seq(0,n-i501)*B_alpha_coef[n-i501]*is_integer(n-i501)*py_sum(sum_arg_3,0,i501))
    
    def sum_arg_2(i493):
        # Child args for sum_arg_2    
        def sum_arg_1(i494):
            # Child args for sum_arg_1
            return((diff(B_psi_coef_cp[i494],'chi',1)*B_denom_coef_c[(-n)-i494+2*i493+2]+B_psi_coef_cp[i494]*diff(B_denom_coef_c[(-n)-i494+2*i493+2],'chi',1))*is_seq(n-i493-2,i493-i494))
        
        return(is_seq(0,n-i493-2)*B_alpha_coef[n-i493-2]*is_integer(n-i493-2)*py_sum(sum_arg_1,0,i493))
    
    
    out = ((is_seq(0,n)*dl_p*n*is_integer(n)*py_sum_parallel(sum_arg_30,0,n)-is_seq(0,n)*dl_p*n*is_integer(n)*py_sum_parallel(sum_arg_29,0,n)+is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_28,0,n)-is_seq(0,n)*dl_p*is_integer(n)*py_sum_parallel(sum_arg_27,0,n))*tau_p+is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_9,0,n)+is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_8,0,n)+is_seq(0,n)*is_integer(n)*py_sum_parallel(sum_arg_7,0,n)+is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_6,0,n)-is_seq(0,n)*dl_p*kap_p*is_integer(n)*py_sum_parallel(sum_arg_5,0,n)+n*py_sum_parallel(sum_arg_4,ceil(n/2),floor(n))+is_seq(0,n)*dl_p*kap_p*n*is_integer(n)*py_sum_parallel(sum_arg_26,0,n)-is_seq(0,n)*dl_p*kap_p*n*is_integer(n)*py_sum_parallel(sum_arg_25,0,n)-n*py_sum_parallel(sum_arg_24,ceil(n/2),floor(n))-is_seq(0,n)*n*is_integer(n)*py_sum_parallel(sum_arg_22,0,n)-n*py_sum_parallel(sum_arg_21,ceil(n/2),floor(n))-2*py_sum_parallel(sum_arg_2,ceil(n/2)-1,floor(n)-2)-is_seq(0,n)*n*is_integer(n)*py_sum_parallel(sum_arg_19,0,n)-n*py_sum_parallel(sum_arg_18,ceil(n/2),floor(n))-is_seq(0,n)*n*is_integer(n)*py_sum_parallel(sum_arg_16,0,n)+py_sum_parallel(sum_arg_15,ceil(n/2),floor(n))+py_sum_parallel(sum_arg_13,ceil(n/2),floor(n))+py_sum_parallel(sum_arg_11,ceil(n/2),floor(n)))/(2*B_alpha_coef[0]*B_denom_coef_c[0])
    return(out)
